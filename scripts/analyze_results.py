#!/usr/bin/env python3
"""
Analyze evaluation results and generate human-readable reports.

Supports both GAIA and HLE datasets, legacy (string) and structured (dict) step formats,
and surfaces adaptive agent features (diagnose_subagent, modify_subagent).

Usage:
    # Terminal summary
    python scripts/analyze_results.py workdir/.../dra.jsonl

    # HTML report
    python scripts/analyze_results.py workdir/.../dra.jsonl --html

    # With explicit config for richer metadata
    python scripts/analyze_results.py workdir/.../dra.jsonl --html --config configs/config_gaia_adaptive_qwen.py

    # Per-question terminal detail
    python scripts/analyze_results.py workdir/.../dra.jsonl --detail
"""

import argparse
import json
import re
import sys
import os
import html as html_mod
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

try:
    from src.metric import question_scorer, arc_question_scorer, get_scorer
except ImportError:
    def question_scorer(pred, truth):
        return str(pred).strip().lower() == str(truth).strip().lower()
    arc_question_scorer = question_scorer
    def get_scorer(dataset_type="gaia"):
        if dataset_type == "arc":
            return arc_question_scorer
        return question_scorer

ADAPTIVE_TOOLS = {"diagnose_subagent", "modify_subagent"}
AGENT_TOOLS = {"deep_researcher_agent", "deep_analyzer_agent", "browser_use_agent"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(filepath: str) -> List[Dict]:
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# Metadata extraction from log.txt and config files
# ---------------------------------------------------------------------------

def parse_log_config(workdir: str) -> Dict[str, Any]:
    """Parse the full config block dumped at the top of log.txt."""
    log_path = os.path.join(workdir, "log.txt")
    if not os.path.exists(log_path):
        return {}

    meta: Dict[str, Any] = {"log_file": log_path}
    try:
        meta["log_lines"] = sum(1 for _ in open(log_path, errors="ignore"))
    except Exception:
        pass

    config_lines = []
    capturing = False
    with open(log_path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 500:
                break
            stripped = line.rstrip()
            if "| Config:" in stripped:
                capturing = True
                continue
            if capturing:
                if stripped and not stripped.startswith("["):
                    config_lines.append(stripped)
                else:
                    if config_lines:
                        capturing = False

    config_text = "\n".join(config_lines)
    meta["config_text"] = config_text

    agent_configs = {}
    current_key = None
    current_lines = []
    for line in config_lines:
        m = re.match(r"^(\w+_config)\s*=\s*dict\(", line)
        if m:
            if current_key and current_lines:
                agent_configs[current_key] = "\n".join(current_lines)
            current_key = m.group(1)
            current_lines = [line]
            continue
        if current_key:
            current_lines.append(line)
            if line.strip() == ")":
                agent_configs[current_key] = "\n".join(current_lines)
                current_key = None
                current_lines = []

    hierarchy = {}
    for key, block in agent_configs.items():
        info: Dict[str, Any] = {}
        for pat, field in [
            (r"name='([^']+)'", "name"),
            (r"type='([^']+)'", "type"),
            (r"model_id='([^']+)'", "model_id"),
            (r"max_steps=(\d+)", "max_steps"),
            (r"description='([^']*)'", "description"),
        ]:
            m_f = re.search(pat, block)
            if m_f:
                info[field] = m_f.group(1)
        tools_m = re.search(r"tools=\[([^\]]*)\]", block)
        if tools_m:
            info["tools"] = [t.strip().strip("'\"") for t in tools_m.group(1).split(",") if t.strip()]
        agents_m = re.search(r"managed_agents=\[([^\]]*)\]", block)
        if agents_m:
            info["managed_agents"] = [a.strip().strip("'\"") for a in agents_m.group(1).split(",") if a.strip()]
        if info:
            hierarchy[key] = info

    meta["agent_hierarchy"] = hierarchy

    for pat, field in [
        (r"split='([^']+)'", "dataset_split"),
        (r"name='([^']+)'.*?path=", "dataset_name"),
        (r"path='([^']+)'", "dataset_path"),
        (r"type='([^']+)'.*?name=", "dataset_type"),
        (r"per_question_timeout_secs\s*=\s*(\d+)", "per_question_timeout"),
        (r"concurrency\s*=\s*(\d+)", "concurrency"),
    ]:
        m_f = re.search(pat, config_text)
        if m_f:
            meta[field] = m_f.group(1)

    return meta


def load_config_metadata(config_path: Optional[str], workdir: str) -> Dict[str, Any]:
    meta = parse_log_config(workdir)

    if config_path and os.path.exists(config_path):
        meta["config_file"] = config_path
        with open(config_path) as f:
            cfg_text = f.read()
        for pat, key in [
            (r'model_id\s*=\s*["\']([^"\']+)', "model_id"),
            (r'split\s*=\s*["\']([^"\']+)', "dataset_split"),
            (r'per_question_timeout_secs\s*=\s*(\d+)', "per_question_timeout"),
        ]:
            m = re.search(pat, cfg_text)
            if m and key not in meta:
                meta[key] = m.group(1)

    slurm_logs = list(Path(workdir).parent.parent.glob("logs/combined_*.out"))
    for slurm_log in sorted(slurm_logs, reverse=True)[:3]:
        try:
            with open(slurm_log, "r", errors="ignore") as f:
                head = f.read(3000)
            m = re.search(r"--model\s+(\S+)", head)
            if m:
                meta["vllm_model_path"] = m.group(1)
            m = re.search(r"--served-model-name\s+(\S+)", head)
            if m:
                meta["vllm_served_name"] = m.group(1)
            if "vllm_model_path" in meta:
                break
        except Exception:
            pass

    return meta


# ---------------------------------------------------------------------------
# Step parsing — supports both legacy string format and structured dict
# ---------------------------------------------------------------------------

def parse_step(step) -> Dict[str, Any]:
    """Parse a step from either dict (new) or string (legacy) format."""
    if isinstance(step, dict):
        return _parse_dict_step(step)
    return _parse_string_step(str(step))


def _parse_dict_step(d: Dict) -> Dict[str, Any]:
    """Parse a structured dict step (from step.dict())."""
    step_type = d.get("_step_type", "")
    parsed: Dict[str, Any] = {"raw_dict": d}

    if step_type == "TaskStep" or "task" in d and "step_number" not in d:
        parsed["type"] = "task"
        parsed["task_preview"] = str(d.get("task", ""))
        return parsed

    if step_type == "PlanningStep" or "plan" in d:
        parsed["type"] = "planning"
        parsed["plan"] = d.get("plan", "")
        timing = d.get("timing", {})
        parsed["duration_s"] = round(timing.get("duration", 0), 1) if isinstance(timing, dict) else None
        token_usage = d.get("token_usage")
        if token_usage:
            parsed["token_usage"] = token_usage
        return parsed

    parsed["type"] = "action"
    parsed["step_number"] = d.get("step_number")
    timing = d.get("timing", {})
    if isinstance(timing, dict):
        parsed["duration_s"] = round(timing.get("duration", 0), 1)
    parsed["is_final_answer"] = d.get("is_final_answer", False)

    tool_calls = []
    for tc in d.get("tool_calls", []):
        func = tc.get("function", {})
        tool_calls.append({
            "name": func.get("name", tc.get("name", "?")),
            "arguments": func.get("arguments", tc.get("arguments", {})),
            "id": tc.get("id", ""),
        })
    parsed["tool_calls"] = tool_calls

    parsed["observations"] = d.get("observations")
    parsed["tool_results"] = d.get("tool_results")
    parsed["model_output"] = d.get("model_output")

    msg = d.get("model_output_message")
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(item.get("text", "") for item in content if isinstance(item, dict))
        parsed["model_content"] = str(content) if content else None

    err = d.get("error")
    if err:
        parsed["error"] = str(err.get("message", err)) if isinstance(err, dict) else str(err)

    token_usage = d.get("token_usage")
    if token_usage:
        parsed["token_usage"] = token_usage

    return parsed


def _parse_string_step(step_str: str) -> Dict[str, Any]:
    """Parse a legacy stringified ActionStep/TaskStep."""
    parsed: Dict[str, Any] = {"raw_str": step_str}

    if step_str.startswith("TaskStep("):
        parsed["type"] = "task"
        m = re.search(r"task='(.*?)', task_images=", step_str, re.DOTALL)
        if m:
            parsed["task_preview"] = m.group(1)
        return parsed

    if step_str.startswith("PlanningStep("):
        parsed["type"] = "planning"
        m = re.search(r"plan='(.*?)'(?:, timing=)", step_str, re.DOTALL)
        if m:
            parsed["plan"] = m.group(1)
        m = re.search(r"duration=([\d.]+)", step_str)
        if m:
            parsed["duration_s"] = round(float(m.group(1)), 1)
        return parsed

    parsed["type"] = "action"

    m = re.search(r"step_number=(\d+)", step_str)
    if m:
        parsed["step_number"] = int(m.group(1))

    m = re.search(r"duration=([\d.]+)", step_str)
    if m:
        parsed["duration_s"] = round(float(m.group(1)), 1)

    tool_calls = []
    for m in re.finditer(r"ToolCall\(name='([^']+)',\s*arguments=(\{[^}]*\})", step_str):
        name = m.group(1)
        try:
            args = eval(m.group(2))
        except Exception:
            args = m.group(2)
        tool_calls.append({"name": name, "arguments": args})
    parsed["tool_calls"] = tool_calls

    obs_match = re.search(r"observations='(.*?)'(?:, observations_images=)", step_str, re.DOTALL)
    if obs_match:
        parsed["observations"] = obs_match.group(1)
    else:
        obs_match = re.search(r'observations="(.*?)"(?:, observations_images=)', step_str, re.DOTALL)
        if obs_match:
            parsed["observations"] = obs_match.group(1)

    content_match = re.search(r"model_output='(.*?)'(?:, observations=)", step_str, re.DOTALL)
    if content_match:
        parsed["model_output"] = content_match.group(1)

    m = re.search(r"content='(.*?)'(?:, tool_calls=)", step_str[:5000])
    if m and m.group(1):
        parsed["model_content"] = m.group(1)

    m = re.search(r"error=AgentParsingError\('([^']*)'", step_str)
    if not m:
        m = re.search(r"error=AgentError\('([^']*)'", step_str)
    if m:
        parsed["error"] = m.group(1)

    parsed["is_final_answer"] = "is_final_answer=True" in step_str

    return parsed


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_result(r: Dict, scorer=None) -> str:
    if scorer is None:
        scorer = question_scorer
    if r.get("agent_error"):
        err = str(r["agent_error"])
        err_lower = err.lower()
        if any(kw in err_lower for kw in ("connection refused", "connection reset",
                "connectionerror", "connection error", "remotedisconnected")):
            return "error_connection"
        if "maximum context length" in err or "32768 tok" in err or "context length" in err_lower:
            return "error_context_length"
        if "timeout" in err_lower or "timeouterror" in err_lower:
            return "error_timeout"
        return "error_other"
    pred = str(r.get("prediction", "") or "")
    if pred in ("None", "", "Unable to determine"):
        return "no_answer"
    truth = str(r.get("true_answer", ""))
    if truth == "?":
        return "unscored"
    if scorer(pred, truth):
        return "correct"
    return "wrong"


# ---------------------------------------------------------------------------
# Adaptive features analysis
# ---------------------------------------------------------------------------

def analyze_adaptive_usage(results: List[Dict]) -> Dict[str, Any]:
    """Analyze usage of adaptive tools across all questions."""
    stats: Dict[str, Any] = {
        "diagnose_count": 0,
        "modify_count": 0,
        "diagnose_questions": 0,
        "modify_questions": 0,
        "modify_actions": Counter(),
        "diagnose_agents": Counter(),
        "modify_agents": Counter(),
        "diagnose_success": 0,
        "diagnose_fail": 0,
        "modify_success": 0,
        "modify_fail": 0,
    }

    for r in results:
        q_has_diag = False
        q_has_mod = False
        for step in r.get("intermediate_steps", []):
            parsed = parse_step(step)
            for tc in parsed.get("tool_calls", []):
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                if name == "diagnose_subagent":
                    stats["diagnose_count"] += 1
                    q_has_diag = True
                    agent_name = args.get("agent_name", "?") if isinstance(args, dict) else "?"
                    stats["diagnose_agents"][agent_name] += 1
                elif name == "modify_subagent":
                    stats["modify_count"] += 1
                    q_has_mod = True
                    action = args.get("action", "?") if isinstance(args, dict) else "?"
                    agent_name = args.get("agent_name", "?") if isinstance(args, dict) else "?"
                    stats["modify_actions"][action] += 1
                    stats["modify_agents"][agent_name] += 1

            obs = parsed.get("observations", "") or ""
            for tc in parsed.get("tool_calls", []):
                if tc.get("name") == "diagnose_subagent":
                    if "Error:" in obs:
                        stats["diagnose_fail"] += 1
                    else:
                        stats["diagnose_success"] += 1
                elif tc.get("name") == "modify_subagent":
                    if "Error:" in obs or "error" in obs.lower()[:50]:
                        stats["modify_fail"] += 1
                    else:
                        stats["modify_success"] += 1

        if q_has_diag:
            stats["diagnose_questions"] += 1
        if q_has_mod:
            stats["modify_questions"] += 1

    return stats


# ---------------------------------------------------------------------------
# Auto-detect dataset type
# ---------------------------------------------------------------------------

def detect_dataset_info(results: List[Dict], meta: Dict) -> Dict[str, str]:
    """Auto-detect dataset type and category labels."""
    info = {"dataset": meta.get("dataset_type", "unknown"), "category_label": "Category"}

    task_values = [r.get("task") for r in results if r.get("task") is not None]
    if task_values:
        try:
            numeric = all(str(t).isdigit() for t in task_values)
            if numeric:
                info["category_label"] = "Level"
            else:
                info["category_label"] = "Category"
        except Exception:
            pass

    if any("gaia" in str(meta.get(k, "")).lower() for k in ("dataset_type", "dataset_name", "config_file")):
        info["dataset"] = "GAIA"
        info["category_label"] = "Level"
    elif any("hle" in str(meta.get(k, "")).lower() for k in ("dataset_type", "dataset_name", "config_file")):
        info["dataset"] = "HLE"
    elif any("arc" in str(meta.get(k, "")).lower() for k in ("dataset_type", "dataset_name", "config_file")):
        info["dataset"] = "ARC-AGI"
        info["category_label"] = "Split"

    # Data-level fallback: detect ARC from grid-shaped true_answer
    if info["dataset"] in ("unknown", None):
        for r in results[:5]:
            truth = str(r.get("true_answer", "")).strip()
            if truth.startswith("[[") and truth.endswith("]]"):
                info["dataset"] = "ARC-AGI"
                info["category_label"] = "Split"
                break

    return info


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def terminal_report(results: List[Dict], meta: Dict[str, Any], scorer=None) -> str:
    classified = [classify_result(r, scorer=scorer) for r in results]
    counts = Counter(classified)
    total = len(results)
    ds_info = detect_dataset_info(results, meta)
    cat_label = ds_info["category_label"]

    n_correct = counts.get("correct", 0)
    n_wrong = counts.get("wrong", 0)
    n_no_answer = counts.get("no_answer", 0)
    n_unscored = counts.get("unscored", 0)
    n_errors = sum(v for k, v in counts.items() if k.startswith("error"))
    scorable = n_correct + n_wrong + n_no_answer

    lines = []
    lines.append("=" * 72)
    lines.append("  EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append("")

    lines.append("  METADATA")
    lines.append("  " + "-" * 40)
    if meta.get("config_file"):
        lines.append(f"  Config:         {meta['config_file']}")
    if meta.get("vllm_model_path"):
        served = meta.get("vllm_served_name", "")
        lines.append(f"  Model:          {meta['vllm_model_path']} (served as \"{served}\" via local vLLM)")
    elif meta.get("model_id"):
        lines.append(f"  Model:          {meta['model_id']}")
    lines.append(f"  Dataset:        {ds_info['dataset']}")
    if meta.get("dataset_split"):
        lines.append(f"  Split:          {meta['dataset_split']}")
    if meta.get("per_question_timeout"):
        lines.append(f"  Per-Q timeout:  {meta['per_question_timeout']}s")
    if meta.get("log_lines"):
        lines.append(f"  Log:            {meta.get('log_file','')} ({meta['log_lines']} lines)")

    hierarchy = meta.get("agent_hierarchy", {})
    if hierarchy:
        lines.append("")
        lines.append("  AGENT HIERARCHY")
        lines.append("  " + "-" * 40)
        planning = hierarchy.get("planning_agent_config") or hierarchy.get("agent_config", {})
        if planning:
            name = planning.get("name", planning.get("type", "planning_agent"))
            lines.append(f"  {name} (max_steps={planning.get('max_steps','?')})")
            for ma in planning.get("managed_agents", []):
                ma_cfg = hierarchy.get(f"{ma}_config", {})
                ms = ma_cfg.get("max_steps", "?")
                lines.append(f"    -> {ma} (max_steps={ms})")

    if results:
        lines.append("")
        lines.append(f"  Run period:     {results[0].get('start_time','?')} -> {results[-1].get('end_time','?')}")
    lines.append("")

    lines.append("  OVERALL RESULTS")
    lines.append("  " + "-" * 40)
    lines.append(f"  Total questions:      {total}")
    lines.append(f"  Scorable:             {scorable}")
    if scorable:
        lines.append(f"  Correct:              {n_correct} / {scorable}  ({100*n_correct/scorable:.1f}%)")
    lines.append(f"  Wrong answer:         {n_wrong}")
    lines.append(f"  No answer / gave up:  {n_no_answer}")
    lines.append(f"  Errors (total):       {n_errors}")
    for k, label in [("error_connection", "Connection error"), ("error_context_length", "Context overflow"),
                     ("error_timeout", "Timeout"), ("error_other", "Other error")]:
        if counts.get(k, 0):
            lines.append(f"    - {label}: {counts[k]}")
    if n_unscored:
        lines.append(f"  Unscored (no truth):  {n_unscored}")
    n_retried = sum(1 for r in results if (r.get("attempts") or 1) > 1)
    if n_retried:
        lines.append(f"  Retried questions:    {n_retried}")
    lines.append("")

    lines.append(f"  BY {cat_label.upper()}")
    lines.append("  " + "-" * 40)
    by_cat = defaultdict(list)
    for r, c in zip(results, classified):
        by_cat[str(r.get("task", "?"))].append(c)
    for cat in sorted(by_cat):
        cats = Counter(by_cat[cat])
        t = len(by_cat[cat])
        cor = cats.get("correct", 0)
        sc = cor + cats.get("wrong", 0) + cats.get("no_answer", 0)
        err = sum(cats.get(k, 0) for k in ("error_connection", "error_context_length", "error_timeout", "error_other"))
        pct = f"{100*cor/sc:.1f}%" if sc else "N/A"
        lines.append(f"  {cat_label} {cat:4s}: {cor}/{sc} correct ({pct})   [{t} total, {err} errors]")
    lines.append("")

    adaptive = analyze_adaptive_usage(results)
    if adaptive["diagnose_count"] or adaptive["modify_count"]:
        lines.append("  ADAPTIVE FEATURES")
        lines.append("  " + "-" * 40)
        lines.append(f"  diagnose_subagent:  {adaptive['diagnose_count']} calls in {adaptive['diagnose_questions']} questions"
                     f"  (success={adaptive['diagnose_success']}, fail={adaptive['diagnose_fail']})")
        lines.append(f"  modify_subagent:    {adaptive['modify_count']} calls in {adaptive['modify_questions']} questions"
                     f"  (success={adaptive['modify_success']}, fail={adaptive['modify_fail']})")
        if adaptive["modify_actions"]:
            lines.append(f"  modify actions:     {dict(adaptive['modify_actions'].most_common())}")
        if adaptive["diagnose_agents"]:
            lines.append(f"  diagnosed agents:   {dict(adaptive['diagnose_agents'].most_common())}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-question terminal detail
# ---------------------------------------------------------------------------

def per_question_text(results: List[Dict], scorer=None) -> str:
    lines = []
    for i, r in enumerate(results):
        cat = classify_result(r, scorer=scorer)
        tag = {"correct": "CORRECT", "wrong": "WRONG", "no_answer": "NO ANSWER",
               "unscored": "UNSCORED"}.get(cat, "ERROR")

        lines.append(f"\n{'='*72}")
        lines.append(f"  Question {i+1}/{len(results)}  [Level {r.get('task','?')}]  [{tag}]")
        lines.append(f"  ID: {r.get('task_id','')}")
        lines.append(f"  Time: {r.get('start_time','?')} -> {r.get('end_time','?')}")
        lines.append(f"{'='*72}\n")
        lines.append(f"  QUESTION: {r['question'][:400]}")
        if "attached file" in str(r.get("augmented_question", "")):
            m = re.search(r"Attached file: (.+)", r["augmented_question"])
            if m:
                lines.append(f"  FILE: {m.group(1).strip()}")
        lines.append(f"\n  PREDICTION: {r.get('prediction','(none)')}")
        lines.append(f"  TRUTH:      {r.get('true_answer','?')}")
        if r.get("agent_error"):
            lines.append(f"  ERROR:      {str(r['agent_error'])[:200]}")
        lines.append("")

        steps = r.get("intermediate_steps", [])
        if steps:
            lines.append(f"  STEPS ({len(steps)}):")
            lines.append(f"  {'-'*40}")
            for j, raw_step in enumerate(steps):
                parsed = parse_step(raw_step)
                if parsed["type"] == "task":
                    lines.append(f"  [Task] {parsed.get('task_preview','')[:150]}")
                    continue
                if parsed["type"] == "planning":
                    lines.append(f"  [Plan] {parsed.get('plan','')[:200]}")
                    continue

                sn = parsed.get("step_number", j)
                dur = parsed.get("duration_s", "?")
                lines.append(f"  Step {sn} ({dur}s):")

                if parsed.get("model_output"):
                    lines.append(f"    [Reasoning] {parsed['model_output'][:200]}")

                for tc in parsed.get("tool_calls", []):
                    name = tc["name"]
                    is_adaptive = name in ADAPTIVE_TOOLS
                    prefix = "** " if is_adaptive else ""
                    args = tc.get("arguments", {})
                    if isinstance(args, dict):
                        arg_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
                    else:
                        arg_str = str(args)[:100]
                    lines.append(f"    {prefix}-> {name}({arg_str})")

                obs = parsed.get("observations")
                if obs:
                    lines.append(f"    [Observation] {obs[:300]}")

                if parsed.get("error"):
                    lines.append(f"    !! Error: {parsed['error']}")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _build_hierarchy_mermaid(meta: Dict) -> str:
    """Build a mermaid flowchart string from agent hierarchy."""
    hierarchy = meta.get("agent_hierarchy", {})
    planning = hierarchy.get("planning_agent_config") or hierarchy.get("agent_config", {})
    if not planning:
        return ""

    name = planning.get("name", planning.get("type", "planning_agent"))
    ms = planning.get("max_steps", "?")
    lines = [
        "flowchart TD",
        f'    planning["{name} (max_steps={ms})"]',
    ]
    for ma in planning.get("managed_agents", []):
        ma_cfg = hierarchy.get(f"{ma}_config", {})
        ma_ms = ma_cfg.get("max_steps", "?")
        safe_id = ma.replace("-", "_")
        lines.append(f'    planning --> {safe_id}["{ma} (max_steps={ma_ms})"]')

    lines.append('    planning --> diagnose["diagnose_subagent (adaptive)"]')
    lines.append('    planning --> modify["modify_subagent (adaptive)"]')
    return "\n".join(lines)


def _expandable(e_fn, content: str, default_collapsed: bool = True) -> str:
    """Wrap content in an expandable container. JS adds Show more/less buttons."""
    cls = "expandable-content collapsed" if default_collapsed else "expandable-content"
    return f'<div class="expandable-wrapper"><pre class="{cls}">{e_fn(content)}</pre></div>'


def generate_html(results: List[Dict], meta: Dict[str, Any], filepath: str, scorer=None) -> str:
    total = len(results)
    classified = [classify_result(r, scorer=scorer) for r in results]
    counts = Counter(classified)
    ds_info = detect_dataset_info(results, meta)
    cat_label = ds_info["category_label"]
    adaptive = analyze_adaptive_usage(results)

    n_correct = counts.get("correct", 0)
    n_wrong = counts.get("wrong", 0)
    n_no_answer = counts.get("no_answer", 0)
    n_errors = sum(v for k, v in counts.items() if k.startswith("error"))
    n_retried = sum(1 for r in results if (r.get("attempts") or 1) > 1)
    scorable = n_correct + n_wrong + n_no_answer
    pct = f"{100*n_correct/scorable:.1f}" if scorable else "0"

    e = html_mod.escape

    # Count adaptive questions for filter button
    n_adaptive = 0
    for r in results:
        for step in r.get("intermediate_steps", []):
            parsed = parse_step(step)
            if any(tc.get("name") in ADAPTIVE_TOOLS for tc in parsed.get("tool_calls", [])):
                n_adaptive += 1
                break

    # --- Metadata table ---
    meta_rows = ""
    if meta.get("config_file"):
        meta_rows += f'<tr><td>Config</td><td><code>{e(meta["config_file"])}</code></td></tr>'
    if meta.get("vllm_model_path"):
        served = meta.get("vllm_served_name", "")
        meta_rows += f'<tr><td>Model</td><td><code>{e(meta["vllm_model_path"])}</code> (served as "{e(served)}" via local vLLM)</td></tr>'
    elif meta.get("model_id"):
        meta_rows += f'<tr><td>Model</td><td><code>{e(meta["model_id"])}</code></td></tr>'

    hierarchy = meta.get("agent_hierarchy", {})
    planning = hierarchy.get("planning_agent_config") or hierarchy.get("agent_config", {})
    if planning:
        pname = planning.get("name", planning.get("type", "?"))
        ptype = planning.get("type", "?")
        meta_rows += f'<tr><td>Agent</td><td><code>{e(pname)}</code> (type: {e(ptype)}, max_steps={planning.get("max_steps","?")})</td></tr>'

    meta_rows += f'<tr><td>Dataset</td><td>{e(ds_info["dataset"])}</td></tr>'
    if meta.get("dataset_split"):
        meta_rows += f'<tr><td>Split</td><td>{e(meta["dataset_split"])}</td></tr>'
    if meta.get("per_question_timeout"):
        meta_rows += f'<tr><td>Per-Q timeout</td><td>{e(str(meta["per_question_timeout"]))}s</td></tr>'
    if meta.get("concurrency"):
        meta_rows += f'<tr><td>Concurrency</td><td>{e(str(meta["concurrency"]))}</td></tr>'
    if meta.get("log_lines"):
        meta_rows += f'<tr><td>Log file</td><td>{e(str(meta.get("log_file","")))} ({meta["log_lines"]} lines)</td></tr>'
    if results:
        meta_rows += f'<tr><td>Run period</td><td>{results[0].get("start_time","?")} &rarr; {results[-1].get("end_time","?")}</td></tr>'

    # --- Sub-agent table ---
    subagent_rows = ""
    for key in sorted(hierarchy):
        cfg = hierarchy[key]
        if "planning" in key:
            continue
        name = cfg.get("name", key.replace("_config", ""))
        subagent_rows += f'<tr><td>{e(name)}</td><td>{e(cfg.get("type",""))}</td><td>{e(str(cfg.get("max_steps","?")))}</td><td>{e(", ".join(cfg.get("tools",[])))}</td></tr>'

    # --- By-level table ---
    by_level = defaultdict(lambda: {"correct": 0, "wrong": 0, "no_answer": 0, "error": 0, "total": 0})
    for r, c in zip(results, classified):
        lvl = str(r.get("task", "?"))
        by_level[lvl]["total"] += 1
        if c == "correct":
            by_level[lvl]["correct"] += 1
        elif c == "wrong":
            by_level[lvl]["wrong"] += 1
        elif c == "no_answer":
            by_level[lvl]["no_answer"] += 1
        elif c.startswith("error"):
            by_level[lvl]["error"] += 1

    level_rows = ""
    for lvl in sorted(by_level):
        s = by_level[lvl]
        sc = s["correct"] + s["wrong"] + s["no_answer"]
        p = f'{100*s["correct"]/sc:.1f}' if sc else "N/A"
        level_rows += f'<tr><td>{e(str(lvl))}</td><td>{s["correct"]}/{sc}</td><td>{p}%</td><td>{s["wrong"]}</td><td>{s["no_answer"]}</td><td>{s["error"]}</td><td>{s["total"]}</td></tr>'

    # --- Error breakdown ---
    err_cats = Counter()
    for r, c in zip(results, classified):
        if c == "error_connection":
            err_cats["Connection error (vLLM down)"] += 1
        elif c == "error_context_length":
            err_cats["Context length exceeded"] += 1
        elif c == "error_timeout":
            err_cats["Timeout"] += 1
        elif c == "error_other":
            err_cats[str(r.get("agent_error", ""))[:80]] += 1
    error_items = "".join(f"<li><b>x{cnt}</b>: {e(msg)}</li>" for msg, cnt in err_cats.most_common())
    error_html = f'<ul>{error_items}</ul>' if error_items else "<p>No errors.</p>"

    # --- Adaptive summary ---
    adaptive_html = ""
    if adaptive["diagnose_count"] or adaptive["modify_count"]:
        mod_actions = ", ".join(f"{k}: {v}" for k, v in adaptive["modify_actions"].most_common())
        diag_agents = ", ".join(f"{k}: {v}" for k, v in adaptive["diagnose_agents"].most_common())
        adaptive_html = f'''
        <h2>Adaptive Features</h2>
        <p>The adaptive planning agent can diagnose sub-agent failures and modify sub-agents at runtime.</p>
        <div class="cards">
            <div class="stat-card purple"><div class="num">{adaptive["diagnose_count"]}</div><div class="label">diagnose calls</div></div>
            <div class="stat-card purple"><div class="num">{adaptive["modify_count"]}</div><div class="label">modify calls</div></div>
            <div class="stat-card"><div class="num">{adaptive["diagnose_questions"]}</div><div class="label">questions w/ diagnose</div></div>
            <div class="stat-card"><div class="num">{adaptive["modify_questions"]}</div><div class="label">questions w/ modify</div></div>
        </div>
        <table>
            <tr><th>Metric</th><th>diagnose_subagent</th><th>modify_subagent</th></tr>
            <tr><td>Success</td><td>{adaptive["diagnose_success"]}</td><td>{adaptive["modify_success"]}</td></tr>
            <tr><td>Fail / Error</td><td>{adaptive["diagnose_fail"]}</td><td>{adaptive["modify_fail"]}</td></tr>
        </table>
        {"<p><b>Modify actions:</b> " + e(mod_actions) + "</p>" if mod_actions else ""}
        {"<p><b>Diagnosed agents:</b> " + e(diag_agents) + "</p>" if diag_agents else ""}
        '''

    # --- Agent hierarchy mermaid ---
    mermaid_code = _build_hierarchy_mermaid(meta)
    hierarchy_html = ""
    if mermaid_code:
        hierarchy_html = f'''
        <h2>Agent Hierarchy</h2>
        <p>Each "step" below is one step of the <b>planning agent</b> (max_steps={planning.get("max_steps","?")}).
           Sub-agent calls appear as tool calls within a step. The sub-agent's internal steps are not shown individually;
           their report is returned as the observation.</p>
        <div class="mermaid">{e(mermaid_code)}</div>
        '''
        if subagent_rows:
            hierarchy_html += f'''
            <table><tr><th>Sub-agent</th><th>Type</th><th>Max steps</th><th>Tools</th></tr>{subagent_rows}</table>
            '''

    # --- Per-question cards ---
    q_rows = []
    for i, (r, cat) in enumerate(zip(results, classified)):
        tag_class = {"correct": "correct", "wrong": "wrong", "no_answer": "noanswer"}.get(cat, "error")
        tag_label = {"correct": "CORRECT", "wrong": "WRONG", "no_answer": "NO ANSWER",
                     "unscored": "UNSCORED"}.get(cat, "ERROR")

        steps = r.get("intermediate_steps", [])
        step_items = []
        q_has_adaptive = False
        q_token_total = 0

        for j, raw_step in enumerate(steps):
            parsed = parse_step(raw_step)

            # --- TaskStep ---
            if parsed["type"] == "task":
                task_text = parsed.get("task_preview", "")
                if task_text:
                    step_items.append(
                        f'<div class="step task-step"><div class="step-header">Task (system prompt)</div>'
                        + _expandable(e, task_text) + '</div>')
                else:
                    step_items.append('<div class="step task-step"><div class="step-header">Task</div><p class="empty-note">No task text captured.</p></div>')
                continue

            # --- PlanningStep ---
            if parsed["type"] == "planning":
                plan_text = parsed.get("plan", "")
                dur = parsed.get("duration_s", "?")
                step_items.append(
                    f'<div class="step plan-step"><div class="step-header">Planning step <span class="duration">({dur}s)</span></div>'
                    + _expandable(e, plan_text) + '</div>')
                continue

            # --- ActionStep ---
            sn = parsed.get("step_number", j)
            dur = parsed.get("duration_s", "?")

            has_adaptive_tc = any(tc.get("name") in ADAPTIVE_TOOLS for tc in parsed.get("tool_calls", []))
            if has_adaptive_tc:
                q_has_adaptive = True
            step_cls = "adaptive-step" if has_adaptive_tc else ""
            final_cls = " final-step" if parsed.get("is_final_answer") else ""

            tu = parsed.get("token_usage")
            tu_html = ""
            if tu and isinstance(tu, dict):
                inp = tu.get("input_tokens") or tu.get("prompt_tokens", 0)
                out = tu.get("output_tokens") or tu.get("completion_tokens", 0)
                q_token_total += (inp or 0) + (out or 0)
                tu_html = f'<span class="token-badge" title="input: {inp}, output: {out}">tokens: {inp}+{out}</span>'

            # -- Model content (actual LLM reasoning text, not tool call summary) --
            model_content_html = ""
            model_content = parsed.get("model_content") or ""
            if model_content and model_content.strip():
                model_content_html = (
                    '<div class="step-section model-content-section"><div class="section-label">Model Content</div>'
                    + _expandable(e, model_content) + '</div>')

            # -- Tool calls section --
            tool_calls = parsed.get("tool_calls", [])
            tc_section_html = ""
            if tool_calls:
                tc_items = []
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc.get("arguments", {})
                    is_adaptive = name in ADAPTIVE_TOOLS
                    is_agent = name in AGENT_TOOLS
                    tool_class = "adaptive-tool" if is_adaptive else ("agent-tool" if is_agent else "")

                    badge_html = ""
                    if is_adaptive:
                        badge_html = ' <span class="adaptive-badge">ADAPTIVE</span>'
                    elif is_agent:
                        badge_html = ' <span class="agent-badge">AGENT</span>'

                    tc_body_parts = []

                    if is_agent and isinstance(args, dict) and "task" in args:
                        agent_msg = args["task"]
                        other_args = {k: v for k, v in args.items() if k != "task"}
                        tc_body_parts.append(
                            '<div class="section-label">Message to agent</div>'
                            + _expandable(e, str(agent_msg)))
                        if other_args:
                            other_fmt = json.dumps(other_args, indent=2, ensure_ascii=False, default=str)
                            tc_body_parts.append(
                                '<div class="section-label">Other arguments</div>'
                                + _expandable(e, other_fmt))
                    else:
                        if isinstance(args, dict):
                            args_fmt = json.dumps(args, indent=2, ensure_ascii=False, default=str)
                        else:
                            args_fmt = str(args)
                        tc_body_parts.append(
                            '<div class="section-label">Arguments</div>'
                            + _expandable(e, args_fmt))

                    tc_items.append(
                        f'<div class="tool-call-item {tool_class}">'
                        f'<div class="tool-call-header"><span class="tool-name">{e(name)}</span>{badge_html}</div>'
                        f'<div class="tool-call-body">{"".join(tc_body_parts)}</div>'
                        f'</div>')

                tc_section_html = '<div class="step-section tool-calls-section">' + ''.join(tc_items) + '</div>'

            # -- Observation / Result --
            obs_html = ""
            tool_results_list = parsed.get("tool_results") or []
            if tool_results_list and len(tool_results_list) > 1:
                obs_parts = []
                for tr in tool_results_list:
                    tr_id = tr.get("id", "")
                    tr_name = ""
                    for tc in tool_calls:
                        if tc.get("id") == tr_id:
                            tr_name = tc.get("name", "")
                            break
                    tr_label = f"Result ({tr_name})" if tr_name else f"Result ({tr_id[:12]})"
                    obs_parts.append(
                        f'<div class="step-section observation-section">'
                        f'<div class="section-label">{e(tr_label)}</div>'
                        + _expandable(e, tr.get("content", "")) + '</div>')
                obs_html = "\n".join(obs_parts)
            else:
                obs = parsed.get("observations") or ""
                if obs:
                    n_tools = len(tool_calls)
                    obs_label = "Result" if n_tools <= 1 else "Combined Result"
                    obs_html = (
                        f'<div class="step-section observation-section">'
                        f'<div class="section-label">{obs_label}</div>'
                        + _expandable(e, obs) + '</div>')

            # -- Error --
            err_html = ""
            if parsed.get("error"):
                err_html = f'<div class="step-error">Step error: {e(parsed["error"])}</div>'

            step_items.append(
                f'<div class="step {step_cls}{final_cls}">'
                f'<div class="step-header">Step {sn} <span class="duration">({dur}s)</span> {tu_html}</div>'
                f'{model_content_html}{tc_section_html}{obs_html}{err_html}'
                f'</div>')

        steps_html = "\n".join(step_items)

        file_html = ""
        if "attached file" in str(r.get("augmented_question", "")):
            m = re.search(r"Attached file: (.+)", r["augmented_question"])
            if m:
                file_html = f'<div class="file-ref">File: <code>{e(m.group(1).strip())}</code></div>'

        error_card_html = ""
        if r.get("agent_error"):
            error_card_html = f'<div class="agent-error"><b>Error:</b> {e(str(r["agent_error"]))}</div>'

        token_summary = ""
        if q_token_total:
            token_summary = f'<span class="token-badge">Total tokens: {q_token_total:,}</span>'

        adaptive_badge = ' <span class="adaptive-badge">ADAPTIVE</span>' if q_has_adaptive else ""
        data_adaptive = ' data-adaptive="true"' if q_has_adaptive else ""

        q_rows.append(f'''
        <div class="question-card {tag_class}" id="q{i}" data-question="{e(r['question'][:200].lower())}"{data_adaptive}>
            <div class="q-header" onclick="this.parentElement.classList.toggle('expanded')">
                <span class="q-num">#{i+1}</span>
                <span class="q-level">{cat_label[0]}{r.get("task","?")}</span>
                <span class="q-tag {tag_class}">{tag_label}</span>
                <span class="q-text">{e(r["question"][:100])}{adaptive_badge}</span>
                <span class="q-time">{r.get("start_time","")}</span>
            </div>
            <div class="q-body">
                <div class="q-full-text"><b>Question:</b><pre class="question-text">{e(r["question"])}</pre></div>
                {file_html}
                <div class="q-answer">
                    <div><b>Prediction:</b> <span class="pred">{e(str(r.get("prediction","(none)")))}</span></div>
                    <div><b>Truth:</b> <span class="truth">{e(str(r.get("true_answer","?")))}</span></div>
                    {token_summary}
                </div>
                {error_card_html}
                <div class="steps-container">
                    <b>Steps ({len(steps)}):</b>
                    {steps_html}
                </div>
            </div>
        </div>''')

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    adaptive_filter_btn = ""
    if n_adaptive:
        adaptive_filter_btn = f'<button class="filter-btn" onclick="filterQ(\'adaptive\')">Adaptive ({n_adaptive})</button>'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Evaluation Report — {e(ds_info["dataset"])}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; line-height: 1.5; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 1.6em; margin-bottom: 6px; }}
h2 {{ font-size: 1.2em; margin: 24px 0 10px; color: #444; border-bottom: 2px solid #e5e7eb; padding-bottom: 4px; }}
h3 {{ font-size: 1em; margin: 10px 0 6px; }}
.report-time {{ color: #888; font-size: 0.85em; margin-bottom: 20px; }}
code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; }}

.cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }}
.stat-card {{ background: white; border-radius: 8px; padding: 16px 20px; min-width: 130px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.stat-card .num {{ font-size: 2em; font-weight: 700; }}
.stat-card .label {{ font-size: 0.8em; color: #777; text-transform: uppercase; }}
.stat-card.green .num {{ color: #16a34a; }}
.stat-card.red .num {{ color: #dc2626; }}
.stat-card.amber .num {{ color: #d97706; }}
.stat-card.blue .num {{ color: #2563eb; }}
.stat-card.purple .num {{ color: #7c3aed; }}

table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 10px 0; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 0.9em; }}
th {{ background: #f9fafb; font-weight: 600; }}

.controls {{ margin: 16px 0; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
.filter-btn {{ padding: 6px 14px; border: 1px solid #ddd; border-radius: 20px; background: white; cursor: pointer; font-size: 0.85em; transition: all .15s; }}
.filter-btn:hover {{ background: #f0f0f0; }}
.filter-btn.active {{ background: #2563eb; color: white; border-color: #2563eb; }}
#searchBox {{ padding: 6px 12px; border: 1px solid #ddd; border-radius: 20px; font-size: 0.85em; width: 220px; }}
.expand-btns {{ margin-left: auto; }}
.expand-btns button {{ padding: 4px 10px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer; font-size: 0.8em; margin-left: 4px; }}

.question-card {{ background: white; border-radius: 8px; margin: 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden; border-left: 4px solid #ddd; }}
.question-card.correct {{ border-left-color: #16a34a; }}
.question-card.wrong {{ border-left-color: #dc2626; }}
.question-card.noanswer {{ border-left-color: #d97706; }}
.question-card.error {{ border-left-color: #9333ea; }}

.q-header {{ padding: 10px 16px; cursor: pointer; display: flex; align-items: center; gap: 10px; user-select: none; }}
.q-header:hover {{ background: #f9fafb; }}
.q-num {{ font-weight: 700; color: #555; min-width: 36px; }}
.q-level {{ background: #e5e7eb; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; font-weight: 600; }}
.q-tag {{ padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 700; color: white; }}
.q-tag.correct {{ background: #16a34a; }}
.q-tag.wrong {{ background: #dc2626; }}
.q-tag.noanswer {{ background: #d97706; }}
.q-tag.error {{ background: #9333ea; }}
.q-text {{ flex: 1; font-size: 0.9em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.q-time {{ font-size: 0.8em; color: #999; white-space: nowrap; }}

.adaptive-badge {{ background: #7c3aed; color: white; padding: 1px 6px; border-radius: 8px; font-size: 0.7em; font-weight: 700; vertical-align: middle; }}
.agent-badge {{ background: #059669; color: white; padding: 1px 6px; border-radius: 8px; font-size: 0.7em; font-weight: 700; vertical-align: middle; }}

.q-body {{ display: none; padding: 12px 16px; border-top: 1px solid #eee; font-size: 0.9em; }}
.question-card.expanded .q-body {{ display: block; }}
.q-full-text {{ margin-bottom: 10px; }}
.question-text {{ white-space: pre-wrap; font-family: inherit; font-size: 0.95em; background: #f9fafb; padding: 8px; border-radius: 6px; max-height: 200px; overflow-y: auto; }}
.q-answer {{ background: #f9fafb; padding: 10px 14px; border-radius: 6px; margin: 10px 0; }}
.q-answer .pred {{ font-weight: 700; color: #1e40af; }}
.q-answer .truth {{ color: #16a34a; font-weight: 600; }}
.file-ref {{ color: #2563eb; font-size: 0.85em; margin: 4px 0; }}
.agent-error {{ background: #fef2f2; padding: 8px 12px; border-radius: 6px; margin: 8px 0; color: #991b1b; font-size: 0.85em; word-wrap: break-word; }}
.token-badge {{ background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 8px; font-size: 0.75em; font-weight: 600; margin-left: 8px; }}

.steps-container {{ margin-top: 12px; }}
.step {{ background: #f9fafb; border-radius: 6px; padding: 10px 14px; margin: 8px 0; border-left: 3px solid #2563eb; }}
.step.task-step {{ border-left-color: #6b7280; }}
.step.plan-step {{ border-left-color: #0891b2; }}
.step.adaptive-step {{ border-left-color: #7c3aed; background: #faf5ff; }}
.step.final-step {{ border-left-color: #16a34a; background: #f0fdf4; }}
.step-header {{ font-weight: 600; font-size: 0.9em; margin-bottom: 6px; }}
.duration {{ color: #888; font-weight: 400; }}
.empty-note {{ color: #999; font-style: italic; font-size: 0.85em; margin: 4px 0; }}

/* Step sections */
.step-section {{ margin: 8px 0; }}
.section-label {{ font-size: 0.75em; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }}

/* Expandable content */
.expandable-wrapper {{ position: relative; }}
.expandable-content {{ font-size: 0.82em; color: #444; background: #fff; padding: 6px 10px; border-radius: 4px; border: 1px solid #e5e7eb; margin: 2px 0; }}
.expandable-content.collapsed {{ max-height: 150px; overflow: hidden; }}
.show-more-btn {{ display: inline-block; margin-top: 3px; padding: 2px 10px; font-size: 0.75em; color: #2563eb; cursor: pointer; border: 1px solid #dbeafe; border-radius: 4px; background: white; }}
.show-more-btn:hover {{ background: #eff6ff; }}

/* Tool call items */
.tool-calls-section {{ margin: 6px 0; }}
.tool-call-item {{ margin: 6px 0; border-left: 2px solid #93c5fd; padding-left: 10px; }}
.tool-call-item.adaptive-tool {{ border-left-color: #a78bfa; background: #faf5ff; border-radius: 0 4px 4px 0; padding: 6px 10px; }}
.tool-call-item.agent-tool {{ border-left-color: #6ee7b7; background: #f0fdf4; border-radius: 0 4px 4px 0; padding: 6px 10px; }}
.tool-call-header {{ margin-bottom: 4px; }}
.tool-name {{ background: #dbeafe; padding: 2px 8px; border-radius: 4px; font-family: monospace; font-size: 0.85em; font-weight: 600; }}
.tool-call-item.adaptive-tool .tool-name {{ background: #ede9fe; color: #5b21b6; }}
.tool-call-item.agent-tool .tool-name {{ background: #d1fae5; color: #065f46; }}
.tool-call-body {{ margin-top: 4px; }}

/* Observation section */
.observation-section .expandable-content {{ background: #f0fdf4; border-color: #bbf7d0; border-left: 2px solid #22c55e; }}

/* Model content section */
.model-content-section .expandable-content {{ background: #fffbeb; border-color: #fde68a; border-left: 2px solid #f59e0b; }}

.step-error {{ color: #dc2626; font-size: 0.85em; margin: 6px 0; padding: 6px 10px; background: #fef2f2; border-radius: 4px; border-left: 2px solid #dc2626; }}

.mermaid {{ background: white; padding: 16px; border-radius: 8px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <h1>Evaluation Report &mdash; {e(ds_info["dataset"])}</h1>
    <div class="report-time">Generated: {now} &nbsp;|&nbsp; Source: <code>{e(filepath)}</code></div>

    <h2>Metadata</h2>
    <table>{meta_rows}</table>

    {hierarchy_html}

    <h2>Summary</h2>
    <div class="cards">
        <div class="stat-card blue"><div class="num">{total}</div><div class="label">Total</div></div>
        <div class="stat-card green"><div class="num">{n_correct}</div><div class="label">Correct</div></div>
        <div class="stat-card red"><div class="num">{n_wrong}</div><div class="label">Wrong</div></div>
        <div class="stat-card amber"><div class="num">{n_no_answer}</div><div class="label">No Answer</div></div>
        <div class="stat-card"><div class="num">{n_errors}</div><div class="label">Errors</div></div>
        <div class="stat-card green"><div class="num">{pct}%</div><div class="label">Accuracy</div></div>
        {"" if not n_retried else f'<div class="stat-card"><div class="num">{n_retried}</div><div class="label">Retried</div></div>'}
    </div>

    <h2>Error Breakdown</h2>
    {error_html}

    <h2>Results by {e(cat_label)}</h2>
    <table>
        <tr><th>{e(cat_label)}</th><th>Correct/Scorable</th><th>Accuracy</th><th>Wrong</th><th>No Answer</th><th>Errors</th><th>Total</th></tr>
        {level_rows}
    </table>

    {adaptive_html}

    <h2>Questions</h2>
    <div class="controls">
        <button class="filter-btn active" onclick="filterQ('all')">All ({total})</button>
        <button class="filter-btn" onclick="filterQ('correct')">Correct ({n_correct})</button>
        <button class="filter-btn" onclick="filterQ('wrong')">Wrong ({n_wrong})</button>
        <button class="filter-btn" onclick="filterQ('noanswer')">No Answer ({n_no_answer})</button>
        <button class="filter-btn" onclick="filterQ('error')">Error ({n_errors})</button>
        {adaptive_filter_btn}
        <input type="text" id="searchBox" placeholder="Search questions..." oninput="searchQ(this.value)">
        <div class="expand-btns">
            <button onclick="expandAll(true)">Expand All</button>
            <button onclick="expandAll(false)">Collapse All</button>
        </div>
    </div>
    <div id="questions">
        {"".join(q_rows)}
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});

function filterQ(type) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    const search = document.getElementById('searchBox').value.toLowerCase();
    document.querySelectorAll('.question-card').forEach(card => {{
        let matchType;
        if (type === 'all') matchType = true;
        else if (type === 'adaptive') matchType = card.dataset.adaptive === 'true';
        else matchType = card.classList.contains(type);
        const matchSearch = !search || (card.getAttribute('data-question') || '').includes(search);
        card.style.display = (matchType && matchSearch) ? '' : 'none';
    }});
}}

function searchQ(query) {{
    const q = query.toLowerCase();
    const activeFilter = document.querySelector('.filter-btn.active');
    const type = activeFilter ? activeFilter.textContent.split(' ')[0].toLowerCase() : 'all';
    document.querySelectorAll('.question-card').forEach(card => {{
        const matchSearch = !q || (card.getAttribute('data-question') || '').includes(q);
        let matchType;
        if (type === 'all') matchType = true;
        else if (type === 'adaptive') matchType = card.dataset.adaptive === 'true';
        else matchType = card.classList.contains(type);
        card.style.display = (matchType && matchSearch) ? '' : 'none';
    }});
}}

function expandAll(expand) {{
    document.querySelectorAll('.question-card').forEach(card => {{
        if (card.style.display !== 'none') {{
            if (expand) card.classList.add('expanded');
            else card.classList.remove('expanded');
        }}
    }});
}}

document.addEventListener('click', function(ev) {{
    if (ev.target.classList.contains('show-more-btn')) {{
        const content = ev.target.previousElementSibling;
        if (!content) return;
        const pre = content.classList.contains('expandable-content') ? content : content.querySelector('.expandable-content');
        if (pre) {{
            pre.classList.toggle('collapsed');
            ev.target.textContent = pre.classList.contains('collapsed') ? 'Show more' : 'Show less';
        }}
    }}
}});

document.addEventListener('DOMContentLoaded', function() {{
    document.querySelectorAll('.expandable-content.collapsed').forEach(el => {{
        if (el.scrollHeight > 160) {{
            const btn = document.createElement('span');
            btn.className = 'show-more-btn';
            btn.textContent = 'Show more';
            el.parentElement.appendChild(btn);
        }} else {{
            el.classList.remove('collapsed');
        }}
    }});
}});
</script>
</body>
</html>'''
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("results_file", help="Path to dra.jsonl results file")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--config", help="Path to config file for metadata extraction")
    parser.add_argument("--detail", action="store_true", help="Print per-question detail in terminal")
    parser.add_argument("-o", "--output", help="Output file path (default: auto-generated)")
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: {args.results_file} not found")
        sys.exit(1)

    results = load_results(args.results_file)
    if not results:
        print("No results found in file.")
        sys.exit(1)

    workdir = str(Path(args.results_file).parent)
    config_path = args.config
    if not config_path:
        log_path = os.path.join(workdir, "log.txt")
        if os.path.exists(log_path):
            with open(log_path, errors="ignore") as f:
                for line_idx, line in enumerate(f):
                    if line_idx > 200:
                        break
                    if any(kw in line for kw in ("config_gaia", "config_arc", "config_hle")) and ".py" in line:
                        m = re.search(r"(configs/\S+\.py)", line)
                        if m:
                            config_path = m.group(1)
                            break

    meta = load_config_metadata(config_path, workdir)

    ds_info = detect_dataset_info(results, meta)
    scorer = get_scorer("arc" if ds_info["dataset"] == "ARC-AGI" else "gaia")

    print(terminal_report(results, meta, scorer=scorer))

    if args.detail:
        print(per_question_text(results, scorer=scorer))

    if args.html:
        out_path = args.output or os.path.join(workdir, "report.html")
        html = generate_html(results, meta, args.results_file, scorer=scorer)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nHTML report saved to: {out_path}")


if __name__ == "__main__":
    main()
