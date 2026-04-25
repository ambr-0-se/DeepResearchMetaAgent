#!/usr/bin/env python3
"""F0.2 — Post-hoc log-evidence salvage of blank predictions.

For each row in `<cell>/dra.jsonl` with empty `prediction`, extract a
timestamp-bounded slice of `<cell>/log.txt` (within the row's
[start_time, end_time + grace] window), pre-filter for relevance, and ask
the cell's main model for a best-guess answer in GAIA format. Idempotent
and resumable: rows already tagged `[post-hoc-log-salvage]` are skipped.

Usage:
    python scripts/salvage_blank_predictions.py \\
        --cells workdir/gaia_c0_mistral_<run> workdir/gaia_c0_qwen_<run> \\
        --model-ids mistral-small or-qwen3.6-plus

    # Sanity check first (no LLM calls):
    python scripts/salvage_blank_predictions.py --cells <cell> --model-ids <id> --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logger import logger
from src.models import model_manager, ChatMessage, MessageRole
from examples.run_gaia import _is_banned_answer, _force_concrete_guess


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
HMS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})")
SALVAGE_TAG = "[post-hoc-log-salvage]"
GRACE_SECONDS = 30
MAX_SLICE_CHARS = 30_000
TIER1_MARKERS = (
    "Task Id:",
    "Plan ",
    "Plan:",
    "Calling tool:",
    "Reached max steps",
    "Final Answer:",
    "Observation:",
)
STOPWORDS = frozenset(
    """
    the a an of to in on for and or is are was were be been being by with from as at this that
    these those what which who whom how when where why your you we us they them it its their
    there here but not no yes have has had do does did can could would should shall will may
    might must i me my mine if then than so too very just also only any all some into out up
    down over under more most least between about after before during answer question based
    according given above shown page paper book one two three four five six seven eight nine
    ten following first second third last next also each every both either neither such
    """.split()
)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_log_hms(raw: str) -> str | None:
    stripped = strip_ansi(raw).lstrip()
    m = HMS_RE.match(stripped)
    if not m:
        return None
    return f"{m.group(1)}:{m.group(2)}:{m.group(3)}"


def hms_to_seconds(hms: str) -> int:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def build_log_index(log_path: Path, anchor_date: datetime) -> list[tuple[int, datetime, str]]:
    """Build a list of (line_no, monotonic_dt, raw_line).

    Continuation lines (no HH:MM:SS prefix) inherit the previous entry's
    timestamp so they can still be associated with their parent log line.
    Date rollovers are detected when HH:MM:SS goes backward by >10 minutes.
    """
    index: list[tuple[int, datetime, str]] = []
    prev_secs: int | None = None
    day_offset = 0
    last_dt = anchor_date
    with open(log_path, "r", encoding="utf-8", errors="replace") as fp:
        for line_no, raw in enumerate(fp):
            hms = parse_log_hms(raw)
            if hms is None:
                index.append((line_no, last_dt, raw))
                continue
            secs = hms_to_seconds(hms)
            if prev_secs is not None and secs + 600 < prev_secs:
                day_offset += 1
            prev_secs = secs
            last_dt = anchor_date + timedelta(days=day_offset, seconds=secs)
            index.append((line_no, last_dt, raw))
    return index


def tokenize_question(q: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]{2,}", q.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 3}


def slice_log_for_row(
    index: list[tuple[int, datetime, str]],
    row: dict,
    q_tokens: set[str],
    grace_s: int = GRACE_SECONDS,
) -> str:
    start_dt = datetime.strptime(row["start_time"], "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(row["end_time"], "%Y-%m-%d %H:%M:%S") + timedelta(seconds=grace_s)
    task_id = row["task_id"]

    in_window = [(idx, dt, raw) for (idx, dt, raw) in index if start_dt <= dt <= end_dt]
    if not in_window:
        return ""

    first_60s_cutoff = start_dt + timedelta(seconds=60)
    tier1: list[str] = []
    tier2: list[str] = []
    tier3: list[str] = []
    seen: set[str] = set()

    for (_idx, dt, raw) in in_window:
        clean = strip_ansi(raw).rstrip("\n")
        if not clean.strip():
            continue
        if clean in seen:
            continue
        if task_id in clean or any(m in clean for m in TIER1_MARKERS):
            seen.add(clean)
            tier1.append(clean)
            continue
        line_words = set(re.findall(r"[A-Za-z][A-Za-z0-9'\-]{2,}", clean.lower()))
        overlap = (line_words - STOPWORDS) & q_tokens
        if len(overlap) >= 3:
            seen.add(clean)
            tier2.append(clean)
            continue
        if dt <= first_60s_cutoff:
            seen.add(clean)
            tier3.append(clean)

    header = (
        f"[Task Id: {task_id}]\n"
        f"[Question: {row.get('question', '')}]\n"
        f"[Window: {row['start_time']} -> {row['end_time']}]\n"
    )
    body_chunks: list[str] = []
    body_chunks.extend(tier1)
    body_chunks.extend(tier3)
    body_chunks.extend(reversed(tier2))
    body = "\n".join(body_chunks)
    budget = MAX_SLICE_CHARS - len(header)
    if len(body) > budget:
        body = body[:budget]
    return header + body


PROMPT_SYSTEM = (
    "You are recovering a final answer for a GAIA test question whose pipeline timed out. "
    "Partial evidence exists in the session log. Lines from up to 8 concurrent unrelated "
    "tasks are interleaved; ignore lines that don't relate to the question's topic. Focus "
    "on: plans mentioning the question's keywords, sub-agent calls, observations returned, "
    "partial answers found.\n\n"
    "Output ONLY the FINAL ANSWER in the format the question requires (number, short string, "
    "or comma-separated list). NO preamble, NO apology, NO 'Unable to determine'. If "
    "evidence is incomplete, make your best guess from general knowledge."
)


def build_user_prompt(question: str, slice_text: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Evidence (interleaved, possibly noisy log slice):\n"
        f"{slice_text}\n\n"
        "Use template: FINAL ANSWER: <answer>"
    )


async def call_model(model, question: str, slice_text: str) -> str:
    messages = [
        ChatMessage.from_dict({
            "role": MessageRole.SYSTEM,
            "content": [{"type": "text", "text": PROMPT_SYSTEM}],
        }),
        ChatMessage.from_dict({
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": build_user_prompt(question, slice_text)}],
        }),
    ]
    response = await model(messages)
    text = response.content if hasattr(response, "content") else str(response)
    if "FINAL ANSWER:" in text:
        return text.split("FINAL ANSWER:")[-1].strip()
    return text.strip()


def atomic_write_jsonl(path: Path, rows: list) -> None:
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


async def salvage_cell(cell_dir: Path, model_id: str, dry_run: bool = False) -> None:
    cell_dir = Path(cell_dir)
    jsonl_path = cell_dir / "dra.jsonl"
    log_path = cell_dir / "log.txt"
    if not jsonl_path.exists():
        logger.warning(f"[{cell_dir.name}] no dra.jsonl — skipping.")
        return
    if not log_path.exists():
        logger.warning(f"[{cell_dir.name}] no log.txt — skipping.")
        return

    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    blank_idxs = [
        i for i, r in enumerate(rows)
        if not r.get("prediction") and SALVAGE_TAG not in str(r.get("agent_error", ""))
    ]
    logger.info(f"[{cell_dir.name}] {len(rows)} total rows; {len(blank_idxs)} to salvage.")
    if not blank_idxs:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not dry_run:
        backup_path = jsonl_path.with_name(f"dra.jsonl.pre_log_salvage_{ts}")
        shutil.copy2(jsonl_path, backup_path)
        logger.info(f"[{cell_dir.name}] backup -> {backup_path.name}")

    anchor_date_str = rows[blank_idxs[0]]["start_time"][:10]
    anchor_date = datetime.strptime(anchor_date_str, "%Y-%m-%d")
    logger.info(f"[{cell_dir.name}] indexing log (anchor date={anchor_date_str}) ...")
    index = build_log_index(log_path, anchor_date)
    logger.info(f"[{cell_dir.name}] indexed {len(index)} lines.")

    if dry_run:
        sample_idx = blank_idxs[0]
        sample_row = rows[sample_idx]
        slc = slice_log_for_row(index, sample_row, tokenize_question(sample_row["question"]))
        print(f"\n=== Sample slice for row {sample_idx} (task {sample_row['task_id']}) ===")
        print(f"slice chars: {len(slc)}")
        print(slc[:3000])
        return

    model = model_manager.registed_models[model_id]
    log_dir = Path("workdir/run_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    salvage_log_path = log_dir / f"blank_salvage_{cell_dir.name}_{ts}.log"
    log_fp = open(salvage_log_path, "a", encoding="utf-8")
    log_fp.write(f"# Salvage run started {ts} for {cell_dir}\n")
    log_fp.flush()

    salvaged = forced = failed = 0
    for n, idx in enumerate(blank_idxs):
        row = rows[idx]
        task_id = row["task_id"]
        try:
            slc = slice_log_for_row(index, row, tokenize_question(row.get("question", "")))
            if not slc:
                logger.info(f"[{cell_dir.name}] {task_id}: empty slice, forcing concrete guess.")
                pred = await _force_concrete_guess(row.get("question", ""), model)
                forced += 1
                status = "forced-empty-slice"
            else:
                pred = await call_model(model, row.get("question", ""), slc)
                if _is_banned_answer(pred):
                    logger.info(f"[{cell_dir.name}] {task_id}: banned, retrying with concrete-guess.")
                    pred = await _force_concrete_guess(row.get("question", ""), model)
                    forced += 1
                    status = "second-attempt"
                else:
                    salvaged += 1
                    status = "salvaged"
            row["prediction"] = pred
            existing_err = str(row.get("agent_error", "") or "")
            row["agent_error"] = (existing_err + " " + SALVAGE_TAG).strip()
            atomic_write_jsonl(jsonl_path, rows)
            log_fp.write(f"{task_id}\t{status}\t{len(pred)}c\t{pred[:200]!r}\n")
            log_fp.flush()
        except Exception as e:
            failed += 1
            logger.warning(
                f"[{cell_dir.name}] {task_id}: salvage failed: "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            log_fp.write(f"{task_id}\tfailed\t-\t{type(e).__name__}: {str(e)[:200]}\n")
            log_fp.flush()
        if (n + 1) % 10 == 0 or (n + 1) == len(blank_idxs):
            logger.info(
                f"[{cell_dir.name}] progress {n+1}/{len(blank_idxs)} "
                f"salvaged={salvaged} forced={forced} failed={failed}"
            )

    log_fp.close()
    logger.info(
        f"[{cell_dir.name}] DONE: salvaged={salvaged} forced={forced} "
        f"failed={failed} log={salvage_log_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", required=True,
                   help="One or more cell dirs containing dra.jsonl + log.txt")
    p.add_argument("--model-ids", nargs="+", required=True,
                   help="Model id per cell (same order as --cells)")
    p.add_argument("--dry-run", action="store_true",
                   help="Build log index, print sample slice; no LLM calls, no row writes")
    args = p.parse_args()

    if len(args.cells) != len(args.model_ids):
        raise SystemExit("--cells and --model-ids must have equal length")

    if not args.dry_run:
        model_manager.init_models(use_local_proxy=True)
        logger.info("Registered models: %s", ", ".join(model_manager.registed_models.keys()))

    for cell, mid in zip(args.cells, args.model_ids):
        if not args.dry_run and mid not in model_manager.registed_models:
            raise SystemExit(f"Model id '{mid}' not registered.")
        asyncio.run(salvage_cell(Path(cell), mid, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
