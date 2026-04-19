#!/usr/bin/env python
"""
E0 v3 monitor tick — comprehensive per-fire assessment.

Emits a structured multi-line report to stdout AND appends a snapshot to
`workdir/E0_MONITORING_STATE.jsonl` so deltas are computable across fires.

READ-ONLY: this script never edits source/config/tests, never kills procs,
never runs git mutating commands. All it does is read + append-log.

Usage:
  python scripts/monitor_tick.py

Designed to be called from a bash heartbeat loop OR from the MCP scheduled
task. Same logic in both paths so results align.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DRA_RUN_ID = "20260420_E0v3"
MODELS = ["mistral", "qwen"]
STATE_FILE = Path("workdir/E0_MONITORING_STATE.jsonl")

# --- Official GAIA scorer (lazy isolated import) -------------------------
# Importing `src.metric.gaia_scorer` via the package path pulls in
# `src/__init__.py`, which imports `crawl4ai` (not always installed in the
# env that invokes this monitor). Load the scorer module directly so the
# monitor stays self-contained and read-only.
_SCORER_PATH = Path(__file__).resolve().parent.parent / "src" / "metric" / "gaia_scorer.py"
_spec = importlib.util.spec_from_file_location("gaia_scorer_iso", _SCORER_PATH)
_gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gs)  # type: ignore[union-attr]
_official_scorer = _gs.question_scorer


def _is_correct_official(pred: str, truth: str) -> bool:
    """Official GAIA scorer, with warnings suppressed for length-mismatch
    lists (expected noise, not signal)."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bool(_official_scorer(pred, truth))
    except Exception:
        return False


_GAVE_UP_RE = re.compile(r"unable to determine|i cannot determine|cannot be determined", re.I)


def _classify_wrong(pred: str, truth: str) -> str:
    """Categorize a wrong (non-errored) answer row."""
    if not pred:
        return "empty_prediction"
    if _GAVE_UP_RE.search(pred):
        return "gave_up"
    # "Close under lax": official scorer said wrong, but a strip+lower match
    # would flip it. This catches residual format brittleness the official
    # normalizer missed (should be rare — the official scorer already strips
    # whitespace + punctuation for strings).
    if pred.strip().lower() == truth.strip().lower():
        return "format_near_miss_lax"
    return "genuine_wrong"


def _read_rows(m: str) -> list:
    path = Path(f"workdir/gaia_c4_{m}_{DRA_RUN_ID}/dra.jsonl")
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open()]


def _count_skills(m: str) -> tuple[int, int]:
    base = Path(f"workdir/gaia_c4_{m}_{DRA_RUN_ID}/skills")
    if not base.exists():
        return 0, 0
    seeded = learned = 0
    for s in base.glob("*/SKILL.md"):
        body = s.read_text(errors="replace")
        if re.search(r"^\s*source:\s*seeded\s*$", body, re.M):
            seeded += 1
        else:
            learned += 1
    return seeded, learned


def _evolution_log_len(m: str) -> int:
    p = Path(f"workdir/gaia_c4_{m}_{DRA_RUN_ID}/skill_evolution.jsonl")
    if not p.exists():
        return 0
    return sum(1 for _ in p.open())


def _recent_activity(m: str) -> str:
    """Last meaningful log line from the stream log."""
    p = Path(f"workdir/run_logs/full_{m}.log")
    if not p.exists():
        return "<no stream log>"
    try:
        # Use list args to avoid shell-quoting issues on paths with spaces
        # (the project root contains "APAI4799 MetaAgent").
        tail = subprocess.run(
            ["tail", "-200", str(p)],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return "<tail failed>"
    # Find last line that isn't MCP JSON-RPC noise
    for line in reversed(tail.splitlines()):
        if any(x in line for x in ("JSONRPC", "mcp.client", "Traceback", 'File "')):
            continue
        if not line.strip():
            continue
        return line[-180:]  # truncate to fit
    return "<only noise in last 200 lines>"


def _proc_count() -> int:
    out = subprocess.getoutput(
        "pgrep -fl 'run_gaia.py --config configs/config_gaia_c4_' 2>/dev/null"
    )
    return sum(1 for l in out.splitlines() if "run_gaia.py" in l)


def _caffeinate_count() -> int:
    out = subprocess.getoutput("pmset -g assertions 2>/dev/null")
    return sum(1 for l in out.splitlines() if "caffeinate command-line" in l)


def snapshot() -> dict:
    now = datetime.now(timezone.utc)
    snap: dict = {"timestamp": now.isoformat(timespec="seconds"), "dra_run_id": DRA_RUN_ID, "models": {}}
    wrong_task_ids_by_model: dict[str, set] = {}

    for m in MODELS:
        rows = _read_rows(m)
        err = sum(1 for r in rows if r.get("agent_error"))
        cap = sum(1 for r in rows if r.get("iteration_limit_exceeded"))

        correct = 0
        wrong_buckets: Counter = Counter()
        wrong_task_ids: set = set()
        wrong_examples: list = []

        for r in rows:
            if r.get("agent_error"):
                continue
            pred = str(r.get("prediction") or "").strip()
            truth = str(r.get("true_answer") or "").strip()
            if not pred or not truth:
                continue
            if _is_correct_official(pred, truth):
                correct += 1
            else:
                category = _classify_wrong(pred, truth)
                wrong_buckets[category] += 1
                tid = str(r.get("task_id") or "")
                if tid:
                    wrong_task_ids.add(tid)
                # Keep a small rotating sample for the report (max 3).
                if len(wrong_examples) < 3:
                    wrong_examples.append({
                        "task_id": tid[:8],
                        "category": category,
                        "pred": pred[:50],
                        "truth": truth[:50],
                    })

        wrong_task_ids_by_model[m] = wrong_task_ids

        error_types: Counter = Counter()
        for r in rows:
            e = r.get("agent_error")
            if e:
                error_types[str(e)[:80]] += 1

        seeded, learned = _count_skills(m)
        snap["models"][m] = {
            "rows": len(rows),
            "correct": correct,  # OFFICIAL GAIA scorer (type-aware normalization)
            "errored": err,
            "iter_limit_hit": cap,
            "seeded_count": seeded,
            "learned_count": learned,
            "evolution_log_entries": _evolution_log_len(m),
            "error_types": dict(error_types),
            "wrong_buckets": dict(wrong_buckets),
            "wrong_examples": wrong_examples,
            "recent_activity": _recent_activity(m),
        }

    # Cross-model shared-wrong task_ids — the "inherently hard" set and a
    # direct signal for the C4 skill library's value add.
    if len(MODELS) >= 2:
        shared = set.intersection(*wrong_task_ids_by_model.values()) if wrong_task_ids_by_model else set()
        snap["shared_wrong_task_ids"] = sorted(shared)
    else:
        snap["shared_wrong_task_ids"] = []

    snap["procs_alive"] = _proc_count()
    snap["caffeinate_assertions"] = _caffeinate_count()
    return snap


def load_history() -> list:
    if not STATE_FILE.exists():
        return []
    out = []
    for line in STATE_FILE.open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def fmt_delta(now: int, prev: int | None) -> str:
    if prev is None:
        return f"{now}"
    d = now - prev
    sign = "+" if d >= 0 else ""
    return f"{now}({sign}{d})"


def main() -> int:
    snap = snapshot()
    history = load_history()
    prev = history[-1] if history else None

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("a") as f:
        f.write(json.dumps(snap) + "\n")

    # Compute NEW error types across ALL prior snapshots
    seen_errors_global: dict[str, set] = {m: set() for m in MODELS}
    for h in history:
        for m in MODELS:
            et = h.get("models", {}).get(m, {}).get("error_types", {})
            seen_errors_global[m].update(et.keys())
    new_errors: dict[str, list] = {m: [] for m in MODELS}
    for m in MODELS:
        current = snap["models"][m].get("error_types", {})
        for et in current:
            if et not in seen_errors_global[m]:
                new_errors[m].append(et)

    # Wall since last
    wall_str = "first fire"
    if prev:
        prev_ts = datetime.fromisoformat(prev["timestamp"])
        now_ts = datetime.fromisoformat(snap["timestamp"])
        delta_min = (now_ts - prev_ts).total_seconds() / 60.0
        wall_str = f"{delta_min:.0f}min since last"

    # Emit the report
    ts = datetime.fromisoformat(snap["timestamp"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"==== MONITOR TICK {ts} ({wall_str}) ====")
    print(
        f"procs={snap['procs_alive']}/2 caffeinate={snap['caffeinate_assertions']}"
    )
    for m in MODELS:
        cur = snap["models"][m]
        prev_m = prev["models"].get(m, {}) if prev else {}
        rows = fmt_delta(cur["rows"], prev_m.get("rows"))
        ok = fmt_delta(cur["correct"], prev_m.get("correct"))
        err = fmt_delta(cur["errored"], prev_m.get("errored"))
        cap = fmt_delta(cur["iter_limit_hit"], prev_m.get("iter_limit_hit"))
        learned = fmt_delta(cur["learned_count"], prev_m.get("learned_count"))
        print(
            f"[{m:7s}] rows={rows}/80  correct={ok}(official)  errored={err}  cap_hit={cap}  learned_skills={learned}  evol_log={cur['evolution_log_entries']}"
        )

        # Wrong-answer bucket breakdown (non-errored rows the official scorer
        # marked wrong). Primary signal for fixable vs inherent-difficulty.
        wb = cur.get("wrong_buckets", {})
        if wb:
            total_wrong = sum(wb.values())
            bucket_str = "  ".join(f"{k}={v}" for k, v in sorted(wb.items(), key=lambda x: -x[1]))
            print(f"         wrong (non-err, n={total_wrong}): {bucket_str}")
            # Show up to 3 representative wrong examples — helps spot drift
            # in failure mode without opening the dra.jsonl.
            for ex in cur.get("wrong_examples", [])[:3]:
                print(
                    f"           {ex['category']:22s} {ex['task_id']}  pred={ex['pred']!r} vs truth={ex['truth']!r}"
                )

        top = sorted(cur["error_types"].items(), key=lambda x: -x[1])[:3]
        if top:
            print(f"         top errors:")
            for e, c in top:
                print(f"           {c:3d} × {e!r}")
        if new_errors[m]:
            print(f"         NEW error types since last fire:")
            for e in new_errors[m]:
                print(f"           + {e!r}")
        print(f"         last_activity: {cur['recent_activity']}")

    # Cross-model analysis — the C4 skill-library-target set.
    shared = snap.get("shared_wrong_task_ids", [])
    if shared:
        prev_shared = set(prev.get("shared_wrong_task_ids", []) if prev else [])
        new_shared = [t for t in shared if t not in prev_shared]
        print(
            f"cross-model: {len(shared)} task_ids wrong under BOTH models "
            f"(C4 skill-library target set){'  +' + str(len(new_shared)) + ' new' if new_shared else ''}"
        )
        for tid in shared[:5]:
            print(f"           {tid[:8]}")

    # Alarms
    alarms = []
    if snap["procs_alive"] < 2:
        alarms.append(f"procs_alive={snap['procs_alive']}/2 (degraded)")
    if snap["caffeinate_assertions"] < 2:
        alarms.append(f"caffeinate_assertions={snap['caffeinate_assertions']}/2 (mac may sleep)")
    if prev:
        for m in MODELS:
            cur_rows = snap["models"][m]["rows"]
            prev_rows = prev["models"].get(m, {}).get("rows", 0)
            if cur_rows == prev_rows:
                alarms.append(f"{m}: stall (no Δrows since last fire)")
    if alarms:
        print("ALARMS:")
        for a in alarms:
            print(f"  ! {a}")
    else:
        print("alarms: none")

    # Completion detection
    done_re = re.compile(r"stream:(mistral|qwen) DONE")
    done_models = set()
    for m in MODELS:
        log = Path(f"workdir/run_logs/full_{m}.log")
        if log.exists():
            for line in subprocess.getoutput(f"tail -50 {log!s}").splitlines():
                if f"stream:{m} DONE" in line:
                    done_models.add(m)
    if done_models == set(MODELS):
        print("")
        print("[E0 v3 COMPLETE] both streams DONE — E1 snapshot + E2 freeze smoke next (user approval needed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
