#!/usr/bin/env python
"""
Per-batch timeout attribution for a GAIA run.

`run_gaia.py` runs `concurrency=4` tasks in parallel via asyncio.gather, so
log.txt lines from concurrent tasks are interleaved. A per-task line-range
slice is therefore meaningless — signatures are properly attributable only
to the *batch* they fall inside. This script:

    1. Groups dra.jsonl rows into batches via their start_time (rows whose
       start_time is within BATCH_WINDOW_SECS of each other share a batch).
    2. Slices log.txt by timestamp (HH:MM:SS in `[HH:MM:SS - logger:...]`)
       so each batch gets the log slice from its first start_time to the
       maximum end_time across the batch.
    3. Counts failure signatures inside that slice.
    4. Attributes the batch's timeout tasks to the dominant bucket.

Outputs per-batch table, global signature totals, and a few deep-dive case
studies.

Usage:
    python scripts/timeout_analysis.py workdir/gaia_c3_mistral_<RUN_ID>
    python scripts/timeout_analysis.py workdir/gaia_c3_mistral_<RUN_ID> --deep 5

    # E0 v3 (2026-04) artifacts may still live under workdir/gaia_c4_* (legacy prefix).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


# Lines inside the same batch all start from roughly the same wall time;
# start_times within this window are treated as one batch.
BATCH_WINDOW_SECS = 15

LOG_TS_RE = re.compile(r"^(?:\x1b\[\d+m)?(\d{2}):(\d{2}):(\d{2})")
TASK_TIMEOUT_RE = re.compile(r"Question timed out after (\d+)s: (.{1,80})")

SIGNATURE_PATTERNS: dict[str, re.Pattern[str]] = {
    # Real HTTP rate-limit / gateway-error signals only. Bare "429" / "502"
    # digit patterns are avoided because skill-prompt text contains strings
    # like "on a 429 error, try archive.org" that are not runtime events.
    "rate_limit_429": re.compile(
        r"(RateLimitError|raw_status_code['\":\s]+429|Error code: 429|HTTP 429|429 Too Many)"
    ),
    "gateway_5xx": re.compile(
        r"(Error code: (502|503|504)|HTTP 50[234]|raw_status_code['\":\s]+50[234]|Bad Gateway|Service Unavailable|Gateway Timeout)"
    ),
    "dr_query_timeout": re.compile(r"deep_researcher\.py:350.*timed out after 60s"),
    "dr_analyze_timeout": re.compile(r"deep_researcher\.py:555.*timed out after 60s"),
    "dr_summary_timeout": re.compile(r"deep_researcher\.py:611.*timed out after 60s"),
    "browser_stall": re.compile(
        r"(auto_browser_use.*cleanup-deadlock|browser_agent.*wait_for.*timeout|cleanup-deadlock)"
    ),
    # Real REVIEW retry verdicts (distinct from skill text that discusses retry).
    "review_retry_loop": re.compile(r'"next_action":\s*\{[^}]*"action":\s*"retry"'),
    "provider_reset": re.compile(r"(upstream connect error|reset reason: overflow)"),
    "context_length": re.compile(
        r"(maximum context length|context length.*exceeded|32768 tok)"
    ),
    "parsing_error": re.compile(r"AgentParsingError"),
}


@dataclass
class Batch:
    index: int
    start: datetime
    end: datetime
    task_ids: list[str] = field(default_factory=list)
    timeout_task_ids: list[str] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    lines: int = 0


def parse_iso(s: str) -> datetime:
    # dra.jsonl uses "YYYY-MM-DD HH:MM:SS"
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def group_batches(rows: list[dict], window: int = BATCH_WINDOW_SECS) -> list[Batch]:
    """Cluster rows by start_time within `window` seconds."""
    rows_sorted = sorted(rows, key=lambda r: parse_iso(r["start_time"]))
    batches: list[Batch] = []
    cur: Batch | None = None
    for r in rows_sorted:
        st = parse_iso(r["start_time"])
        et = parse_iso(r["end_time"])
        if cur is None or (st - cur.start).total_seconds() > window:
            cur = Batch(index=len(batches), start=st, end=et)
            batches.append(cur)
        cur.task_ids.append(r["task_id"])
        if "timeout" in (r.get("agent_error") or "").lower():
            cur.timeout_task_ids.append(r["task_id"])
        if et > cur.end:
            cur.end = et
    return batches


def scan_log(log_path: Path, batches: list[Batch]) -> None:
    """
    Walk log.txt once; for each line parse its HH:MM:SS and assign it to
    the first batch whose [start, end] interval contains that clock time
    (same calendar day as the batch's start). Count signatures in place.
    """
    # Index batches by (start_clock, end_clock); since runs span > 24h we
    # also need to treat end_clock < start_clock as crossing midnight.
    anchors = [(b.start, b.end, b) for b in batches]

    current: Batch | None = None
    with log_path.open(errors="replace") as f:
        for line in f:
            m = LOG_TS_RE.match(line)
            if not m:
                if current is not None:
                    _count_line(line, current)
                continue
            hh, mm, ss = map(int, m.groups())
            # Seek a batch whose clock window covers (hh:mm:ss).
            # Because runs can span 24h+, we don't reconstruct the date; we
            # instead advance `current` forward-only through `anchors`.
            if current is None or not _covers_clock(current, hh, mm, ss):
                current = None
                for _s, _e, b in anchors:
                    if _covers_clock(b, hh, mm, ss):
                        current = b
                        break
            if current is not None:
                current.lines += 1
                _count_line(line, current)


def _covers_clock(batch: Batch, hh: int, mm: int, ss: int) -> bool:
    """
    Return True if HH:MM:SS falls inside the batch's [start,end] interval,
    modulo-24h tolerant (a batch that spans midnight covers both sides).
    """
    start_sec = batch.start.hour * 3600 + batch.start.minute * 60 + batch.start.second
    end_sec = batch.end.hour * 3600 + batch.end.minute * 60 + batch.end.second
    ts = hh * 3600 + mm * 60 + ss
    if batch.start.date() == batch.end.date():
        return start_sec <= ts <= end_sec
    # Crosses midnight
    return ts >= start_sec or ts <= end_sec


def _count_line(line: str, batch: Batch) -> None:
    for name, pat in SIGNATURE_PATTERNS.items():
        if pat.search(line):
            batch.counts[name] += 1


def attribute(batch: Batch) -> str:
    total = sum(batch.counts.values())
    if total == 0:
        return "cap_only"
    top = batch.counts.most_common()
    leader, ln = top[0]
    runner, rn = top[1] if len(top) > 1 else (None, 0)
    if ln / total > 0.5 or ln >= max(3 * rn, 3):
        return leader
    return f"mixed({leader}+{runner})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--window", type=int, default=BATCH_WINDOW_SECS,
                    help="Seconds that count as the same batch (default 15)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    dra_path = run_dir / "dra.jsonl"
    log_path = run_dir / "log.txt"
    if not dra_path.exists() or not log_path.exists():
        print(f"missing dra.jsonl or log.txt under {run_dir}", file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in dra_path.open()]
    n_timeouts = sum(1 for r in rows if "timeout" in (r.get("agent_error") or "").lower())
    batches = group_batches(rows, window=args.window)
    print(f"# Timeout attribution — {run_dir.name}")
    print(f"  {len(rows)} rows  |  {n_timeouts} timeouts  |  {len(batches)} batches  (concurrency=4)")
    print(f"  scanning log.txt ({log_path.stat().st_size / 1e6:.0f} MB) …", flush=True)
    scan_log(log_path, batches)

    # Per-batch table (only batches with at least one timeout)
    print()
    print("## Per-batch signature counts (batches with ≥1 timeout)")
    header = f"{'#':>3}  {'start':<19}  {'mins':>4}  {'n_to':>4}  {'attribution':<38}  top 3 signatures"
    print(header)
    print("-" * len(header))

    per_label = Counter()
    task_attrib: dict[str, str] = {}
    for b in batches:
        if not b.timeout_task_ids:
            continue
        mins = int((b.end - b.start).total_seconds() / 60)
        label = attribute(b)
        per_label[label] += len(b.timeout_task_ids)
        for tid in b.timeout_task_ids:
            task_attrib[tid] = label
        top3 = ", ".join(f"{k}={v}" for k, v in b.counts.most_common(3))
        print(f"{b.index:>3}  {b.start.strftime('%Y-%m-%d %H:%M:%S'):<19}  {mins:>4}  "
              f"{len(b.timeout_task_ids):>4}  {label:<38}  {top3}")

    print()
    print("## Timeout-task attribution summary (per task, inherited from batch)")
    print(f"{'bucket':<42} {'n_timeouts':>10}  {'share':>7}")
    for k, v in per_label.most_common():
        print(f"{k:<42} {v:>10}  {v/n_timeouts*100:>6.1f}%")

    # Global rates across all timeout-containing batches
    print()
    print("## Global signature totals (only batches that contained ≥1 timeout)")
    global_counts = Counter()
    global_mins = 0
    for b in batches:
        if b.timeout_task_ids:
            global_counts.update(b.counts)
            global_mins += (b.end - b.start).total_seconds() / 60
    print(f"{'signature':<22} {'total':>8}  {'per_min':>9}")
    for k, v in global_counts.most_common():
        rate = v / global_mins if global_mins else 0
        print(f"{k:<22} {v:>8}  {rate:>9.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
