#!/usr/bin/env python3
"""
Export a GAIA evaluation JSONL (dra.jsonl) to the leaderboard submission format.

The GAIA leaderboard expects each JSON line to include at least:
    {"task_id": "...", "model_answer": "..."}
Optional key: `reasoning_trace`.

This script reads the agent's output JSONL, maps `prediction` -> `model_answer`
(intentional abstentions such as `Unable to determine` stay as non-empty `model_answer`).
By default it strips markdown blobs by taking the segment after the last `FINAL ANSWER`
marker when present (`--no-sanitize` keeps predictions verbatim).

Validates against the official 2023 test split counts (Level 1: 93, 2: 159, 3: 49 = 301).

Usage:
    python scripts/export_gaia_submission.py workdir/<run_dir>/dra.jsonl

    python scripts/export_gaia_submission.py workdir/<run_dir>/dra.jsonl -o submission.jsonl

    python scripts/export_gaia_submission.py workdir/<run_dir>/dra.jsonl --no-validate

    python scripts/export_gaia_submission.py run.jsonl --reasoning-trace
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter

EXPECTED_COUNTS = {"1": 93, "2": 159, "3": 49}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())  # 301

_FINAL_ANSWER_SPLIT_RE = re.compile(r"(?is)\bfinal\s+answer\s*:?\s*")


def sanitize_prediction_to_model_answer(raw: str) -> str:
    """
    If `prediction` contains a FINAL ANSWER template (possibly markdown-wrapped),
    take the span after the *last* occurrence — mirrors reformulator / run_gaia salvage.

    Short predictions without that marker are returned stripped unchanged.
    """
    s = raw.strip()
    if not s:
        return ""
    lower = s.lower()
    if "final answer" not in lower:
        return s

    parts = _FINAL_ANSWER_SPLIT_RE.split(s)
    if len(parts) < 2:
        return s

    tail = parts[-1].strip()
    # First logical line often holds the benchmark answer
    line = tail.split("\n")[0].strip()
    # Trim wrapping markdown emphasis / bullets
    line = re.sub(r"^\*+\s*", "", line)
    line = re.sub(r"\*+$", "", line).strip()
    return line if line else s


def main():
    parser = argparse.ArgumentParser(description="Export GAIA results to leaderboard submission format")
    parser.add_argument("input", help="Path to dra.jsonl from evaluation run")
    parser.add_argument("-o", "--output", help="Output submission JSONL path (default: <input_dir>/submission.jsonl)")
    parser.add_argument("--no-validate", action="store_true", help="Skip question count validation")
    parser.add_argument(
        "--sanitize",
        action="store_true",
        default=True,
        help="Extract text after the last 'FINAL ANSWER:' when present (default: on)",
    )
    parser.add_argument(
        "--no-sanitize",
        dest="sanitize",
        action="store_false",
        help="Use prediction verbatim (strip only outer whitespace)",
    )
    parser.add_argument(
        "--reasoning-trace",
        action="store_true",
        help="Include optional reasoning_trace from intermediate_steps (JSON text; large)",
    )
    parser.add_argument(
        "--reasoning-max-chars",
        type=int,
        default=50000,
        help="Truncate each reasoning_trace to this many chars (default 50000)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        results = [json.loads(line) for line in f if line.strip()]

    submission_lines = []
    level_counts = Counter()
    errors = 0
    no_answer = 0
    sanitized_rows = 0
    task_ids_seen = set()

    for r in results:
        task_id = r.get("task_id")
        prediction = r.get("prediction")
        level = str(r.get("task", ""))
        agent_error = r.get("agent_error")

        if not task_id:
            print(f"Warning: skipping entry with no task_id", file=sys.stderr)
            continue

        if task_id in task_ids_seen:
            print(f"Warning: duplicate task_id {task_id}, keeping first occurrence", file=sys.stderr)
            continue
        task_ids_seen.add(task_id)

        if agent_error:
            errors += 1
        if prediction is None or str(prediction).strip() == "":
            no_answer += 1
            model_answer = ""
        else:
            raw = str(prediction).strip()
            if args.sanitize:
                model_answer = sanitize_prediction_to_model_answer(raw)
                if model_answer != raw:
                    sanitized_rows += 1
            else:
                model_answer = raw

        row = {"task_id": task_id, "model_answer": model_answer}
        if args.reasoning_trace:
            steps = r.get("intermediate_steps")
            if steps is not None:
                trace = json.dumps(steps, ensure_ascii=False)
                if len(trace) > args.reasoning_max_chars:
                    trace = trace[: args.reasoning_max_chars] + "…[truncated]"
                row["reasoning_trace"] = trace
        submission_lines.append(row)
        level_counts[level] += 1

    output_path = Path(args.output) if args.output else input_path.parent / "submission.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for line in submission_lines:
            f.write(json.dumps(line) + "\n")

    print(f"Submission exported: {output_path}")
    print(f"  Total entries:  {len(submission_lines)}")
    print(f"  With answers:   {len(submission_lines) - no_answer}")
    print(f"  No answer:      {no_answer}")
    print(f"  Agent errors:   {errors}")
    print(f"  Sanitized rows: {sanitized_rows}" if args.sanitize else "  Sanitize:       off")
    print(f"  Per level:      {dict(sorted(level_counts.items()))}")

    if not args.no_validate:
        if len(submission_lines) != EXPECTED_TOTAL:
            print(f"\n  WARNING: Expected {EXPECTED_TOTAL} entries, got {len(submission_lines)}")
            print(f"  The leaderboard requires exactly {EXPECTED_TOTAL} test questions.")
            missing = EXPECTED_TOTAL - len(submission_lines)
            if missing > 0:
                print(f"  {missing} questions are missing — the submission may be rejected.")
        else:
            for level, expected in sorted(EXPECTED_COUNTS.items()):
                actual = level_counts.get(level, 0)
                status = "OK" if actual == expected else f"MISMATCH (expected {expected})"
                print(f"  Level {level}: {actual} — {status}")
            print(f"\n  Submission is complete and ready for upload at:")
            print(f"  https://huggingface.co/spaces/gaia-benchmark/leaderboard")


if __name__ == "__main__":
    main()
