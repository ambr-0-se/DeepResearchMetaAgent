#!/usr/bin/env python3
"""
Export a GAIA evaluation JSONL (dra.jsonl) to the leaderboard submission format.

The GAIA leaderboard expects a JSONL file with exactly two fields per line:
    {"task_id": "...", "model_answer": "..."}

This script reads the agent's output JSONL, maps `prediction` -> `model_answer`,
and validates the result against the expected test-split question counts
(Level 1: 93, Level 2: 159, Level 3: 49 = 301 total).

Usage:
    python scripts/export_gaia_submission.py workdir/<run_dir>/dra.jsonl

    # Custom output path
    python scripts/export_gaia_submission.py workdir/<run_dir>/dra.jsonl -o submission.jsonl

    # Skip validation (e.g. partial runs)
    python scripts/export_gaia_submission.py workdir/<run_dir>/dra.jsonl --no-validate
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

EXPECTED_COUNTS = {"1": 93, "2": 159, "3": 49}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())  # 301


def main():
    parser = argparse.ArgumentParser(description="Export GAIA results to leaderboard submission format")
    parser.add_argument("input", help="Path to dra.jsonl from evaluation run")
    parser.add_argument("-o", "--output", help="Output submission JSONL path (default: <input_dir>/submission.jsonl)")
    parser.add_argument("--no-validate", action="store_true", help="Skip question count validation")
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
            model_answer = str(prediction).strip()

        level_counts[level] += 1
        submission_lines.append({"task_id": task_id, "model_answer": model_answer})

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
