#!/usr/bin/env bash
# Resume an in-flight E0 v3 C4 training run.
#
# Why this wrapper exists:
#   On 2026-04-20 a resume was attempted from a cold `nohup bash` shell
#   (no conda activation). Two silent failures followed:
#
#   1. `filter_answers` (run_gaia.py:64) deletes every wrong/errored row
#      and re-attempts them on resume. Re-runs cost per-Q timeout budget
#      AND give the model an unfair "second chance" at a task it failed
#      once, contaminating E0 skill-library training.
#
#   2. `configs/base.py:41` spawns the local MCP server via
#      `"command": "python"`. With no conda env active, PATH resolves
#      `python` to `/Users/ahbo/miniconda3/bin/python` (base env, no
#      `fastmcp`). All 80 questions then crash with `ModuleNotFoundError`
#      wrapped as `Connection closed` in ~1s, leaving a file full of
#      errored rows and a false "DONE rc=0" marker.
#
# Both failures are fixable with environment alone — no code change.
# This script bakes in the fix so resume works from any shell.
#
# Usage:
#   bash scripts/resume_e0.sh              # resume default run
#   DRA_RUN_ID=<other_id> bash scripts/resume_e0.sh   # resume a specific run
#
# Preconditions:
#   - `dra` conda env exists at /Users/ahbo/miniconda3/envs/dra/bin/python
#     with `fastmcp`, `pandas`, `mmengine`, `crawl4ai` installed.
#   - Existing workdir/gaia_c4_{mistral,qwen}_<DRA_RUN_ID>/dra.jsonl
#     carries the rows to preserve.
#   - No other run_gaia.py / run_eval_matrix.sh procs alive.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRA_PY="/Users/ahbo/miniconda3/envs/dra/bin/python"
DRA_BIN_DIR="$(dirname "$DRA_PY")"

if [ ! -x "$DRA_PY" ]; then
  echo "FATAL: dra conda python not found at $DRA_PY" >&2
  exit 2
fi

# Verify fastmcp is importable in the dra env (the subprocess MCP server
# needs it; the earlier catastrophic resume failed precisely here).
if ! "$DRA_PY" -c "import fastmcp, pandas, mmengine" >/dev/null 2>&1; then
  echo "FATAL: dra env missing one of {fastmcp, pandas, mmengine}" >&2
  exit 2
fi

if pgrep -f 'run_gaia.py\|run_eval_matrix' >/dev/null; then
  echo "FATAL: existing run_gaia / run_eval_matrix procs alive — refusing to double-launch" >&2
  pgrep -fl 'run_gaia.py\|run_eval_matrix' >&2
  exit 2
fi

: "${DRA_RUN_ID:=20260420_E0v3}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="workdir/run_logs/launcher_resume_${TS}.log"
mkdir -p workdir/run_logs

# Belt-and-braces: a fresh caffeinate assertion for this launch so the
# Mac doesn't sleep mid-run. Detached so closing the shell doesn't kill it.
if ! pmset -g assertions 2>/dev/null | grep -q "caffeinate command-line"; then
  caffeinate -dimsu &
  disown
  echo "Started caffeinate."
fi

echo "[resume_e0] repo:       $REPO_ROOT"
echo "[resume_e0] DRA_RUN_ID: $DRA_RUN_ID"
echo "[resume_e0] log:        $LOG"
echo "[resume_e0] rows preserved:"
for m in mistral qwen; do
  f="workdir/gaia_c4_${m}_${DRA_RUN_ID}/dra.jsonl"
  [ -f "$f" ] && printf "  %-8s %4s rows\n" "$m" "$(wc -l <"$f")" || printf "  %-8s no dra.jsonl yet\n" "$m"
done

# Target rows per model. E0 v3 methodology: 80 Qs per (model, C4) cell
# on the shuffled validation subsample.
TARGET_PER_MODEL=80

# Launch.
#   PATH prefix = dra bin    → MCP subprocess "python" resolves to dra.
#   PYTHON=dra py            → the outer launcher uses the right interpreter.
#   DRA_RESUME_PRESERVE_ALL  → skip filter_answers; no reruns of prior attempts.
#   DATASET_SPLIT=validation → required by run_eval_matrix strict-guard.
#   Per-model max_samples    → dynamically computed from current row count
#                              so we stop at TARGET_PER_MODEL total (not
#                              TARGET_PER_MODEL *new* past what's already done).
#                              Required because run_gaia.py's max_samples
#                              slices the REMAINING tasks after excluding
#                              done_questions.
LAUNCH_TS="$(date +%H%M%S)"
for model in mistral qwen; do
  dra_jsonl="workdir/gaia_c4_${model}_${DRA_RUN_ID}/dra.jsonl"
  done_rows=0
  [ -f "$dra_jsonl" ] && done_rows=$(wc -l <"$dra_jsonl" | tr -d ' ')
  remaining=$(( TARGET_PER_MODEL - done_rows ))
  if [ "$remaining" -le 0 ]; then
    echo "[resume_e0] $model: already at $done_rows/$TARGET_PER_MODEL — skipping launch"
    continue
  fi
  echo "[resume_e0] $model: $done_rows done, will attempt up to $remaining more (target $TARGET_PER_MODEL)"
  model_log="workdir/run_logs/launcher_${model}_${LAUNCH_TS}.log"
  PATH="$DRA_BIN_DIR:$PATH" \
  DRA_RUN_ID="$DRA_RUN_ID" \
  DATASET_SPLIT=validation \
  PYTHON="$DRA_PY" \
  DRA_RESUME_PRESERVE_ALL=1 \
  FULL_CFG_OPTIONS="max_samples=${remaining} dataset.shuffle=True dataset.seed=42" \
    nohup bash scripts/run_eval_matrix.sh full "$model" c4 >"$model_log" 2>&1 &
  disown
  echo "[resume_e0] $model: launched, log=$model_log"
done

echo "[resume_e0] tail streams:  tail -f workdir/run_logs/full_{mistral,qwen}.log"
echo "[resume_e0] monitor tick:  $DRA_PY scripts/monitor_tick.py"
