#!/usr/bin/env bash
# E0 v3 — Phase 2: one-shot infra-error re-attempt.
#
# Pre-registered in docs/handoffs/HANDOFF_E0_V3_PHASE2_RERUN.md (commit 8f7cd17).
# Applies the frozen eligibility regex to the target model's dra.jsonl,
# backs up dra.jsonl + skills/, drops eligible rows, and re-launches
# exactly those rows via run_eval_matrix.sh full <model> c4 with
# max_samples=N_infra (seed=42 shuffle invariant guarantees the
# first-N-of-remaining slice = the dropped task_ids).
#
# Usage:
#   bash scripts/rerun_e0_infra_errors.sh <model>
#     where model ∈ {mistral, qwen}
#
# Preconditions:
#   - workdir/gaia_c4_<model>_<DRA_RUN_ID>/dra.jsonl has exactly 80 rows
#   - No other run_eval_matrix / run_gaia proc is active on the same <model>
#     (a run on a DIFFERENT model is fine — separate workdirs, separate streams)
#
# Rule invariants (from pre-registration):
#   - Eligibility regex is FROZEN at pre-registration time — do not edit
#   - One re-attempt per eligible row; no second re-attempts
#   - Same DRA_RUN_ID, same skills_dir (skill library inherits Phase 1 state)
#   - Per-model workdir isolation (safe to run Mistral while Qwen is still in Phase 1)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/rerun_e0_infra_errors.sh <mistral|qwen>" >&2
  exit 2
fi
MODEL="$1"
if [[ "$MODEL" != "mistral" && "$MODEL" != "qwen" ]]; then
  echo "FATAL: model must be 'mistral' or 'qwen' (got: $MODEL)" >&2
  exit 2
fi

DRA_PY="/Users/ahbo/miniconda3/envs/dra/bin/python"
DRA_BIN_DIR="$(dirname "$DRA_PY")"
if [ ! -x "$DRA_PY" ]; then
  echo "FATAL: dra conda python not found at $DRA_PY" >&2
  exit 2
fi
if ! "$DRA_PY" -c "import fastmcp, pandas, mmengine" >/dev/null 2>&1; then
  echo "FATAL: dra env missing one of {fastmcp, pandas, mmengine}" >&2
  exit 2
fi

: "${DRA_RUN_ID:=20260420_E0v3}"
WORKDIR="workdir/gaia_c4_${MODEL}_${DRA_RUN_ID}"
DRA_JSONL="${WORKDIR}/dra.jsonl"
SKILLS_DIR="${WORKDIR}/skills"
TARGET_ROWS=80

if [ ! -f "$DRA_JSONL" ]; then
  echo "FATAL: $DRA_JSONL not found" >&2
  exit 2
fi

# Refuse double-launch on the same model. A run on a different model is OK.
if pgrep -f "run_eval_matrix.sh full ${MODEL}\b" >/dev/null \
   || pgrep -f "run_gaia.py.*config_gaia_c4_${MODEL}\.py" >/dev/null; then
  echo "FATAL: existing ${MODEL} eval proc alive — refusing to double-launch" >&2
  pgrep -fl "run_eval_matrix.sh full ${MODEL}\b|run_gaia.py.*config_gaia_c4_${MODEL}\.py" >&2 || true
  exit 2
fi

# Precondition: exactly TARGET_ROWS rows in dra.jsonl (Phase 1 complete).
ROW_COUNT=$(wc -l <"$DRA_JSONL" | tr -d ' ')
if [ "$ROW_COUNT" -ne "$TARGET_ROWS" ]; then
  echo "FATAL: $DRA_JSONL has $ROW_COUNT rows, expected $TARGET_ROWS (Phase 1 incomplete?)" >&2
  exit 2
fi

# Frozen eligibility regex (see HANDOFF_E0_V3_PHASE2_RERUN.md).
INFRA_REGEX='upstream connect error|reset reason: overflow|HTTP 5\d{2}|Service Unavailable|Bad Gateway|provider.*unavailable|model_overloaded'

# Count eligible rows via Python (identical semantics to pre-registration).
N_INFRA=$("$DRA_PY" - <<PY
import json, re
regex = re.compile(r'''${INFRA_REGEX}''')
n = 0
with open('${DRA_JSONL}') as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        err = r.get('agent_error') or ''
        if regex.search(err):
            n += 1
print(n)
PY
)

echo "[rerun_e0] model=${MODEL}  DRA_RUN_ID=${DRA_RUN_ID}"
echo "[rerun_e0] rows=${ROW_COUNT}  infra-eligible=${N_INFRA}"

if [ "$N_INFRA" -eq 0 ]; then
  echo "[rerun_e0] ${MODEL}: no infra-eligible rows — no-op, exiting cleanly."
  exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
BAK_JSONL="${DRA_JSONL}.pre_rerun_${TS}.bak"
BAK_SKILLS="${WORKDIR}/skills.pre_rerun_${TS}"
LOG="workdir/run_logs/rerun_infra_${MODEL}_${TS}.log"
mkdir -p workdir/run_logs

# Step 1: back up dra.jsonl (atomic-safe: we read-then-write, never truncate the source).
cp -p "$DRA_JSONL" "$BAK_JSONL"
echo "[rerun_e0] backed up dra.jsonl → $BAK_JSONL"

# Step 2: snapshot skills/ (cp -a preserves metadata; rsync-like).
if [ -d "$SKILLS_DIR" ]; then
  cp -a "$SKILLS_DIR" "$BAK_SKILLS"
  echo "[rerun_e0] snapshotted skills → $BAK_SKILLS"
else
  echo "[rerun_e0] note: $SKILLS_DIR does not exist — skipping skills snapshot"
fi

# Step 3: atomic rewrite of dra.jsonl (drop infra-eligible rows).
"$DRA_PY" - <<PY
import json, os, re, tempfile
regex = re.compile(r'''${INFRA_REGEX}''')
src = '${DRA_JSONL}'
dropped = []
kept = []
with open(src) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        err = r.get('agent_error') or ''
        if regex.search(err):
            dropped.append(r.get('task_id', '?'))
        else:
            kept.append(line)
# Write atomically.
dirn = os.path.dirname(src) or '.'
fd, tmp = tempfile.mkstemp(prefix='.dra.', suffix='.jsonl.tmp', dir=dirn)
with os.fdopen(fd, 'w') as f:
    f.writelines(kept)
os.replace(tmp, src)
print(f'DROPPED={len(dropped)} KEPT={len(kept)}')
for tid in dropped:
    print(f'  dropped: {tid}')
PY

# Step 4: assert post-rewrite row count = TARGET_ROWS - N_INFRA. Rollback on mismatch.
EXPECTED=$(( TARGET_ROWS - N_INFRA ))
AFTER=$(wc -l <"$DRA_JSONL" | tr -d ' ')
if [ "$AFTER" -ne "$EXPECTED" ]; then
  echo "FATAL: post-rewrite row count ${AFTER} != expected ${EXPECTED} — rolling back" >&2
  cp -p "$BAK_JSONL" "$DRA_JSONL"
  exit 3
fi
echo "[rerun_e0] rewrite OK: ${AFTER} rows remain (expected ${EXPECTED})"

# Step 5: re-launch run_eval_matrix.sh full <model> c4 with max_samples=N_INFRA.
# Identical FULL_CFG_OPTIONS shape as Phase 1 (shuffle=True, seed=42).
# DRA_RESUME_PRESERVE_ALL=1 so the new rows append and the 64 kept rows are
# not touched. DATASET_SPLIT=validation because C4 is a learning condition.
LAUNCH_TS="$(date +%H%M%S)"

# Keep-awake: start caffeinate if none asserting.
if ! pmset -g assertions 2>/dev/null | grep -q "caffeinate command-line"; then
  caffeinate -dimsu &
  disown
  echo "[rerun_e0] started caffeinate"
fi

echo "[rerun_e0] launching Phase 2: max_samples=${N_INFRA}"
echo "[rerun_e0] log: $LOG"

PATH="$DRA_BIN_DIR:$PATH" \
DRA_RUN_ID="$DRA_RUN_ID" \
DATASET_SPLIT=validation \
PYTHON="$DRA_PY" \
DRA_RESUME_PRESERVE_ALL=1 \
FULL_CFG_OPTIONS="max_samples=${N_INFRA} dataset.shuffle=True dataset.seed=42" \
  nohup bash scripts/run_eval_matrix.sh full "$MODEL" c4 >"$LOG" 2>&1 &
disown
LAUNCH_PID=$!

echo "[rerun_e0] ${MODEL}: launched (pid=${LAUNCH_PID}); tail with:"
echo "  tail -f $LOG"
echo "  tail -f workdir/run_logs/full_${MODEL}.log"
echo
echo "[rerun_e0] post-run verify (run after stream DONE):"
echo "  wc -l $DRA_JSONL   # expect ${TARGET_ROWS}"
echo "  $DRA_PY scripts/monitor_tick.py"
