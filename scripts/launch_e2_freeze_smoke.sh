#!/usr/bin/env bash
# E2 freeze smoke — validate the frozen-library C3 eval path.
#
# Runs 3 validation-split questions per model with:
#   - skills_dir pinned to the trained snapshot (workdir/c4_trained_libraries/<m>_skills_v3)
#   - enable_skill_extraction=False (library frozen, no new writes)
#
# Uses the CLAUDE.md-documented override pattern: `agent_config.*` (not
# `planning_agent_config.*`, which is silently ignored by `create_agent()`).
#
# Output: workdir/gaia_c3_<model>_20260420_E2freeze/dra.jsonl — isolated
# from E0 training rows via a fresh DRA_RUN_ID.
#
# Usage: bash scripts/launch_e2_freeze_smoke.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRA_PY="/Users/ahbo/miniconda3/envs/dra/bin/python"
DRA_BIN_DIR="$(dirname "$DRA_PY")"
[ -x "$DRA_PY" ] || { echo "FATAL: $DRA_PY not found" >&2; exit 2; }
"$DRA_PY" -c "import fastmcp, pandas, mmengine" >/dev/null 2>&1 \
  || { echo "FATAL: dra env missing {fastmcp, pandas, mmengine}" >&2; exit 2; }

# Refuse double-launch.
if pgrep -f 'run_gaia.py\|run_eval_matrix' >/dev/null; then
  echo "FATAL: run_gaia / run_eval_matrix procs alive — refusing to double-launch" >&2
  pgrep -fl 'run_gaia.py\|run_eval_matrix' >&2
  exit 2
fi

# Verify the frozen libraries exist and look sane.
for m in mistral qwen; do
  lib="workdir/c4_trained_libraries/${m}_skills_v3"
  [ -d "$lib" ] || { echo "FATAL: frozen library $lib missing" >&2; exit 2; }
  [ -f "$lib/.seeded" ] || { echo "FATAL: $lib/.seeded missing — library may not be usable" >&2; exit 2; }
  skills=$(find "$lib" -name SKILL.md | wc -l | tr -d ' ')
  echo "[e2] ${m}: ${skills} skills in $lib"
done

DRA_RUN_ID="${DRA_RUN_ID:-20260420_E2freeze}"
export DRA_RUN_ID DATASET_SPLIT=validation PYTHON="$DRA_PY"
export PATH="$DRA_BIN_DIR:$PATH"

mkdir -p workdir/run_logs

# Keep-awake.
if ! pmset -g assertions 2>/dev/null | grep -q "caffeinate command-line"; then
  caffeinate -dimsu & disown
  echo "[e2] started caffeinate"
fi

echo "[e2] DRA_RUN_ID=${DRA_RUN_ID}  DATASET_SPLIT=${DATASET_SPLIT}"
TS="$(date +%H%M%S)"

for m in mistral qwen; do
  lib="workdir/c4_trained_libraries/${m}_skills_v3"
  log="workdir/run_logs/e2_freeze_${m}_${TS}.log"
  FULL_CFG_OPTIONS="max_samples=3 dataset.shuffle=True dataset.seed=42 agent_config.skills_dir=${lib} agent_config.enable_skill_extraction=False" \
    nohup bash scripts/run_eval_matrix.sh full "$m" c3 > "$log" 2>&1 &
  disown
  echo "[e2] launched $m (log=$log)"
done

echo "[e2] tail streams:  tail -f workdir/run_logs/full_{mistral,qwen}.log"
echo "[e2] outputs:       workdir/gaia_c3_{mistral,qwen}_${DRA_RUN_ID}/dra.jsonl"
echo "[e2] monitor:       $DRA_PY scripts/monitor_tick.py"
