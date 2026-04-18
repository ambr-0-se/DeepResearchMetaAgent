#!/bin/bash
#SBATCH --job-name=gaia-matrix
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/matrix_%j.out
#SBATCH --error=logs/matrix_%j.err
#
# SLURM wrapper for the API-only 16-cell GAIA eval matrix
# (Mistral / Kimi / Qwen / Gemma × C0 / C2 / C3 / C4).
#
# All four matrix models are API-based (no local vLLM), so this job
# doesn't request GPUs — just CPU + RAM + network. The wall-clock cap is
# 24h which is plenty for smoke (default 3 Q × 16 cells on validation) or for
# orchestrating full test-split runs per scripts/run_eval_matrix.sh.
#
# Track naming (see docs/handoffs/HANDOFF_TEST_EVAL.md): **I2** = `smoke` 16-cell
# matrix on validation; **E3** = `full` test-split submission; **E0** C4 val
# training = `full '' c4` with `DATASET_SPLIT=validation` (see eval handoff).
#
# Usage (on the HKU CS Phase-3 gateway, e.g. gpu3gate1.cs.hku.hk):
#
#   cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
#   git pull origin main
#
#   # Full matrix (all 16 cells, test split):
#   sbatch run_matrix_slurm.sh full
#
#   # Just one condition — e.g. only C3 across all 4 models:
#   sbatch run_matrix_slurm.sh full '' c3
#
#   # Validation-split smoke (default 3 Q/cell; override LIMIT=5):
#   sbatch run_matrix_slurm.sh smoke
#
#   # One model, one condition (cheapest):
#   sbatch run_matrix_slurm.sh full mistral c3
#
# Job survives SSH disconnect by construction. Check progress with:
#
#   squeue -u $USER                       # is it running?
#   scontrol show job <JOBID> | grep RunT # elapsed time
#   tail -f logs/matrix_<JOBID>.out       # live log
#   scancel <JOBID>                       # abort
#
# Results land in workdir/gaia_<condition>_<model>_<run_id>/dra.jsonl
# Per-cell greps via: bash scripts/validate_handoffs.sh <run_id>

set -u

MODE="${1:-full}"            # smoke | full
ONLY_MODEL="${2:-}"          # mistral | kimi | qwen | gemma | '' (all)
ONLY_CONDITION="${3:-}"      # c0 | c2 | c3 | c4 | '' (all)

echo "========================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURM_NODELIST:-$(hostname)}"
echo "MODE:          $MODE"
echo "ONLY_MODEL:    '${ONLY_MODEL}'"
echo "ONLY_CONDITION:'${ONLY_CONDITION}'"
if [[ "$MODE" == "smoke" ]]; then
  echo "Smoke LIMIT:   ${LIMIT:-3}  (override: export LIMIT=5 before sbatch)"
  echo "Smoke caps:    SMOKE_CFG_OPTIONS ${SMOKE_CFG_OPTIONS:+set}${SMOKE_CFG_OPTIONS:-unset→defaults in run_eval_matrix.sh}"
elif [[ "$MODE" == "full" && -n "${DATASET_SPLIT:-}" ]]; then
  echo "DATASET_SPLIT: ${DATASET_SPLIT}  (full-mode override; empty unset=test default)"
fi
echo "Started:       $(date)"
echo "========================================"

# Activate the shared conda env (same layout used by run_combined_eval.sh)
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate dra

mkdir -p logs workdir workdir/run_logs

# Pin a single DRA_RUN_ID for the whole matrix so all 16 cells land under
# the same workdir/gaia_<cond>_<model>_<run_id>/ tree — easy to compare.
RUN_ID="${DRA_RUN_ID:-matrix_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)}"
export DRA_RUN_ID="$RUN_ID"
echo "DRA_RUN_ID=$RUN_ID"
echo "DRA_RUN_ID=$RUN_ID" > workdir/run_logs/matrix_runid.txt

# Playwright is only strictly needed for browser_use_agent tests; keep it
# optional so the script can run on farm nodes that don't ship Chromium.
if [ -f scripts/ensure_playwright_browsers.sh ]; then
    bash scripts/ensure_playwright_browsers.sh || echo "(playwright not installed; browser tool will fall back)"
fi

# Kick off the matrix runner with the user-supplied filters.
bash scripts/run_eval_matrix.sh "$MODE" "$ONLY_MODEL" "$ONLY_CONDITION"
MATRIX_RC=$?

echo "========================================"
echo "matrix runner exit: $MATRIX_RC"
echo "Finished: $(date)"
echo "========================================"

# Always run the grep sweep; surfaces per-handoff evidence in the SLURM output.
bash scripts/validate_handoffs.sh "$RUN_ID" 2>&1 | sed 's/^/[validate] /' || true

exit $MATRIX_RC
