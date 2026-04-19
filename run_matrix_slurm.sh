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
# SLURM wrapper for the API-only 12-cell GAIA eval matrix
# (Mistral / Qwen / Gemma × C0 / C2 / C3 / C4).
#
# Kimi K2.5 was dropped from the default matrix 2026-04-19 due to persistent
# OpenRouter→Moonshot AI SSE streaming stalls (see HANDOFF_TEST_EVAL.md
# methodology note). Kimi configs remain in the repo; run explicitly with
# `full kimi <cond>` if the provider-side stability improves.
#
# All three default matrix models are API-based (no local vLLM), so this job
# doesn't request GPUs — just CPU + RAM + network. The wall-clock cap is
# 24h which is plenty for smoke (default 3 Q × 12 cells on validation) or for
# orchestrating full test-split runs per scripts/run_eval_matrix.sh.
#
# Track naming (see docs/handoffs/HANDOFF_TEST_EVAL.md): **I2** = `smoke` 12-cell
# matrix on validation; **E3** = `full` test-split submission; **E0** C4 val
# training = `full '' c4` with `DATASET_SPLIT=validation` (see eval handoff).
#
# Usage (on the HKU CS Phase-3 gateway, e.g. gpu3gate1.cs.hku.hk):
#
#   cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
#   git pull origin main
#
#   # Full mode requires an explicit DATASET_SPLIT (enforced by the matrix
#   # runner — no implicit default). Use --export=ALL if your site strips env.
#
#   # E0 — C4 val training (4 models, full validation split):
#   DATASET_SPLIT=validation sbatch --export=ALL run_matrix_slurm.sh full '' c4
#
#   # E3 — C0/C2/C3 test-split submission (run each condition separately so C4
#   # cannot silently tag along and train on test — the matrix runner refuses
#   # the all-4-conditions shape on test by design):
#   DATASET_SPLIT=test sbatch --export=ALL run_matrix_slurm.sh full '' c0
#   DATASET_SPLIT=test sbatch --export=ALL run_matrix_slurm.sh full '' c2
#   DATASET_SPLIT=test sbatch --export=ALL run_matrix_slurm.sh full '' c3
#   # Frozen C4 test eval is a separate path — examples/run_gaia.py direct with
#   # agent_config.enable_skill_extraction=False — see HANDOFF_TEST_EVAL.md §E2/§E3.
#
#   # Validation-split smoke (default 3 Q/cell; override LIMIT=5). Smoke mode
#   # ignores DATASET_SPLIT (always validation + capped via max_samples).
#   sbatch run_matrix_slurm.sh smoke
#
#   # One model, one condition (cheapest, e.g. Mistral C3 on test):
#   DATASET_SPLIT=test sbatch --export=ALL run_matrix_slurm.sh full mistral c3
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
elif [[ "$MODE" == "full" ]]; then
  # DATASET_SPLIT is mandatory in full mode — the matrix runner aborts if
  # unset or inconsistent with the condition selection. Echo what was
  # received so a dropped `--export=ALL` is obvious in the job header.
  echo "DATASET_SPLIT: ${DATASET_SPLIT:-<unset — run_eval_matrix.sh will refuse>}"
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
