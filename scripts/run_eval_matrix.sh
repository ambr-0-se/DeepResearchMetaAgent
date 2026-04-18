#!/usr/bin/env bash
# Parallel GAIA evaluation runner across the 4-models × 4-conditions matrix.
#
# Strategy: 4 model streams run in parallel (different API keys, no rate-limit
# contention between streams). WITHIN each model stream, the four conditions
# C0/C2/C3/C4 run sequentially because they share one API key per model.
#
# Usage:
#   # Smoke test — default 3 validation-split Q/cell, all 16 cells (4 model streams in parallel):
#   bash scripts/run_eval_matrix.sh smoke
#
#   # Full test-split submission run (NO max_samples cap, default split=test):
#   bash scripts/run_eval_matrix.sh full
#
#   # Single model only (e.g. just Qwen — useful while burning DashScope free tier):
#   bash scripts/run_eval_matrix.sh full qwen
#
#   # Single condition only (e.g. just C0 across all models):
#   bash scripts/run_eval_matrix.sh smoke '' c0
#
# Env:
#   PYTHON     — interpreter (default: python)
#   LIMIT      — max_samples for `smoke` mode (default: 3). Set LIMIT=5 to match older docs.
#   SMOKE_CFG_OPTIONS — extra mmengine keys appended in smoke mode only. If **unset**,
#                       defaults to tight planner/browser/sub-agent caps (cost control).
#                       Set to empty string before launch to omit caps: `export SMOKE_CFG_OPTIONS=`
#                       (then only max_samples + validation split apply).
#   DATASET_SPLIT — for `full` mode only: if set (e.g. `validation`), passed as
#                   `dataset.split=...` so runs do not use the config default (`test`).
#                   **C4 skill training:** export `DATASET_SPLIT=validation` before
#                   `sbatch run_matrix_slurm.sh full '' c4` so each model trains on the
#                   **full validation set**; omit before S4 test submission.
#   LOG_DIR    — where to tee per-cell stdout/stderr (default: workdir/run_logs)
#   GEMMA_CONCURRENCY — per-Gemma-cell concurrency cap (default: 4). Workaround
#                       for vLLM #39392 (gemma4 tool parser emits all-<pad>
#                       tokens non-deterministically under parallel load).
#
# Parallelism: one bash process per model (mistral / kimi / qwen / gemma) runs in the
# background; conditions C0→C2→C3→C4 are sequential within a model. Four models ⇒ four
# parallel streams whenever all models are selected.

set -uo pipefail   # -e omitted on purpose: a failure in one cell shouldn't kill others

MODE="${1:-smoke}"            # smoke | full
ONLY_MODEL="${2:-}"           # mistral | kimi | qwen | gemma | '' (all)
ONLY_CONDITION="${3:-}"       # c0 | c2 | c3 | c4 | '' (all)

PYTHON="${PYTHON:-python}"
LIMIT="${LIMIT:-3}"
LOG_DIR="${LOG_DIR:-workdir/run_logs}"
GEMMA_CONCURRENCY="${GEMMA_CONCURRENCY:-4}"
mkdir -p "$LOG_DIR"

# Smoke-only step caps (unset SMOKE_CFG_OPTIONS entirely to get these defaults).
if [ -z "${SMOKE_CFG_OPTIONS+x}" ]; then
  SMOKE_CFG_OPTIONS="agent_config.max_steps=10 auto_browser_use_tool_config.max_steps=8 deep_analyzer_agent_config.max_steps=2 deep_researcher_agent_config.max_steps=2 browser_use_agent_config.max_steps=3 deep_researcher_tool_config.time_limit_seconds=30"
fi

# Shared run id for every cell in this invocation. Every generated config
# reads this env var (with a fresh-timestamp fallback at config load) to
# stamp its output directory. Exporting it here guarantees all 16 cells of
# one matrix invocation share a single RUN_ID so results correlate cleanly.
# Without it each cell would generate its own timestamp. Operators resume
# a prior run by exporting DRA_RUN_ID before invocation.
export DRA_RUN_ID="${DRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

ALL_MODELS=(mistral kimi qwen gemma)
ALL_CONDITIONS=(c0 c2 c3 c4)

[[ -n "$ONLY_MODEL" ]] && ALL_MODELS=("$ONLY_MODEL")
[[ -n "$ONLY_CONDITION" ]] && ALL_CONDITIONS=("$ONLY_CONDITION")

# Build the per-cell command. In smoke mode, cap question count and use the
# labeled validation split so we can score immediately. In full mode, no
# `max_samples` cap; split is `test` unless `DATASET_SPLIT` is set (use
# `DATASET_SPLIT=validation` for C4 skill training on the full validation set).
#
# Per-model overrides:
#   - gemma: concurrency capped via GEMMA_CONCURRENCY (default 4) to dodge
#     vLLM #39392 (gemma4 tool parser pad-bug under parallel load). Other
#     models fall through to the config file's own `concurrency` setting.
cell_cmd() {
  local cfg="$1"
  local model="$2"
  if [[ "$MODE" == "smoke" ]]; then
    local smoke_tail=""
    if [[ -n "${SMOKE_CFG_OPTIONS}" ]]; then
      smoke_tail=" ${SMOKE_CFG_OPTIONS}"
    fi
    local gem=""
    [[ "$model" == "gemma" ]] && gem=" concurrency=$GEMMA_CONCURRENCY"
    echo "$PYTHON examples/run_gaia.py --config $cfg --cfg-options max_samples=$LIMIT dataset.split=validation${smoke_tail}${gem}"
  else
    local opts=()
    [[ -n "${DATASET_SPLIT:-}" ]] && opts+=("dataset.split=${DATASET_SPLIT}")
    [[ "$model" == "gemma" ]] && opts+=("concurrency=$GEMMA_CONCURRENCY")
    if [[ ${#opts[@]} -gt 0 ]]; then
      echo "$PYTHON examples/run_gaia.py --config $cfg --cfg-options ${opts[*]}"
    else
      echo "$PYTHON examples/run_gaia.py --config $cfg"
    fi
  fi
}

# Per-model worker — runs all selected conditions sequentially for one model.
run_model_stream() {
  local model="$1"
  local stream_log="$LOG_DIR/${MODE}_${model}.log"
  echo "[stream:$model] starting → $stream_log"
  {
    echo "=== model=$model mode=$MODE conditions=${ALL_CONDITIONS[*]} ==="
    date
    for condition in "${ALL_CONDITIONS[@]}"; do
      local cfg="configs/config_gaia_${condition}_${model}.py"
      if [[ ! -f "$cfg" ]]; then
        echo "[stream:$model] SKIP missing $cfg"
        continue
      fi
      local cmd
      cmd="$(cell_cmd "$cfg" "$model")"
      echo "----"
      echo "[stream:$model] CELL=$condition CMD: $cmd"
      date
      eval "$cmd"
      local rc=$?
      echo "[stream:$model] CELL=$condition rc=$rc"

      # Every condition writes into a timestamped directory now. Maintain
      # a `_latest` symlink per (condition, model) so "inspect the last
      # run for model X, condition Y" is a single stable path. `ln -sfn`
      # retargets atomically. Swallow errors silently: on filesystems that
      # refuse symlinks (some NFS setups) the user can still find the run
      # by its timestamped name.
      if [[ $rc -eq 0 ]]; then
        local target="gaia_${condition}_${model}_${DRA_RUN_ID}"
        if [[ -d "workdir/$target" ]]; then
          ln -sfn "$target" "workdir/gaia_${condition}_${model}_latest" 2>/dev/null || true
        fi
      fi
    done
    date
    echo "=== stream:$model DONE ==="
  } >>"$stream_log" 2>&1 &
  echo "$!"
}

# Launch the model streams in parallel and wait for all.
echo "Launching ${#ALL_MODELS[@]} parallel model streams: ${ALL_MODELS[*]}"
echo "Conditions per stream: ${ALL_CONDITIONS[*]}"
echo "Mode: $MODE  (smoke=cap $LIMIT on validation; full=no cap on test)"
echo "Logs: $LOG_DIR/${MODE}_<model>.log"
echo "DRA_RUN_ID: $DRA_RUN_ID  (every cell writes into workdir/gaia_<cond>_<model>_\$DRA_RUN_ID/)"
echo ""

pids=()
for model in "${ALL_MODELS[@]}"; do
  pid="$(run_model_stream "$model")"
  pids+=("$pid")
done

# Wait for all streams; collect exit codes.
fail=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  model="${ALL_MODELS[$i]}"
  if wait "$pid"; then
    echo "[stream:$model] ✓ completed"
  else
    echo "[stream:$model] ✗ exit=$?"
    fail=1
  fi
done

echo ""
echo "All streams finished. Per-stream logs in $LOG_DIR/."
echo "Results:"
echo "  All cells: workdir/gaia_<cond>_<model>_${DRA_RUN_ID}/dra.jsonl"
echo "  C4 skills: workdir/gaia_c4_<model>_${DRA_RUN_ID}/skills/"
echo "  Latest:    workdir/gaia_<cond>_<model>_latest  (symlink to most recent run)"
exit "$fail"
