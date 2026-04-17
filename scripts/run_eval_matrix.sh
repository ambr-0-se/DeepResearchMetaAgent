#!/usr/bin/env bash
# Parallel GAIA evaluation runner across the 3-models × 4-conditions matrix.
#
# Strategy: 3 model streams run in parallel (different API keys, no rate-limit
# contention between streams). WITHIN each model stream, the four conditions
# C0/C2/C3/C4 run sequentially because they share one API key per model.
#
# Usage:
#   # Smoke test — 5 validation-split questions per cell, all 12 cells:
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
#   LIMIT      — override max_samples for `smoke` mode (default: 5)
#   LOG_DIR    — where to tee per-cell stdout/stderr (default: workdir/run_logs)

set -uo pipefail   # -e omitted on purpose: a failure in one cell shouldn't kill others

MODE="${1:-smoke}"            # smoke | full
ONLY_MODEL="${2:-}"           # mistral | kimi | qwen | '' (all)
ONLY_CONDITION="${3:-}"       # c0 | c2 | c3 | c4 | '' (all)

PYTHON="${PYTHON:-python}"
LIMIT="${LIMIT:-5}"
LOG_DIR="${LOG_DIR:-workdir/run_logs}"
mkdir -p "$LOG_DIR"

# Shared run id for every C4 cell in this invocation. The env var is
# exported so each child process sees the same value and writes to the
# same per-cell dir — without it each cell would generate its own timestamp
# at config load and you could not correlate Mistral/Kimi/Qwen C4 runs.
# Operators resume a prior run by exporting DRA_RUN_ID before invocation.
export DRA_RUN_ID="${DRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

ALL_MODELS=(mistral kimi qwen)
ALL_CONDITIONS=(c0 c2 c3 c4)

[[ -n "$ONLY_MODEL" ]] && ALL_MODELS=("$ONLY_MODEL")
[[ -n "$ONLY_CONDITION" ]] && ALL_CONDITIONS=("$ONLY_CONDITION")

# Build the per-cell command. In smoke mode, cap question count and use the
# labeled validation split so we can score immediately. In full mode, no cap
# and the test split (default in config_gaia.py).
cell_cmd() {
  local cfg="$1"
  if [[ "$MODE" == "smoke" ]]; then
    echo "$PYTHON examples/run_gaia.py --config $cfg --cfg-options max_samples=$LIMIT dataset.split=validation"
  else
    echo "$PYTHON examples/run_gaia.py --config $cfg"
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
      cmd="$(cell_cmd "$cfg")"
      echo "----"
      echo "[stream:$model] CELL=$condition CMD: $cmd"
      date
      eval "$cmd"
      local rc=$?
      echo "[stream:$model] CELL=$condition rc=$rc"

      # C4 runs write into a timestamped directory (per-run skill isolation).
      # Maintain a `_latest` symlink so "inspect the last C4 run for model X"
      # is a single stable path. `ln -sfn` retargets atomically. Swallow
      # errors silently: on filesystems that refuse symlinks (some NFS
      # setups) the user can still find the run by timestamped name.
      if [[ "$condition" == "c4" && $rc -eq 0 ]]; then
        local target="gaia_c4_${model}_${DRA_RUN_ID}"
        if [[ -d "workdir/$target" ]]; then
          ln -sfn "$target" "workdir/gaia_c4_${model}_latest" 2>/dev/null || true
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
echo "DRA_RUN_ID: $DRA_RUN_ID  (C4 writes into workdir/gaia_c4_<model>_\$DRA_RUN_ID/)"
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
echo "  C0/C2/C3: workdir/gaia_<condition>_<model>/dra.jsonl"
echo "  C4:       workdir/gaia_c4_<model>_${DRA_RUN_ID}/dra.jsonl"
echo "  C4 skills: workdir/gaia_c4_<model>_${DRA_RUN_ID}/skills/"
echo "  C4 latest: workdir/gaia_c4_<model>_latest  (symlink to most recent run)"
exit "$fail"
