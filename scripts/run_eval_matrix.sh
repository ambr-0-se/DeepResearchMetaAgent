#!/usr/bin/env bash
# Parallel GAIA evaluation runner.
#
# DEFAULT MATRIX (2026-04-19 onward): 3 models × 4 conditions = **12 cells**
# (mistral / qwen / gemma × C0/C2/C3/C4). Kimi K2.5 was dropped from the
# default set after persistent OpenRouter→Moonshot AI SSE streaming stalls
# during E0 validation training made its data unreliable. Kimi configs
# remain in the repo — pass `model=kimi` explicitly to re-enable.
#
# Track naming (see docs/handoffs/HANDOFF_TEST_EVAL.md): **`smoke`** = integration
# **I2** (12-cell default validation matrix); **`full`** (default test split) =
# evaluation **E3** submission when scoring the official matrix. C4 val
# training before E3 is **E0** (`DATASET_SPLIT=validation`, `full '' c4`).
#
# Strategy: 4 model streams run in parallel (different API keys, no rate-limit
# contention between streams). WITHIN each model stream, the four conditions
# C0/C2/C3/C4 run sequentially because they share one API key per model.
#
# Usage:
#   # Smoke test — default 3 validation-split Q/cell, all 12 cells (3 model streams in parallel):
#   bash scripts/run_eval_matrix.sh smoke
#
#   # Include Kimi (opt-in — slow/flaky on OR):
#   bash scripts/run_eval_matrix.sh smoke kimi
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
#   DATASET_SPLIT — **REQUIRED** in `full` mode. Must be explicitly set to one of
#                   {`validation`, `test`}; no implicit default. The script refuses
#                   to launch otherwise. Rules (enforced, not advisory):
#                     - If C4 is among the selected conditions → must be
#                       `validation` (C4 runs in extraction-on / training mode here;
#                       training on `test` would leak the test set into the learned
#                       skill library and invalidate every downstream test score).
#                       Frozen-C4 test evaluation does not go through this runner —
#                       see `docs/handoffs/HANDOFF_TEST_EVAL.md` §E2/E3.
#                     - If C4 is NOT among the selected conditions (`c0` | `c2` | `c3`
#                       alone) → must be `test` (all reported evaluation scores live
#                       on the test split).
#                   Rationale: the two catastrophic misconfigurations ("train C4 on
#                   test" and "score C0–C3 on validation") are structurally
#                   indistinguishable from valid invocations except for this env
#                   var, so the operator must commit to a split before the job
#                   starts spending money. Smoke mode is unaffected — it always
#                   uses `validation` with a `max_samples` cap.
#   FULL_CFG_OPTIONS — extra mmengine keys appended in `full` mode only. Unset
#                      by default (no overrides). Same syntax as SMOKE_CFG_OPTIONS.
#                      Intended for one-off cap tightening at launch time
#                      without regenerating checked-in configs (e.g. shorten
#                      E0 per-Q wall by reducing planner max_steps).
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
# I-track smokes verify the pipeline wires up and runs — not accuracy. Caps
# are intentionally tight: hitting `max_steps` on every question is a
# SUCCESS state for validation (proves the planner loop and its cap enforce
# correctly, and the question exits in bounded time). An earlier tighter
# pass (planner=6/browser=4) thrashed only because firecrawl-py 4.22 was
# crashing the search path; that's fixed in 42583af. With a working
# pipeline, tight caps just mean "run hits the ceiling, next Q starts."
if [ -z "${SMOKE_CFG_OPTIONS+x}" ]; then
  SMOKE_CFG_OPTIONS="agent_config.max_steps=4 auto_browser_use_tool_config.max_steps=3 deep_analyzer_agent_config.max_steps=2 deep_researcher_agent_config.max_steps=2 browser_use_agent_config.max_steps=2 deep_researcher_tool_config.time_limit_seconds=20"
fi

# Shared run id for every cell in this invocation. Every generated config
# reads this env var (with a fresh-timestamp fallback at config load) to
# stamp its output directory. Exporting it here guarantees all 12 cells of
# one default matrix invocation share a single RUN_ID so results correlate cleanly.
# Without it each cell would generate its own timestamp. Operators resume
# a prior run by exporting DRA_RUN_ID before invocation.
export DRA_RUN_ID="${DRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# Kimi excluded from the default model set as of 2026-04-19. The
# OpenRouter→Moonshot AI route exhibited persistent SSE streaming stalls
# during E0 training (58-min post-timeout cleanup deadlocks, 9% effective
# throughput across 5.8h). Provider-side reliability issues — outside our
# code's control — would have carried into E3 test scoring and contaminated
# the C0/C2/C3/C4 ablation deltas. Kimi configs remain in the repo and can
# be re-enabled by passing model=kimi explicitly (e.g., `smoke kimi c0`);
# they are just not run by default. See `docs/handoffs/HANDOFF_TEST_EVAL.md`
# methodology note and HANDOFF_INDEX.md commit row for full rationale.
ALL_MODELS=(mistral qwen gemma)
ALL_CONDITIONS=(c0 c2 c3 c4)

[[ -n "$ONLY_MODEL" ]] && ALL_MODELS=("$ONLY_MODEL")
[[ -n "$ONLY_CONDITION" ]] && ALL_CONDITIONS=("$ONLY_CONDITION")

# --- Full-mode split mandate -------------------------------------------------
# `DATASET_SPLIT` is required and must match the condition set. This block
# enforces the two invariants the eval protocol depends on:
#   (1) never train on test   — C4 (extraction-on here) ⇒ split must be validation
#   (2) always score on test  — c0 | c2 | c3           ⇒ split must be test
# There is intentionally NO escape hatch: both misconfigurations silently
# invalidate the entire experimental matrix, and neither is recoverable after
# the run has spent compute. Smoke mode is exempt (always validation + capped
# via max_samples).
if [[ "$MODE" == "full" ]]; then
  c4_in_run=0
  for _c in "${ALL_CONDITIONS[@]}"; do
    [[ "$_c" == "c4" ]] && c4_in_run=1
  done

  if [[ -z "${DATASET_SPLIT:-}" ]]; then
    {
      echo "ERROR: DATASET_SPLIT is not set."
      echo ""
      echo "Full-mode runs must commit to a split explicitly. Required values:"
      echo "  - C4 in conditions (training)  → export DATASET_SPLIT=validation"
      echo "  - C0/C2/C3 alone (evaluation)  → export DATASET_SPLIT=test"
      echo ""
      echo "Selected conditions: ${ALL_CONDITIONS[*]}"
      echo "See docs/handoffs/HANDOFF_TEST_EVAL.md §E0 / §E3."
    } >&2
    exit 2
  fi

  case "$DATASET_SPLIT" in
    validation|test) ;;
    *)
      echo "ERROR: DATASET_SPLIT must be 'validation' or 'test' (got: '$DATASET_SPLIT')." >&2
      exit 2
      ;;
  esac

  if [[ "$c4_in_run" == "1" && "$DATASET_SPLIT" != "validation" ]]; then
    {
      echo "ERROR: C4 is in the selected conditions (${ALL_CONDITIONS[*]}) but DATASET_SPLIT=$DATASET_SPLIT."
      echo ""
      echo "C4 runs with skill extraction ON via this matrix runner (training mode)."
      echo "Training on the test split leaks test content into the learned skill"
      echo "library and invalidates every downstream test score."
      echo ""
      echo "Fix one of:"
      echo "  (a) export DATASET_SPLIT=validation  # if this is an E0 C4 training run"
      echo "  (b) drop c4 from the conditions and rerun C0/C2/C3 on test"
      echo "      (frozen C4 test evaluation goes through examples/run_gaia.py"
      echo "       with agent_config.enable_skill_extraction=False overrides —"
      echo "       see HANDOFF_TEST_EVAL.md §E2/§E3, not this script)"
    } >&2
    exit 2
  fi

  if [[ "$c4_in_run" == "0" && "$DATASET_SPLIT" != "test" ]]; then
    {
      echo "ERROR: Conditions ${ALL_CONDITIONS[*]} do not include C4 but DATASET_SPLIT=$DATASET_SPLIT."
      echo ""
      echo "Reported evaluation scores are on the test split only. Running C0/C2/C3"
      echo "at full scale on validation burns budget without producing a submittable"
      echo "number."
      echo ""
      echo "Fix: export DATASET_SPLIT=test"
    } >&2
    exit 2
  fi
fi
# ---------------------------------------------------------------------------

# Build the per-cell command. In smoke mode, cap question count and use the
# labeled validation split so we can score immediately. In full mode,
# `DATASET_SPLIT` is already validated above (required, and matches the
# conditions), so we can pass it through unconditionally.
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
    # Full mode: DATASET_SPLIT is mandatory (enforced earlier).
    local opts=("dataset.split=${DATASET_SPLIT}")
    [[ "$model" == "gemma" ]] && opts+=("concurrency=$GEMMA_CONCURRENCY")
    # Optional `FULL_CFG_OPTIONS` env lets the operator tighten max_steps
    # etc. without regenerating checked-in configs. Used 2026-04-19 to
    # shorten E0 resume's per-Q wall from ~20 min (hitting the 1200s
    # per-Q safety timeout) to ~10 min (hitting max_steps cleanly, which
    # writes a valid row instead of `agent_error`).
    if [[ -n "${FULL_CFG_OPTIONS:-}" ]]; then
      # shellcheck disable=SC2206
      opts+=(${FULL_CFG_OPTIONS})
    fi
    echo "$PYTHON examples/run_gaia.py --config $cfg --cfg-options ${opts[*]}"
  fi
}

# Per-model worker — runs all selected conditions sequentially for one model.
# Called as a backgrounded function in the launch loop below (`foo &`); do NOT
# background anything inside — the caller owns the child PID via `$!` and must
# be able to `wait` on it. An earlier version tried to capture the PID via
# `$(run_model_stream "$model")`, which put the function (and its background
# child) in a subshell the parent can't wait on — breaking both PID handling
# and the exit-code roll-up.
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
  } >>"$stream_log" 2>&1
}

# Launch the model streams in parallel and wait for all.
echo "Launching ${#ALL_MODELS[@]} parallel model streams: ${ALL_MODELS[*]}"
echo "Conditions per stream: ${ALL_CONDITIONS[*]}"
if [[ "$MODE" == "smoke" ]]; then
  echo "Mode: smoke  (cap $LIMIT on validation)"
else
  echo "Mode: full   (no cap on ${DATASET_SPLIT})"
fi
echo "Logs: $LOG_DIR/${MODE}_<model>.log"
echo "DRA_RUN_ID: $DRA_RUN_ID  (every cell writes into workdir/gaia_<cond>_<model>_\$DRA_RUN_ID/)"
echo ""

pids=()
for model in "${ALL_MODELS[@]}"; do
  run_model_stream "$model" &
  pids+=("$!")
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
