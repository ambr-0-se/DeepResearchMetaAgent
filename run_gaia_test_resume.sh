#!/bin/bash
#SBATCH --job-name=gaia-test-resume
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Default: batch + 2×RTX 4080. If that fails to schedule, submit with e.g.
#   sbatch --partition=q-3090 --gres=gpu:rtx_3090:2 run_gaia_test_resume.sh
#SBATCH --gres=gpu:rtx_4080:2
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/gaia_test_resume_%j.out
#SBATCH --error=logs/gaia_test_resume_%j.err

# Resume a partial GAIA *test* split run by reusing the same workdir tag as the original job.
# run_gaia.py skips task_ids already in dra.jsonl (after filter_answers). Test-split rows with
# agent_error set are dropped on reload so those questions are retried; completed rows (including
# Unable to determine) are kept so abstentions are not re-run.
#
# Usage (TAG = directory name under workdir/, e.g. gaia_test_127877_20260327_230408):
#
#   sbatch --export=ALL,TAG=gaia_test_127877_20260327_230408 run_gaia_test_resume.sh
#
# Or from your shell:
#   export TAG=gaia_test_127877_20260327_230408
#   sbatch --export=ALL,TAG run_gaia_test_resume.sh
#
# Auto-chain (no manual resubmit after OOM / 18h wall): submit the next job from this script when
# dra.jsonl still has fewer than 301 lines. Cap with MAX_CHAIN_DEPTH to avoid infinite loops.
#   sbatch --partition=q-3090 --gres=gpu:rtx_3090:2 --time=18:00:00 \
#     --export=ALL,TAG=...,EVAL_TIMEOUT_SECS=64800,AUTO_RESUBMIT=1 run_gaia_test_resume.sh

REPO_ROOT="/userhome/cs2/ambr0se/DeepResearchMetaAgent"

if [ -z "${TAG:-}" ]; then
    echo "ERROR: TAG is not set."
    echo ""
    echo "Usage:"
    echo "  sbatch --export=ALL,TAG=<run_tag> $(basename "$0")"
    echo ""
    echo "Example:"
    echo "  sbatch --export=ALL,TAG=gaia_test_127877_20260327_230408 $(basename "$0")"
    echo ""
    echo "<run_tag> must match an existing directory: workdir/<run_tag>/"
    exit 1
fi

RESULT_DIR="workdir/${TAG}"
RESULT_FILE="${RESULT_DIR}/dra.jsonl"
SUBMISSION_FILE="${RESULT_DIR}/submission.jsonl"

if [ ! -d "$RESULT_DIR" ]; then
    echo "ERROR: workdir not found: $RESULT_DIR"
    exit 1
fi

echo "========================================"
echo "GAIA Test-Split — RESUME"
echo "========================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURM_NODELIST:-N/A}"
echo "TAG:         $TAG"
echo "Results:     $RESULT_FILE"
echo "Submission:  $SUBMISSION_FILE"
if [ -f "$RESULT_FILE" ]; then
    echo "Existing rows: $(wc -l < "$RESULT_FILE")"
else
    echo "Existing rows: 0 (no dra.jsonl yet — full run will start)"
fi
echo "Started:     $(date)"
echo "========================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra

mkdir -p logs workdir

nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1

cd "$REPO_ROOT"

bash scripts/ensure_playwright_browsers.sh || {
    echo "ERROR: Playwright Chromium missing (needed for browser_use_agent). See scripts/ensure_playwright_browsers.sh"
    exit 1
}

# ---------------------------------------------------------------------------
# vLLM lifecycle (same as run_gaia_test_eval.sh)
# ---------------------------------------------------------------------------
VLLM_PID_FILE=$(mktemp /tmp/vllm_pid.XXXXXX)
MONITOR_PID=""
MAX_VLLM_RESTARTS=5

cleanup() {
    echo ""
    echo "Cleanup triggered..."
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null
        wait "$MONITOR_PID" 2>/dev/null
    fi
    if [ -f "$VLLM_PID_FILE" ]; then
        local pid
        pid=$(cat "$VLLM_PID_FILE" 2>/dev/null)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
        fi
        rm -f "$VLLM_PID_FILE"
    fi
    echo "Cleanup done. Exiting."
}
trap cleanup EXIT INT TERM

launch_vllm() {
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-VL-4B-Instruct \
      --served-model-name Qwen \
      --host 0.0.0.0 \
      --port 8000 \
      --max-model-len 32768 \
      --max-num-seqs 4 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --limit-mm-per-prompt '{"image": 4}' &
    echo $! > "$VLLM_PID_FILE"
}

wait_for_vllm() {
    local max_wait=${1:-600}
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s --max-time 5 http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "vLLM server is ready! (took ${waited}s)"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        echo "Waiting for vLLM... (${waited}s / ${max_wait}s)"
    done
    echo "vLLM server not ready after ${max_wait}s."
    return 1
}

monitor_vllm() {
    local fail_count=0
    local restart_count=0
    while true; do
        sleep 30
        if ! curl -s --max-time 5 http://localhost:8000/v1/models > /dev/null 2>&1; then
            fail_count=$((fail_count + 1))
            echo "[watchdog] vLLM health check failed ($fail_count/3)"
            if [ $fail_count -ge 3 ]; then
                if [ $restart_count -ge $MAX_VLLM_RESTARTS ]; then
                    echo "[watchdog] Max restarts ($MAX_VLLM_RESTARTS) reached. Giving up."
                    return 1
                fi
                restart_count=$((restart_count + 1))
                echo "[watchdog] vLLM unresponsive for 90s. Restarting (attempt $restart_count/$MAX_VLLM_RESTARTS)..."
                local old_pid
                old_pid=$(cat "$VLLM_PID_FILE" 2>/dev/null)
                if [ -n "$old_pid" ]; then
                    kill "$old_pid" 2>/dev/null
                    wait "$old_pid" 2>/dev/null
                fi
                sleep 5
                launch_vllm
                local new_pid
                new_pid=$(cat "$VLLM_PID_FILE" 2>/dev/null)
                echo "[watchdog] Launched new vLLM (PID: $new_pid). Waiting for readiness..."
                if ! wait_for_vllm 600; then
                    echo "[watchdog] vLLM failed to start after restart."
                    continue
                fi
                fail_count=0
            fi
        else
            fail_count=0
        fi
    done
}

echo "Starting vLLM server on port 8000..."
launch_vllm
echo "vLLM server started with PID: $(cat "$VLLM_PID_FILE")"

echo ""
echo "Waiting for vLLM server to be ready..."
if ! wait_for_vllm 600; then
    echo "vLLM server not ready after 10 minutes. Aborting."
    exit 1
fi

monitor_vllm &
MONITOR_PID=$!

# Default: 72h (matches batch partition). For q-3090 (max 18h), pass e.g.
#   EVAL_TIMEOUT_SECS=64800 sbatch --partition=q-3090 --time=18:00:00 ...
EVAL_TIMEOUT_SECS=${EVAL_TIMEOUT_SECS:-259200}

echo ""
echo "========================================"
echo "Resuming GAIA test evaluation (tag=$TAG, timeout: ${EVAL_TIMEOUT_SECS}s)..."
echo "========================================"
timeout $EVAL_TIMEOUT_SECS python examples/run_gaia.py \
    --config configs/config_gaia_test_qwen.py \
    --cfg-options tag="$TAG"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -eq 124 ]; then
    echo "WARNING: Evaluation timed out after ${EVAL_TIMEOUT_SECS}s!"
fi

echo ""
echo "========================================"
echo "Exporting leaderboard submission..."
echo "========================================"
if [ -f "$RESULT_FILE" ]; then
    python scripts/export_gaia_submission.py "$RESULT_FILE" -o "$SUBMISSION_FILE"
else
    echo "WARNING: Result file not found at $RESULT_FILE"
fi

echo ""
echo "========================================"
echo "Resume job finished with exit code: $EVAL_EXIT_CODE"
echo "Finished: $(date)"
echo "Results:    $RESULT_FILE"
echo "Submission: $SUBMISSION_FILE"
if [ -f "$RESULT_FILE" ]; then
    python3 -c "
import json
with open('$RESULT_FILE') as f:
    results = [json.loads(l) for l in f]
errors = sum(1 for r in results if r.get('agent_error'))
no_answer = sum(1 for r in results if not r.get('prediction'))
print(f'  Total lines: {len(results)} / 301')
print(f'  Errors:      {errors}')
print(f'  No answer:   {no_answer}')
"
fi
echo "Leaderboard: https://huggingface.co/spaces/gaia-benchmark/leaderboard"
echo "========================================"

# Optional: submit another resume job if the benchmark is not finished (OOM/timeouts stop the
# inner python process; this starts a fresh allocation so RSS resets).
AUTO_RESUBMIT="${AUTO_RESUBMIT:-0}"
CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
MAX_CHAIN_DEPTH="${MAX_CHAIN_DEPTH:-20}"
REQUIRED_LINES="${REQUIRED_LINES:-301}"
if [ -f "$RESULT_FILE" ]; then
    LINE_COUNT=$(wc -l < "$RESULT_FILE")
else
    LINE_COUNT=0
fi

if [ "$AUTO_RESUBMIT" = "1" ] && [ "$LINE_COUNT" -lt "$REQUIRED_LINES" ] && [ "$CHAIN_DEPTH" -lt "$MAX_CHAIN_DEPTH" ]; then
    NEXT_DEPTH=$((CHAIN_DEPTH + 1))
    RP="${RESUBMIT_PARTITION:-q-3090}"
    RG="${RESUBMIT_GRES:-gpu:rtx_3090:2}"
    RM="${RESUBMIT_MEM:-128G}"
    RC="${RESUBMIT_CPUS:-8}"
    RT="${RESUBMIT_TIME:-18:00:00}"
    ETS="${EVAL_TIMEOUT_SECS:-64800}"
    echo ""
    echo "========================================"
    echo "AUTO_RESUBMIT: progress ${LINE_COUNT}/${REQUIRED_LINES} (chain ${NEXT_DEPTH}/${MAX_CHAIN_DEPTH})"
    echo "Submitting follow-up job to ${RP}..."
    echo "========================================"
    NEXT_ID=$(sbatch --parsable \
        --partition="$RP" \
        --gres="$RG" \
        --mem="$RM" \
        --cpus-per-task="$RC" \
        --time="$RT" \
        --job-name="gaia-resume-${NEXT_DEPTH}" \
        --output=logs/gaia_test_resume_%j.out \
        --error=logs/gaia_test_resume_%j.err \
        --export=ALL,TAG="$TAG",EVAL_TIMEOUT_SECS="$ETS",AUTO_RESUBMIT=1,CHAIN_DEPTH="$NEXT_DEPTH",MAX_CHAIN_DEPTH="$MAX_CHAIN_DEPTH",REQUIRED_LINES="$REQUIRED_LINES",RESUBMIT_PARTITION="$RP",RESUBMIT_GRES="$RG",RESUBMIT_MEM="$RM",RESUBMIT_CPUS="$RC",RESUBMIT_TIME="$RT" \
        "$REPO_ROOT/run_gaia_test_resume.sh") || NEXT_ID=""
    if [ -n "$NEXT_ID" ]; then
        echo "Submitted follow-up job: $NEXT_ID  (squeue -j $NEXT_ID)"
    else
        echo "WARNING: sbatch follow-up failed (check permissions / partition)."
    fi
elif [ "$AUTO_RESUBMIT" = "1" ] && [ "$LINE_COUNT" -ge "$REQUIRED_LINES" ]; then
    echo ""
    echo "AUTO_RESUBMIT: complete (${LINE_COUNT}/${REQUIRED_LINES}). No follow-up job."
elif [ "$AUTO_RESUBMIT" = "1" ] && [ "$CHAIN_DEPTH" -ge "$MAX_CHAIN_DEPTH" ]; then
    echo ""
    echo "AUTO_RESUBMIT: MAX_CHAIN_DEPTH ($MAX_CHAIN_DEPTH) reached with ${LINE_COUNT}/${REQUIRED_LINES} lines. Stop."
fi

exit $EVAL_EXIT_CODE
