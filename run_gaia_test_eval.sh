#!/bin/bash
#SBATCH --job-name=gaia-test-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_4080:2
# Use at least 128G: 64G caused cgroup OOM kills mid-run (slurmstepd: oom_kill events) on job 127877
# — Python eval + vLLM + tokenizer/MCP subprocesses exceed 64G RSS over multi-hour runs.
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/gaia_test_eval_%j.out
#SBATCH --error=logs/gaia_test_eval_%j.err

# Full GAIA test-split evaluation (301 questions) with Qwen3-VL-4B-Instruct.
# After evaluation, exports a submission JSONL for the GAIA leaderboard.
#
# Usage:
#   sbatch run_gaia_test_eval.sh
#
# Results:
#   workdir/gaia_test_<JOBID>_<timestamp>/dra.jsonl          — full agent output
#   workdir/gaia_test_<JOBID>_<timestamp>/submission.jsonl   — leaderboard submission
#
# Submit to leaderboard:
#   https://huggingface.co/spaces/gaia-benchmark/leaderboard

RUN_TAG="gaia_test_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="workdir/${RUN_TAG}"
RESULT_FILE="${RESULT_DIR}/dra.jsonl"
SUBMISSION_FILE="${RESULT_DIR}/submission.jsonl"

echo "========================================"
echo "GAIA Test-Split Full Evaluation"
echo "========================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Run tag:     $RUN_TAG"
echo "Results:     $RESULT_FILE"
echo "Submission:  $SUBMISSION_FILE"
echo "Started:     $(date)"
echo "========================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra

mkdir -p logs workdir

nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1

cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

bash scripts/ensure_playwright_browsers.sh || {
    echo "ERROR: Playwright Chromium missing (needed for browser_use_agent). See scripts/ensure_playwright_browsers.sh"
    exit 1
}

# ---------------------------------------------------------------------------
# vLLM lifecycle management
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

# ---------------------------------------------------------------------------
# Start vLLM
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Run evaluation — 301 test questions
# ---------------------------------------------------------------------------
# Worst case: 301 questions × 20 min = ~100 h (sequential).
# Realistic: ~4 min avg = ~20 h. 72h SBATCH wall-clock provides headroom.
EVAL_TIMEOUT_SECS=259200  # 72h, matches SBATCH --time

echo ""
echo "========================================"
echo "Starting GAIA test evaluation (301 questions, timeout: ${EVAL_TIMEOUT_SECS}s)..."
echo "========================================"
timeout $EVAL_TIMEOUT_SECS python examples/run_gaia.py \
    --config configs/config_gaia_test_qwen.py \
    --cfg-options tag=$RUN_TAG

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -eq 124 ]; then
    echo "WARNING: Evaluation timed out after ${EVAL_TIMEOUT_SECS}s!"
fi

# ---------------------------------------------------------------------------
# Export submission file
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Exporting leaderboard submission..."
echo "========================================"
if [ -f "$RESULT_FILE" ]; then
    python scripts/export_gaia_submission.py "$RESULT_FILE" -o "$SUBMISSION_FILE"
else
    echo "WARNING: Result file not found at $RESULT_FILE"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Evaluation finished with exit code: $EVAL_EXIT_CODE"
echo "Finished: $(date)"
echo ""
echo "Results:    $RESULT_FILE"
echo "Submission: $SUBMISSION_FILE"
if [ -f "$RESULT_FILE" ]; then
    TOTAL=$(wc -l < "$RESULT_FILE")
    ERRORS=$(python3 -c "
import json
with open('$RESULT_FILE') as f:
    results = [json.loads(l) for l in f]
errors = sum(1 for r in results if r.get('agent_error'))
no_answer = sum(1 for r in results if not r.get('prediction'))
print(f'  Total:      {len(results)} / 301 questions')
print(f'  Errors:     {errors}')
print(f'  No answer:  {no_answer}')
")
    echo "$ERRORS"
fi
echo ""
echo "To submit to the GAIA leaderboard:"
echo "  1. Go to https://huggingface.co/spaces/gaia-benchmark/leaderboard"
echo "  2. Log in with your HuggingFace account"
echo "  3. Upload: $SUBMISSION_FILE"
echo "========================================"

exit $EVAL_EXIT_CODE
