#!/bin/bash
#SBATCH --job-name=arc-test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_4080:2
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/arc_test_%j.out
#SBATCH --error=logs/arc_test_%j.err

# Smoke test: 10 ARC test cases with Qwen on vLLM (same lifecycle as run_combined_test.sh).
# Results: workdir/arc_test_<JOBID>_<timestamp>/dra.jsonl
RUN_TAG="arc_test_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
RESULT_FILE="workdir/${RUN_TAG}/dra.jsonl"

echo "========================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Run tag:  $RUN_TAG"
echo "Results:  $RESULT_FILE"
echo "Started:  $(date)"
echo "========================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra

mkdir -p logs workdir

nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1

cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

VLLM_PID_FILE=$(mktemp /tmp/vllm_arc_pid.XXXXXX)
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
    local max_wait=${1:-300}
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
                if ! wait_for_vllm 300; then
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

# 10 cases × 20 min max each + buffer
EVAL_TIMEOUT_SECS=14400

echo ""
echo "========================================"
echo "Starting ARC-AGI evaluation (10 test cases, timeout: ${EVAL_TIMEOUT_SECS}s)..."
echo "========================================"
timeout $EVAL_TIMEOUT_SECS python examples/run_arc.py \
    --config configs/config_arc_qwen.py \
    --cfg-options tag=$RUN_TAG max_samples=10

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -eq 124 ]; then
    echo "WARNING: Evaluation timed out after ${EVAL_TIMEOUT_SECS}s!"
fi

echo ""
echo "========================================"
echo "Evaluation finished with exit code: $EVAL_EXIT_CODE"
echo "Finished: $(date)"

echo ""
echo "Results: $RESULT_FILE"
if [ -f "$RESULT_FILE" ]; then
    echo ""
    echo "=== Final Score (ARC grid match) ==="
    export ARC_RESULT_FILE="$RESULT_FILE"
    python3 << 'PYEOF'
import json, os, sys
sys.path.insert(0, os.getcwd())
from src.metric import arc_question_scorer
path = os.environ["ARC_RESULT_FILE"]
with open(path) as f:
    results = [json.loads(l) for l in f]
total = len(results)
scorable = [r for r in results if str(r.get("true_answer", "")).strip() not in ("", "?")]
correct = sum(
    1
    for r in scorable
    if r.get("prediction") is not None
    and arc_question_scorer(str(r["prediction"]), str(r["true_answer"]))
)
errors = sum(1 for r in results if r.get("agent_error"))
retried = sum(1 for r in results if (r.get("attempts") or 1) > 1)
print(f"  Total:    {total} test case(s)")
if scorable:
    print(f"  Correct:  {correct} / {len(scorable)}  ({100*correct/len(scorable):.1f}%)")
else:
    print("  Correct:  N/A")
print(f"  Errors:   {errors}")
if retried:
    print(f"  Retried:  {retried} case(s) had transient-error retries")
if results:
    print(f"  From:     {results[0].get('start_time', 'N/A')}")
    print(f"  To:       {results[-1].get('end_time', 'N/A')}")
PYEOF
fi
echo "========================================"

exit $EVAL_EXIT_CODE
