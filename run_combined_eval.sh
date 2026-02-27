#!/bin/bash
#SBATCH --job-name=gaia-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_4080:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/combined_eval_%j.out
#SBATCH --error=logs/combined_eval_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Full GAIA Evaluation - AdaptivePlanningAgent + Qwen"
echo "Started: $(date)"
echo "========================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra

mkdir -p logs workdir

nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1

cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

# Start vLLM server in background
echo "Starting vLLM server on port 8000..."
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --served-model-name Qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 4}' &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready (up to 10 minutes)
echo ""
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM server is ready! (took ${WAITED}s)"
        break
    fi
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "✗ vLLM server not ready after 10 minutes. Aborting."
        kill $VLLM_PID
        exit 1
    fi
    echo "Waiting... (${WAITED}s / ${MAX_WAIT}s)"
    sleep 10
    WAITED=$((WAITED + 10))
done

# Run full evaluation
echo ""
echo "========================================"
echo "Starting full GAIA evaluation..."
echo "========================================"
python examples/run_gaia.py \
    --config configs/config_gaia_adaptive_qwen.py

EVAL_EXIT_CODE=$?

# Cleanup
echo ""
echo "========================================"
echo "Evaluation finished with exit code: $EVAL_EXIT_CODE"
echo "Finished: $(date)"
echo "Shutting down vLLM server..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

echo "Results saved to: workdir/gaia_adaptive_qwen/dra.jsonl"
echo "========================================"

exit $EVAL_EXIT_CODE
