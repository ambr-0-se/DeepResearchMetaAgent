#!/bin/bash
#SBATCH --job-name=gaia-test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_4080:2
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/combined_test_%j.out
#SBATCH --error=logs/combined_test_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Combined vLLM Server + GAIA Evaluation"
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

# Wait for vLLM server to be ready
echo ""
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=300  # 5 minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM server is ready!"
        break
    fi
    if [ $WAITED -eq $MAX_WAIT ]; then
        echo "✗ vLLM server not ready after 5 minutes"
        kill $VLLM_PID
        exit 1
    fi
    echo "Waiting... (${WAITED}s / ${MAX_WAIT}s)"
    sleep 10
    WAITED=$((WAITED + 10))
done

# Show available models
echo ""
echo "Available models:"
curl -s http://localhost:8000/v1/models | python -m json.tool

# Run evaluation
echo ""
echo "========================================"
echo "Starting GAIA evaluation (1 question)..."
echo "========================================"
python examples/run_gaia.py \
    --config configs/config_gaia_adaptive_qwen.py \
    --cfg-options tag=gaia_test_single dataset_config.max_samples=1

EVAL_EXIT_CODE=$?

# Cleanup
echo ""
echo "========================================"
echo "Evaluation finished with exit code: $EVAL_EXIT_CODE"
echo "Shutting down vLLM server..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

echo "Results saved to: workdir/gaia_test_single/dra.jsonl"
echo "========================================"

exit $EVAL_EXIT_CODE
