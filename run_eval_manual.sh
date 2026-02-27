#!/bin/bash
# Manual evaluation script to run on gpu-4080-410
# Usage: 
#   1. ssh gpu-4080-410
#   2. bash /userhome/cs2/ambr0se/DeepResearchMetaAgent/run_eval_manual.sh

echo "========================================"
echo "Running Single Question GAIA Evaluation"
echo "Node: $(hostname)"
echo "========================================"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra

cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

# Wait for vLLM server to be ready
echo "Checking if vLLM server is ready on localhost:8000..."
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "✗ vLLM server not responding after 5 minutes"
        exit 1
    fi
    echo "Waiting for vLLM server... (attempt $i/60)"
    sleep 5
done

# Show available models
echo ""
echo "Available models:"
curl -s http://localhost:8000/v1/models | python -m json.tool

# Run single question evaluation
echo ""
echo "Starting GAIA evaluation with 1 question..."
python examples/run_gaia.py \
    --config configs/config_gaia_adaptive_qwen.py \
    --cfg-options tag=gaia_test_single dataset_config.max_samples=1

echo ""
echo "========================================"
echo "Evaluation complete! Check results:"
echo "  workdir/gaia_test_single/dra.jsonl"
echo "========================================"
