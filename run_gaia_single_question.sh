#!/bin/bash
#SBATCH --job-name=gaia-test-qwen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gaia_test_%j.out
#SBATCH --error=logs/gaia_test_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GAIA Single Question Test"
echo "========================================"

# Load conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra

# Create logs directory
mkdir -p logs

# Change to project directory
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

# Wait for vLLM server to be available
echo "Waiting for vLLM server at http://localhost:8000..."
MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server not available after $MAX_RETRIES attempts"
        echo "Please ensure the vLLM server job is running on the same node"
        exit 1
    fi
    echo "  Attempt $RETRY_COUNT/$MAX_RETRIES - server not ready, waiting 5s..."
    sleep 5
done

echo "✓ vLLM server is ready!"
echo ""

# Test connection
echo "Testing vLLM server connection:"
curl -s http://localhost:8000/v1/models
echo ""

# Run single question test
echo "Starting single question test (max_samples=1)..."
echo ""

python examples/run_gaia.py \
    --config configs/config_gaia_adaptive_qwen.py \
    --cfg-options tag=gaia_test_single dataset_config.max_samples=1

echo ""
echo "========================================"
echo "Single question test completed!"
echo "Check results in: workdir/gaia_test_single/"
echo "========================================"
