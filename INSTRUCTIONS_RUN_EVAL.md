# How to Run Evaluation on GPU Node

## The Problem
Direct SSH to compute nodes is blocked. You must use SLURM commands.

## Solution 1: Interactive Shell (Recommended for Testing)

```bash
# Get a shell on the node where your vLLM server is running
srun --jobid=125979 --overlap --pty bash

# Once you're on the node, run:
bash /userhome/cs2/ambr0se/DeepResearchMetaAgent/run_eval_manual.sh

# Or run commands directly:
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

# Check if vLLM is ready
curl http://localhost:8000/v1/models

# When ready, run evaluation
python examples/run_gaia.py \
    --config configs/config_gaia_adaptive_qwen.py \
    --cfg-options tag=gaia_test_single dataset_config.max_samples=1
```

## Solution 2: Run Command Directly (No Interactive Shell)

```bash
# Run the evaluation script directly on the node
srun --jobid=125979 --overlap bash /userhome/cs2/ambr0se/DeepResearchMetaAgent/run_eval_manual.sh
```

## Solution 3: Modified SLURM Script (For Future Runs)

For future runs, you can combine both vLLM server and evaluation in a single job.
See `run_combined_eval.sh` (will be created).

## Check Server Status Before Running

```bash
# Monitor vLLM server logs
tail -f logs/vllm_server_125979.out

# Look for "Uvicorn running" message - that means server is ready
```
