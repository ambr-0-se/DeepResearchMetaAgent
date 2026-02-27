# Quick Start Guide: GAIA Evaluation with Qwen3-VL-4B on SLURM

## Current Status

✅ **Completed:**
- Environment setup (`dra` conda environment)
- All Python dependencies installed
- Agents registered (including `adaptive_planning_agent`)
- Dataset downloaded (`data/GAIA/`)
- Configuration files created
- SLURM job scripts created

⏳ **In Progress:**
- vLLM installation (large package, ~509MB)

⚠️ **Still Needed:**
- OpenAI API key (for audio transcription fallback)

---

## Step 1: Complete vLLM Installation

```bash
# Check if installation is still running
ps aux | grep "pip install vllm" | grep -v grep

# If not running, install vLLM:
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra
pip install vllm

# Verify installation:
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

---

## Step 2: Add OpenAI API Key (Optional but Recommended)

Edit `.env` file and replace line 12:

```bash
# Current:
OPENAI_API_KEY=xxxxxx

# Replace with your actual key:
OPENAI_API_KEY=sk-proj-...
```

**Why needed?** For transcribing audio files (.mp3, .m4a) in GAIA dataset via OpenAI API.

---

## Step 3: Submit SLURM Jobs

### Option A: Test with Single Question (Recommended First)

```bash
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

# Start vLLM server on GPU node
sbatch run_vllm_server.sh

# Wait ~30 seconds for server to start, then check:
squeue -u $USER

# Get the node name where vLLM is running:
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Submit test job on SAME node as vLLM:
sbatch --nodelist=<NODE_NAME> run_gaia_single_question.sh
```

### Option B: Full GAIA Evaluation

```bash
# After successful single-question test:
sbatch --nodelist=<NODE_NAME> run_gaia_eval.sh
```

---

## Step 4: Monitor Jobs

### Check job status:
```bash
squeue -u $USER
```

### View vLLM server logs (real-time):
```bash
tail -f logs/vllm_server_<JOBID>.out
```

### View evaluation logs:
```bash
tail -f logs/gaia_test_<JOBID>.out  # For single question
tail -f logs/gaia_eval_<JOBID>.out  # For full evaluation
```

---

## Step 5: Check Results

### Single question test results:
```bash
ls -la workdir/gaia_test_single/
cat workdir/gaia_test_single/dra.jsonl
```

### Full evaluation results:
```bash
ls -la workdir/gaia_adaptive_qwen/
cat workdir/gaia_adaptive_qwen/dra.jsonl
```

---

## Alternative: Interactive Session (For Debugging)

If you prefer interactive testing:

```bash
# Request interactive GPU session
srun --partition=batch --gres=gpu:rtx_4080:2 --cpus-per-task=8 --mem=64G --time=4:00:00 --pty bash

# Once on GPU node:
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

# Start vLLM server (will run in foreground):
./start_vllm.sh

# In another terminal, SSH to same node and run:
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

# Test single question:
python examples/run_gaia.py \
    --config configs/config_gaia_adaptive_qwen.py \
    --cfg-options tag=gaia_test_single dataset_config.max_samples=1
```

---

## GPU Access on HKU CS GPU Farm

### How GPU Allocation Works

**SLURM automatically handles GPU allocation** - you don't need to manually configure GPU access:

1. **Batch Jobs (`sbatch`)**: GPU allocation specified in script headers
   ```bash
   #SBATCH --gres=gpu:rtx_4080:2  # Requests 2× RTX 4080 GPUs
   ```
   
2. **Interactive Sessions (`srun`)**: GPU allocation in command
   ```bash
   srun --gres=gpu:rtx_4080:2 --pty bash  # Allocates 2 GPUs immediately
   ```

### Verify GPU Access on Compute Node

Once your job is running (or you're in an interactive session):

```bash
# Check GPU visibility and status
nvidia-smi

# Expected output: Shows 2× RTX 4080 GPUs with memory usage
# GPU 0: NVIDIA GeForce RTX 4080 (16GB)
# GPU 1: NVIDIA GeForce RTX 4080 (16GB)

# Check which GPUs are visible to CUDA
echo $CUDA_VISIBLE_DEVICES  # Should show: 0,1

# Verify PyTorch can see GPUs
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
# Expected: GPUs available: 2

# Check vLLM can detect GPUs
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

### Understanding CUDA_VISIBLE_DEVICES

The SLURM scripts automatically set `CUDA_VISIBLE_DEVICES=0,1`:
- **What it does:** Controls which GPUs are visible to CUDA applications
- **Why needed:** Ensures vLLM uses both allocated GPUs for tensor parallelism
- **Already configured:** Set in `run_vllm_server.sh` and `start_vllm.sh`

### Check Available GPUs Before Submitting

```bash
# View all GPU nodes and their availability
sinfo -o "%20N %10c %10m %25f %10G %10a %10l %20C"

# Check specifically for RTX 4080 availability
sinfo | grep rtx_4080

# See which nodes have free GPUs
squeue -t RUNNING -o "%.18i %.9P %.20j %.8u %.10M %N %b" | grep gpu

# Check your current GPU allocation
scontrol show job $SLURM_JOB_ID | grep Gres
```

### GPU Usage Best Practices

1. **Always use SLURM allocation** - Never try to access GPUs on login nodes
2. **Match node placement** - Ensure vLLM server and evaluation run on the **same node**
3. **Monitor GPU usage** - Use `nvidia-smi` to check memory and utilization
4. **Release unused GPUs** - Cancel jobs when done: `scancel <JOBID>`

### Login Node vs Compute Node

| Environment | GPU Access | Use Case |
|------------|------------|----------|
| **Login Node** | ❌ No GPUs | Development, submitting jobs, checking status |
| **Compute Node (sbatch)** | ✅ Via SLURM | Running vLLM server, batch evaluations |
| **Compute Node (srun)** | ✅ Via SLURM | Interactive debugging, testing |

**Important:** You're on a login node if `nvidia-smi` returns "command not found" or "no devices found". Use `sbatch` or `srun` to access GPUs.

---

## Troubleshooting

### Issue: vLLM server fails to start
- **Check:** GPU availability with `nvidia-smi`
- **Check:** CUDA version compatibility
- **Solution:** Look at error logs in `logs/vllm_server_*.err`

### Issue: Model not found
- **Symptom:** "Model Qwen/Qwen3-VL-4B-Instruct not found"
- **Solution:** Model will be auto-downloaded on first run (requires HuggingFace token in `.env`)

### Issue: Out of memory / KV cache memory error
- **Symptom:** `ValueError: ... KV cache is needed, which is larger than the available KV cache memory`
- **Root Cause:** Model's default max sequence length (262144) requires too much memory
- **Solution:** Already fixed - scripts now use `--max-model-len 32768` to fit in available GPU memory
- **If still having issues:** Reduce `max-num-seqs` in scripts (e.g., from 8 to 4)

### Issue: Evaluation job can't connect to vLLM
- **Symptom:** "vLLM server not available"
- **Solution:** Ensure both jobs run on the **same node** using `--nodelist=<NODE_NAME>`

---

## Expected Runtime

- **vLLM server startup:** 2-5 minutes (includes model download ~8GB)
- **Single question test:** 5-15 minutes
- **Full GAIA evaluation (~450 questions):** 24-48 hours

---

## Configuration Files

### SLURM Scripts:
- `run_vllm_server.sh` - Start vLLM server with 2× RTX 4080
- `run_gaia_single_question.sh` - Test with 1 question
- `run_gaia_eval.sh` - Full GAIA evaluation

### Agent Configuration:
- `configs/config_gaia_adaptive_qwen.py` - AdaptivePlanningAgent with Qwen model

### Environment:
- `.env` - API keys and configuration
- `start_vllm.sh` - Direct vLLM startup script (for interactive use)

---

## Quick Commands Reference

```bash
# === GPU Management ===
# Check GPU availability and status (on compute node)
nvidia-smi

# Check which GPUs are allocated to your job
echo $CUDA_VISIBLE_DEVICES

# View available RTX 4080 nodes
sinfo | grep rtx_4080

# Check your running jobs and their GPU allocation
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"

# === vLLM Server ===
# Check vLLM installation
python -c "import vllm; print(vllm.__version__)"

# Test vLLM server connection
curl http://localhost:8000/v1/models

# === Job Management ===
# Check job queue
squeue -u $USER

# Cancel all jobs
scancel -u $USER

# Cancel specific job
scancel <JOBID>

# View detailed job info
scontrol show job <JOBID>
```

---

## Next Steps After Successful Test

1. Review single-question results
2. Adjust configuration if needed (`max_steps`, `max_insights`, etc.)
3. Run full evaluation
4. Compare with baseline: `python scripts/compare_results.py workdir/gaia/dra.jsonl workdir/gaia_adaptive_qwen/dra.jsonl`
