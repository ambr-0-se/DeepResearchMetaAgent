# Running GAIA Evaluation on GPU Node

## Quick Start

### Single-question test (sanity check)
```bash
sbatch run_combined_test.sh
```
Runs 1 question with a 30-minute SLURM wall-clock. Results in `workdir/gaia_test_<JOBID>_<timestamp>/dra.jsonl`.

### Full evaluation (165 questions)
```bash
sbatch run_combined_eval.sh
```
Runs all questions with 72-hour wall-clock. Results in `workdir/gaia_eval_<JOBID>_<timestamp>/dra.jsonl`.

## What the Scripts Do

Both scripts handle the full lifecycle:
1. **Launch vLLM** server (Qwen3-VL-4B-Instruct, tensor-parallel on 2x RTX 4080)
2. **Wait for readiness** (polls `/v1/models` endpoint)
3. **Start health watchdog** — monitors vLLM every 30s, auto-restarts after 90s of failures (up to 5 restarts)
4. **Run evaluation** with per-question timeout (20 min default) and transient-error retry (3 attempts, 60s backoff)
5. **Print score summary** at the end
6. **Cleanup** — kills vLLM and watchdog on exit/signal

## Analyzing Results

```bash
# Terminal summary
python scripts/analyze_results.py workdir/<run_dir>/dra.jsonl

# Interactive HTML report
python scripts/analyze_results.py workdir/<run_dir>/dra.jsonl --html

# Per-question detail in terminal
python scripts/analyze_results.py workdir/<run_dir>/dra.jsonl --detail

# With explicit config for richer metadata
python scripts/analyze_results.py workdir/<run_dir>/dra.jsonl --html --config configs/config_gaia_adaptive_qwen.py
```

## Monitoring a Running Job

```bash
# Check SLURM job status
squeue -u $USER

# Watch live output
tail -f logs/combined_eval_<JOBID>.out

# Watch vLLM errors
tail -f logs/combined_eval_<JOBID>.err

# Check how many questions are done
wc -l workdir/gaia_eval_<JOBID>_*/dra.jsonl
```

## Configuration

The eval config is `configs/config_gaia_adaptive_qwen.py`. Key settings:

| Setting | Value | Description |
|---------|-------|-------------|
| `model_id` | `"Qwen"` | vLLM-served Qwen3-VL-4B-Instruct |
| `split` | `"validation"` | GAIA split with ground-truth answers |
| `per_question_timeout_secs` | `1200` | 20-minute wall-clock per question |
| `max_steps` (planning agent) | `25` | Max reasoning steps |
| `max_steps` (sub-agents) | `3-5` | Per sub-agent step limit |

### Overriding Config at Runtime
```bash
# Limit to N questions (useful for quick tests)
python examples/run_gaia.py --config configs/config_gaia_adaptive_qwen.py \
    --cfg-options max_samples=5

# Change tag (affects output directory)
python examples/run_gaia.py --config configs/config_gaia_adaptive_qwen.py \
    --cfg-options tag=my_experiment
```

## Comparing Results

```bash
python scripts/compare_results.py workdir/run_a/dra.jsonl workdir/run_b/dra.jsonl
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "vLLM server not ready after N minutes" | Model download or GPU issue | Check `.err` log for CUDA/OOM errors |
| Many "Connection refused" errors | vLLM crashed mid-run | Watchdog should auto-restart; check if max restarts (5) was hit |
| "Per-question timeout exceeded" | Agent stuck in loop | Normal for hard questions; timeout prevents stalling |
| Results file is empty | All questions errored | Check `.err` log and `agent_error` field in JSONL |
