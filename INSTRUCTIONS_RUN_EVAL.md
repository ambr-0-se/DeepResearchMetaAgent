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

### Full GAIA **test** split (301 questions, leaderboard submission)

```bash
sbatch run_gaia_test_eval.sh
```

Outputs `workdir/gaia_test_<JOBID>_<timestamp>/dra.jsonl` and `submission.jsonl` (upload to the [GAIA leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)). Config: `configs/config_gaia_test_qwen.py` (`split="test"`).

**Resume after a crash (GPU farm):** pass the existing run directory name as `TAG` (the folder under `workdir/`, e.g. `gaia_test_127877_20260327_230408`):

```bash
sbatch --export=ALL,TAG=gaia_test_<JOBID>_<timestamp> run_gaia_test_resume.sh
```

This launches vLLM + watchdog like the full test job, then continues `run_gaia.py` with the same `tag` so results append to the same `dra.jsonl`. It re-exports `submission.jsonl` at the end.

If `batch` + 2×RTX 4080 does not schedule on your cluster, use **`q-3090`** (2×RTX 3090). That partition often has a **18-hour** max wall time — match the inner timeout and re-submit until the run finishes:

```bash
sbatch --partition=q-3090 --gres=gpu:rtx_3090:2 --mem=128G --cpus-per-task=8 --time=18:00:00 \
  --export=ALL,TAG=gaia_test_<JOBID>_<timestamp>,EVAL_TIMEOUT_SECS=64800 \
  run_gaia_test_resume.sh
```

**Auto-chain (no manual resubmit after each OOM / 18h wall):** the same script can submit the next `sbatch` when `dra.jsonl` still has fewer than 301 lines (capped by `MAX_CHAIN_DEPTH`, default 20):

```bash
sbatch --partition=q-3090 --gres=gpu:rtx_3090:2 --mem=128G --cpus-per-task=8 --time=18:00:00 \
  --export=ALL,TAG=gaia_test_<JOBID>_<timestamp>,EVAL_TIMEOUT_SECS=64800,AUTO_RESUBMIT=1 \
  run_gaia_test_resume.sh
```

The compute node must allow `sbatch` from the job (same cluster policy as your login node). Tune `RESUBMIT_PARTITION`, `RESUBMIT_MEM`, etc. if needed. This does **not** fix OOM inside a single allocation; it only starts a **new** job so RAM resets.

**Local resume** (vLLM already running on port 8000):

```bash
python examples/run_gaia.py --config configs/config_gaia_test_qwen.py \
  --cfg-options tag=gaia_test_<JOBID>_<timestamp>
```

(`run_gaia.py` skips `task_id`s already present after `filter_answers`. For the **test** split, any row with a **completed** run (`agent_error` empty) is kept—including final answers **`Unable to determine`**—so resumes do not re-run abstentions. Rows with **`agent_error` set** (timeout, OOM, API failure after retries, etc.) are **removed** on load so only those questions are retried. Validation split behavior is unchanged: only **scorer-correct** rows are kept for resume.)

### ARC-AGI (GPU farm, Qwen on vLLM)

Requires `data/arc-agi/evaluation/` (and optionally `training/`) with standard ARC JSON task files.

```bash
# Smoke test: 10 test cases, 6 h wall-clock
sbatch run_arc_test.sh

# Full evaluation split (one JSONL row per test case), 72 h wall-clock
sbatch run_arc_eval.sh
```

Results: `workdir/arc_test_<JOBID>_<timestamp>/dra.jsonl` or `workdir/arc_eval_<JOBID>_<timestamp>/dra.jsonl`. Config: `configs/config_arc_qwen.py` (same vLLM + watchdog pattern as GAIA). Scoring in the job log uses exact grid match (`arc_question_scorer`).

For API models (no GPU job), run locally: `python examples/run_arc.py --config configs/config_arc.py`.

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
python scripts/analyze_results.py workdir/<run_dir>/dra.jsonl --html --config configs/config_gaia_c1_qwen_local.py
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

The eval config is `configs/config_gaia_c1_qwen_local.py`. Key settings:

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
python examples/run_gaia.py --config configs/config_gaia_c1_qwen_local.py \
    --cfg-options max_samples=5

# Change tag (affects output directory)
python examples/run_gaia.py --config configs/config_gaia_c1_qwen_local.py \
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
| Job exits **137**, `.err` shows `oom_kill` / `Killed` | SLURM cgroup RAM exceeded (common with **64G** + vLLM + agent) | Use **`--mem=128G`** (see `run_gaia_test_eval.sh`); avoid running heavy extra processes on the same allocation |
| `Connection error` mid-run then failures | vLLM unreachable or client HTTP error | Watchdog restarts vLLM; `run_gaia.py` also retries transient errors including `"Connection error"` |
| `BrowserType.launch: Executable doesn't exist` under `~/.cache/ms-playwright/.../chrome` | Chromium was never downloaded for Playwright | **Automated:** GAIA/combined SLURM scripts run `scripts/ensure_playwright_browsers.sh` after `conda activate` (idempotent). **Manual:** same env: `bash scripts/ensure_playwright_browsers.sh` or `python -m playwright install chromium`. On Linux nodes missing system libs: `PLAYWRIGHT_WITH_DEPS=1 bash scripts/ensure_playwright_browsers.sh`. Ensure `playwright` is installed (`pip install playwright`; listed in `requirements.txt`). |
