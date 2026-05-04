"""
Configuration for AdaptivePlanningAgent with Qwen3-VL-4B-Instruct on the GAIA test split.

The test split has 301 questions with no ground-truth answers (true_answer = "?").
Results must be submitted to the GAIA leaderboard for scoring:
    https://huggingface.co/spaces/gaia-benchmark/leaderboard

Usage:
    python examples/run_gaia.py --config configs/config_gaia_test_qwen.py

After evaluation, export the submission file:
    python scripts/export_gaia_submission.py workdir/<run_dir>/dra.jsonl
"""

_base_ = './config_gaia_c1_qwen_local.py'

tag = "gaia_test_qwen"

dataset = dict(
    type="gaia_dataset",
    name="2023_all",
    path="data/GAIA",
    split="test",
)
