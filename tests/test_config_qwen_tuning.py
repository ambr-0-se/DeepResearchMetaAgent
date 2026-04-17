"""Pass 2 regression test: Qwen-4B (vLLM) config tuning.

Guards the three Pass 2 values on `configs/config_gaia_adaptive_qwen.py`:
- Sub-agent max_steps raised from 3/3/5 to 7/7/7 (was the cap Qwen-4B kept
  hitting — 56+ AgentMaxStepsError lines in the 4.98% test-set run).
- context_prune_threshold_ratio=0.75 on every agent config dict (earlier
  than the 0.85 default to protect the 32k vLLM context wall).

This is a source-level parse test; it does not instantiate agents or
import the heavy stack (mmengine, huggingface_hub, etc.). A future refactor
that legitimately renames the config dicts is welcome — update this test
alongside it.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/config_gaia_adaptive_qwen.py"

EXPECTED_MAX_STEPS = {
    "deep_researcher_agent_config": 7,
    "deep_analyzer_agent_config": 7,
    "browser_use_agent_config": 7,
    "planning_agent_config": 25,
}

EXPECTED_PRUNE_RATIO = 0.75

AGENT_CONFIGS = {
    "deep_researcher_agent_config",
    "deep_analyzer_agent_config",
    "browser_use_agent_config",
    "planning_agent_config",
}


def _config_dict_kwargs(cfg_name: str) -> dict:
    """Extract the kwargs of `<cfg_name> = dict(...)` assignment."""
    module = ast.parse(CFG.read_text())
    for node in module.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == cfg_name):
            continue
        if not (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "dict"):
            continue
        return {kw.arg: kw.value for kw in node.value.keywords}
    raise AssertionError(f"`{cfg_name} = dict(...)` not found in {CFG}")


def test_sub_agent_max_steps_raised_to_seven():
    """Pass 2.1 — sub-agents 3/3/5 → 7/7/7."""
    for cfg, expected in EXPECTED_MAX_STEPS.items():
        kwargs = _config_dict_kwargs(cfg)
        assert "max_steps" in kwargs, f"{cfg} missing max_steps"
        val_node = kwargs["max_steps"]
        assert isinstance(val_node, ast.Constant), f"{cfg}.max_steps not a literal"
        assert val_node.value == expected, (
            f"{cfg}.max_steps = {val_node.value}, expected {expected}. "
            "Pass 2.1 raised sub-agents to 7 (pending P95 measurement); "
            "planner stays at 25. Regression."
        )


def test_context_prune_threshold_ratio_override_on_every_agent_config():
    """Pass 2.2 — every agent dict has an explicit 0.75 override."""
    for cfg in AGENT_CONFIGS:
        kwargs = _config_dict_kwargs(cfg)
        assert "context_prune_threshold_ratio" in kwargs, (
            f"{cfg} missing context_prune_threshold_ratio override. Pass 2.2 "
            "adds this on every agent dict (Qwen-4B has a 32k context wall; "
            "default 0.85 prunes too late). Regression."
        )
        val_node = kwargs["context_prune_threshold_ratio"]
        assert isinstance(val_node, ast.Constant), (
            f"{cfg}.context_prune_threshold_ratio not a literal"
        )
        assert val_node.value == EXPECTED_PRUNE_RATIO, (
            f"{cfg}.context_prune_threshold_ratio = {val_node.value}, "
            f"expected {EXPECTED_PRUNE_RATIO}"
        )
