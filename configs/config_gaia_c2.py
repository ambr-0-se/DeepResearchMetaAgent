"""
Condition C2 — GAIA with AdaptivePlanningAgent + structural REVIEW step.

Extends C1 (configs/config_gaia_c1.py) with:
- A new prompt template that describes the automatic [REVIEW] block.
- `enable_review=True` on the planner config, which causes
  AdaptivePlanningAgent.__init__ to construct a `ReviewStep` internally and
  fire it from `_post_action_hook` after every sub-agent delegation.

No new tools are added to the planner — REVIEW runs automatically. The
existing reactive tools (diagnose_subagent, modify_subagent) remain
available as overrides; their guidance in the C2 prompt tells the planner
to prefer following REVIEW's recommendation over invoking them manually.

Results land in `workdir/gaia_c2_<RUN_ID>/dra.jsonl` via the separate tag.

Usage:
    python examples/run_gaia.py --config configs/config_gaia_c2.py

    # Compare C2 vs C1 on the same subset
    python scripts/compare_results.py workdir/gaia_c1_<RUN_ID>/dra.jsonl workdir/gaia_c2_<RUN_ID>/dra.jsonl
"""

_base_ = './config_gaia_c1.py'

# Per-run output isolation: `workdir/gaia_c2_<RUN_ID>/`.
# Set `DRA_RUN_ID` explicitly to resume a prior run.
import os as _os
from datetime import datetime as _datetime
_RUN_ID = _os.environ.get("DRA_RUN_ID") or _datetime.now().strftime("%Y%m%d_%H%M%S")
# mmengine's Config.pretty_text would otherwise emit the module/class values of
# these imports as invalid Python (e.g. `_datetime=<class ...>`), crashing yapf.
del _os, _datetime

tag = f"gaia_c2_{_RUN_ID}"

# Override the adaptive planning agent config for C2:
# - Swap in the C2-specific YAML template (documents the [REVIEW] block)
# - Enable the structural REVIEW step via config flag
planning_agent_config = dict(
    type="adaptive_planning_agent",
    name="adaptive_planning_agent",
    model_id="claude-3.7-sonnet-thinking",
    description=(
        "An adaptive planning agent with reactive diagnose/modify tools "
        "plus a structural REVIEW step that automatically assesses each "
        "sub-agent delegation (condition C2)."
    ),
    max_steps=25,  # Same budget as C1; REVIEW is counted against planner steps via injection, not extra loop iterations
    template_path="src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c2.yaml",
    provide_run_summary=True,
    tools=["planning_tool"],  # diagnose_subagent and modify_subagent added automatically by AdaptivePlanningAgent
    managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"],
    # C2 flag — AdaptivePlanningAgent.__init__ reads this via getattr and
    # builds a ReviewStep when True. Default False keeps C0/C1 unaffected.
    enable_review=True,
)

# Use the C2 planning agent config as the main agent
agent_config = planning_agent_config

# Per-question wall clock timeout (secs). Pinned 2026-04-20 for fairness
# with E0 v3 C3 training (which used 1800s). Previously inherited
# run_gaia.py default 1200, creating a training/test asymmetry that
# could bias the C0–C2 vs C3 ablation at test time.
per_question_timeout_secs = 1800
