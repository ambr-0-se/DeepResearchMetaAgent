"""
Condition C2 — GAIA with AdaptivePlanningAgent (reactive diagnose + modify tools).

This config inherits from config_gaia.py and overrides the planning agent
to use the AdaptivePlanningAgent. The planner has `diagnose_subagent` and
`modify_subagent` tools available and may invoke them reactively when a
sub-agent delegation fails. There is no structural REVIEW step — that is
added by C3 (configs/config_gaia_c3.py).

All architectural modifications remain task-scoped via
AdaptiveMixin._reset_to_original_state(); there is no cross-task
persistence. For persistent skill library behavior, see C4
(configs/config_gaia_c4.py).

Usage:
    # Run C2 evaluation
    python examples/run_gaia.py --config configs/config_gaia_adaptive.py

    # Compare C2 vs C0 baseline
    python scripts/compare_results.py workdir/gaia_c0/dra.jsonl workdir/gaia_adaptive/dra.jsonl
"""

_base_ = './config_gaia.py'

# Per-run output isolation: `workdir/gaia_adaptive_<RUN_ID>/`.
# Set `DRA_RUN_ID` explicitly to resume a prior run.
import os as _os
from datetime import datetime as _datetime
_RUN_ID = _os.environ.get("DRA_RUN_ID") or _datetime.now().strftime("%Y%m%d_%H%M%S")

tag = f"gaia_adaptive_{_RUN_ID}"

# Override planning agent to use adaptive version
planning_agent_config = dict(
    type="adaptive_planning_agent",  # Use the new adaptive agent type
    name="adaptive_planning_agent",
    model_id="claude-3.7-sonnet-thinking",
    description="An adaptive planning agent that can diagnose and modify sub-agents at runtime.",
    max_steps=25,  # Slightly more steps than C0 (20) to allow for reactive adaptation
    template_path="src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml",
    provide_run_summary=True,
    tools=["planning_tool"],  # diagnose_subagent and modify_subagent added automatically
    managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"]
)

# Use the adaptive planning agent config as the main agent
agent_config = planning_agent_config
