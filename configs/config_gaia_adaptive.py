"""
Configuration for AdaptivePlanningAgent evaluation on GAIA benchmark.

This config inherits from config_gaia.py and overrides the planning agent
to use the AdaptivePlanningAgent with self-modification capabilities.

Usage:
    # Run evaluation with adaptive agent
    python examples/run_gaia.py --config configs/config_gaia_adaptive.py
    
    # Compare with baseline
    python scripts/compare_results.py workdir/gaia/dra.jsonl workdir/gaia_adaptive/dra.jsonl
"""

_base_ = './config_gaia.py'

# Different tag for separate results directory
tag = "gaia_adaptive"

# Override planning agent to use adaptive version
planning_agent_config = dict(
    type="adaptive_planning_agent",  # Use the new adaptive agent type
    name="adaptive_planning_agent",
    model_id="claude-3.7-sonnet-thinking",
    description="An adaptive planning agent that can diagnose and modify sub-agents at runtime.",
    max_steps=25,  # Slightly more steps to allow for reflection and adaptation
    template_path="src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml",
    provide_run_summary=True,
    tools=["planning_tool"],  # diagnose_subagent and modify_subagent added automatically
    managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"]
)

# Use the adaptive planning agent config as the main agent
agent_config = planning_agent_config
