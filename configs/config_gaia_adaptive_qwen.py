"""
Configuration for AdaptivePlanningAgent with Qwen3-VL-4B-Instruct on vLLM.

Usage:
    # Run evaluation with adaptive agent using Qwen
    python examples/run_gaia.py --config configs/config_gaia_adaptive_qwen.py
    
    # Compare with baseline
    python scripts/compare_results.py workdir/gaia/dra.jsonl workdir/gaia_adaptive_qwen/dra.jsonl
"""

_base_ = './config_gaia.py'

tag = "gaia_adaptive_qwen"

dataset = dict(
    type="gaia_dataset",
    name="2023_all",
    path="data/GAIA",
    split="test",
)

# Override tool configs to use Qwen model
deep_researcher_tool_config = dict(
    type="deep_researcher_tool",
    model_id="Qwen",  # Changed from "gpt-4.1"
    max_depth=2,
    max_insights=20,
    time_limit_seconds=60,
    max_follow_ups=3,
)

deep_analyzer_tool_config = dict(
    type="deep_analyzer_tool",
    analyzer_model_ids=["Qwen"],  # Changed from ["gemini-2.5-pro"]
    summarizer_model_id="Qwen",   # Changed from "gemini-2.5-pro"
)

auto_browser_use_tool_config = dict(
    type="auto_browser_use_tool",
    model_id="Qwen"  # Changed from "gpt-4.1"
)

# Override all agents to use Qwen (vLLM)
deep_researcher_agent_config = dict(
    type="deep_researcher_agent",
    name="deep_researcher_agent",
    model_id="Qwen",  # Use vLLM served model
    description="A deep researcher agent that can conduct extensive web searches.",
    max_steps=3,
    template_path="src/agent/deep_researcher_agent/prompts/deep_researcher_agent.yaml",
    provide_run_summary=True,
    tools=["deep_researcher_tool", "python_interpreter_tool"],
)

deep_analyzer_agent_config = dict(
    type="deep_analyzer_agent",
    name="deep_analyzer_agent",
    model_id="Qwen",
    description="A deep analyzer agent that can perform systematic analysis.",
    max_steps=3,
    template_path="src/agent/deep_analyzer_agent/prompts/deep_analyzer_agent.yaml",
    provide_run_summary=True,
    tools=["deep_analyzer_tool", "python_interpreter_tool"],
)

browser_use_agent_config = dict(
    type="browser_use_agent",
    name="browser_use_agent",
    model_id="Qwen",
    description="A browser use agent for web interaction.",
    max_steps=5,
    template_path="src/agent/browser_use_agent/prompts/browser_use_agent.yaml",
    provide_run_summary=True,
    tools=["auto_browser_use_tool", "python_interpreter_tool"],
)

planning_agent_config = dict(
    type="adaptive_planning_agent",
    name="adaptive_planning_agent",
    model_id="Qwen",
    description="An adaptive planning agent with self-modification capabilities.",
    max_steps=25,
    template_path="src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml",
    provide_run_summary=True,
    tools=["planning_tool"],
    managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"]
)

agent_config = planning_agent_config
