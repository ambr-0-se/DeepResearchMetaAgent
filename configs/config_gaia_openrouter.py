"""
Configuration for GAIA evaluation using OpenRouter gpt-oss-120b (free tier).

Model: openai/gpt-oss-120b:free via OpenRouter
  - 117B parameter Mixture-of-Experts, activates 5.1B params per forward pass
  - Free: $0/M input tokens, $0/M output tokens
  - 131,072 context window, native tool use, function calling
  - https://openrouter.ai/openai/gpt-oss-120b:free

Requirements:
  - Set OPENROUTER_API_KEY in .env

Usage:
    python examples/run_gaia.py --config configs/config_gaia_openrouter.py

    # Compare with baseline
    python scripts/compare_results.py workdir/gaia/dra.jsonl workdir/gaia_openrouter/dra.jsonl
"""

_base_ = './config_gaia.py'

tag = "gaia_openrouter"

# Override tool configs to use gpt-oss-120b
deep_researcher_tool_config = dict(
    type="deep_researcher_tool",
    model_id="gpt-oss-120b",
    max_depth=2,
    max_insights=20,
    time_limit_seconds=60,
    max_follow_ups=3,
)

deep_analyzer_tool_config = dict(
    type="deep_analyzer_tool",
    analyzer_model_ids=["gpt-oss-120b"],
    summarizer_model_id="gpt-oss-120b",
)

auto_browser_use_tool_config = dict(
    type="auto_browser_use_tool",
    model_id="gpt-oss-120b",
)

# Override all agents to use gpt-oss-120b
deep_researcher_agent_config = dict(
    type="deep_researcher_agent",
    name="deep_researcher_agent",
    model_id="gpt-oss-120b",
    description="A deep researcher agent that can conduct extensive web searches.",
    max_steps=3,
    template_path="src/agent/deep_researcher_agent/prompts/deep_researcher_agent.yaml",
    provide_run_summary=True,
    tools=["deep_researcher_tool", "python_interpreter_tool"],
)

deep_analyzer_agent_config = dict(
    type="deep_analyzer_agent",
    name="deep_analyzer_agent",
    model_id="gpt-oss-120b",
    description="A deep analyzer agent that can perform systematic, step-by-step analysis.",
    max_steps=3,
    template_path="src/agent/deep_analyzer_agent/prompts/deep_analyzer_agent.yaml",
    provide_run_summary=True,
    tools=["deep_analyzer_tool", "python_interpreter_tool"],
)

browser_use_agent_config = dict(
    type="browser_use_agent",
    name="browser_use_agent",
    model_id="gpt-oss-120b",
    description="A browser use agent that can search relevant web pages and interact with them.",
    max_steps=5,
    template_path="src/agent/browser_use_agent/prompts/browser_use_agent.yaml",
    provide_run_summary=True,
    tools=["auto_browser_use_tool", "python_interpreter_tool"],
)

planning_agent_config = dict(
    type="planning_agent",
    name="planning_agent",
    model_id="gpt-oss-120b",
    description="A planning agent that can plan the steps to complete the task.",
    max_steps=20,
    template_path="src/agent/planning_agent/prompts/planning_agent.yaml",
    provide_run_summary=True,
    tools=["planning_tool"],
    managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"],
)

agent_config = planning_agent_config
