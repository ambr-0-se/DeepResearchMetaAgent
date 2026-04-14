_base_ = './base.py'

# General Config
tag = "arc"
concurrency = 1
workdir = "workdir"
log_path = "log.txt"
save_path = "dra.jsonl"
use_local_proxy = False
per_question_timeout_secs = 600

use_hierarchical_agent = True

dataset = dict(
    type="arc_dataset",
    path="data/arc-agi",
    split="evaluation",
)

deep_analyzer_agent_config = dict(
    type="deep_analyzer_agent",
    name="deep_analyzer_agent",
    model_id="claude-3.7-sonnet-thinking",
    description="A deep analyzer agent that can perform systematic, step-by-step analysis of ARC grid patterns.",
    max_steps=5,
    template_path="src/agent/deep_analyzer_agent/prompts/deep_analyzer_agent.yaml",
    provide_run_summary=True,
    tools=["deep_analyzer_tool", "python_interpreter_tool"],
)

general_agent_config = dict(
    type="general_agent",
    name="general_agent",
    model_id="gpt-4.1",
    description="A general-purpose agent with Python execution for testing grid transformations.",
    max_steps=5,
    template_path="src/agent/general_agent/prompts/general_agent.yaml",
    provide_run_summary=True,
    tools=["python_interpreter_tool"],
)

planning_agent_config = dict(
    type="planning_agent",
    name="planning_agent",
    model_id="claude-3.7-sonnet-thinking",
    description="A planning agent that coordinates sub-agents to solve ARC-AGI tasks.",
    max_steps=15,
    template_path="src/agent/planning_agent/prompts/planning_agent.yaml",
    provide_run_summary=True,
    tools=["planning_tool"],
    managed_agents=["deep_analyzer_agent", "general_agent"],
)

agent_config = planning_agent_config
