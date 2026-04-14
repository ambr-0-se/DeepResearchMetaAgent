"""
ARC-AGI evaluation with Qwen3-VL-4B-Instruct on local vLLM (HKU CS GPU farm).

Usage:
    python examples/run_arc.py --config configs/config_arc_qwen.py

    # With SLURM (starts vLLM + watchdog, then evaluation):
    sbatch run_arc_test.sh
    sbatch run_arc_eval.sh
"""

_base_ = "./config_arc.py"

tag = "arc_qwen"

use_local_proxy = True

deep_analyzer_tool_config = dict(
    type="deep_analyzer_tool",
    analyzer_model_ids=["Qwen"],
    summarizer_model_id="Qwen",
)

deep_analyzer_agent_config = dict(
    type="deep_analyzer_agent",
    name="deep_analyzer_agent",
    model_id="Qwen",
    description="A deep analyzer agent that can perform systematic, step-by-step analysis of ARC grid patterns.",
    max_steps=5,
    template_path="src/agent/deep_analyzer_agent/prompts/deep_analyzer_agent.yaml",
    provide_run_summary=True,
    tools=["deep_analyzer_tool", "python_interpreter_tool"],
)

general_agent_config = dict(
    type="general_agent",
    name="general_agent",
    model_id="Qwen",
    description="A general-purpose agent with Python execution for testing grid transformations.",
    max_steps=5,
    template_path="src/agent/general_agent/prompts/general_agent.yaml",
    provide_run_summary=True,
    tools=["python_interpreter_tool"],
)

planning_agent_config = dict(
    type="planning_agent",
    name="planning_agent",
    model_id="Qwen",
    description="A planning agent that coordinates sub-agents to solve ARC-AGI tasks.",
    max_steps=15,
    template_path="src/agent/planning_agent/prompts/planning_agent.yaml",
    provide_run_summary=True,
    tools=["planning_tool"],
    managed_agents=["deep_analyzer_agent", "general_agent"],
)

agent_config = planning_agent_config

per_question_timeout_secs = 1200
