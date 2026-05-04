"""
Configuration for AdaptivePlanningAgent with Qwen3-VL-4B-Instruct on vLLM (local).

Extends configs/config_gaia_c1.py (reactive C1 stack) but overrides all models
to Qwen (vLLM). Fixed tag `gaia_c1_qwen_local` — distinct from matrix API configs
`config_gaia_c1_qwen.py`.

Usage:
    python examples/run_gaia.py --config configs/config_gaia_c1_qwen_local.py

    # Compare with baseline
    python scripts/compare_results.py workdir/gaia/dra.jsonl workdir/gaia_c1_qwen_local/dra.jsonl
"""

_base_ = './config_gaia_c1.py'

tag = "gaia_c1_qwen_local"

dataset = dict(
    type="gaia_dataset",
    name="2023_all",
    path="data/GAIA",
    split="validation",  # use validation split - has real answers for scoring
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

# Pass 2 — Qwen-4B (vLLM, 32k context) specific tuning. Rationale:
#   - max_steps 3/3/5 was hit 56+ times in the 4.98% test-set run; 7 is a
#     starting point (pending GPU-farm P95 measurement of successful-run
#     step counts per sub-agent).
#   - context_prune_threshold_ratio 0.85 → 0.75 starts pruning at ~20 480
#     effective tokens instead of ~23 757, giving ~8 k headroom before the
#     32 k wall. Observed overflows reached 45 k–111 k tokens; tighter
#     pruning is the first step, deeper pruning (tail_segments 4→2) is
#     the out-of-scope follow-up if 0.75 does not close the gap.
# Scope: this file ONLY. The 12 matrix configs use qwen3.6-plus-failover
# (128 k API context), not the vLLM 4B — different context envelope,
# different step-budget needs, pending their own measurement.

# Override all agents to use Qwen (vLLM)
deep_researcher_agent_config = dict(
    type="deep_researcher_agent",
    name="deep_researcher_agent",
    model_id="Qwen",  # Use vLLM served model
    description="A deep researcher agent that can conduct extensive web searches.",
    max_steps=7,  # Pass 2.1: was 3, pending P95 measurement from first re-run
    context_prune_threshold_ratio=0.75,  # Pass 2.2: earlier than 0.85 default for Qwen 32k
    template_path="src/agent/deep_researcher_agent/prompts/deep_researcher_agent.yaml",
    provide_run_summary=True,
    tools=["deep_researcher_tool", "python_interpreter_tool"],
)

deep_analyzer_agent_config = dict(
    type="deep_analyzer_agent",
    name="deep_analyzer_agent",
    model_id="Qwen",
    description="A deep analyzer agent that can perform systematic analysis.",
    max_steps=7,  # Pass 2.1: was 3
    context_prune_threshold_ratio=0.75,  # Pass 2.2
    template_path="src/agent/deep_analyzer_agent/prompts/deep_analyzer_agent.yaml",
    provide_run_summary=True,
    tools=["deep_analyzer_tool", "python_interpreter_tool"],
)

browser_use_agent_config = dict(
    type="browser_use_agent",
    name="browser_use_agent",
    model_id="Qwen",
    description="A browser use agent for web interaction.",
    max_steps=7,  # Pass 2.1: was 5 (browser navigation is step-hungry)
    context_prune_threshold_ratio=0.75,  # Pass 2.2
    template_path="src/agent/browser_use_agent/prompts/browser_use_agent.yaml",
    provide_run_summary=True,
    tools=["auto_browser_use_tool", "python_interpreter_tool"],
)

planning_agent_config = dict(
    type="adaptive_planning_agent",
    name="adaptive_planning_agent",
    model_id="Qwen",
    description="An adaptive planning agent with self-modification capabilities.",
    max_steps=25,  # Planner budget unchanged; raising sub-agents gives them room within this ceiling
    context_prune_threshold_ratio=0.75,  # Pass 2.2 — planner accumulates more context than sub-agents
    template_path="src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml",
    provide_run_summary=True,
    tools=["planning_tool"],
    managed_agents=["deep_analyzer_agent", "browser_use_agent", "deep_researcher_agent"]
)

# Hard wall-clock limit per question (seconds). Prevents any single question from
# stalling indefinitely. 20 min is generous for a 4B model with max_steps=25.
per_question_timeout_secs = 1800

agent_config = planning_agent_config
