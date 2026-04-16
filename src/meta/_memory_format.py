"""
Shared memory / tools formatting helpers.

Module-level utilities for rendering an agent's execution memory and
available tools into human-readable strings. Used by:

- `DiagnoseSubAgentTool` (src/meta/diagnose_tool.py) — reactive diagnosis
- `ReviewAgent`           (src/meta/review_agent.py) — structural review (C3)

Keeping these as module-level functions (rather than instance methods) lets
both callers share the same logic without duplicating code or coupling the
ReviewAgent to DiagnoseSubAgentTool's class hierarchy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.base.async_multistep_agent import AsyncMultiStepAgent


# Truncation caps for rendered fields; kept as module-level constants so
# callers can reference them if they want to know the contract without
# duplicating magic numbers.
MAX_TASK_CHARS: int = 500
MAX_REASONING_CHARS: int = 800
MAX_TOOL_ARGS_CHARS: int = 300
MAX_OBSERVATION_CHARS: int = 500
MAX_TOOL_DESCRIPTION_CHARS: int = 200


def _truncate(text: str, limit: int) -> str:
    """Truncate `text` to `limit` chars, appending '...' if truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def format_execution_history(agent: "AsyncMultiStepAgent") -> str:
    """
    Format an agent's execution history (memory.steps) into a readable string.

    Handles heterogeneous step types (TaskStep, PlanningStep, ActionStep, etc.)
    by defensively checking for each relevant attribute via `hasattr`. Missing
    fields are silently omitted.

    Args:
        agent: Any agent with a `.memory.steps` list (AsyncMultiStepAgent or
            its subclasses).

    Returns:
        Multi-line string rendering each step with its reasoning, tool calls,
        observations, and errors. Returns a placeholder message when memory
        is empty or the agent has no memory.
    """
    if not hasattr(agent, "memory") or agent.memory is None:
        return "No execution history available (memory is empty)."

    steps = getattr(agent.memory, "steps", [])
    if not steps:
        return "No execution steps recorded."

    history_parts: list[str] = []
    for i, step in enumerate(steps):
        step_info: list[str] = [f"=== Step {i + 1} ==="]
        step_info.append(f"Type: {type(step).__name__}")

        if hasattr(step, "task"):
            step_info.append(f"Task: {_truncate(str(step.task), MAX_TASK_CHARS)}")

        if hasattr(step, "model_output") and step.model_output:
            step_info.append(
                f"Agent Reasoning: {_truncate(str(step.model_output), MAX_REASONING_CHARS)}"
            )

        if hasattr(step, "tool_calls") and step.tool_calls:
            for tc in step.tool_calls:
                if hasattr(tc, "name") and hasattr(tc, "arguments"):
                    args_str = _truncate(str(tc.arguments), MAX_TOOL_ARGS_CHARS)
                    step_info.append(f"Tool Called: {tc.name}({args_str})")

        if hasattr(step, "observations") and step.observations:
            step_info.append(
                f"Observation: {_truncate(str(step.observations), MAX_OBSERVATION_CHARS)}"
            )

        if hasattr(step, "error") and step.error:
            step_info.append(f"ERROR: {step.error}")

        history_parts.append("\n".join(step_info))

    return "\n\n".join(history_parts)


def format_agent_tools(agent: "AsyncMultiStepAgent") -> str:
    """
    Format an agent's available tools into a readable bullet list.

    Args:
        agent: Any agent with a `.tools` dict mapping name -> tool.

    Returns:
        Multi-line string with one bullet per tool. Returns a placeholder
        when the agent has no tools.
    """
    if not hasattr(agent, "tools") or not agent.tools:
        return "No tools available."

    tool_info: list[str] = []
    for name, tool in agent.tools.items():
        desc = getattr(tool, "description", "No description")
        tool_info.append(f"- {name}: {_truncate(desc, MAX_TOOL_DESCRIPTION_CHARS)}")

    return "\n".join(tool_info)
