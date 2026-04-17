"""
Pydantic schemas for the structural REVIEW step (condition C3).

The `ReviewResult` is produced by the `ReviewAgent` (see `review_agent.py`) after
each sub-agent delegation and describes:

1. A verdict on whether the sub-agent's response was satisfactory
2. Root-cause classification when unsatisfactory (fixed taxonomy, not free text —
   enables aggregate analysis across runs)
3. A polymorphic `next_action` specifying what the planning agent should do next

The `next_action` is a discriminated union so the planner can dispatch on
`action` without type-checking. `ModifyAgentSpec`'s fields match
`ModifySubAgentTool.forward()` exactly, so a `modify_agent` next_action can be
passed through to `modify_subagent` without reformatting.

Design invariants:
- Review apparatus is SEALED — these schemas are fixed at import time and are
  NOT modifiable via `modify_subagent`. This prevents reward hacking where the
  agent makes the reviewer lenient instead of solving the task.
- Same model as the planner produces these — no separate LLM.
"""

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# -- Root-cause taxonomy -----------------------------------------------------

class RootCauseCategory(str, Enum):
    """
    Fixed taxonomy of failure categories. Kept small (8 entries) and mapped
    1:1 to the kinds of remediation the planning agent can actually perform.

    Do NOT add free-text categories here — the research value of the taxonomy
    depends on stability across runs. If a failure doesn't fit any category,
    use the closest match and put the detail in `ReviewResult.root_cause_detail`.
    """
    MISSING_TOOL = "missing_tool"              # Agent lacked a needed capability
    WRONG_TOOL_SELECTION = "wrong_tool"        # Had the tool, picked wrong one
    INSUFFICIENT_INSTRUCTION = "bad_instruction"  # Prompt/task was underspecified
    TASK_MISUNDERSTANDING = "misread_task"     # Agent misinterpreted objective
    EXTERNAL_FAILURE = "external"              # Network, rate limit, paywall, etc.
    MODEL_LIMITATION = "model_limit"           # Reasoning capacity exceeded
    UNCLEAR_OBJECTIVE = "unclear_goal"         # Manager's task was ambiguous
    INCOMPLETE_OUTPUT = "incomplete"           # Correct direction, not finished


# -- next_action discriminated union -----------------------------------------

class ProceedSpec(BaseModel):
    """Satisfactory result — continue with the plan. No payload."""
    model_config = ConfigDict(extra="forbid")

    action: Literal["proceed"] = "proceed"


class RetrySpec(BaseModel):
    """
    Retry the same sub-agent with a reformulated task.

    Use when the failure was due to unclear instructions or misread task, and
    the sub-agent is still the right one for the job.
    """
    model_config = ConfigDict(extra="forbid")

    action: Literal["retry"] = "retry"
    agent_name: str
    revised_task: str
    additional_guidance: str
    avoid_patterns: list[str] = Field(
        default_factory=list,
        description="Behaviours/approaches the sub-agent should NOT repeat.",
    )


class ModifyAgentSpec(BaseModel):
    """
    Modify a sub-agent's capability before re-delegating.

    Fields mirror `ModifySubAgentTool.forward()` so a `ModifyAgentSpec` can be
    passed through to `modify_subagent` without reformatting. Keep this model
    aligned with any action changes in `src/meta/modify_tool.py`.
    """
    model_config = ConfigDict(extra="forbid")

    action: Literal["modify_agent"] = "modify_agent"
    modify_action: Literal[
        "add_existing_tool_to_agent",
        "add_new_tool_to_agent",
        "remove_tool_from_agent",
        "modify_agent_instructions",
        "add_agent",
        "remove_agent",
        "set_agent_max_steps",
    ]
    agent_name: str
    specification: str
    followup_retry: bool = Field(
        default=True,
        description="Whether the planner should re-delegate after the modification.",
    )


class EscalateSpec(BaseModel):
    """
    Switch to a different sub-agent (e.g. because the original is unsuited).

    The `to_agent` MUST be an existing managed-agent name — ReviewStep
    validates this before returning; a hallucinated name falls back to
    ProceedSpec with a warning in `summary`.
    """
    model_config = ConfigDict(extra="forbid")

    action: Literal["escalate"] = "escalate"
    from_agent: str
    to_agent: str
    reason: str
    task: str


NextAction = Union[ProceedSpec, RetrySpec, ModifyAgentSpec, EscalateSpec]


# -- ReviewResult ------------------------------------------------------------

class ReviewResult(BaseModel):
    """
    Output of the ReviewAgent for a single sub-agent delegation.

    This object is serialised into the delegating action step's observations
    (with a `[REVIEW]` marker) so the planning agent sees it in its next THINK.
    See `AdaptivePlanningAgent._post_action_hook`.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["satisfactory", "partial", "unsatisfactory"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(description="One-line human-readable assessment.")

    # Populated only when verdict != "satisfactory".
    root_cause_primary: Optional[RootCauseCategory] = None
    root_cause_secondary: Optional[RootCauseCategory] = None
    root_cause_detail: Optional[str] = None

    next_action: NextAction = Field(discriminator="action")

    def render(self) -> str:
        """
        Compact text rendering for inclusion in `action_step.observations`.

        Kept short so it doesn't dominate the observation context. Full
        structured data remains available via the pydantic model if downstream
        code needs it.
        """
        lines = [
            f"verdict: {self.verdict} (confidence={self.confidence:.2f})",
            f"summary: {self.summary}",
        ]
        if self.root_cause_primary is not None:
            rc_line = f"root_cause: {self.root_cause_primary.value}"
            if self.root_cause_secondary is not None:
                rc_line += f" (+{self.root_cause_secondary.value})"
            lines.append(rc_line)
        if self.root_cause_detail:
            lines.append(f"detail: {self.root_cause_detail}")
        lines.append(f"next_action: {self._render_next_action()}")
        return "\n".join(lines)

    def _render_next_action(self) -> str:
        """One-line rendering of the next action, dispatched on type."""
        na = self.next_action
        if isinstance(na, ProceedSpec):
            return "proceed"
        if isinstance(na, RetrySpec):
            return f"retry {na.agent_name} with: {na.revised_task[:120]}"
        if isinstance(na, ModifyAgentSpec):
            return (
                f"modify_agent {na.agent_name} "
                f"[{na.modify_action}]: {na.specification[:120]}"
            )
        if isinstance(na, EscalateSpec):
            return f"escalate {na.from_agent} -> {na.to_agent}: {na.reason[:120]}"
        return "unknown"
