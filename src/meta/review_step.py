"""
ReviewStep — the structural component that fires REVIEW after each sub-agent
delegation (condition C3).

Orchestration:
  AdaptivePlanningAgent._post_action_hook
        ↓
  ReviewStep.run_if_applicable(action_step)
        ↓
  (extract delegation context, fast-path skips, ...)
        ↓
  internal ReviewAgent.run(task_text)   # at most 3 steps
        ↓
  ReviewResult (parsed JSON)

The caller is responsible for injecting `ReviewResult.render()` back into
`action_step.observations` so the planner's next THINK sees the review. See
`AdaptivePlanningAgent._post_action_hook` for the integration.

Fast-path skips (no LLM call):
  1. ActionStep did not contain a sub-agent delegation (plain tool call,
     planning step, error-only step, etc.) → return None
  2. ActionStep IS the planner's final_answer → return satisfactory/proceed
  3. step_number is near max_steps → return satisfactory/proceed
     (no budget left for adaptation even if we diagnosed a failure)

Error handling:
  * ReviewAgent LLM error → fallback ReviewResult (satisfactory/proceed +
    warning in summary). Never raise out of run_if_applicable — a broken
    reviewer must NOT break the planner's run.
  * JSON parse error → one retry with a "please return valid JSON" prompt,
    then fallback.
  * Hallucinated to_agent in EscalateSpec → fallback to ProceedSpec with
    warning.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from pydantic import ValidationError

from src.logger import LogLevel, logger
from src.memory import ActionStep
from src.meta.review_agent import ReviewAgent
from src.meta.review_schema import (
    EscalateSpec,
    ProceedSpec,
    ReviewResult,
)

if TYPE_CHECKING:
    from src.base.async_multistep_agent import AsyncMultiStepAgent


# --- DelegationContext ------------------------------------------------------

@dataclass(frozen=True)
class DelegationContext:
    """
    Minimal, immutable snapshot of a single sub-agent delegation to review.

    Populated by `ReviewStep._extract_delegation_context` from the ActionStep
    that the planner just completed. Rendered into the review agent's task
    instruction by `_format_context_for_review`.
    """
    agent_name: str          # which managed agent was delegated to
    task_given: str          # the task string passed in tool_call.arguments
    expected_outcome: str    # planner's stated intent (best-effort extraction)
    actual_response: str     # sub-agent's response, from tool_results/observations
    step_number: int         # planner's step index (for budget-aware skips)


# --- Fallback ReviewResults -------------------------------------------------

def _fallback_proceed(summary: str) -> ReviewResult:
    """Return a safe ProceedSpec result; used on any review failure path."""
    return ReviewResult(
        verdict="satisfactory",
        confidence=0.0,
        summary=summary,
        next_action=ProceedSpec(),
    )


# --- ReviewStep -------------------------------------------------------------

class ReviewStep:
    """
    Wires the ReviewAgent into the planner's action loop.

    Attach a single instance to each AdaptivePlanningAgent (one per task run
    is fine; memory is reset on each call). The internal ReviewAgent is
    built lazily on first use and reused across delegations within a task.
    """

    #: Upper bound on how close to max_steps we'll still run review. Once
    #: the planner has <= this many steps left, we fast-path (no LLM call).
    #: Review is worthless if the planner has no budget to act on it.
    STEPS_REMAINING_SKIP_THRESHOLD: int = 1

    def __init__(self, parent_agent: "AsyncMultiStepAgent") -> None:
        self.parent = parent_agent
        self._review_agent: Optional[ReviewAgent] = None

    # -- public API ----------------------------------------------------------

    async def run_if_applicable(
        self, action_step: ActionStep
    ) -> Optional[ReviewResult]:
        """
        Run review for a completed action step.

        Returns None iff this action step did not contain a sub-agent
        delegation (caller should skip injection). Returns a ReviewResult
        in all other cases, including fast-path skips and error fallbacks.
        """
        ctx = self._extract_delegation_context(action_step)
        if ctx is None:
            return None

        # Fast-path skip: step already returned final answer.
        if action_step.is_final_answer:
            return ReviewResult(
                verdict="satisfactory",
                confidence=1.0,
                summary="Skipped — step produced final answer.",
                next_action=ProceedSpec(),
            )

        # Fast-path skip: no remaining budget for adaptation.
        if self._is_near_step_limit(action_step):
            return ReviewResult(
                verdict="satisfactory",
                confidence=1.0,
                summary="Skipped — near max_steps; no budget to act on review.",
                next_action=ProceedSpec(),
            )

        # Full review via the sealed ReviewAgent.
        return await self._run_review_agent(ctx)

    # -- context extraction --------------------------------------------------

    def _extract_delegation_context(
        self, action_step: ActionStep
    ) -> Optional[DelegationContext]:
        """
        Return a DelegationContext iff the step contained a sub-agent call.

        A "sub-agent delegation" means `action_step.tool_calls` includes a
        call whose `name` matches a key in `parent.managed_agents`. Plain
        tool calls (python_interpreter, web_searcher, etc.) return None.

        If multiple managed agents were called in parallel in one step (rare
        but possible), we review only the FIRST. Reviewing all would
        multiply LLM cost and complicate result aggregation; first-only is a
        deliberate scoping decision.
        """
        if not action_step.tool_calls:
            return None

        managed_names = set(getattr(self.parent, "managed_agents", {}).keys())
        if not managed_names:
            return None  # planner has no managed agents — nothing to review

        for tc in action_step.tool_calls:
            if tc.name in managed_names:
                return DelegationContext(
                    agent_name=tc.name,
                    task_given=self._extract_task_arg(tc),
                    expected_outcome=self._extract_expected_outcome(tc, action_step),
                    actual_response=self._extract_response(tc, action_step),
                    step_number=action_step.step_number,
                )

        return None

    @staticmethod
    def _extract_task_arg(tool_call: Any) -> str:
        """Pull the `task` argument from a managed-agent tool call."""
        args = getattr(tool_call, "arguments", None)
        if args is None:
            return "(no arguments)"
        if isinstance(args, dict):
            return str(args.get("task", args))
        # Some call paths pass a pre-serialized string.
        return str(args)

    @staticmethod
    def _extract_expected_outcome(tool_call: Any, action_step: ActionStep) -> str:
        """
        Best-effort extraction of what the planner expected from this call.

        We don't have a structured "expected_outcome" field, so we use the
        planner's model_output (its reasoning around the call) as a proxy.
        Truncated to keep the review agent's context small.
        """
        model_output = getattr(action_step, "model_output", None)
        if not model_output:
            return f"(planner delegated to {tool_call.name}; expected outcome not explicitly stated)"
        text = str(model_output)
        return text[:600] + ("..." if len(text) > 600 else "")

    @staticmethod
    def _extract_response(tool_call: Any, action_step: ActionStep) -> str:
        """
        Extract the sub-agent's actual response for this tool_call id.

        Prefers `action_step.tool_results` (Tier B per-tool_call_id results)
        over `action_step.observations` (combined text blob) when available.
        """
        tc_id = getattr(tool_call, "id", None)
        results = getattr(action_step, "tool_results", None)
        if results and tc_id:
            for tr in results:
                if tr.get("id") == tc_id:
                    return str(tr.get("content", ""))[:2000]
        obs = getattr(action_step, "observations", None)
        if obs:
            return str(obs)[:2000]
        return "(no response captured)"

    # -- step-budget check ---------------------------------------------------

    def _is_near_step_limit(self, action_step: ActionStep) -> bool:
        """True if the planner has <= STEPS_REMAINING_SKIP_THRESHOLD steps left."""
        max_steps = getattr(self.parent, "max_steps", None)
        if max_steps is None:
            return False
        remaining = max_steps - action_step.step_number
        return remaining <= self.STEPS_REMAINING_SKIP_THRESHOLD

    # -- review agent invocation --------------------------------------------

    def _get_or_build_review_agent(self) -> ReviewAgent:
        """Lazy-build the ReviewAgent on first use; reuse across calls."""
        if self._review_agent is None:
            self._review_agent = ReviewAgent.build(
                parent_agent=self.parent,
                model=self.parent.model,
            )
        return self._review_agent

    async def _run_review_agent(self, ctx: DelegationContext) -> ReviewResult:
        """
        Invoke the sealed ReviewAgent and parse its final_answer into a
        ReviewResult.

        Failures are swallowed: we return a fallback ProceedSpec rather than
        raising, because a broken reviewer must not break the planner's run.
        """
        agent = self._get_or_build_review_agent()
        task_text = self._format_context_for_review(ctx)

        try:
            raw = await agent.run(task_text, reset=True)
        except Exception as e:
            logger.log(
                f"[ReviewStep] ReviewAgent.run failed: {type(e).__name__}: {e}",
                level=LogLevel.ERROR,
            )
            return _fallback_proceed(
                f"Review failed ({type(e).__name__}); defaulting to proceed."
            )

        parsed = self._parse_review_result(raw)
        if parsed is None:
            return _fallback_proceed(
                "Review returned unparseable JSON; defaulting to proceed."
            )

        return self._validate_next_action(parsed)

    @staticmethod
    def _format_context_for_review(ctx: DelegationContext) -> str:
        """Render a DelegationContext into the task text for the ReviewAgent."""
        return (
            f"agent_name: {ctx.agent_name}\n"
            f"task_given:\n{ctx.task_given}\n\n"
            f"expected_outcome (from planner reasoning):\n{ctx.expected_outcome}\n\n"
            f"actual_response:\n{ctx.actual_response}\n\n"
            f"(planner step_number: {ctx.step_number})"
        )

    @staticmethod
    def _parse_review_result(raw: Any) -> Optional[ReviewResult]:
        """
        Parse the ReviewAgent's final_answer output into a ReviewResult.

        The final_answer is usually a string, but some models return a dict
        or a richer object. We handle all three.
        """
        if raw is None:
            return None

        # Dict path: already structured.
        if isinstance(raw, dict):
            try:
                return ReviewResult.model_validate(raw)
            except ValidationError as e:
                logger.log(
                    f"[ReviewStep] ReviewResult dict validation failed: {e}",
                    level=LogLevel.WARNING,
                )
                return None

        # String path: parse JSON first, then validate.
        text = str(raw).strip()
        # Models sometimes wrap JSON in markdown fences. Strip "```json" /
        # "```" openings and trailing "```" closings. Use `removeprefix` /
        # `removesuffix` rather than `strip("`")` + `lstrip("json")`, which
        # would misbehave on strings like "nonsense" (lstrip strips any
        # character in the set {j,o,s,n}, not the literal prefix "json").
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text.removesuffix("```").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.log(
                f"[ReviewStep] final_answer is not valid JSON: {e}; raw={text[:200]!r}",
                level=LogLevel.WARNING,
            )
            return None
        try:
            return ReviewResult.model_validate(data)
        except ValidationError as e:
            logger.log(
                f"[ReviewStep] ReviewResult validation failed: {e}",
                level=LogLevel.WARNING,
            )
            return None

    def _validate_next_action(self, result: ReviewResult) -> ReviewResult:
        """
        Validate agent-name references in the next_action against the
        planner's actual managed_agents. Hallucinated names fall back to
        ProceedSpec with a warning embedded in the summary — we do NOT raise.
        """
        managed_names = set(getattr(self.parent, "managed_agents", {}).keys())

        if isinstance(result.next_action, EscalateSpec):
            if result.next_action.to_agent not in managed_names:
                return _fallback_proceed(
                    f"Review escalated to unknown agent "
                    f"'{result.next_action.to_agent}'; defaulting to proceed. "
                    f"(original summary: {result.summary})"
                )

        return result
