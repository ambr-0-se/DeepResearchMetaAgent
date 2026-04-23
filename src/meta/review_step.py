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
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from pydantic import ValidationError

from src.logger import LogLevel, logger
from src.memory import ActionStep
from src.meta.review_schema import (
    EscalateSpec,
    ModifyAgentSpec,
    ProceedSpec,
    RetrySpec,
    ReviewResult,
    RootCauseCategory,
)

if TYPE_CHECKING:
    from src.base.async_multistep_agent import AsyncMultiStepAgent
    from src.meta.review_agent import ReviewAgent  # runtime import deferred
                                                    # (see _get_or_build_review_agent)


# --- Retry cap table (per-root-cause) ---------------------------------------

#: Asymmetric retry budgets by root cause. Cap=0 means RetrySpec for that
#: cause is immediately coerced to ProceedSpec — rephrasing can't unlock
#: paywalls (external), fix reasoning gaps (model_limit), or synthesise
#: missing capabilities (missing_tool / wrong_tool). The advisory table
#: in REVIEW_AGENT_SYSTEM_PROMPT mirrors these values.
RETRY_CAP_BY_ROOT_CAUSE: dict[RootCauseCategory, int] = {
    RootCauseCategory.INSUFFICIENT_INSTRUCTION: 2,  # bad_instruction
    RootCauseCategory.TASK_MISUNDERSTANDING:    2,  # misread_task
    RootCauseCategory.UNCLEAR_OBJECTIVE:        2,  # unclear_goal
    RootCauseCategory.INCOMPLETE_OUTPUT:        1,  # incomplete
    RootCauseCategory.EXTERNAL_FAILURE:         0,  # external
    RootCauseCategory.MODEL_LIMITATION:         0,  # model_limit
    RootCauseCategory.MISSING_TOOL:             0,  # missing_tool
    RootCauseCategory.WRONG_TOOL_SELECTION:     0,  # wrong_tool
}

#: Default cap for unknown root causes (defensive — the pydantic enum should
#: prevent this, but a defensive 0 means "never retry" if somehow an unknown
#: string slips through).
_DEFAULT_RETRY_CAP: int = 0


#: Ordered list of metric keys. Used by on_task_start to reset and by
#: tests to assert the schema. Must match the plan's §Layer 2 table.
_METRIC_KEYS: tuple[str, ...] = (
    "retry_chains_started",
    "retry_chains_capped",
    "retry_coercions_to_proceed",
    "blocklist_coercions",
    "modify_agent_emitted",
    "escalate_emitted",
    "proceed_emitted",
    "max_chain_length",
)


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


# --- PriorAttempt (chain-ledger history row) --------------------------------

@dataclass(frozen=True)
class PriorAttempt:
    """
    A single earlier attempt on the same delegation lineage, stored in the
    chain ledger's `prior` list. Rendered into the reviewer task text so
    the reviewer can see what has already been tried on this intent.

    Only the fields needed for the reviewer's decision are included —
    full ReviewResults are not retained.
    """
    attempt_idx: int                # 1-based index within the chain
    verdict: str                    # "satisfactory" | "partial" | "unsatisfactory"
    root_cause: Optional[str]       # str value of RootCauseCategory, or None
    revised_task_digest: str        # task_given truncated to PRIOR_ATTEMPT_DIGEST_MAX


# --- ChainState (retry-ledger node) -----------------------------------------

@dataclass
class ChainState:
    """
    Per-lineage retry-chain state. Keyed by (agent_name, intent_anchor) in
    ReviewStep._chains. A chain starts fresh on a new delegation to an agent
    without a pending retry flag; continues on the next delegation that
    inherits the anchor via _pending_retry_anchor.
    """
    anchor: str                              # UUID, the chain id
    count: int = 0                           # retries consumed
    capped: bool = False                     # True once count reaches the cap
    last_root_cause: Optional[str] = None    # str value of last emitted root cause
    prior: list[PriorAttempt] = field(default_factory=list)


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

    #: Max chars of a revised_task kept in the prior_attempts digest. 150
    #: chars is enough to identify intent; keeping 5 priors × 150 chars ≈
    #: 1 KB of history context, well under the reviewer's 3-step budget.
    PRIOR_ATTEMPT_DIGEST_MAX: int = 150

    #: Hard cap on how many prior attempts we retain per chain. Beyond this,
    #: retry is categorically capped by Layer 2's per-root-cause limits, so
    #: more history adds tokens without adding signal.
    PRIOR_ATTEMPTS_KEEP_LAST: int = 5

    def __init__(self, parent_agent: "AsyncMultiStepAgent") -> None:
        self.parent = parent_agent
        self._review_agent: Optional["ReviewAgent"] = None
        #: Original user task for the current GAIA question. Set by
        #: `on_task_start` (called from the planner); rendered into the
        #: reviewer task text so the reviewer can tell "sub-agent failed its
        #: sub-task" from "planner gave wrong sub-task". Empty string when
        #: not yet initialised — harmless (the rendering guards on it).
        self._original_user_task: str = ""
        #: Chain ledger: (agent_name, intent_anchor) -> ChainState.
        #: New delegations without a pending retry flag mint a fresh anchor;
        #: review-driven continuations inherit the anchor from
        #: _pending_retry_anchor. See _resolve_anchor.
        self._chains: dict[tuple[str, str], ChainState] = {}
        #: When the prior ReviewResult for `agent_name` was a RetrySpec,
        #: the chain's anchor is stashed here so the NEXT delegation to
        #: that agent inherits it (same logical intent). Popped on read.
        self._pending_retry_anchor: dict[str, str] = {}
        #: Chains that have reached their per-root-cause cap. Any RetrySpec
        #: targeting a key in this set is coerced to ProceedSpec.
        self._capped_anchors: set[tuple[str, str]] = set()
        #: Task-wide blocklist: (agent_name, root_cause_str). Once an agent
        #: has failed with a cap=0 root cause, ALL future delegations to
        #: that (agent, root_cause) combination are blocked — regardless of
        #: whether the delegation is review-driven (pending_retry_anchor)
        #: or planner-initiated (fresh anchor). Closes the planner re-entry
        #: bypass described in the plan.
        self._task_blocklist: set[tuple[str, str]] = set()
        #: Per-task metrics. Flushed by the caller (run_gaia.py) from
        #: `agent.review_step._metrics` after the task completes — see
        #: plan §Layer 3.
        self._metrics: dict[str, int] = {k: 0 for k in _METRIC_KEYS}

    # -- task lifecycle -----------------------------------------------------

    def on_task_start(self, original_user_task: str) -> None:
        """
        Initialise per-task state for a new GAIA question.

        Called at the top of `AdaptivePlanningAgent.run()`. Idempotent —
        safe to call multiple times in succession; all state is cleared
        each call. In practice the planner is re-created per question, so
        this method is primarily a (a) init hook for `_original_user_task`,
        (b) metric zero-ing hook, and (c) defensive cleanup in case a
        future refactor reuses a single planner across tasks.

        Intentionally does NOT call `_flush_metrics` — metric extraction
        happens at the caller level (run_gaia.py) after `agent.run()`
        returns, because under P1 the agent task may be cancelled before
        any in-agent `finally` could fire.
        """
        self._chains.clear()
        self._pending_retry_anchor.clear()
        self._capped_anchors.clear()
        self._task_blocklist.clear()
        self._review_agent = None   # rebuild on next use (fresh Monitor/AgentLogger)
        self._metrics = {k: 0 for k in _METRIC_KEYS}
        self._original_user_task = original_user_task or ""

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

        # Resolve the chain for this delegation BEFORE the LLM call so we
        # can inject prior_attempts and blocklist directives into the
        # reviewer's task text. Chains are identified by (agent_name, anchor)
        # where anchor is a UUID inherited on pending retries or fresh
        # otherwise.
        anchor, is_new_chain = self._resolve_anchor(ctx.agent_name)
        chain_key = (ctx.agent_name, anchor)
        if is_new_chain:
            self._chains[chain_key] = ChainState(anchor=anchor)
            self._metrics["retry_chains_started"] += 1
        chain = self._chains[chain_key]

        # Compose directives for the reviewer.
        task_blocklist_directive = self._render_task_blocklist_directive(ctx.agent_name)
        chain_capped_directive = (
            self._render_chain_capped_directive(chain)
            if chain_key in self._capped_anchors
            else ""
        )

        # Full review via the sealed ReviewAgent.
        result = await self._run_review_agent(
            ctx,
            prior_attempts=list(chain.prior),
            task_blocklist_directive=task_blocklist_directive,
            chain_capped_directive=chain_capped_directive,
        )

        # Dispatch on next_action to update the chain ledger.
        return self._dispatch_review_result(result, ctx, chain_key, chain)

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

    # -- chain ledger --------------------------------------------------------

    def _resolve_anchor(self, agent_name: str) -> tuple[str, bool]:
        """
        Return (intent_anchor, is_new_chain) for a delegation to `agent_name`.

        If the prior ReviewResult for this agent was a RetrySpec targeting
        the same agent, `_pending_retry_anchor[agent_name]` holds the
        anchor from that chain; we pop and reuse it (continuation). Else
        we mint a new UUID (fresh chain).
        """
        pending = self._pending_retry_anchor.pop(agent_name, None)
        if pending is not None:
            return pending, False
        return str(uuid.uuid4()), True

    def _retry_cap_for(self, root_cause: Optional[RootCauseCategory]) -> int:
        """Look up the retry cap for a root cause; default 0 for unknown/None."""
        if root_cause is None:
            return _DEFAULT_RETRY_CAP
        return RETRY_CAP_BY_ROOT_CAUSE.get(root_cause, _DEFAULT_RETRY_CAP)

    def _render_task_blocklist_directive(self, agent_name: str) -> str:
        """
        Render a directive listing all (root_cause)s this agent is blocked
        for in the current task. Empty string when the agent has no block.
        """
        blocked = sorted(rc for (a, rc) in self._task_blocklist if a == agent_name)
        if not blocked:
            return ""
        causes = ", ".join(blocked)
        return (
            f"IMPORTANT: agent '{agent_name}' previously failed in this task with "
            f"root_cause(s)=[{causes}]. Retry is unavailable for this combination; "
            f"choose modify_agent, escalate, or proceed."
        )

    @staticmethod
    def _render_chain_capped_directive(chain: ChainState) -> str:
        """Render a directive noting the chain has reached its retry cap."""
        rc = chain.last_root_cause or "unknown"
        return (
            f"CHAIN CAPPED: this delegation lineage has reached its retry cap "
            f"(last root_cause={rc}, attempts={chain.count}). Any further "
            f"RetrySpec will be coerced to proceed. Choose modify_agent, "
            f"escalate, or proceed."
        )

    def _dispatch_review_result(
        self,
        result: ReviewResult,
        ctx: DelegationContext,
        chain_key: tuple[str, str],
        chain: ChainState,
    ) -> ReviewResult:
        """
        Update chain ledger + task blocklist + metrics based on the
        reviewer's `next_action`. Coerces invalid RetrySpec emissions to
        ProceedSpec (cap=0 root causes, already-capped chains, blocklisted
        (agent, cause) combinations). Returns the possibly-rewritten
        ReviewResult.
        """
        agent_name = ctx.agent_name
        next_action = result.next_action

        # --- RetrySpec: the path that drives most chain mutation -----------
        if isinstance(next_action, RetrySpec):
            root_cause = result.root_cause_primary
            cap = self._retry_cap_for(root_cause)
            rc_str = root_cause.value if root_cause is not None else None

            # Check blocklist (task-wide) BEFORE cap — blocklist is the
            # broader guard that also covers planner-initiated re-entry.
            if rc_str is not None and (agent_name, rc_str) in self._task_blocklist:
                self._metrics["blocklist_coercions"] += 1
                return self._coerce_to_proceed(
                    result,
                    summary=(
                        f"Retry cap hit for chain on '{agent_name}' "
                        f"(task-blocklist for root_cause={rc_str}); coerced to proceed. "
                        f"Original summary: {result.summary}"
                    ),
                )

            # Check per-chain cap.
            if chain_key in self._capped_anchors:
                self._metrics["blocklist_coercions"] += 1
                return self._coerce_to_proceed(
                    result,
                    summary=(
                        f"Retry cap hit for chain on '{agent_name}' "
                        f"(anchor capped, last_root_cause={chain.last_root_cause}); "
                        f"coerced to proceed. Original summary: {result.summary}"
                    ),
                )

            # cap == 0: this root cause never allows retry.
            if cap == 0:
                if rc_str is not None:
                    self._task_blocklist.add((agent_name, rc_str))
                self._metrics["retry_coercions_to_proceed"] += 1
                chain.capped = True
                chain.last_root_cause = rc_str
                self._capped_anchors.add(chain_key)
                return self._coerce_to_proceed(
                    result,
                    summary=(
                        f"Retry unavailable for root_cause={rc_str or 'unknown'} "
                        f"on '{agent_name}'; coerced to proceed. "
                        f"Original summary: {result.summary}"
                    ),
                )

            # cap > 0: increment and check if we've now reached the cap.
            chain.count += 1
            chain.last_root_cause = rc_str
            chain.prior.append(
                PriorAttempt(
                    attempt_idx=chain.count,
                    verdict=result.verdict,
                    root_cause=rc_str,
                    revised_task_digest=self._digest_task(next_action.revised_task),
                )
            )
            # Bound history size.
            if len(chain.prior) > self.PRIOR_ATTEMPTS_KEEP_LAST:
                chain.prior = chain.prior[-self.PRIOR_ATTEMPTS_KEEP_LAST:]
            if chain.count > self._metrics["max_chain_length"]:
                self._metrics["max_chain_length"] = chain.count
            # Pending flag so the next delegation to this agent inherits the anchor.
            self._pending_retry_anchor[agent_name] = chain.anchor

            if chain.count >= cap:
                chain.capped = True
                self._capped_anchors.add(chain_key)
                if rc_str is not None:
                    self._task_blocklist.add((agent_name, rc_str))
                self._metrics["retry_chains_capped"] += 1
            return result

        # --- EscalateSpec: chain terminates; escalate implies "stop this agent" ---
        if isinstance(next_action, EscalateSpec):
            from_agent = next_action.from_agent
            if result.root_cause_primary is not None:
                self._task_blocklist.add(
                    (from_agent, result.root_cause_primary.value)
                )
            # Mark the CURRENT chain as capped (not the escalate target's chain;
            # target gets a fresh chain on its next delegation).
            if from_agent == agent_name:
                self._capped_anchors.add(chain_key)
                chain.capped = True
            self._pending_retry_anchor.pop(from_agent, None)
            self._metrics["escalate_emitted"] += 1
            return result

        # --- ModifyAgentSpec: chain terminates; next delegation is fresh ----
        if isinstance(next_action, ModifyAgentSpec):
            self._pending_retry_anchor.pop(next_action.agent_name, None)
            # followup_retry=True: next delegation mints a new chain; we do
            # NOT pre-seed _pending_retry_anchor, so _resolve_anchor on the
            # follow-up will report is_new_chain=True. Task blocklist is
            # NOT auto-cleared by modify (modify doesn't prove the prior
            # root cause is fixed).
            self._metrics["modify_agent_emitted"] += 1
            return result

        # --- ProceedSpec: chain terminates cleanly ---------------------------
        if isinstance(next_action, ProceedSpec):
            self._pending_retry_anchor.pop(agent_name, None)
            self._metrics["proceed_emitted"] += 1
            return result

        # Unknown next_action type — should be unreachable given the
        # discriminated union, but be defensive.
        logger.log(
            f"[ReviewStep] unknown next_action type {type(next_action).__name__}; "
            f"treating as proceed.",
            level=LogLevel.WARNING,
        )
        self._metrics["proceed_emitted"] += 1
        return result

    @staticmethod
    def _coerce_to_proceed(original: ReviewResult, *, summary: str) -> ReviewResult:
        """
        Rewrite a ReviewResult's next_action to ProceedSpec while preserving
        the verdict / confidence / root_cause fields for downstream logging.
        """
        return ReviewResult(
            verdict=original.verdict,
            confidence=original.confidence,
            summary=summary,
            root_cause_primary=original.root_cause_primary,
            root_cause_secondary=original.root_cause_secondary,
            root_cause_detail=original.root_cause_detail,
            next_action=ProceedSpec(),
        )

    # -- step-budget check ---------------------------------------------------

    def _is_near_step_limit(self, action_step: ActionStep) -> bool:
        """True if the planner has <= STEPS_REMAINING_SKIP_THRESHOLD steps left."""
        max_steps = getattr(self.parent, "max_steps", None)
        if max_steps is None:
            return False
        remaining = max_steps - action_step.step_number
        return remaining <= self.STEPS_REMAINING_SKIP_THRESHOLD

    # -- review agent invocation --------------------------------------------

    def _get_or_build_review_agent(self) -> "ReviewAgent":
        """
        Lazy-build the ReviewAgent on first use; reuse across calls.

        The sub-agent catalog is rendered lazily at build time from the
        planner's current `managed_agents` dict, so it reflects whatever
        the planner was given at construction time. Because `build()` is
        idempotent and each GAIA question gets a fresh planner (and thus
        a fresh `ReviewStep`), the catalog is effectively per-task.

        `ReviewAgent` is imported lazily here to break the circular import
        chain (review_step → review_agent → general_agent → agent.__init__
        → adaptive_planning_agent → review_step). The lazy import keeps
        `src.meta.review_step` importable without transitively pulling in
        the agent construction graph, which enables unit-testing of the
        chain ledger without instantiating real agents.
        """
        if self._review_agent is None:
            from src.meta.review_agent import ReviewAgent
            catalog = self._render_sub_agent_catalog()
            self._review_agent = ReviewAgent.build(
                parent_agent=self.parent,
                model=self.parent.model,
                sub_agent_catalog=catalog,
            )
        return self._review_agent

    def _render_sub_agent_catalog(self) -> str:
        """
        Produce a compact string listing the planner's managed agents:
        name, description, and tool names. Rendered once per ReviewAgent
        build and injected into REVIEW_AGENT_SYSTEM_PROMPT under
        AVAILABLE SUB-AGENTS.

        Returns an empty string when the planner has no managed agents —
        the prompt's `{%- if sub_agent_catalog %}` guards then omit the
        entire section (and the references to it) cleanly.
        """
        managed = getattr(self.parent, "managed_agents", None) or {}
        if not managed:
            return ""
        lines: list[str] = []
        for name in sorted(managed.keys()):
            agent = managed[name]
            desc = (getattr(agent, "description", None) or "").strip()
            # Tool names: agent.tools is a dict[name, Tool]. Skip
            # final_answer_tool — it's ubiquitous and adds no signal.
            tools = getattr(agent, "tools", None) or {}
            tool_names = sorted(t for t in tools.keys() if t != "final_answer_tool")
            # Keep the block compact (<=500 chars total for 3 agents is
            # the budget in the plan): single-line description, tool
            # list on the next indented line.
            desc_line = f"* {name} — {desc[:160]}" if desc else f"* {name}"
            lines.append(desc_line)
            if tool_names:
                lines.append(f"  tools: [{', '.join(tool_names)}]")
        return "\n".join(lines)

    async def _run_review_agent(
        self,
        ctx: DelegationContext,
        *,
        prior_attempts: Optional[list[PriorAttempt]] = None,
        task_blocklist_directive: str = "",
        chain_capped_directive: str = "",
    ) -> ReviewResult:
        """
        Invoke the sealed ReviewAgent and parse its final_answer into a
        ReviewResult.

        Extra kwargs (all optional; default to empty/None so current behaviour
        is unchanged when Layer 2 chain ledger is not yet wiring them):

        - `prior_attempts`: earlier attempts on the same chain; rendered into
          the task text under a "PRIOR ATTEMPTS" block so the reviewer can
          tell "already tried and failed that" from "fresh delegation".
        - `task_blocklist_directive`: rendered verbatim (when non-empty) as
          an explicit "retry unavailable — agent X previously failed with
          root_cause=Y" note at the top of the task text.
        - `chain_capped_directive`: same, for the "this specific delegation
          lineage is capped" case.

        Failures are swallowed: we return a fallback ProceedSpec rather than
        raising, because a broken reviewer must not break the planner's run.
        """
        agent = self._get_or_build_review_agent()
        task_text = self._format_context_for_review(
            ctx,
            original_user_task=self._original_user_task,
            prior_attempts=prior_attempts,
            task_blocklist_directive=task_blocklist_directive,
            chain_capped_directive=chain_capped_directive,
        )

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
    def _format_context_for_review(
        ctx: DelegationContext,
        *,
        original_user_task: str = "",
        prior_attempts: Optional[list[PriorAttempt]] = None,
        task_blocklist_directive: str = "",
        chain_capped_directive: str = "",
    ) -> str:
        """
        Render a DelegationContext (plus optional task-wide signals) into the
        task text for the ReviewAgent.

        Sections are ordered so the highest-salience items come first — the
        reviewer is bounded to 3 steps and may not scroll back:

            1. Cap directives (task blocklist, chain capped) — tells reviewer
               up front which next_action types are unavailable.
            2. Original user task — what the outer GAIA question is.
            3. Prior attempts — compact log of earlier attempts on this chain.
            4. Core delegation fields (agent_name, task_given, ...).

        Any optional section that is empty is omitted entirely; the rendering
        must stay identical to the pre-Layer-3 shape when all optional
        arguments are empty/None (so commit 2 is behaviour-preserving — the
        chain ledger in commit 3 is what actually populates these fields).
        """
        parts: list[str] = []

        if task_blocklist_directive:
            parts.append(task_blocklist_directive.strip())
        if chain_capped_directive:
            parts.append(chain_capped_directive.strip())

        if original_user_task:
            # Keep the outer task bounded so it can't dominate the reviewer's
            # context. 1500 chars is roughly 300-400 tokens, enough to convey
            # intent without crowding out prior_attempts / actual_response.
            trimmed = original_user_task.strip()
            if len(trimmed) > 1500:
                trimmed = trimmed[:1500].rstrip() + " …"
            parts.append(f"ORIGINAL USER TASK:\n{trimmed}")

        if prior_attempts:
            lines = ["PRIOR ATTEMPTS (same delegation lineage, oldest → newest):"]
            for pa in prior_attempts:
                rc = pa.root_cause if pa.root_cause is not None else "—"
                lines.append(
                    f"  #{pa.attempt_idx}  verdict={pa.verdict}  root_cause={rc}\n"
                    f"    revised_task_digest: {pa.revised_task_digest!r}"
                )
            parts.append("\n".join(lines))

        parts.append(
            f"agent_name: {ctx.agent_name}\n"
            f"task_given:\n{ctx.task_given}\n\n"
            f"expected_outcome (from planner reasoning):\n{ctx.expected_outcome}\n\n"
            f"actual_response:\n{ctx.actual_response}\n\n"
            f"(planner step_number: {ctx.step_number})"
        )

        return "\n\n".join(parts)

    @classmethod
    def _digest_task(cls, task_given: str, cap: Optional[int] = None) -> str:
        """
        Condense a task_given string into a fixed-max digest for inclusion in
        the prior_attempts block. Collapses whitespace and truncates.
        """
        effective_cap = cap if cap is not None else cls.PRIOR_ATTEMPT_DIGEST_MAX
        if not task_given:
            return ""
        collapsed = " ".join(task_given.split())
        if len(collapsed) > effective_cap:
            collapsed = collapsed[: effective_cap - 1].rstrip() + "…"
        return collapsed

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
