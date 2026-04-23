"""
Test the P1/P4 planner-initiated re-entry pathology fix.

Scenario: after an (agent, root_cause) pair hits a cap=0 coercion and gets
blocklisted, any future delegation to the same agent with the same cause —
regardless of whether the chain is review-driven (pending_retry_anchor) or
planner-initiated (fresh anchor) — must be coerced to proceed via the
blocklist branch.

This is the defence-in-depth layer above the per-anchor cap. Without it,
a planner that re-issues a new delegation to a blocklisted agent would
mint a fresh chain (new anchor, count=0), bypass the capped_anchors check,
and trigger another cap=0 coercion repeatedly at ~1 LLM call/iteration
until the per-Q wall clock.
"""
import sys
from pathlib import Path

from unittest.mock import MagicMock

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.meta.review_schema import (  # noqa: E402
    ProceedSpec, RetrySpec, ReviewResult, RootCauseCategory,
)
from src.meta.review_step import (  # noqa: E402
    ChainState, DelegationContext, ReviewStep,
)


def _fake_parent():
    p = MagicMock()
    p.managed_agents = {}
    p.max_steps = 15
    return p


def _rs():
    rs = ReviewStep(_fake_parent())
    rs.on_task_start("task X")
    return rs


def _ctx(agent_name="browser_use_agent"):
    return DelegationContext(
        agent_name=agent_name,
        task_given="find X",
        expected_outcome="",
        actual_response="paywall",
        step_number=3,
    )


def _retry(agent_name, rc):
    return ReviewResult(
        verdict="unsatisfactory",
        confidence=0.7,
        summary="try again",
        root_cause_primary=rc,
        next_action=RetrySpec(
            agent_name=agent_name,
            revised_task="v",
            additional_guidance="x",
            avoid_patterns=[],
        ),
    )


class TestPlannerReentryBlocklist:
    def test_blocklisted_external_blocks_planner_reentry(self):
        """The canonical re-entry pathology: external coerce → blocklist →
        planner fresh delegation → reviewer suggests retry(external) →
        coerced via blocklist (not anchor) branch."""
        rs = _rs()
        ctx = _ctx()
        # Stage: prior attempt already hit cap=0 and blocklisted
        rs._task_blocklist.add((ctx.agent_name, "external"))

        # Planner independently re-delegates (no pending retry flag)
        a, is_new = rs._resolve_anchor(ctx.agent_name)
        assert is_new is True, "should be a fresh chain"
        rs._chains[(ctx.agent_name, a)] = ChainState(anchor=a)
        chain = rs._chains[(ctx.agent_name, a)]

        # Reviewer again returns RetrySpec on external
        r = _retry(ctx.agent_name, RootCauseCategory.EXTERNAL_FAILURE)
        out = rs._dispatch_review_result(r, ctx, (ctx.agent_name, a), chain)

        # Coerced
        assert isinstance(out.next_action, ProceedSpec)
        # Via the blocklist branch, NOT the anchor branch
        assert rs._metrics["blocklist_coercions"] == 1
        # retry_coercions_to_proceed NOT incremented (that's the cap=0
        # path for a first-time encounter)
        assert rs._metrics["retry_coercions_to_proceed"] == 0

    def test_blocklisted_for_one_cause_does_not_block_different_cause(self):
        """A block on (browser, external) must NOT block (browser,
        bad_instruction) — different causes have different blocklist
        entries."""
        rs = _rs()
        ctx = _ctx()
        rs._task_blocklist.add((ctx.agent_name, "external"))

        a, _ = rs._resolve_anchor(ctx.agent_name)
        rs._chains[(ctx.agent_name, a)] = ChainState(anchor=a)
        chain = rs._chains[(ctx.agent_name, a)]

        # bad_instruction has cap=2, so first retry should pass through
        r = _retry(ctx.agent_name, RootCauseCategory.INSUFFICIENT_INSTRUCTION)
        out = rs._dispatch_review_result(r, ctx, (ctx.agent_name, a), chain)
        assert isinstance(out.next_action, RetrySpec)
        assert rs._metrics["blocklist_coercions"] == 0
        assert chain.count == 1

    def test_blocklist_persists_across_chain_pivots(self):
        """Planner: A(external, blocked) → B → A(external, again).
        Second A delegation must still hit the blocklist."""
        rs = _rs()
        rs._task_blocklist.add(("A", "external"))

        # Delegation to B (unrelated)
        b_anchor, _ = rs._resolve_anchor("B")

        # Pivot back to A — fresh anchor because no pending retry flag
        a_anchor, is_new = rs._resolve_anchor("A")
        assert is_new is True
        rs._chains[("A", a_anchor)] = ChainState(anchor=a_anchor)
        chain = rs._chains[("A", a_anchor)]

        ctx_a = _ctx("A")
        r = _retry("A", RootCauseCategory.EXTERNAL_FAILURE)
        out = rs._dispatch_review_result(r, ctx_a, ("A", a_anchor), chain)
        assert isinstance(out.next_action, ProceedSpec)
        assert rs._metrics["blocklist_coercions"] == 1

    def test_blocklist_cleared_on_task_start(self):
        rs = _rs()
        rs._task_blocklist.add(("A", "external"))
        rs.on_task_start("new task")
        assert rs._task_blocklist == set()
