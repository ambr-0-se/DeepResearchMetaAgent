"""
Test reviewer-driven A→B→A alternation cannot loop infinitely.

Two branches covered:
  1. Reviewer-driven continuation via `_pending_retry_anchor[A]` targeting
     an already-capped chain → coerced via `_capped_anchors` branch.
  2. `EscalateSpec` correctly transitions state: from_agent's chain is
     marked capped, (from_agent, root_cause) joins the task blocklist,
     and the pending retry flag for from_agent is cleared.
"""
import sys
from pathlib import Path

from unittest.mock import MagicMock

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.meta.review_schema import (  # noqa: E402
    EscalateSpec, ProceedSpec, RetrySpec, ReviewResult, RootCauseCategory,
)
from src.meta.review_step import (  # noqa: E402
    ChainState, DelegationContext, ReviewStep,
)


def _fake_parent():
    p = MagicMock()
    p.managed_agents = {"A": object(), "B": object()}
    p.max_steps = 15
    return p


def _rs():
    rs = ReviewStep(_fake_parent())
    rs.on_task_start("task X")
    return rs


def _ctx(agent_name):
    return DelegationContext(
        agent_name=agent_name,
        task_given="find X",
        expected_outcome="",
        actual_response="no",
        step_number=3,
    )


class TestEscalateTransition:
    def test_escalate_caps_from_agent_chain(self):
        rs = _rs()
        ctx_a = _ctx("A")
        a_anchor, _ = rs._resolve_anchor("A")
        rs._chains[("A", a_anchor)] = ChainState(anchor=a_anchor)
        chain = rs._chains[("A", a_anchor)]

        result = ReviewResult(
            verdict="unsatisfactory",
            confidence=0.8,
            summary="switch",
            root_cause_primary=RootCauseCategory.EXTERNAL_FAILURE,
            next_action=EscalateSpec(
                from_agent="A", to_agent="B",
                reason="better", task="try this",
            ),
        )
        out = rs._dispatch_review_result(result, ctx_a, ("A", a_anchor), chain)
        assert isinstance(out.next_action, EscalateSpec)
        # from_agent's chain marked capped
        assert chain.capped is True
        assert ("A", a_anchor) in rs._capped_anchors
        # (from_agent, root_cause) added to task blocklist
        assert ("A", "external") in rs._task_blocklist
        # Pending retry flag for from_agent cleared (if any was set)
        assert "A" not in rs._pending_retry_anchor
        # Metric
        assert rs._metrics["escalate_emitted"] == 1


class TestAToBToA:
    def test_reviewer_retry_a_after_escalate_is_coerced(self):
        """A capped → escalate to B → reviewer (somehow) tries RetrySpec(A)
        via pending_retry_anchor → coerced via _capped_anchors."""
        rs = _rs()

        # Step 1: A hit cap (simulate directly)
        a1_anchor, _ = rs._resolve_anchor("A")
        chain_a = ChainState(
            anchor=a1_anchor, count=2, capped=True,
            last_root_cause="bad_instruction",
        )
        rs._chains[("A", a1_anchor)] = chain_a
        rs._capped_anchors.add(("A", a1_anchor))

        # Step 2: escalate to B (mock directly — no dispatch needed for this branch)
        # The pending_retry_anchor for A should be pre-set to simulate a
        # (buggy) reviewer that tries to continue A's chain.
        rs._pending_retry_anchor["A"] = a1_anchor

        # Step 3: Planner dispatches again to A (review-driven via pending flag)
        a2_anchor, is_new = rs._resolve_anchor("A")
        assert is_new is False
        assert a2_anchor == a1_anchor, "continuation should reuse capped anchor"

        # Step 4: Reviewer emits another RetrySpec; dispatch should coerce
        ctx_a = _ctx("A")
        r = ReviewResult(
            verdict="unsatisfactory", confidence=0.6,
            summary="try again",
            root_cause_primary=RootCauseCategory.INSUFFICIENT_INSTRUCTION,
            next_action=RetrySpec(
                agent_name="A", revised_task="v",
                additional_guidance="", avoid_patterns=[],
            ),
        )
        out = rs._dispatch_review_result(r, ctx_a, ("A", a2_anchor), chain_a)
        assert isinstance(out.next_action, ProceedSpec)
        assert rs._metrics["blocklist_coercions"] == 1

    def test_planner_independent_reentry_after_a_capped_mints_fresh_chain(self):
        """Planner: A(capped) → B → A (no pending flag, different intent).
        Fresh chain allowed — but blocklist still applies if root_cause
        matches a cap=0 cause. See test_review_planner_reentry_blocklist
        for the blocklist-blocking path; this test covers the NON-blocked
        fresh chain path."""
        rs = _rs()

        # A hits cap=2 for bad_instruction (NOT cap=0, so NOT blocklisted)
        a1, _ = rs._resolve_anchor("A")
        rs._chains[("A", a1)] = ChainState(
            anchor=a1, count=2, capped=True,
            last_root_cause="bad_instruction",
        )
        rs._capped_anchors.add(("A", a1))
        # No task_blocklist entry (cap=2 doesn't populate it until coercion)

        # Planner delegates to B
        b, _ = rs._resolve_anchor("B")
        # Planner independently delegates back to A (no pending flag)
        a2, is_new = rs._resolve_anchor("A")
        assert is_new is True
        assert a2 != a1, "new chain should have new anchor"

        # The new chain starts fresh — count=0; retry allowed
        rs._chains[("A", a2)] = ChainState(anchor=a2)
        chain2 = rs._chains[("A", a2)]
        ctx_a = _ctx("A")
        r = ReviewResult(
            verdict="unsatisfactory", confidence=0.6,
            summary="fresh try",
            root_cause_primary=RootCauseCategory.INSUFFICIENT_INSTRUCTION,
            next_action=RetrySpec(
                agent_name="A", revised_task="v", additional_guidance="",
                avoid_patterns=[],
            ),
        )
        out = rs._dispatch_review_result(r, ctx_a, ("A", a2), chain2)
        assert isinstance(out.next_action, RetrySpec)
        assert chain2.count == 1
