"""
Tests for ModifyAgentSpec interactions with the chain ledger.

Key invariants:
  - followup_retry=True: chain effectively terminates; next delegation to
    the same agent with a fresh anchor is a new chain (count=0).
  - followup_retry=False: same — both clear the pending flag; differ only
    at the caller (planner) in whether a follow-up delegation is dispatched.
  - Modify does NOT clear the task blocklist. A prior cap=0 coercion on
    (agent, root_cause) remains blocklisted even after modify — modify
    doesn't prove the root cause is resolved.
  - modify_agent_emitted metric bumped.
"""
import sys
from pathlib import Path

from unittest.mock import MagicMock

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.meta.review_schema import (  # noqa: E402
    ModifyAgentSpec, ProceedSpec, RetrySpec, ReviewResult, RootCauseCategory,
)
from src.meta.review_step import (  # noqa: E402
    ChainState, DelegationContext, ReviewStep,
)


def _fake_parent():
    p = MagicMock()
    p.managed_agents = {"A": object()}
    p.max_steps = 15
    return p


def _rs():
    rs = ReviewStep(_fake_parent())
    rs.on_task_start("task X")
    return rs


def _ctx(agent_name="A"):
    return DelegationContext(
        agent_name=agent_name, task_given="t",
        expected_outcome="", actual_response="",
        step_number=3,
    )


def _modify(agent_name="A", followup_retry=True,
            modify_action="add_existing_tool_to_agent",
            specification="python_interpreter_tool"):
    return ReviewResult(
        verdict="unsatisfactory", confidence=0.8,
        summary="add tool", root_cause_primary=RootCauseCategory.MISSING_TOOL,
        next_action=ModifyAgentSpec(
            modify_action=modify_action,
            agent_name=agent_name,
            specification=specification,
            followup_retry=followup_retry,
        ),
    )


class TestModifyDispatch:
    def test_modify_clears_pending_retry_flag(self):
        rs = _rs()
        ctx = _ctx("A")
        a, _ = rs._resolve_anchor("A")
        rs._chains[("A", a)] = ChainState(anchor=a)
        chain = rs._chains[("A", a)]
        # Plant a pending retry anchor that modify should clear
        rs._pending_retry_anchor["A"] = a

        rs._dispatch_review_result(_modify("A"), ctx, ("A", a), chain)
        assert "A" not in rs._pending_retry_anchor

    def test_modify_bumps_metric(self):
        rs = _rs()
        ctx = _ctx("A")
        a, _ = rs._resolve_anchor("A")
        rs._chains[("A", a)] = ChainState(anchor=a)
        chain = rs._chains[("A", a)]
        rs._dispatch_review_result(_modify("A"), ctx, ("A", a), chain)
        assert rs._metrics["modify_agent_emitted"] == 1

    def test_modify_does_not_populate_task_blocklist(self):
        """Modify is not a coercion; the prior missing_tool / wrong_tool
        root cause that triggered modify is, by itself, not a reason to
        blocklist (add_*_tool might resolve it). Only cap=0 coercion
        and escalate populate the blocklist."""
        rs = _rs()
        ctx = _ctx("A")
        a, _ = rs._resolve_anchor("A")
        rs._chains[("A", a)] = ChainState(anchor=a)
        chain = rs._chains[("A", a)]
        rs._dispatch_review_result(_modify("A"), ctx, ("A", a), chain)
        assert rs._task_blocklist == set()

    def test_modify_preserves_prior_blocklist(self):
        """If the blocklist had entries from earlier in the task, modify
        must NOT clear them — modify doesn't prove the prior cause is
        resolved."""
        rs = _rs()
        rs._task_blocklist.add(("A", "external"))
        ctx = _ctx("A")
        a, _ = rs._resolve_anchor("A")
        rs._chains[("A", a)] = ChainState(anchor=a)
        chain = rs._chains[("A", a)]
        rs._dispatch_review_result(_modify("A"), ctx, ("A", a), chain)
        assert ("A", "external") in rs._task_blocklist

    def test_next_delegation_after_modify_is_fresh_chain(self):
        """Post-modify, the next delegation to A without a pending retry
        flag should mint a new chain with count=0."""
        rs = _rs()
        ctx = _ctx("A")
        a1, _ = rs._resolve_anchor("A")
        rs._chains[("A", a1)] = ChainState(anchor=a1, count=1)
        chain = rs._chains[("A", a1)]
        rs._dispatch_review_result(_modify("A"), ctx, ("A", a1), chain)

        # Next delegation (followup_retry would dispatch, but that's the
        # planner's job — here we simulate the resolver call)
        a2, is_new = rs._resolve_anchor("A")
        assert is_new is True, (
            "modify should terminate the chain; next delegation "
            "must be fresh (no pending retry flag)"
        )
        assert a2 != a1

    def test_modify_with_followup_retry_false_same_behaviour_at_ledger(self):
        rs = _rs()
        ctx = _ctx("A")
        a, _ = rs._resolve_anchor("A")
        rs._chains[("A", a)] = ChainState(anchor=a)
        chain = rs._chains[("A", a)]
        rs._dispatch_review_result(
            _modify("A", followup_retry=False), ctx, ("A", a), chain,
        )
        # Identical ledger effect to followup_retry=True: pending cleared,
        # metric bumped. (The difference is purely at the planner — whether
        # it re-delegates.)
        assert "A" not in rs._pending_retry_anchor
        assert rs._metrics["modify_agent_emitted"] == 1
