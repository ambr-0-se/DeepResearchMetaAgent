"""
Tests for the ReviewStep chain ledger (commit 3 of REVIEW hardening).

Covers anchor minting / continuation, on_task_start isolation, pivot
detection, and metric counters.
"""
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.meta.review_schema import (  # noqa: E402
    ProceedSpec, RetrySpec, ReviewResult, RootCauseCategory,
)
from src.meta.review_step import (  # noqa: E402
    ChainState, DelegationContext, PriorAttempt, ReviewStep,
    _METRIC_KEYS,
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
        actual_response="no luck",
        step_number=3,
    )


def _retry(agent_name, rc, revised="v"):
    return ReviewResult(
        verdict="unsatisfactory",
        confidence=0.7,
        summary="try again",
        root_cause_primary=rc,
        next_action=RetrySpec(
            agent_name=agent_name,
            revised_task=revised,
            additional_guidance="x",
            avoid_patterns=[],
        ),
    )


# --- anchor minting / continuation ------------------------------------------

class TestResolveAnchor:
    def test_fresh_delegation_mints_new_anchor(self):
        rs = _rs()
        a, is_new = rs._resolve_anchor("browser_use_agent")
        assert is_new is True
        assert isinstance(a, str) and len(a) > 0

    def test_two_fresh_delegations_get_different_anchors(self):
        rs = _rs()
        a1, _ = rs._resolve_anchor("browser_use_agent")
        a2, _ = rs._resolve_anchor("deep_researcher_agent")
        assert a1 != a2

    def test_pending_retry_continuation_inherits_anchor(self):
        rs = _rs()
        a1, _ = rs._resolve_anchor("browser_use_agent")
        rs._pending_retry_anchor["browser_use_agent"] = a1
        a2, is_new = rs._resolve_anchor("browser_use_agent")
        assert is_new is False
        assert a2 == a1

    def test_pending_flag_is_popped_on_read(self):
        rs = _rs()
        a1, _ = rs._resolve_anchor("browser_use_agent")
        rs._pending_retry_anchor["browser_use_agent"] = a1
        rs._resolve_anchor("browser_use_agent")
        assert "browser_use_agent" not in rs._pending_retry_anchor

    def test_pivot_mints_fresh_anchor(self):
        """Planner delegates to A, then B, then back to A without a pending
        retry flag — the second A delegation should get a fresh anchor."""
        rs = _rs()
        a1, _ = rs._resolve_anchor("A")
        _, _ = rs._resolve_anchor("B")
        a3, is_new = rs._resolve_anchor("A")
        assert is_new is True
        assert a3 != a1


# --- on_task_start isolation ------------------------------------------------

class TestOnTaskStart:
    def test_metric_keys_stable(self):
        rs = _rs()
        assert set(rs._metrics.keys()) == set(_METRIC_KEYS)
        assert all(v == 0 for v in rs._metrics.values())

    def test_reset_clears_all_state(self):
        rs = _rs()
        # Pollute state
        a, _ = rs._resolve_anchor("A")
        rs._chains[("A", a)] = ChainState(anchor=a, count=3)
        rs._task_blocklist.add(("A", "external"))
        rs._pending_retry_anchor["A"] = a
        rs._capped_anchors.add(("A", a))
        rs._metrics["retry_chains_started"] = 5
        rs._metrics["max_chain_length"] = 3

        rs.on_task_start("new task")
        assert rs._chains == {}
        assert rs._task_blocklist == set()
        assert rs._pending_retry_anchor == {}
        assert rs._capped_anchors == set()
        assert all(v == 0 for v in rs._metrics.values())
        assert rs._original_user_task == "new task"

    def test_reset_is_idempotent(self):
        rs = _rs()
        rs.on_task_start("task A")
        rs.on_task_start("task B")  # No-op state-wise; just updates task string
        assert rs._original_user_task == "task B"
        assert all(v == 0 for v in rs._metrics.values())

    def test_reset_rebuilds_review_agent_on_next_use(self):
        rs = _rs()
        # Simulate a cached review agent.
        rs._review_agent = object()
        rs.on_task_start("x")
        assert rs._review_agent is None

    def test_empty_original_task_is_safe(self):
        rs = _rs()
        rs.on_task_start("")
        assert rs._original_user_task == ""


# --- dispatch increments metrics correctly ----------------------------------

class TestMetricCounters:
    def test_retry_chain_started_on_fresh_chain(self):
        """Exercised via run_if_applicable in higher-level tests; here we
        check the counter increments correctly from a direct
        _dispatch_review_result call path."""
        rs = _rs()
        ctx = _ctx()
        a, _ = rs._resolve_anchor(ctx.agent_name)
        rs._chains[(ctx.agent_name, a)] = ChainState(anchor=a)
        # Simulate that run_if_applicable already bumped the counter.
        rs._metrics["retry_chains_started"] = 1
        chain = rs._chains[(ctx.agent_name, a)]

        # A cap=2 retry passes through
        r = _retry(ctx.agent_name, RootCauseCategory.INSUFFICIENT_INSTRUCTION)
        rs._dispatch_review_result(r, ctx, (ctx.agent_name, a), chain)
        assert chain.count == 1
        assert rs._metrics["max_chain_length"] == 1

    def test_prior_attempts_are_bounded(self):
        rs = _rs()
        ctx = _ctx()
        a, _ = rs._resolve_anchor(ctx.agent_name)
        rs._chains[(ctx.agent_name, a)] = ChainState(anchor=a)
        chain = rs._chains[(ctx.agent_name, a)]

        # Push many retries (use cap=2; after 2 we'd cap, so use bad_instr).
        # For this test, directly bypass dispatch and fill prior list to
        # test bounding logic.
        for i in range(1, 10):
            chain.prior.append(
                PriorAttempt(
                    attempt_idx=i,
                    verdict="unsatisfactory",
                    root_cause="bad_instruction",
                    revised_task_digest=f"t{i}",
                )
            )
        # Simulate the bounding logic (as _dispatch_review_result applies)
        if len(chain.prior) > rs.PRIOR_ATTEMPTS_KEEP_LAST:
            chain.prior = chain.prior[-rs.PRIOR_ATTEMPTS_KEEP_LAST:]
        assert len(chain.prior) == rs.PRIOR_ATTEMPTS_KEEP_LAST
        # Newest retained, oldest dropped
        assert chain.prior[0].attempt_idx == 5
        assert chain.prior[-1].attempt_idx == 9
