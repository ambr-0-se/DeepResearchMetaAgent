"""
Tests for asymmetric per-root-cause retry caps + coercion in ReviewStep.

Covers every root cause in the taxonomy and asserts the correct cap +
coercion behaviour.
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
    ChainState, DelegationContext, ReviewStep,
    RETRY_CAP_BY_ROOT_CAUSE,
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
        actual_response="no",
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


def _setup_chain(rs, ctx):
    a, _ = rs._resolve_anchor(ctx.agent_name)
    chain = ChainState(anchor=a)
    rs._chains[(ctx.agent_name, a)] = chain
    rs._metrics["retry_chains_started"] += 1
    return a, chain


# --- cap table sanity -------------------------------------------------------

class TestCapTable:
    def test_all_taxonomy_members_have_cap(self):
        for rc in RootCauseCategory:
            assert rc in RETRY_CAP_BY_ROOT_CAUSE, f"missing cap for {rc}"

    def test_cap_values_match_plan(self):
        assert RETRY_CAP_BY_ROOT_CAUSE[RootCauseCategory.INSUFFICIENT_INSTRUCTION] == 2
        assert RETRY_CAP_BY_ROOT_CAUSE[RootCauseCategory.TASK_MISUNDERSTANDING] == 2
        assert RETRY_CAP_BY_ROOT_CAUSE[RootCauseCategory.UNCLEAR_OBJECTIVE] == 2
        assert RETRY_CAP_BY_ROOT_CAUSE[RootCauseCategory.INCOMPLETE_OUTPUT] == 1
        assert RETRY_CAP_BY_ROOT_CAUSE[RootCauseCategory.EXTERNAL_FAILURE] == 0
        assert RETRY_CAP_BY_ROOT_CAUSE[RootCauseCategory.MODEL_LIMITATION] == 0
        assert RETRY_CAP_BY_ROOT_CAUSE[RootCauseCategory.MISSING_TOOL] == 0
        assert RETRY_CAP_BY_ROOT_CAUSE[RootCauseCategory.WRONG_TOOL_SELECTION] == 0


# --- cap=0 immediate coercion -----------------------------------------------

@pytest.mark.parametrize(
    "rc",
    [
        RootCauseCategory.EXTERNAL_FAILURE,
        RootCauseCategory.MODEL_LIMITATION,
        RootCauseCategory.MISSING_TOOL,
        RootCauseCategory.WRONG_TOOL_SELECTION,
    ],
    ids=["external", "model_limit", "missing_tool", "wrong_tool"],
)
class TestCap0Causes:
    def test_first_retry_immediately_coerced(self, rc):
        rs = _rs()
        ctx = _ctx()
        a, chain = _setup_chain(rs, ctx)
        r = _retry(ctx.agent_name, rc)
        out = rs._dispatch_review_result(r, ctx, (ctx.agent_name, a), chain)
        assert isinstance(out.next_action, ProceedSpec), (
            f"cap=0 cause {rc.value} should coerce to ProceedSpec"
        )

    def test_coercion_adds_to_task_blocklist(self, rc):
        rs = _rs()
        ctx = _ctx()
        a, chain = _setup_chain(rs, ctx)
        r = _retry(ctx.agent_name, rc)
        rs._dispatch_review_result(r, ctx, (ctx.agent_name, a), chain)
        assert (ctx.agent_name, rc.value) in rs._task_blocklist

    def test_coercion_marks_chain_capped(self, rc):
        rs = _rs()
        ctx = _ctx()
        a, chain = _setup_chain(rs, ctx)
        r = _retry(ctx.agent_name, rc)
        rs._dispatch_review_result(r, ctx, (ctx.agent_name, a), chain)
        assert chain.capped is True
        assert (ctx.agent_name, a) in rs._capped_anchors

    def test_coercion_bumps_metric(self, rc):
        rs = _rs()
        ctx = _ctx()
        a, chain = _setup_chain(rs, ctx)
        r = _retry(ctx.agent_name, rc)
        rs._dispatch_review_result(r, ctx, (ctx.agent_name, a), chain)
        assert rs._metrics["retry_coercions_to_proceed"] == 1


# --- cap=2 passes through, 3rd coerced --------------------------------------

@pytest.mark.parametrize(
    "rc",
    [
        RootCauseCategory.INSUFFICIENT_INSTRUCTION,
        RootCauseCategory.TASK_MISUNDERSTANDING,
        RootCauseCategory.UNCLEAR_OBJECTIVE,
    ],
    ids=["bad_instruction", "misread_task", "unclear_goal"],
)
class TestCap2Causes:
    def test_two_retries_allowed_third_coerced(self, rc):
        rs = _rs()
        ctx = _ctx()
        a, chain = _setup_chain(rs, ctx)
        # Attempt 1
        r1 = _retry(ctx.agent_name, rc)
        out1 = rs._dispatch_review_result(r1, ctx, (ctx.agent_name, a), chain)
        assert isinstance(out1.next_action, RetrySpec)
        # Attempt 2 (reaches cap)
        r2 = _retry(ctx.agent_name, rc)
        out2 = rs._dispatch_review_result(r2, ctx, (ctx.agent_name, a), chain)
        assert isinstance(out2.next_action, RetrySpec)
        # Attempt 3 (coerced via anchor branch)
        r3 = _retry(ctx.agent_name, rc)
        out3 = rs._dispatch_review_result(r3, ctx, (ctx.agent_name, a), chain)
        assert isinstance(out3.next_action, ProceedSpec)

    def test_chain_count_tracks_retries(self, rc):
        rs = _rs()
        ctx = _ctx()
        a, chain = _setup_chain(rs, ctx)
        for _ in range(2):
            r = _retry(ctx.agent_name, rc)
            rs._dispatch_review_result(r, ctx, (ctx.agent_name, a), chain)
        assert chain.count == 2
        assert rs._metrics["max_chain_length"] == 2
        assert rs._metrics["retry_chains_capped"] == 1


# --- cap=1 incomplete --------------------------------------------------------

class TestCap1Incomplete:
    def test_one_retry_allowed_second_coerced(self):
        rs = _rs()
        ctx = _ctx()
        a, chain = _setup_chain(rs, ctx)
        rc = RootCauseCategory.INCOMPLETE_OUTPUT
        r1 = _retry(ctx.agent_name, rc)
        out1 = rs._dispatch_review_result(r1, ctx, (ctx.agent_name, a), chain)
        assert isinstance(out1.next_action, RetrySpec)
        r2 = _retry(ctx.agent_name, rc)
        out2 = rs._dispatch_review_result(r2, ctx, (ctx.agent_name, a), chain)
        assert isinstance(out2.next_action, ProceedSpec)


# --- coercion preserves other fields ----------------------------------------

class TestCoercionPreservesContext:
    def test_coerced_result_retains_root_cause(self):
        rs = _rs()
        ctx = _ctx()
        a, chain = _setup_chain(rs, ctx)
        r = _retry(ctx.agent_name, RootCauseCategory.EXTERNAL_FAILURE)
        out = rs._dispatch_review_result(r, ctx, (ctx.agent_name, a), chain)
        assert out.root_cause_primary == RootCauseCategory.EXTERNAL_FAILURE
        assert out.verdict == "unsatisfactory"
        # Summary is rewritten to explain the coercion
        assert "Retry unavailable" in out.summary or "coerced to proceed" in out.summary
