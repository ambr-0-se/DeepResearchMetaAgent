"""
Tests for the `review_metrics` counters and their end-to-end flow.

These complement the per-dispatch tests in test_review_retry_cap_coercion
and test_review_a_to_b_to_a by verifying:

- Every next_action type bumps the right counter (exactly one per call).
- The metric dict schema matches `_METRIC_KEYS` at all times.
- `max_chain_length` tracks the high-water mark across many chains.
"""
import sys
from pathlib import Path

from unittest.mock import MagicMock

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.meta.review_schema import (  # noqa: E402
    EscalateSpec, ModifyAgentSpec, ProceedSpec, RetrySpec, ReviewResult,
    RootCauseCategory,
)
from src.meta.review_step import (  # noqa: E402
    ChainState, DelegationContext, ReviewStep, _METRIC_KEYS,
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


def _ctx(agent_name="A"):
    return DelegationContext(
        agent_name=agent_name, task_given="t",
        expected_outcome="", actual_response="",
        step_number=3,
    )


def _anchor_and_chain(rs, agent):
    a, _ = rs._resolve_anchor(agent)
    chain = ChainState(anchor=a)
    rs._chains[(agent, a)] = chain
    return a, chain


def _retry(agent, rc=RootCauseCategory.INSUFFICIENT_INSTRUCTION):
    return ReviewResult(
        verdict="unsatisfactory", confidence=0.7, summary="x",
        root_cause_primary=rc,
        next_action=RetrySpec(
            agent_name=agent, revised_task="v",
            additional_guidance="", avoid_patterns=[],
        ),
    )


def _proceed():
    return ReviewResult(
        verdict="satisfactory", confidence=1.0, summary="ok",
        next_action=ProceedSpec(),
    )


def _modify():
    return ReviewResult(
        verdict="unsatisfactory", confidence=0.8, summary="add",
        root_cause_primary=RootCauseCategory.MISSING_TOOL,
        next_action=ModifyAgentSpec(
            modify_action="add_existing_tool_to_agent",
            agent_name="A", specification="python_interpreter_tool",
            followup_retry=True,
        ),
    )


def _escalate():
    return ReviewResult(
        verdict="unsatisfactory", confidence=0.8, summary="switch",
        root_cause_primary=RootCauseCategory.EXTERNAL_FAILURE,
        next_action=EscalateSpec(
            from_agent="A", to_agent="B", reason="r", task="t",
        ),
    )


class TestPerActionCounters:
    def test_retry_pass_through_does_not_bump_special_counters(self):
        rs = _rs()
        ctx = _ctx()
        a, chain = _anchor_and_chain(rs, "A")
        rs._dispatch_review_result(_retry("A"), ctx, ("A", a), chain)
        # Pass-through retry: no coercion, no escalate, no modify
        assert rs._metrics["retry_coercions_to_proceed"] == 0
        assert rs._metrics["blocklist_coercions"] == 0
        assert rs._metrics["escalate_emitted"] == 0
        assert rs._metrics["modify_agent_emitted"] == 0
        assert rs._metrics["proceed_emitted"] == 0
        assert rs._metrics["max_chain_length"] == 1

    def test_proceed_bumps_proceed_emitted(self):
        rs = _rs()
        ctx = _ctx()
        a, chain = _anchor_and_chain(rs, "A")
        rs._dispatch_review_result(_proceed(), ctx, ("A", a), chain)
        assert rs._metrics["proceed_emitted"] == 1

    def test_modify_bumps_modify_emitted(self):
        rs = _rs()
        ctx = _ctx()
        a, chain = _anchor_and_chain(rs, "A")
        rs._dispatch_review_result(_modify(), ctx, ("A", a), chain)
        assert rs._metrics["modify_agent_emitted"] == 1

    def test_escalate_bumps_escalate_emitted(self):
        rs = _rs()
        ctx = _ctx()
        a, chain = _anchor_and_chain(rs, "A")
        rs._dispatch_review_result(_escalate(), ctx, ("A", a), chain)
        assert rs._metrics["escalate_emitted"] == 1


class TestMaxChainLength:
    def test_tracks_high_water_mark_across_distinct_chains(self):
        """Two independent chains; max_chain_length reflects the longest."""
        rs = _rs()
        # Chain 1 on A: 2 retries (reaches cap, gets marked capped)
        a1, chain1 = _anchor_and_chain(rs, "A")
        ctx_a = _ctx("A")
        for _ in range(2):
            rs._dispatch_review_result(_retry("A"), ctx_a, ("A", a1), chain1)
        # Chain 2 on B: 1 retry
        b1, chain2 = _anchor_and_chain(rs, "B")
        ctx_b = _ctx("B")
        rs._dispatch_review_result(_retry("B"), ctx_b, ("B", b1), chain2)
        assert rs._metrics["max_chain_length"] == 2


class TestMetricSchema:
    def test_only_documented_keys_present(self):
        rs = _rs()
        assert set(rs._metrics.keys()) == set(_METRIC_KEYS)

    def test_all_values_integer(self):
        rs = _rs()
        ctx = _ctx()
        a, chain = _anchor_and_chain(rs, "A")
        rs._dispatch_review_result(_retry("A"), ctx, ("A", a), chain)
        rs._dispatch_review_result(_proceed(), ctx, ("A", a), chain)
        rs._dispatch_review_result(_modify(), ctx, ("A", a), chain)
        for k, v in rs._metrics.items():
            assert isinstance(v, int), f"metric {k} is {type(v).__name__}, expected int"

    def test_reset_returns_to_zero(self):
        rs = _rs()
        ctx = _ctx()
        a, chain = _anchor_and_chain(rs, "A")
        rs._dispatch_review_result(_retry("A"), ctx, ("A", a), chain)
        rs.on_task_start("next task")
        assert all(v == 0 for v in rs._metrics.values())
