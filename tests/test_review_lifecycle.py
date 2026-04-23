"""
Tests for per-task lifecycle + metric emission (commits 4 + 5).

Covers:
- on_task_start idempotence and state clearing
- Metric extraction path from `agent.review_step._metrics`
- C0/C2 guard: review_step is None → extraction returns None cleanly
- Simulation of post-timeout extraction (agent object survives after
  `asyncio.TimeoutError`, so _metrics is still accessible)
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
    ChainState, DelegationContext, ReviewStep, _METRIC_KEYS,
)


def _fake_parent():
    p = MagicMock()
    p.managed_agents = {}
    p.max_steps = 15
    return p


# --- Idempotence + rebuild --------------------------------------------------

class TestLifecycle:
    def test_on_task_start_is_idempotent(self):
        rs = ReviewStep(_fake_parent())
        rs.on_task_start("task A")
        snap_1 = dict(rs._metrics)
        rs.on_task_start("task B")
        snap_2 = dict(rs._metrics)
        assert snap_1 == snap_2 == {k: 0 for k in _METRIC_KEYS}
        assert rs._original_user_task == "task B"

    def test_review_agent_rebuilds_after_reset(self):
        rs = ReviewStep(_fake_parent())
        sentinel = object()
        rs._review_agent = sentinel
        rs.on_task_start("x")
        assert rs._review_agent is None

    def test_metric_schema_stable(self):
        rs = ReviewStep(_fake_parent())
        rs.on_task_start("x")
        assert set(rs._metrics.keys()) == set(_METRIC_KEYS)
        # All int, all zero
        assert all(isinstance(v, int) and v == 0 for v in rs._metrics.values())


# --- Metric extraction path (mimics run_gaia.py post-loop handler) ---------

class TestMetricExtraction:
    def _extract(self, agent):
        """Replicates the defensive getattr in examples/run_gaia.py:296."""
        try:
            rev = getattr(agent, "review_step", None) if agent is not None else None
            if rev is not None:
                return dict(rev._metrics)
        except Exception:
            return None
        return None

    def test_c0_c2_guard_returns_none(self):
        """Agents built under C0/C2 have review_step=None."""
        agent = MagicMock()
        agent.review_step = None
        assert self._extract(agent) is None

    def test_pre_agent_crash_returns_none(self):
        """If agent is None (e.g. create_agent raised), extraction is safe."""
        assert self._extract(None) is None

    def test_c3_c4_returns_dict_matching_metric_schema(self):
        agent = MagicMock()
        rs = ReviewStep(_fake_parent())
        rs.on_task_start("task")
        agent.review_step = rs
        extracted = self._extract(agent)
        assert isinstance(extracted, dict)
        assert set(extracted.keys()) == set(_METRIC_KEYS)

    def test_extraction_captures_accumulated_counts(self):
        """Populate some counts via a dispatch path; extraction returns
        the same numbers."""
        agent = MagicMock()
        rs = ReviewStep(_fake_parent())
        rs.on_task_start("task")
        agent.review_step = rs

        ctx = DelegationContext(
            agent_name="A", task_given="t",
            expected_outcome="", actual_response="",
            step_number=1,
        )
        a, _ = rs._resolve_anchor("A")
        rs._chains[("A", a)] = ChainState(anchor=a)
        rs._metrics["retry_chains_started"] = 1
        chain = rs._chains[("A", a)]

        # Emit a cap=0 external retry → coerces, bumps retry_coercions
        r = ReviewResult(
            verdict="unsatisfactory", confidence=0.7,
            summary="try", root_cause_primary=RootCauseCategory.EXTERNAL_FAILURE,
            next_action=RetrySpec(
                agent_name="A", revised_task="v",
                additional_guidance="", avoid_patterns=[],
            ),
        )
        rs._dispatch_review_result(r, ctx, ("A", a), chain)

        extracted = self._extract(agent)
        assert extracted["retry_chains_started"] == 1
        assert extracted["retry_coercions_to_proceed"] == 1
        assert extracted["proceed_emitted"] == 0  # coercion path bumps retry_coercions, not proceed_emitted

    def test_extraction_survives_timeout_scenario(self):
        """Simulates the P1 semantics: the agent's run() was cancelled and
        abandoned, but the agent object itself remains in scope (because
        run_gaia.py holds the reference via `agent = await create_agent(config)`).
        Metrics should still be extractable."""
        agent = MagicMock()
        rs = ReviewStep(_fake_parent())
        rs.on_task_start("task")
        # Pretend some chain activity happened before the timeout
        rs._metrics["retry_chains_started"] = 2
        rs._metrics["max_chain_length"] = 1
        agent.review_step = rs

        # Now simulate the P1 cancel/abandon: raise-and-catch at the caller
        # level doesn't invalidate the agent reference.
        import asyncio
        try:
            raise asyncio.TimeoutError("simulated cleanup timeout")
        except asyncio.TimeoutError:
            pass

        extracted = self._extract(agent)
        assert extracted["retry_chains_started"] == 2
        assert extracted["max_chain_length"] == 1

    def test_extraction_swallows_unexpected_errors(self):
        """A broken _metrics attribute must not break the row write."""
        class _Boom:
            @property
            def _metrics(self):
                raise RuntimeError("simulated breakage")
        agent = MagicMock()
        agent.review_step = _Boom()
        # Should return None rather than raise
        assert self._extract(agent) is None
