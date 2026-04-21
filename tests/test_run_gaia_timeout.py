"""Unit tests for the hard wall-clock guard in `examples/run_gaia.py`
(§P1 of `docs/handoffs/HANDOFF_THROUGHPUT_REFACTOR.md`).

Motivating evidence: E2 task 023e9d44 ran 3298 s against a nominal 1800 s
`per_question_timeout_secs` because `asyncio.wait_for` alone does not
enforce a hard wall-clock cap — `finally:` cleanup in sub-tools (e.g.
`src/tools/auto_browser.py:152-160`) runs after cancellation before
control returns.

Design under test: `answer_single_question` now builds the agent call as
an explicit task, shields it, and on timeout cancels + waits at most
`CLEANUP_GRACE_SECS` for cleanup before abandoning.

Scale: these tests use `per_question_timeout_secs=1.0` and
monkeypatched `CLEANUP_GRACE_SECS=2.0` to keep the whole suite well
under 10 s.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from examples import run_gaia


class _FakeMemoryStep:
    """Lightweight stand-in for a MemoryStep in `agent.memory.steps`."""

    def __init__(self, text: str = "stub-step") -> None:
        self.text = text
        self.model_input_messages = "placeholder"  # zeroed by answer_single_question

    def __str__(self) -> str:  # used by the parsing_error check
        return self.text


class _FakeMemory:
    def __init__(self, steps: list[_FakeMemoryStep] | None = None) -> None:
        self.steps = steps or []


class _FakeAgent:
    """Stub `AsyncMultiStepAgent`-shaped object.

    `run` behaviour is parameterised by the test via `behaviour`:
      - "fast":             returns "done" immediately.
      - "cooperative":      sleeps but honours CancelledError.
      - "ignore_cancel":    swallows CancelledError in a tight loop — the
                            worst-case pathology that P1 must cap.

    `write_memory_to_messages` returns an empty list; `memory.steps` is
    set to a minimal shape so the happy-path branch after `run()`
    doesn't blow up.
    """

    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour
        self.memory = _FakeMemory([_FakeMemoryStep()])

    async def run(self, task: str) -> str:
        if self.behaviour == "fast":
            return "done"
        if self.behaviour == "cooperative":
            await asyncio.sleep(100)  # will honour cancel
            return "unreachable"
        if self.behaviour == "ignore_cancel":
            # Simulate a stuck sub-tool that ignores cancellation up to a
            # bounded budget. The production pathology is infinite-swallow;
            # we bound at 4.0 s so pytest-asyncio teardown can complete
            # (an immortal task would block the event-loop close). 4.0 s
            # is just past P1's expected upper bound for the assertion
            # (`per_question_timeout=1.0 + CLEANUP_GRACE_SECS=2.0 + slack`),
            # so the guard's `shield`-bounded wait still fires before the
            # task self-exits — we verify the upper wall-clock bound on
            # `answer_single_question`'s return, not task death.
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < 4.0:
                try:
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    pass
            return "eventually_done_out_of_test_scope"
        raise AssertionError(f"unknown behaviour: {self.behaviour}")

    async def write_memory_to_messages(self, summary_mode: bool = False) -> list:
        return []


def _install_fake_agent(monkeypatch, behaviour: str) -> None:
    """Replace `create_agent`, `prepare_response`, `model_manager`, and
    `logger.visualize_agent_tree` with stubs so `answer_single_question`
    runs without touching the real agent / model / logging stack.
    """

    async def _fake_create_agent(cfg):
        return _FakeAgent(behaviour)

    async def _fake_prepare_response(question, memory, reformulation_model):
        return "done"

    monkeypatch.setattr(run_gaia, "create_agent", _fake_create_agent)
    monkeypatch.setattr(run_gaia, "prepare_response", _fake_prepare_response)
    monkeypatch.setattr(
        run_gaia,
        "model_manager",
        SimpleNamespace(registed_models={"test-model": object()}),
    )
    # `visualize_agent_tree` inspects .model / .tools / .managed_agents,
    # none of which our minimal _FakeAgent exposes. Replace with a no-op
    # so the test focus stays on the timeout mechanics.
    monkeypatch.setattr(run_gaia.logger, "visualize_agent_tree", lambda agent: None)


def _read_jsonl_row(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        lines = [ln for ln in fp if ln.strip()]
    assert len(lines) == 1, f"expected exactly 1 jsonl row, got {len(lines)}"
    return json.loads(lines[0])


@pytest.mark.asyncio
async def test_hard_cap_on_cancel_ignoring_task(
    monkeypatch, fake_config, fake_example
):
    """The pathological case: agent.run swallows CancelledError. P1 must
    cap total wall at `per_question_timeout + CLEANUP_GRACE_SECS + slack`.
    """
    _install_fake_agent(monkeypatch, behaviour="ignore_cancel")
    monkeypatch.setattr(run_gaia, "CLEANUP_GRACE_SECS", 2.0)
    fake_config.per_question_timeout_secs = 1.0

    t0 = time.monotonic()
    await run_gaia.answer_single_question(fake_config, fake_example)
    elapsed = time.monotonic() - t0

    # Upper bound: 1.0 (timeout) + 2.0 (cleanup grace) + 1.0 (slack for
    # retry-loop overhead, create_agent stub, jsonl write). Lower bound:
    # at least the timeout itself.
    assert 1.0 <= elapsed <= 4.0, (
        f"wall time {elapsed:.2f}s outside expected 1.0-4.0s bounds; "
        "hard wall-clock guard is not capping correctly."
    )

    row = _read_jsonl_row(fake_config.save_path)
    assert row["agent_error"] == "Per-question timeout (1.0s) exceeded"
    assert row["iteration_limit_exceeded"] is True
    assert row["prediction"] is None
    assert row["task_id"] == fake_example["task_id"]


@pytest.mark.asyncio
async def test_happy_path_unchanged(monkeypatch, fake_config, fake_example):
    """A fast agent returns normally; jsonl row shape is identical to
    the pre-P1 happy path (no error, prediction populated, attempts=1).
    """
    _install_fake_agent(monkeypatch, behaviour="fast")
    monkeypatch.setattr(run_gaia, "CLEANUP_GRACE_SECS", 2.0)
    fake_config.per_question_timeout_secs = 5.0

    t0 = time.monotonic()
    await run_gaia.answer_single_question(fake_config, fake_example)
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0, (
        f"fast path took {elapsed:.2f}s; should be near-instant"
    )

    row = _read_jsonl_row(fake_config.save_path)
    assert row["agent_error"] is None
    assert row["prediction"] == "done"
    assert row["attempts"] == 1
    assert row["iteration_limit_exceeded"] is False


@pytest.mark.asyncio
async def test_cooperative_cancel_completes_promptly(
    monkeypatch, fake_config, fake_example
):
    """A task that honours CancelledError unwinds well under the
    cleanup-grace window — the overshoot for well-behaved tools is tiny.
    """
    _install_fake_agent(monkeypatch, behaviour="cooperative")
    monkeypatch.setattr(run_gaia, "CLEANUP_GRACE_SECS", 2.0)
    fake_config.per_question_timeout_secs = 1.0

    t0 = time.monotonic()
    await run_gaia.answer_single_question(fake_config, fake_example)
    elapsed = time.monotonic() - t0

    # Cooperative cancel: should complete very close to the timeout
    # itself. Budget: timeout + small epsilon for cleanup + jsonl write.
    assert elapsed < 1.5, (
        f"cooperative cancel took {elapsed:.2f}s; expected near-1.0s. "
        "If this fails, CLEANUP_GRACE_SECS is blocking even cooperative "
        "tasks."
    )

    row = _read_jsonl_row(fake_config.save_path)
    assert row["agent_error"] == "Per-question timeout (1.0s) exceeded"
