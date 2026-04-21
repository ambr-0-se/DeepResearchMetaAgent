"""Shared pytest fixtures for the DeepResearchMetaAgent suite.

Only fixtures used by ≥2 test files belong here. Fixtures specific to a
single file stay local per the existing convention in the repo
(`test_skill_registry.py`'s `skills_dir`, `test_eval_fixes.py`'s
`_MockAgent`, etc.).

Environment requirement: these fixtures assume the `dra` conda env so
heavy runtime deps (mmengine, crawl4ai transitively via `src.*`) are
available. Tests that don't need `src.*` do not depend on this conftest
and can continue to use the `importlib.util.spec_from_file_location`
pattern documented in `test_tool_choice_dispatch.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def fake_example() -> dict[str, Any]:
    """Minimal GAIA example dict matching the shape `answer_single_question`
    consumes (from `get_tasks_to_run` in `examples/run_gaia.py`).

    No file attachment; keeps the augmented-question branch simple.
    """
    return {
        "task_id": "00000000-test-task-id",
        "question": "What is 2 + 2?",
        "task": "stub-task",
        "true_answer": "4",
        "file_name": "",
    }


@pytest.fixture
def fake_config(tmp_path) -> SimpleNamespace:
    """Minimal config shim exposing only the attributes
    `answer_single_question` reads.

    `save_path` points at `tmp_path` so the jsonl write side-effect is
    isolated per-test and teardown is automatic.
    """
    return SimpleNamespace(
        save_path=str(tmp_path / "dra.jsonl"),
        per_question_timeout_secs=60,  # tests override per-case
        agent_config=SimpleNamespace(
            name="test-agent",
            model_id="test-model",
        ),
    )
