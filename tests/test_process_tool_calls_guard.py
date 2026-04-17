"""Unit tests for the premature / duplicate `final_answer_tool` guard.

Exercises `apply_final_answer_guard` — the pure filter that decides which
tool calls from a single model turn should actually run. The guard protects
against small-model failure modes where `final_answer_tool` is emitted
alongside other tool calls (with a fabricated `answer` argument), or
multiple times in one turn.

This test file imports the helper directly via a minimal module-loading
pattern (mirrors `tests/test_tier_b_tool_messages.py`) so the test
is independent of the heavier `src.*` package __init__ chain.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_guard():
    """Load `apply_final_answer_guard` directly, bypassing `src/__init__.py`."""
    if getattr(_load_guard, "_cached", None) is not None:
        return _load_guard._cached

    spec = importlib.util.spec_from_file_location(
        "tcguard",
        SRC / "agent/general_agent/_tool_call_guard.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _load_guard._cached = mod.apply_final_answer_guard
    return _load_guard._cached


apply_final_answer_guard = _load_guard()


def _tc(name: str, arguments="{}", tc_id: str | None = None):
    """Build a duck-typed tool_call with `.function.name` / `.function.arguments` / `.id`."""
    return SimpleNamespace(
        id=tc_id or f"tc_{name}",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


# ---------------------------------------------------------------------------
# Ordering cases: final_answer_tool + at least one sibling -> "premature"
# ---------------------------------------------------------------------------


def test_premature_guard_final_answer_last():
    """Case 1 — sibling first, final_answer second. Classic 'premature emission'."""
    tool_calls = [_tc("planning_tool"), _tc("final_answer_tool", {"answer": "fake"})]
    effective, status = apply_final_answer_guard(tool_calls)
    assert status == "premature"
    assert len(effective) == 1
    assert effective[0].function.name == "planning_tool"


def test_premature_guard_final_answer_first():
    """Case 2 — final_answer first (Codex's regression case the old plan missed)."""
    tool_calls = [_tc("final_answer_tool", {"answer": "fake"}), _tc("planning_tool")]
    effective, status = apply_final_answer_guard(tool_calls)
    assert status == "premature"
    assert len(effective) == 1
    assert effective[0].function.name == "planning_tool"


def test_premature_guard_final_answer_middle():
    """Case 3 — final_answer sandwiched between other calls."""
    tool_calls = [
        _tc("tool_a"),
        _tc("final_answer_tool", {"answer": "fake"}),
        _tc("tool_b"),
    ]
    effective, status = apply_final_answer_guard(tool_calls)
    assert status == "premature"
    assert len(effective) == 2
    assert {tc.function.name for tc in effective} == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# Duplicate-finals cases
# ---------------------------------------------------------------------------


def test_duplicate_guard_only_finals_no_siblings():
    """Case 4 — two final_answer calls, no other tools. Drop all; force regeneration.

    Codex's round-2 point: 'keep first of duplicate-only-finals is arbitrary;
    if they disagree you may keep the worse one.' Drop all.
    """
    tool_calls = [
        _tc("final_answer_tool", {"answer": "first"}),
        _tc("final_answer_tool", {"answer": "second"}),
    ]
    effective, status = apply_final_answer_guard(tool_calls)
    assert status == "duplicate"
    assert effective == []


def test_duplicate_guard_finals_with_siblings_is_premature_not_duplicate():
    """Case 5 — duplicate finals AND a sibling. Treat as premature (drop all finals)."""
    tool_calls = [
        _tc("final_answer_tool", {"answer": "a"}),
        _tc("planning_tool"),
        _tc("final_answer_tool", {"answer": "b"}),
    ]
    effective, status = apply_final_answer_guard(tool_calls)
    assert status == "premature"
    assert len(effective) == 1
    assert effective[0].function.name == "planning_tool"


# ---------------------------------------------------------------------------
# Happy-path cases
# ---------------------------------------------------------------------------


def test_lone_final_answer_preserved():
    """Case 6 — a solitary final_answer_tool is the normal completion path."""
    tool_calls = [_tc("final_answer_tool", {"answer": "real answer"})]
    effective, status = apply_final_answer_guard(tool_calls)
    assert status == "none"
    # Identity-equal: caller can use `is` check to skip ChatMessage mutation.
    assert effective is tool_calls
    assert len(effective) == 1


def test_no_final_answer_turn_unaffected():
    """Case 7 — a normal tool-only turn is untouched by the guard."""
    tool_calls = [_tc("planning_tool"), _tc("search_tool")]
    effective, status = apply_final_answer_guard(tool_calls)
    assert status == "none"
    assert effective is tool_calls
    assert len(effective) == 2


def test_empty_tool_calls():
    """Degenerate case: empty list. Guard should not fire."""
    effective, status = apply_final_answer_guard([])
    assert status == "none"
    assert effective == []
