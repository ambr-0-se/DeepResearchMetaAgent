"""
Unit tests for token_utils without importing the full ``src`` package (avoids heavy/broken optional deps).

Run: python -m pytest tests/test_token_utils.py -v
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest


def _load_token_utils():
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "src/utils/token_utils.py"
    spec = importlib.util.spec_from_file_location("token_utils_isolated", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register under a unique name so repeated loads behave
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tu():
    return _load_token_utils()


def _msg(**kwargs):
    return types.SimpleNamespace(**kwargs)


def test_normalize_and_tiktoken_smoke(tu):
    assert tu.normalize_model_id_for_tiktoken("anthropic/claude-3-5-sonnet-20241022") == "claude-3-5-sonnet-20241022"
    assert tu.get_token_count("hello world", "gpt-4o") > 0


def test_estimate_heuristic_matches_chars_div_35(tu):
    m = _msg(role="user", content="a" * 350, tool_calls=None)
    est = tu.estimate_messages_tokens([m], None, mode="heuristic", context_image_token_estimate=1024)
    assert est == 100


def test_group_assistant_tools_single_segment(tu):
    tc = types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="f", arguments="{}"))
    assistant = _msg(role="assistant", content="x", tool_calls=[tc])
    tool_m = _msg(role="tool", content="out", tool_call_id="c1")
    user = _msg(role="user", content="y", tool_calls=None)
    segs = tu.group_messages_for_pruning([assistant, tool_m, user])
    assert len(segs) == 2
    assert len(segs[0]) == 2
    assert segs[1] == [user]


def test_group_incomplete_tool_chain_stops(tu):
    """Assistant with tool_calls but no following tool messages is one segment."""
    tc = types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="f", arguments="{}"))
    assistant = _msg(role="assistant", content="x", tool_calls=[tc])
    user = _msg(role="user", content="y", tool_calls=None)
    segs = tu.group_messages_for_pruning([assistant, user])
    assert len(segs) == 2


def test_prune_integration_requires_chatmessage(tu):
    """prune_messages_to_budget lazy-imports ChatMessage; skip if full stack unavailable."""
    try:
        from src.models.base import ChatMessage, MessageRole
    except Exception:
        pytest.skip("Full src stack not importable in this environment")

    system = ChatMessage(role=MessageRole.SYSTEM, content="sys")
    long_u = ChatMessage(role=MessageRole.USER, content="Z" * 12000)
    tail = ChatMessage(role=MessageRole.USER, content="tail")
    msgs = [system, long_u, tail]
    out = tu.prune_messages_to_budget(
        msgs,
        "gpt-4o",
        max_model_len=1000,
        context_prune_threshold_ratio=0.85,
        context_prune_reserve_tokens=0,
        context_prune_tail_segments=1,
        token_estimation_mode="heuristic",
        context_image_token_estimate=1024,
    )
    assert len(out) >= 2
    text_blob = " ".join(
        str(getattr(m, "content", "")) for m in out
    )
    assert "truncated" in text_blob.lower() or "[Earlier conversation truncated" in text_blob
