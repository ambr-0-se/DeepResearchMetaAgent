"""Unit tests for the hybrid ``tool_choice`` dispatch helpers.

Covers:
- :func:`pick_tool_choice` named-entry / prefix / passthrough branches.
- One-shot downgrade INFO log dedup (``log_downgrade_once``).
- Retry-guard behaviour for the async (``GeneralAgent``) and sync
  (``ToolCallingAgent``) paths, using a mock ``model`` that alternates
  text-only and tool-call responses.

Loads the ``tool_choice`` module directly to keep the tests independent of
the heavier ``src`` init chain (mirrors the pattern in
``tests/test_process_tool_calls_guard.py``).
"""

from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_tool_choice_module():
    spec = importlib.util.spec_from_file_location(
        "tool_choice_dispatch_under_test",
        SRC / "models/tool_choice.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


tc = _load_tool_choice_module()


# ---------------------------------------------------------------------------
# pick_tool_choice — named entry / prefix / passthrough
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wire_id",
    [
        "qwen/qwen3.6-plus",
        "qwen/qwen3-next-80b-a3b-instruct",
        "qwen/qwen3-coder-next",
        "qwen/qwen3-max",
    ],
)
def test_pick_tool_choice_qwen_prefix_downgrades_to_auto(wire_id):
    assert tc.pick_tool_choice(wire_id) == "auto"


def test_models_rejecting_required_is_empty_after_gemma_probe():
    # Live smoke probe 2026-04-18 against OR `google/gemma-4-31b-it` with the
    # DeepInfra/Together provider pin + reasoning disabled passed
    # `tool_choice="required"` cleanly (finish_reason="tool_calls", proper
    # chat-template rendering). Gemma is therefore NOT in the defensive set;
    # the set itself stays declared as an extensibility hook for future
    # non-Qwen entries that escape the prefix rule.
    assert len(tc.MODELS_REJECTING_REQUIRED) == 0
    assert "google/gemma-4-31b-it" not in tc.MODELS_REJECTING_REQUIRED
    assert tc.pick_tool_choice("google/gemma-4-31b-it") == "required"


@pytest.mark.parametrize(
    "wire_id",
    [
        "moonshotai/kimi-k2.5",
        "mistralai/mistral-small-2603",
        "mistral-small",
        "openai/gpt-4.1",
        "anthropic/claude-3.7-sonnet",
        "deepseek/deepseek-v3.2",
    ],
)
def test_pick_tool_choice_passthrough_keeps_default(wire_id):
    assert tc.pick_tool_choice(wire_id) == "required"
    assert tc.pick_tool_choice(wire_id, default="auto") == "auto"
    assert tc.pick_tool_choice(wire_id, default=None) is None


def test_pick_tool_choice_empty_model_id_returns_default():
    assert tc.pick_tool_choice(None) == "required"
    assert tc.pick_tool_choice("") == "required"


def test_pick_tool_choice_named_dict_default_returned_unchanged():
    # Callers may pass a dict tool_choice (e.g. named function). Non-matching
    # models should still receive the dict back untouched.
    named = {"type": "function", "function": {"name": "final_answer"}}
    assert tc.pick_tool_choice("moonshotai/kimi-k2.5", default=named) is named


def test_qwen_prefix_tuple_only_contains_expected_prefixes():
    # Guard against accidental widening — e.g. "qwen-" (no slash) would match
    # non-OR wire ids and break dispatch for native DashScope Qwen.
    assert tc._AUTO_WIRE_PREFIXES == ("qwen/",)


# ---------------------------------------------------------------------------
# log_downgrade_once — fires once per model_id per process
# ---------------------------------------------------------------------------


def test_log_downgrade_once_dedups_per_model_id(caplog):
    tc._reset_logged_downgrades_for_tests()
    with caplog.at_level("INFO", logger=tc.logger.name):
        tc.log_downgrade_once("qwen/qwen3.6-plus")
        tc.log_downgrade_once("qwen/qwen3.6-plus")  # dup, should be silent
        tc.log_downgrade_once("google/gemma-4-31b-it")
    downgrade_records = [
        r for r in caplog.records if "tool_choice" in r.getMessage()
    ]
    assert len(downgrade_records) == 2
    logged_ids = {r.getMessage() for r in downgrade_records}
    assert any("qwen/qwen3.6-plus" in m for m in logged_ids)
    assert any("google/gemma-4-31b-it" in m for m in logged_ids)


# ---------------------------------------------------------------------------
# Retry-guard semantics — exercised via a pared-down agent fixture
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolCallFunction:
    name: str
    arguments: Any


@dataclass
class _FakeToolCall:
    function: _FakeToolCallFunction
    id: str = "call_1"
    type: str = "function"


@dataclass
class _FakeChatMessage:
    """Subset of src.models.base.ChatMessage sufficient for the retry guard.

    The production guard only reads `.tool_calls` and `.content` and
    constructs new ChatMessage instances with `role=` + `content=`. We rely on
    duck-typing to avoid pulling the heavier `src` init into this test file.
    """
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[_FakeToolCall] | None = None
    token_usage: Any = None
    raw: Any = None


class _ScriptedModel:
    """Mock model that plays a queue of responses per call."""

    def __init__(self, responses: list[_FakeChatMessage], model_id: str):
        self._responses = list(responses)
        self.model_id = model_id
        self.calls: list[list[Any]] = []

    async def __call__(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self._responses.pop(0)

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self._responses.pop(0)


def _build_async_guard(model, tools_and_managed_agents):
    """Build a standalone async retry-guard callable bound to (model, tools).

    Reproduces the production guard's logic without instantiating a full
    GeneralAgent (which pulls in the entire src init chain).
    """
    async def guard(chat_message, input_messages):
        model_id = getattr(model, "model_id", None)
        if tc.pick_tool_choice(model_id, default="required") != "auto":
            return chat_message
        retries = 0
        conversation = list(input_messages)
        MAX = 2
        while (
            (chat_message.tool_calls is None or len(chat_message.tool_calls) == 0)
            and retries < MAX
        ):
            retries += 1
            conversation = conversation + [
                _FakeChatMessage(role="assistant", content=chat_message.content or ""),
                _FakeChatMessage(role="user", content="RETRY_PROMPT"),
            ]
            chat_message = await model(
                conversation,
                stop_sequences=["Observation:", "Calling tools:"],
                tools_to_call_from=tools_and_managed_agents,
            )
        return chat_message

    return guard


def _build_sync_guard(model, tools_and_managed_agents):
    def guard(chat_message, input_messages):
        model_id = getattr(model, "model_id", None)
        if tc.pick_tool_choice(model_id, default="required") != "auto":
            return chat_message
        retries = 0
        conversation = list(input_messages)
        MAX = 2
        while (
            (chat_message.tool_calls is None or len(chat_message.tool_calls) == 0)
            and retries < MAX
        ):
            retries += 1
            conversation = conversation + [
                _FakeChatMessage(role="assistant", content=chat_message.content or ""),
                _FakeChatMessage(role="user", content="RETRY_PROMPT"),
            ]
            chat_message = model.generate(
                conversation,
                stop_sequences=["Observation:", "Calling tools:"],
                tools_to_call_from=tools_and_managed_agents,
            )
        return chat_message

    return guard


# -- non-auto model: guard is a no-op even with empty tool_calls --------------


def test_retry_guard_noop_when_tool_choice_stays_required():
    model = _ScriptedModel(
        responses=[],  # no retries should be attempted
        model_id="moonshotai/kimi-k2.5",
    )
    guard = _build_async_guard(model, tools_and_managed_agents=[])
    initial = _FakeChatMessage(content="plain text reply", tool_calls=None)
    out = asyncio.run(guard(initial, input_messages=[]))
    assert out is initial
    assert model.calls == []


def test_retry_guard_sync_noop_when_tool_choice_stays_required():
    model = _ScriptedModel(
        responses=[],
        model_id="mistral-small",
    )
    guard = _build_sync_guard(model, tools_and_managed_agents=[])
    initial = _FakeChatMessage(content="plain text reply", tool_calls=None)
    out = guard(initial, input_messages=[])
    assert out is initial
    assert model.calls == []


# -- auto model: retry succeeds on first re-prompt ----------------------------


def test_retry_guard_fires_once_when_model_recovers_on_first_retry():
    success = _FakeChatMessage(
        content=None,
        tool_calls=[_FakeToolCall(function=_FakeToolCallFunction(
            name="deep_analyzer_tool", arguments={"task": "x"}))]
    )
    model = _ScriptedModel(
        responses=[success],
        model_id="qwen/qwen3.6-plus",
    )
    guard = _build_async_guard(model, tools_and_managed_agents=[])
    initial = _FakeChatMessage(content="I think I'll answer in text.", tool_calls=None)
    out = asyncio.run(guard(initial, input_messages=[_FakeChatMessage(role="user", content="Q")]))

    assert out is success
    assert out.tool_calls is not None and len(out.tool_calls) == 1
    assert len(model.calls) == 1
    # Retry conversation: original user turn + assistant echo + corrective user
    sent = model.calls[0]
    assert len(sent) == 3
    assert sent[-1].role == "user" and sent[-1].content == "RETRY_PROMPT"


# -- auto model: both retries fail, guard falls through to existing path ------


def test_retry_guard_exhausts_retries_and_returns_empty_tool_calls():
    fail1 = _FakeChatMessage(content="still just text", tool_calls=None)
    fail2 = _FakeChatMessage(content="", tool_calls=[])
    model = _ScriptedModel(
        responses=[fail1, fail2],
        model_id="qwen/qwen3.6-plus",
    )
    guard = _build_async_guard(model, tools_and_managed_agents=[])
    initial = _FakeChatMessage(content="no tools here", tool_calls=None)
    out = asyncio.run(guard(initial, input_messages=[]))

    assert out is fail2
    assert out.tool_calls == []
    assert len(model.calls) == 2  # MAX_TOOL_RETRIES


def test_retry_guard_sync_recovers_on_retry():
    success = _FakeChatMessage(
        content=None,
        tool_calls=[_FakeToolCall(function=_FakeToolCallFunction(
            name="web_fetcher_tool", arguments={"url": "x"}))]
    )
    model = _ScriptedModel(
        responses=[success],
        model_id="qwen/qwen3-coder-next",  # prefix path — Qwen is auto-dispatched
    )
    guard = _build_sync_guard(model, tools_and_managed_agents=[])
    initial = _FakeChatMessage(content="text response", tool_calls=None)
    out = guard(initial, input_messages=[_FakeChatMessage(role="user", content="Q")])
    assert out is success
    assert len(model.calls) == 1


# -- production code still has a ChatMessage-shaped retry payload -------------


def test_production_chat_message_shape_matches_retry_assumptions():
    """Regression: the async + sync guards both construct fresh ChatMessage
    objects with only ``role`` + ``content``. Ensure the production ChatMessage
    dataclass still accepts that call shape — read from the source rather
    than importing, so the test stays independent of the heavier ``src.*``
    init chain (which pulls optional deps like crawl4ai).
    """
    base_py = (SRC / "models/base.py").read_text()
    # The dataclass has a number of fields but only role is required, and
    # role + content is the call shape used by the retry guard. We verify by
    # static-check: the field decls must allow role and content without any
    # other required positionals between them.
    assert "class ChatMessage:" in base_py
    assert "role: str" in base_py
    # ``content`` field declaration (required to be after role per dataclass
    # ordering, with a default so role+content-only construction is legal).
    assert "content: str | list[dict[str, Any]] | None" in base_py
    # Every field after content must have a default (defaulted order).
    # Quick sanity: ``tool_calls: ... = None`` follows the content line.
    assert "tool_calls: list[ChatMessageToolCall] | None = None" in base_py
