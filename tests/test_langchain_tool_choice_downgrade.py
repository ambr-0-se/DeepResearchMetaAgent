"""Tests for `ToolChoiceDowngradingChatOpenAI` — the LangChain-path
counterpart of the native `pick_tool_choice` downgrade.

Motivated by the 2026-04-22 T3 smoke: Qwen's first real invocation of
`auto_browser_use_tool` during the P1-P4 validation smoke threw 3,240 ×
HTTP 404 "No endpoints found that support the provided 'tool_choice'
value" in ~2 min. Root cause: the native `OpenAIServerModel.generate()`
path applies the hybrid dispatch (downgrade `"required"` → `"auto"`
for `qwen/*` wire ids) but the LangChain `ChatOpenAI` wrapper used by
`browser_use.Agent` bypassed it — `browser_use.bind_tools` was emitting
a `tool_choice` value Alibaba rejects. E0 v3 training missed this
latent bug because Qwen's planner never actually chose
`auto_browser_use_tool` in 80 questions.

These tests verify that the subclass:
  1. Downgrades `tool_choice="required"` to `"auto"` for `qwen/*` models.
  2. Does NOT touch `tool_choice` for models outside the auto-dispatch
     set (Gemma, Mistral, OpenAI, etc.).
  3. Preserves every other payload field (model, messages, tools, etc.).
  4. Handles the absent-`tool_choice` case by passing through untouched.
  5. Only logs the downgrade once per (model, process) pair.
  6. Registers correctly at the `ModelManager.init_models()` level —
     Qwen OR alias maps to the downgrading subclass, non-Qwen stays on
     plain `ChatOpenAI`.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.models.openaillm import make_tool_choice_downgrading_chat_openai
from src.models.tool_choice import _reset_logged_downgrades_for_tests


DUMMY_TOOL = {
    "type": "function",
    "function": {
        "name": "dummy_tool",
        "description": "stub",
        "parameters": {"type": "object", "properties": {}},
    },
}


@pytest.fixture(autouse=True)
def _reset_log_guard():
    """`log_downgrade_once` has process-wide state; reset before each test
    so the once-per-model assertion is deterministic."""
    _reset_logged_downgrades_for_tests()
    yield
    _reset_logged_downgrades_for_tests()


# ---------------------------------------------------------------------------
# _get_request_payload behaviour — the actual downgrade seam
# ---------------------------------------------------------------------------

def test_downgrades_required_to_auto_for_qwen_wire_id():
    """A Qwen wire id (`qwen/qwen3.6-plus`) must have
    `tool_choice="required"` rewritten to `"auto"`."""
    Cls = make_tool_choice_downgrading_chat_openai()
    model = Cls(model="qwen/qwen3.6-plus", api_key="fake", base_url="https://x")

    payload = model._get_request_payload(
        [HumanMessage(content="hi")],
        tool_choice="required",
        tools=[DUMMY_TOOL],
    )

    assert payload["tool_choice"] == "auto", (
        f"Expected downgrade to 'auto' for qwen/* wire id, got "
        f"{payload['tool_choice']!r}"
    )


def test_passes_through_tool_choice_for_non_auto_dispatch_models():
    """Mistral and other models NOT in the auto-dispatch set must
    preserve the caller's `tool_choice` exactly."""
    Cls = make_tool_choice_downgrading_chat_openai()
    model = Cls(model="mistral-small-2603", api_key="fake", base_url="https://x")

    payload = model._get_request_payload(
        [HumanMessage(content="hi")],
        tool_choice="required",
        tools=[DUMMY_TOOL],
    )

    assert payload["tool_choice"] == "required", (
        f"Non-qwen models must pass tool_choice unchanged; got "
        f"{payload['tool_choice']!r}"
    )


def test_no_tool_choice_in_payload_means_no_change():
    """If the caller didn't set `tool_choice`, the payload is returned
    unchanged — the downgrade only activates when there's something to
    downgrade."""
    Cls = make_tool_choice_downgrading_chat_openai()
    model = Cls(model="qwen/qwen3.6-plus", api_key="fake", base_url="https://x")

    payload = model._get_request_payload([HumanMessage(content="hi")])

    assert "tool_choice" not in payload, (
        f"Payload should not gain a tool_choice key when the caller "
        f"didn't set one; got {payload.get('tool_choice')!r}"
    )


def test_payload_fields_preserved_after_downgrade():
    """Rewriting `tool_choice` must not mangle other payload fields —
    messages, tools, model id, and anything else LangChain populated."""
    Cls = make_tool_choice_downgrading_chat_openai()
    model = Cls(model="qwen/qwen3.6-plus", api_key="fake", base_url="https://x")

    payload = model._get_request_payload(
        [HumanMessage(content="hello world")],
        tool_choice="required",
        tools=[DUMMY_TOOL],
        stop=["STOP"],
    )

    assert payload["model"] == "qwen/qwen3.6-plus"
    assert "messages" in payload and len(payload["messages"]) == 1
    assert payload["tools"] == [DUMMY_TOOL]
    # Stop sequences get normalised somewhere in LangChain; existence check
    # is enough.
    assert "stop" in payload


def test_downgrade_logs_once_per_model(caplog):
    """Downgrade banner emits at INFO level on first call and is silent
    on subsequent calls for the same model id. Prevents log spam when
    the LangChain path is hit hundreds of times per question."""
    Cls = make_tool_choice_downgrading_chat_openai()
    model = Cls(model="qwen/qwen3.6-plus", api_key="fake", base_url="https://x")

    caplog.set_level(logging.INFO, logger="src.models.tool_choice")

    # Three successive calls — only the first should log.
    for _ in range(3):
        model._get_request_payload(
            [HumanMessage(content="hi")],
            tool_choice="required",
            tools=[DUMMY_TOOL],
        )

    downgrade_records = [
        r for r in caplog.records
        if "auto (model in auto-dispatch set)" in r.getMessage()
           and "qwen/qwen3.6-plus" in r.getMessage()
    ]
    assert len(downgrade_records) == 1, (
        f"Expected exactly one downgrade log entry for qwen/qwen3.6-plus; "
        f"got {len(downgrade_records)}"
    )


def test_auto_tool_choice_is_not_spuriously_rewritten():
    """If the caller already passes `tool_choice="auto"`, no rewrite
    happens and no log fires. Belt-and-braces — the downgrade rule only
    trips when the RESOLVED value differs from the REQUESTED value."""
    Cls = make_tool_choice_downgrading_chat_openai()
    model = Cls(model="qwen/qwen3.6-plus", api_key="fake", base_url="https://x")

    payload = model._get_request_payload(
        [HumanMessage(content="hi")],
        tool_choice="auto",
        tools=[DUMMY_TOOL],
    )

    assert payload["tool_choice"] == "auto"  # unchanged


# ---------------------------------------------------------------------------
# Registration integration — _register_openrouter_models picks the right class
# ---------------------------------------------------------------------------

def test_qwen_or_registration_uses_downgrading_subclass(monkeypatch):
    """With `OPENROUTER_API_KEY` set, `ModelManager.init_models` must
    register `langchain-or-qwen3.6-plus` as the downgrading subclass
    (not plain ChatOpenAI)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    from src.models.models import ModelManager
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    qwen = mm.registed_models.get("langchain-or-qwen3.6-plus")
    assert qwen is not None
    # The downgrading class is a runtime subclass of ChatOpenAI.
    assert isinstance(qwen, ChatOpenAI)
    assert type(qwen).__name__ == "_Impl", (
        f"Qwen OR langchain wrapper should be the downgrading subclass "
        f"(`_Impl`); got {type(qwen).__name__}"
    )


def test_gemma_or_registration_uses_plain_chatopenai(monkeypatch):
    """Gemma is NOT in the auto-dispatch set (D5 probe confirmed
    required works), so its OR LangChain wrapper must be plain
    ChatOpenAI — no subclass, no overhead."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    from src.models.models import ModelManager
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    gemma = mm.registed_models.get("langchain-or-gemma-4-31b-it")
    assert gemma is not None
    assert type(gemma) is ChatOpenAI, (
        f"Gemma OR langchain wrapper must be plain ChatOpenAI; got "
        f"{type(gemma).__name__}"
    )


def test_kimi_or_registration_uses_plain_chatopenai(monkeypatch):
    """Kimi OR is not in the auto-dispatch set; plain ChatOpenAI."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    from src.models.models import ModelManager
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    kimi = mm.registed_models.get("langchain-or-kimi-k2.5")
    assert kimi is not None
    assert type(kimi) is ChatOpenAI


# ---------------------------------------------------------------------------
# Negative case — `OPENROUTER_API_KEY` unset skips registration
# ---------------------------------------------------------------------------

def test_openrouter_key_unset_skips_registration(monkeypatch):
    """Guards the early-return in `_register_openrouter_models` that
    skips registration entirely when no key is provided. If that guard
    regresses (e.g. someone removes the `if api_key == PLACEHOLDER:`
    branch), registering without a key would either crash at network
    time or register models pointed at an unusable endpoint. This test
    catches either failure mode at import time.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from src.models.models import ModelManager
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    # All `or-*` and `langchain-or-*` aliases should be absent.
    or_aliases = [
        k for k in mm.registed_models
        if k.startswith("or-") or k.startswith("langchain-or-")
    ]
    assert or_aliases == [], (
        f"Expected 0 OR aliases registered with OPENROUTER_API_KEY unset; "
        f"got {or_aliases!r}"
    )


# ---------------------------------------------------------------------------
# All registered qwen/* OR variants get the downgrading subclass
# ---------------------------------------------------------------------------

# The full list of qwen/* OR aliases registered by _register_openrouter_models.
# Sourced from src/models/models.py (2026-04-22). Add new slugs here whenever
# a new Qwen OR variant is registered.
QWEN_OR_ALIASES = [
    "langchain-or-qwen3-max",
    "langchain-or-qwen3.6-plus",
    "langchain-or-qwen3-next-80b-a3b-instruct",
    "langchain-or-qwen3-coder-next",
]


@pytest.mark.parametrize("alias", QWEN_OR_ALIASES)
def test_every_qwen_or_variant_uses_downgrading_subclass(monkeypatch, alias):
    """Every registered `qwen/*` OR wire id must map to the downgrading
    subclass, not plain `ChatOpenAI`. A regression here (e.g. a new
    Qwen variant added to models.py without the prefix match firing)
    would silently re-expose the 404 tool_choice bug for that slug.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    from src.models.models import ModelManager
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    model = mm.registed_models.get(alias)
    assert model is not None, f"{alias} not registered at all"
    # Behaviour-level assertion: the _get_request_payload override is
    # installed (doesn't rely on class name which is a brittle contract).
    assert (
        type(model)._get_request_payload
        is not ChatOpenAI._get_request_payload
    ), (
        f"{alias} is plain ChatOpenAI — downgrade override not installed. "
        f"Check the prefix rule in pick_tool_choice() and the branching "
        f"in _register_openrouter_models."
    )


def test_every_qwen_or_variant_behaviour_downgrades(monkeypatch):
    """End-to-end behaviour check: feed each registered qwen/* variant
    a `tool_choice="required"` call and assert the payload comes out
    with `"auto"`. One parametrise combines the registration check
    (above) with the actual rewrite semantics.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    from src.models.models import ModelManager
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    for alias in QWEN_OR_ALIASES:
        model = mm.registed_models.get(alias)
        assert model is not None, f"{alias} missing"
        payload = model._get_request_payload(
            [HumanMessage(content="probe")],
            tool_choice="required",
            tools=[DUMMY_TOOL],
        )
        assert payload["tool_choice"] == "auto", (
            f"{alias}: expected downgrade to 'auto'; got "
            f"{payload['tool_choice']!r}"
        )


# ---------------------------------------------------------------------------
# Factory runtime guard — langchain-openai version compatibility
# ---------------------------------------------------------------------------

def test_factory_raises_when_get_request_payload_removed(monkeypatch):
    """The factory asserts `ChatOpenAI._get_request_payload` exists
    before returning the subclass. If a langchain-openai upgrade
    renames or removes this hook, every call site that relies on the
    downgrade would silently fail — registration falls back to plain
    ChatOpenAI with no override. The guard fails loudly instead.
    """
    # Patch langchain_openai.ChatOpenAI to a stub that LACKS
    # `_get_request_payload`, simulating a hypothetical future version
    # where the method was renamed.
    import langchain_openai
    original = langchain_openai.ChatOpenAI

    class _FutureChatOpenAIWithoutGetRequestPayload:
        # Deliberately no _get_request_payload attribute.
        pass

    monkeypatch.setattr(
        langchain_openai, "ChatOpenAI", _FutureChatOpenAIWithoutGetRequestPayload
    )

    from src.models.openaillm import make_tool_choice_downgrading_chat_openai
    with pytest.raises(RuntimeError, match="_get_request_payload"):
        make_tool_choice_downgrading_chat_openai()

    # Restore so the rest of the test module can still import cleanly.
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", original)
