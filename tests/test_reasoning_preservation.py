"""Unit tests for the three correctness fixes applied alongside DeepSeek V3.2 /
Mistral / Qwen3 / Kimi / MiniMax integration:

- Risk A: reasoning_content round-trips through ChatMessage + MessageManager
- Risk B: Moonshot Kimi / MiniMax sampling params are stripped after caller merge
- Risk C: extra_body flows from OpenAIServerModel.__init__ to completion_kwargs

These tests avoid `src/__init__.py` side-effects (which pull in crawl4ai and other
heavy browser deps) by loading `src.models.base` and `src.models.message_manager`
directly via importlib. This keeps the tests runnable in a minimal env and makes
the unit boundary explicit — we test the model-layer contracts, not the agent.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _REPO_ROOT / "src" / "models"


def _load_isolated(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a module without triggering its package's __init__.py side effects."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-create empty placeholder packages so relative imports inside base.py /
# message_manager.py that reference `src.utils` resolve to a minimal stub.
def _install_minimal_src_packages() -> None:
    if "src" not in sys.modules:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [str(_REPO_ROOT / "src")]
        sys.modules["src"] = src_pkg
    if "src.models" not in sys.modules:
        models_pkg = types.ModuleType("src.models")
        models_pkg.__path__ = [str(_MODELS_DIR)]
        sys.modules["src.models"] = models_pkg
    # src.utils is imported inside message_manager; stub encode_image_base64 and make_image_url.
    if "src.utils" not in sys.modules:
        utils_stub = types.ModuleType("src.utils")
        utils_stub.encode_image_base64 = lambda x: "stub-b64"
        utils_stub.make_image_url = lambda x: f"data:image/png;base64,{x}"
        utils_stub._is_package_available = lambda *a, **k: False
        utils_stub.encode_image_base64 = lambda x: "stub-b64"
        utils_stub.parse_json_blob = lambda x: {}
        utils_stub.get_dict_from_nested_dataclasses = None  # assigned below
        sys.modules["src.utils"] = utils_stub
    # src.logger stub — base.py only needs TokenUsage from it.
    if "src.logger" not in sys.modules:
        from dataclasses import dataclass as _dc

        logger_stub = types.ModuleType("src.logger")

        @_dc
        class _TokenUsage:
            input_tokens: int = 0
            output_tokens: int = 0

            @property
            def total_tokens(self) -> int:
                return self.input_tokens + self.output_tokens

        logger_stub.TokenUsage = _TokenUsage
        sys.modules["src.logger"] = logger_stub
    # Provide get_dict_from_nested_dataclasses used by ChatMessage.model_dump_json
    if not hasattr(sys.modules["src.utils"], "get_dict_from_nested_dataclasses") or \
            sys.modules["src.utils"].get_dict_from_nested_dataclasses is None:
        from dataclasses import asdict, is_dataclass

        def _get_dict(obj, ignore_key: str | None = None):
            if is_dataclass(obj):
                d = asdict(obj)
                if ignore_key:
                    d.pop(ignore_key, None)
                return d
            return obj

        sys.modules["src.utils"].get_dict_from_nested_dataclasses = _get_dict


@pytest.fixture(scope="module", autouse=True)
def _load_modules():
    _install_minimal_src_packages()
    _load_isolated("src.models.base", _MODELS_DIR / "base.py")
    _load_isolated("src.models.message_manager", _MODELS_DIR / "message_manager.py")
    yield


# ---------------------------------------------------------------------------
# Risk A — reasoning_content preservation
# ---------------------------------------------------------------------------


def test_chatmessage_from_dict_preserves_reasoning_content():
    from src.models.base import ChatMessage

    msg = ChatMessage.from_dict(
        {
            "role": "assistant",
            "content": "The answer is 42.",
            "reasoning_content": "Let me think step by step...",
        }
    )
    assert msg.reasoning_content == "Let me think step by step..."
    assert msg.content == "The answer is 42."


def test_chatmessage_json_round_trip_preserves_reasoning():
    from src.models.base import ChatMessage

    original = ChatMessage(
        role="assistant",
        content="Result",
        reasoning_content="Deep reasoning trace",
    )
    as_json = original.model_dump_json()
    data = json.loads(as_json)
    restored = ChatMessage.from_dict(data)
    assert restored.reasoning_content == "Deep reasoning trace"


def test_needs_reasoning_echo_predicate():
    from src.models.message_manager import needs_reasoning_echo

    # Positive cases
    assert needs_reasoning_echo("deepseek-reasoner") is True
    assert needs_reasoning_echo("deepseek-v3.2") is True
    assert needs_reasoning_echo("deepseek-v3.2-exp") is True
    assert needs_reasoning_echo("qwen3-max-thinking") is True
    # Negative cases
    assert needs_reasoning_echo("gpt-4.1") is False
    assert needs_reasoning_echo("claude-4-sonnet") is False
    assert needs_reasoning_echo("qwen3-max") is False  # non-thinking variant
    assert needs_reasoning_echo("kimi-k2.5") is False


def test_reasoning_echo_included_for_deepseek_reasoner():
    from src.models.base import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole
    from src.models.message_manager import MessageManager

    mm = MessageManager(model_id="deepseek-reasoner")
    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="I'll call the tool.",
        reasoning_content="Step 1: decompose. Step 2: delegate.",
        tool_calls=[
            ChatMessageToolCall(
                id="call_abc",
                type="function",
                function=ChatMessageToolCallFunction(name="search", arguments={"q": "x"}),
            )
        ],
    )
    out = mm._get_chat_completions_message_list([msg])
    assert len(out) == 1
    assert out[0]["role"] == "assistant"
    assert out[0]["reasoning_content"] == "Step 1: decompose. Step 2: delegate."
    assert out[0]["tool_calls"][0]["id"] == "call_abc"


def test_reasoning_echo_suppressed_for_non_thinking_model():
    from src.models.base import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole
    from src.models.message_manager import MessageManager

    mm = MessageManager(model_id="gpt-4.1")
    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="ok",
        reasoning_content="leaked reasoning",
        tool_calls=[
            ChatMessageToolCall(
                id="c1",
                type="function",
                function=ChatMessageToolCallFunction(name="f", arguments="{}"),
            )
        ],
    )
    out = mm._get_chat_completions_message_list([msg])
    # No reasoning_content should be emitted to providers that don't need it
    assert "reasoning_content" not in out[0]


def test_stream_delta_accumulates_reasoning():
    from src.models.base import ChatMessageStreamDelta, agglomerate_stream_deltas

    deltas = [
        ChatMessageStreamDelta(content="Hello ", reasoning_content="Think A. "),
        ChatMessageStreamDelta(content="world", reasoning_content="Think B."),
    ]
    msg = agglomerate_stream_deltas(deltas)
    assert msg.content == "Hello world"
    assert msg.reasoning_content == "Think A. Think B."


def test_assistant_messages_not_merged_when_reasoning_present():
    """Reviewer [MEDIUM]: role-merging would fabricate reasoning attribution."""
    from src.models.base import ChatMessage, MessageRole
    from src.models.message_manager import MessageManager

    mm = MessageManager(model_id="deepseek-reasoner")
    msgs = [
        ChatMessage(role=MessageRole.ASSISTANT, content="first", reasoning_content="r1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="second", reasoning_content="r2"),
    ]
    out = mm._get_chat_completions_message_list(msgs)
    # Both must remain as separate assistant messages
    assert len(out) == 2
    assert out[0]["reasoning_content"] == "r1"
    assert out[1]["reasoning_content"] == "r2"


# ---------------------------------------------------------------------------
# Risk B — Kimi / MiniMax sampling-param strip
# ---------------------------------------------------------------------------


def test_kimi_strips_banned_sampling_params():
    from src.models.message_manager import MessageManager

    mm = MessageManager(model_id="kimi-k2.5")
    payload = {
        "model": "kimi-k2.5",
        "temperature": 0.5,
        "top_p": 0.9,
        "n": 3,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.1,
        "messages": [],
    }
    cleaned = mm.get_clean_completion_kwargs(payload)
    for banned in ("temperature", "top_p", "n", "presence_penalty", "frequency_penalty"):
        assert banned not in cleaned, f"Kimi should strip {banned}"
    assert cleaned["model"] == "kimi-k2.5"


def test_kimi_via_openrouter_slug_also_stripped():
    from src.models.message_manager import MessageManager

    mm = MessageManager(model_id="moonshotai/kimi-k2.5")
    payload = {"temperature": 0.3, "messages": []}
    cleaned = mm.get_clean_completion_kwargs(payload)
    assert "temperature" not in cleaned


def test_non_kimi_model_preserves_temperature():
    from src.models.message_manager import MessageManager

    mm = MessageManager(model_id="gpt-4.1")
    payload = {"temperature": 0.3, "messages": []}
    cleaned = mm.get_clean_completion_kwargs(payload)
    assert cleaned["temperature"] == 0.3


def test_minimax_clamps_temperature_and_forces_n_one():
    from src.models.message_manager import MessageManager

    mm = MessageManager(model_id="minimax-m2.7")
    # temp=0 is invalid on MiniMax — should be clamped up
    p1 = mm.get_clean_completion_kwargs({"temperature": 0, "n": 5, "messages": []})
    assert p1["temperature"] > 0
    assert "n" not in p1
    # temp > 1.0 should be clamped down
    p2 = mm.get_clean_completion_kwargs({"temperature": 1.8, "messages": []})
    assert p2["temperature"] == 1.0


# ---------------------------------------------------------------------------
# Risk C — extra_body forwarding (without requiring network / openai SDK install)
# ---------------------------------------------------------------------------


def test_extra_body_merged_into_self_kwargs():
    """OpenAIServerModel should stash extra_body into self.kwargs so that
    _prepare_completion_kwargs forwards it to chat.completions.create."""
    try:
        from src.models.openaillm import OpenAIServerModel
    except Exception as exc:
        pytest.skip(f"OpenAIServerModel import blocked in this env: {exc}")

    # Pass a stub http_client so no real client is constructed.
    class _StubClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("should not be called in this test")

    model = OpenAIServerModel(
        model_id="qwen3-max",
        http_client=_StubClient(),
        extra_body={"enable_thinking": True},
    )
    assert model.kwargs.get("extra_body") == {"enable_thinking": True}
