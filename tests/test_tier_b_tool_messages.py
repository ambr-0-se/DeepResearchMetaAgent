"""
Tier B tests: OpenAI Chat Completions native tool messages (tool + tool_call_id).

Loads `models/base` and `models/message_manager` without `src/__init__.py` so tests
run without optional stack (langchain, crawl4ai, etc.).
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_tier_b_modules():
    if getattr(_load_tier_b_modules, "_done", False):
        return

    def ensure_pkg(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        m = ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m

    ensure_pkg("src", SRC)
    ensure_pkg("src.models", SRC / "models")
    ensure_pkg("src.logger", SRC / "logger")
    ensure_pkg("src.utils", SRC / "utils")

    # Minimal src.utils for models/base.py imports
    utils_pkg = ModuleType("src.utils")
    utils_pkg.encode_image_base64 = lambda x: x
    utils_pkg.make_image_url = lambda x: x
    utils_pkg.parse_json_blob = lambda x: ({}, None)
    utils_pkg._is_package_available = lambda *_a, **_k: False
    sys.modules["src.utils"] = utils_pkg

    # TokenUsage from monitor (base.py needs it)
    spec_mon = importlib.util.spec_from_file_location(
        "src.logger.monitor", SRC / "logger/monitor.py"
    )
    mon = importlib.util.module_from_spec(spec_mon)
    assert spec_mon.loader
    sys.modules["src.logger.monitor"] = mon
    spec_mon.loader.exec_module(mon)

    log_pkg = ModuleType("src.logger")
    log_pkg.TokenUsage = mon.TokenUsage
    sys.modules["src.logger"] = log_pkg

    # src.models.base
    spec_base = importlib.util.spec_from_file_location(
        "src.models.base", SRC / "models/base.py"
    )
    base_mod = importlib.util.module_from_spec(spec_base)
    assert spec_base.loader
    sys.modules["src.models.base"] = base_mod
    spec_base.loader.exec_module(base_mod)

    # Fake src.models package so `import src.models.message_manager` resolves base
    models_pkg = sys.modules["src.models"]
    models_pkg.base = base_mod

    # src.models.message_manager
    spec_mm = importlib.util.spec_from_file_location(
        "src.models.message_manager", SRC / "models/message_manager.py"
    )
    mm_mod = importlib.util.module_from_spec(spec_mm)
    assert spec_mm.loader
    sys.modules["src.models.message_manager"] = mm_mod
    spec_mm.loader.exec_module(mm_mod)
    models_pkg.message_manager = mm_mod

    _load_tier_b_modules._done = True


_load_tier_b_modules()

from src.models.base import (  # noqa: E402
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
)
from src.models.message_manager import MessageManager, _tool_calls_to_openai_api_format  # noqa: E402


def test_tool_calls_to_openai_api_format_serializes_arguments_as_json_string():
    tc = ChatMessageToolCall(
        id="call_abc",
        type="function",
        function=ChatMessageToolCallFunction(name="planning_tool", arguments={"action": "create"}),
    )
    out = _tool_calls_to_openai_api_format([tc])
    assert len(out) == 1
    assert out[0]["id"] == "call_abc"
    assert out[0]["function"]["name"] == "planning_tool"
    assert out[0]["function"]["arguments"] == json.dumps({"action": "create"})


def test_message_manager_emits_assistant_tool_calls_and_per_tool_results():
    mm = MessageManager(model_id="gpt-4", api_type="chat/completions")
    msgs = [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                ChatMessageToolCall(
                    id="call_1",
                    type="function",
                    function=ChatMessageToolCallFunction(name="a", arguments="{}"),
                ),
                ChatMessageToolCall(
                    id="call_2",
                    type="function",
                    function=ChatMessageToolCallFunction(name="b", arguments="{}"),
                ),
            ],
        ),
        ChatMessage(role=MessageRole.TOOL, tool_call_id="call_1", content="out one"),
        ChatMessage(role=MessageRole.TOOL, tool_call_id="call_2", content="out two"),
    ]
    out = mm.get_clean_message_list(msgs)
    assert len(out) == 3
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] is None
    assert "tool_calls" in out[0]
    assert len(out[0]["tool_calls"]) == 2
    assert out[1] == {"role": "tool", "tool_call_id": "call_1", "content": "out one"}
    assert out[2] == {"role": "tool", "tool_call_id": "call_2", "content": "out two"}


def test_message_manager_tool_role_requires_tool_call_id():
    mm = MessageManager(model_id="gpt-4", api_type="chat/completions")
    bad = [ChatMessage(role=MessageRole.TOOL, content="x", tool_call_id=None)]
    with pytest.raises(ValueError, match="tool_call_id"):
        mm.get_clean_message_list(bad)


def test_tier_b_same_as_action_step_to_messages_shape():
    """Mirrors ActionStep.to_messages (Tier B): assistant+tool_calls then one tool message per result."""
    assistant = ChatMessage(
        role="assistant",
        content="thinking",
        tool_calls=[
            ChatMessageToolCall(
                id="id1",
                type="function",
                function=ChatMessageToolCallFunction(name="t1", arguments={}),
            )
        ],
    )
    msgs = [
        ChatMessage(role=assistant.role, content=assistant.content, tool_calls=assistant.tool_calls),
        ChatMessage(role=MessageRole.TOOL, tool_call_id="id1", content="result text"),
    ]
    mm = MessageManager(model_id="gpt-4", api_type="chat/completions")
    api = mm.get_clean_message_list(msgs)
    assert api[0]["role"] == "assistant" and "tool_calls" in api[0]
    assert api[1] == {"role": "tool", "tool_call_id": "id1", "content": "result text"}
