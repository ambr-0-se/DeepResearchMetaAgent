"""Tests for ToolGenerator and dynamic Tool.from_code selection."""
import asyncio
import sys

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.models.base import ChatMessage, MessageRole
from src.meta.tool_generator import (
    ToolGenerator,
    _validate_imports_ast,
    allowed_top_level_modules,
    format_allowlist_for_prompt,
)
from src.tools import Tool


MINIMAL_VALID_CODE = '''
from src.tools import Tool

class AlphaBetaTool(Tool):
    """Minimal valid generated tool."""
    name = "alpha_beta_tool"
    description = "Echo input."
    inputs = {"text": {"type": "string", "description": "Text to echo"}}
    output_type = "string"

    def forward(self, text: str) -> str:
        return text
'''

BAD_SUBPROCESS = '''
import subprocess
from src.tools import Tool

class BadTool(Tool):
    name = "bad_tool"
    description = "bad"
    inputs = {}
    output_type = "string"
    def forward(self) -> str:
        return "x"
'''

FENCED_CODE = "```python\n" + MINIMAL_VALID_CODE.strip() + "\n```"


class _MockModel:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    async def __call__(self, messages, stop_sequences=None):
        self.calls.append(list(messages))
        content = self._responses.pop(0)
        return ChatMessage(role=MessageRole.ASSISTANT, content=content)


def test_format_allowlist_includes_requests_not_pandas_by_default():
    s = format_allowlist_for_prompt(False)
    assert "requests" in s
    assert "pandas" not in s
    assert "pathlib" in s


def test_format_allowlist_includes_pandas_when_data_science():
    s = format_allowlist_for_prompt(True)
    assert "pandas" in s
    assert "numpy" in s
    assert "openpyxl" in s
    assert "yaml" in s


def test_allowed_top_level_modules_frozen():
    assert isinstance(allowed_top_level_modules(False), frozenset)


def test_validate_imports_rejects_subprocess():
    with pytest.raises(ValueError) as exc:
        _validate_imports_ast(BAD_SUBPROCESS, allowed_top_level_modules(False))
    assert "subprocess" in str(exc.value).lower() or "allowlist" in str(exc.value).lower()


def test_validate_imports_rejects_non_allowlisted_root():
    code = """
from src.tools import Tool
import not_allowed_xyz
class T(Tool):
    name = "t_tool"
    description = "d"
    inputs = {}
    output_type = "string"
    def forward(self) -> str:
        return not_allowed_xyz.__name__
"""
    try:
        _validate_imports_ast(code, allowed_top_level_modules(False))
    except ValueError as e:
        assert "not_allowed_xyz" in str(e) or "allowlist" in str(e).lower()


def test_clean_code_strips_fence():
    gen = ToolGenerator(AsyncMock())
    cleaned = gen._clean_code(FENCED_CODE)
    assert cleaned.startswith("from src.tools import Tool")


def test_validate_code_accepts_minimal():
    gen = ToolGenerator(AsyncMock())
    gen._validate_code(MINIMAL_VALID_CODE.strip())


def test_from_code_loads_minimal():
    t = Tool.from_code(MINIMAL_VALID_CODE.strip(), expected_tool_name="alpha_beta_tool")
    assert t.name == "alpha_beta_tool"
    assert t.forward(text="hi") == "hi"


def test_from_code_disambiguates_with_expected_name():
    dual = MINIMAL_VALID_CODE + '''

class OtherTool(Tool):
    name = "other_tool"
    description = "other"
    inputs = {"y": {"type": "string", "description": "y"}}
    output_type = "string"
    def forward(self, y: str) -> str:
        return y + "!"
'''
    t = Tool.from_code(dual.strip(), expected_tool_name="other_tool")
    assert t.name == "other_tool"


def test_from_code_errors_without_expected_when_multiple():
    dual = MINIMAL_VALID_CODE + '''

class OtherTool(Tool):
    name = "other_tool"
    description = "other"
    inputs = {"y": {"type": "string", "description": "y"}}
    output_type = "string"
    def forward(self, y: str) -> str:
        return y
'''
    with pytest.raises(ValueError) as exc:
        Tool.from_code(dual.strip())
    assert "expected_tool_name" in str(exc.value).lower() or "Multiple" in str(exc.value)


def test_modify_subagent_generate_tool_name_collision_suffix():
    from src.meta.modify_tool import ModifySubAgentTool

    parent = MagicMock()
    mod_tool = ModifySubAgentTool(parent)
    agent = MagicMock()
    agent.tools = {"foo_bar_baz_tool": MagicMock()}
    name = mod_tool._generate_tool_name("foo bar baz qux extra", agent)
    assert name != "foo_bar_baz_tool"
    assert name.startswith("foo_bar_baz_")
    assert name.endswith("_tool")


def test_generate_tool_code_retry_on_first_invalid():
    """Repair path must alternate user/assistant/user for chat APIs."""
    bad_then_good = [
        """
import subprocess
from src.tools import Tool

class XTool(Tool):
    name = "x_tool"
    description = "d"
    inputs = {"a": {"type": "string", "description": "a"}}
    output_type = "string"

    def forward(self, a: str) -> str:
        return a + subprocess.__name__
""".strip(),
        MINIMAL_VALID_CODE.strip(),
    ]
    model = _MockModel(bad_then_good)
    gen = ToolGenerator(model)
    out = asyncio.run(
        gen.generate_tool_code("echo text", tool_name="alpha_beta_tool")
    )
    assert len(model.calls) == 2
    assert model.calls[0][0].role == MessageRole.USER
    assert len(model.calls[1]) == 3
    assert model.calls[1][0].role == MessageRole.USER
    assert model.calls[1][1].role == MessageRole.ASSISTANT
    assert model.calls[1][2].role == MessageRole.USER
    assert "subprocess" not in out
    assert "alpha_beta_tool" in out
    Tool.from_code(out, expected_tool_name="alpha_beta_tool")
