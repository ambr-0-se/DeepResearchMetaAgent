"""Tests for the Qwen-specific `auto_browser_use_tool` two-layer fix.

Context
-------
Qwen via OpenRouter+Alibaba rejects every non-"auto" tool_choice value
(HTTP 404). P5 downgrades to "auto" via the LangChain path, but under
browser_use's default `tool_calling_method='function_calling'` this
causes Qwen to emit every optional field of `AgentOutput.action`
simultaneously — `done` fires first, agent dies at Step 1. The fix
routes Qwen through `tool_calling_method='raw'` and installs a tolerant
JSON extractor to handle Qwen's one-closing-fence output quirk.

See `docs/handoffs/HANDOFF_QWEN_BROWSER_RAW_MODE.md` for the full root
cause + 2026-04-23 live-probe evidence.

Test classes
------------
T1 `TestTolerantJsonExtractor` — pure parser behaviour (9 cases).
T2 `TestPatchInstaller` — install / idempotence / reset / upgrade guard.
T3 `TestToolCallingMethodSelection` — wire-id → kwarg mapping (4 cases).
T4 `TestMistralUnchanged` — regression guard; Mistral path byte-identical.
T5 `TestModelIdResolution` — wire-id resolver across wrapper shapes.
T6 `TestEndToEndFixture` — `AutoBrowserUseTool.forward` with a mocked
    `browser_use.Agent` replaying captured Qwen / Mistral trajectories.

All tests are offline — no API keys, no browser launches.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Repo-root on sys.path so the sweep (`conda run -n dra pytest <file>`)
# can resolve `src.*` imports. Convention matches
# `tests/test_skill_registry.py`.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_extractor_patch():
    """Every test starts + ends with the vendored parser restored. Scopes
    the patch to the test that installs it so later tests can't see
    stale wrapping (R8 in the plan).
    """
    from src.tools import _browser_json_extractor as mod
    mod._reset_for_tests()
    yield
    mod._reset_for_tests()


@pytest.fixture
def stub_pdf_server(monkeypatch):
    """Suppress the http.server subprocess spawn in `AutoBrowserUseTool`'s
    `_init_pdf_server`. Each test suite instantiates the tool without a
    real subprocess so CI and local runs don't leak port 8080 children.
    """
    def _noop(self):
        return None

    from src.tools import auto_browser as ab
    monkeypatch.setattr(ab.AutoBrowserUseTool, "_init_pdf_server", _noop)
    yield


# ---------------------------------------------------------------------------
# T1 — TestTolerantJsonExtractor (9)
# ---------------------------------------------------------------------------


class TestTolerantJsonExtractor:
    """Pure-parser tests. Uses the original browser_use function as the
    fast-path reference, so we install the patch at class entry and let
    the autouse fixture restore afterwards.
    """

    @pytest.fixture(autouse=True)
    def _install(self):
        from src.tools._browser_json_extractor import install_tolerant_extractor
        install_tolerant_extractor()
        yield

    @staticmethod
    def _parse(text: Any) -> dict:
        from src.tools._browser_json_extractor import tolerant_extract_json_from_model_output
        return tolerant_extract_json_from_model_output(text)

    # T1.1
    def test_plain_json_passes_through(self):
        out = self._parse('{"a": 1, "b": {"c": 2}}')
        assert out == {"a": 1, "b": {"c": 2}}

    # T1.2
    def test_full_fence_json_passes_through(self):
        out = self._parse('```json\n{"action": [{"go_to_url": {"url": "x"}}]}\n```')
        assert out == {"action": [{"go_to_url": {"url": "x"}}]}

    # T1.3 — the Qwen case from the 2026-04-23 raw-mode probe
    def test_one_closing_fence_qwen_case(self):
        # Captured verbatim from the live probe.
        content = (
            'json\n{\n  "current_state": {\n    "memory": "I am on blank"\n  },\n'
            '  "action": [\n    {\n      "go_to_url": {\n        "url": '
            '"https://en.wikipedia.org/wiki/Kangaroo"\n      }\n    }\n  ]\n}\n```'
        )
        out = self._parse(content)
        assert out["action"][0]["go_to_url"]["url"] == \
            "https://en.wikipedia.org/wiki/Kangaroo"
        assert "current_state" in out

    # T1.4
    def test_one_opening_fence_no_close(self):
        content = '```json\n{"a": 1}'
        out = self._parse(content)
        assert out == {"a": 1}

    # T1.5
    def test_prose_with_embedded_json(self):
        content = (
            "Sure — here's the response you asked for:\n\n"
            '{"action": [{"done": {"text": "hi", "success": true}}]}\n\n'
            "Let me know if anything else."
        )
        out = self._parse(content)
        assert out["action"][0]["done"]["text"] == "hi"

    # T1.6
    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse response."):
            self._parse("")

    # T1.7
    def test_non_dict_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse response."):
            self._parse("null")

    # T1.8
    def test_single_element_list_wrapper_unwraps(self):
        # browser_use utils.py:44 unwraps this case — we must preserve it
        # (precedent: https://github.com/browser-use/browser-use/issues/1458)
        out = self._parse('[{"action": [{"wait": {"seconds": 1}}]}]')
        assert out == {"action": [{"wait": {"seconds": 1}}]}

    # T1.9
    def test_anthropic_content_blocks_flattened(self):
        raw = [{"type": "text", "text": '{"ok": true}'}]
        out = self._parse(raw)
        assert out == {"ok": True}


# ---------------------------------------------------------------------------
# T2 — TestPatchInstaller (4)
# ---------------------------------------------------------------------------


class TestPatchInstaller:
    """Module patch install / reset / idempotence behaviour."""

    # T2.1
    def test_install_replaces_module_function(self):
        from browser_use.agent.message_manager import utils as bu_utils
        from src.tools import _browser_json_extractor as mod

        pristine = bu_utils.extract_json_from_model_output
        mod.install_tolerant_extractor()
        try:
            assert mod.is_patched() is True
            assert bu_utils.extract_json_from_model_output is \
                mod.tolerant_extract_json_from_model_output
            assert mod._ORIGINAL is pristine
        finally:
            mod._reset_for_tests()

    # T2.2
    def test_install_is_idempotent(self):
        from browser_use.agent.message_manager import utils as bu_utils
        from src.tools import _browser_json_extractor as mod

        mod.install_tolerant_extractor()
        ref_after_first = bu_utils.extract_json_from_model_output
        original_after_first = mod._ORIGINAL
        mod.install_tolerant_extractor()
        assert bu_utils.extract_json_from_model_output is ref_after_first
        # Critical: _ORIGINAL must NOT be overwritten on second install.
        # If it were, calling _reset_for_tests() would restore the
        # wrapper as "original" and leak into later tests.
        assert mod._ORIGINAL is original_after_first

    # T2.3
    def test_reset_for_tests_restores_original(self):
        from browser_use.agent.message_manager import utils as bu_utils
        from src.tools import _browser_json_extractor as mod

        pristine = bu_utils.extract_json_from_model_output
        mod.install_tolerant_extractor()
        assert bu_utils.extract_json_from_model_output is not pristine
        mod._reset_for_tests()
        assert bu_utils.extract_json_from_model_output is pristine
        assert mod.is_patched() is False
        assert mod._ORIGINAL is None

    # T2.4
    def test_install_raises_if_upstream_renamed(self, monkeypatch):
        from browser_use.agent.message_manager import utils as bu_utils
        from src.tools import _browser_json_extractor as mod

        # Simulate an upstream library rename by deleting the attribute.
        monkeypatch.delattr(bu_utils, "extract_json_from_model_output")
        with pytest.raises(RuntimeError, match="no longer exposes"):
            mod.install_tolerant_extractor()


# ---------------------------------------------------------------------------
# T3 — TestToolCallingMethodSelection (4)
# ---------------------------------------------------------------------------


class TestToolCallingMethodSelection:
    """Wire-id → browser_use `tool_calling_method` kwarg."""

    # T3.1
    def test_qwen_wire_id_picks_raw(self):
        from src.tools.auto_browser import _pick_browser_tool_calling_method
        assert _pick_browser_tool_calling_method("qwen/qwen3.6-plus") == "raw"
        assert _pick_browser_tool_calling_method("qwen/qwen3-coder-next") == "raw"

    # T3.2
    def test_non_qwen_returns_none(self):
        from src.tools.auto_browser import _pick_browser_tool_calling_method
        assert _pick_browser_tool_calling_method("mistral-small-latest") is None
        assert _pick_browser_tool_calling_method("google/gemma-4-31b-it") is None
        assert _pick_browser_tool_calling_method("moonshotai/kimi-k2.5") is None
        assert _pick_browser_tool_calling_method("gpt-4.1") is None

    # T3.3 — wire id resolution falls through model_id → model_name
    def test_qwen_detected_via_model_name_fallback(self):
        from src.tools.auto_browser import _resolve_wire_id, _pick_browser_tool_calling_method
        stub = types.SimpleNamespace(model_id=None, model_name="qwen/qwen3.6-plus")
        wire_id = _resolve_wire_id(stub)
        assert _pick_browser_tool_calling_method(wire_id) == "raw"

    # T3.4
    def test_empty_or_none_returns_none(self):
        from src.tools.auto_browser import _pick_browser_tool_calling_method
        assert _pick_browser_tool_calling_method("") is None
        assert _pick_browser_tool_calling_method(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T4 — TestMistralUnchanged (2)
# ---------------------------------------------------------------------------


class TestMistralUnchanged:
    """Regression guard: Mistral's browser path is byte-identical pre/post-fix."""

    # T4.1
    def test_mistral_well_formed_output_bypasses_tolerant_branch(self):
        """Well-formed fenced JSON uses the original parser's fast path
        (the tolerant wrapper only activates on ValueError). Assert
        behavioural equivalence by comparing dict equality AND by
        confirming the result came from the original parser (via a
        spy)."""
        from browser_use.agent.message_manager import utils as bu_utils
        from src.tools import _browser_json_extractor as mod

        pristine = bu_utils.extract_json_from_model_output
        calls = []

        def spy(content):
            calls.append(content)
            return pristine(content)

        mod._ORIGINAL = spy
        mod._PATCHED = True
        bu_utils.extract_json_from_model_output = mod.tolerant_extract_json_from_model_output

        try:
            mistral_output = '```json\n{"action": [{"go_to_url": {"url": "https://example.com"}}]}\n```'
            result = bu_utils.extract_json_from_model_output(mistral_output)
            assert result == {"action": [{"go_to_url": {"url": "https://example.com"}}]}
            # The original parser (our spy) was invoked exactly once —
            # tolerant strategies didn't run. This is the fairness
            # guarantee (R6).
            assert len(calls) == 1
        finally:
            mod._reset_for_tests()

    # T4.2
    def test_mistral_construction_kwargs_have_no_tool_calling_method(self):
        from src.tools.auto_browser import _pick_browser_tool_calling_method
        # A Mistral wire id would return None from the selector,
        # which makes auto_browser.py build Agent(...) WITHOUT the
        # tool_calling_method kwarg — preserving browser_use's own
        # 'auto' detection path (the function_calling default for
        # ChatOpenAI).
        assert _pick_browser_tool_calling_method("mistral-small-latest") is None
        assert _pick_browser_tool_calling_method("mistral-small") is None


# ---------------------------------------------------------------------------
# T5 — TestModelIdResolution (3)
# ---------------------------------------------------------------------------


class TestModelIdResolution:
    """Wire-id resolution across the three wrapper shapes we register."""

    # T5.1
    def test_resolve_from_model_id_attribute(self):
        from src.tools.auto_browser import _resolve_wire_id
        stub = types.SimpleNamespace(model_id="qwen/qwen3.6-plus")
        assert _resolve_wire_id(stub) == "qwen/qwen3.6-plus"

    # T5.2
    def test_resolve_with_only_model_name(self):
        from src.tools.auto_browser import _resolve_wire_id
        stub = types.SimpleNamespace(model_name="mistral-small-latest")
        assert _resolve_wire_id(stub) == "mistral-small-latest"

    # T5.3 — KeyRotatingChatOpenAI delegates via __getattr__ (openaillm.py:788)
    def test_resolve_through_getattr_delegation(self):
        from src.tools.auto_browser import _resolve_wire_id

        class Rotating:
            def __init__(self):
                self._inner = types.SimpleNamespace(model_name="mistral-small-latest")

            def __getattr__(self, name):
                return getattr(self._inner, name)

        assert _resolve_wire_id(Rotating()) == "mistral-small-latest"


# ---------------------------------------------------------------------------
# T7 — TestUnwrapForBrowserUse (3) — KeyRotatingChatOpenAI Pydantic shim
# ---------------------------------------------------------------------------


class TestUnwrapForBrowserUse:
    """browser_use's AgentSettings.page_extraction_llm is typed as
    BaseChatModel | None. KeyRotatingChatOpenAI isn't a BaseChatModel
    subclass, so Pydantic v2 strict validation rejects it. `_unwrap_for_browser_use`
    returns instance[0] (plain ChatOpenAI) for rotating wrappers and
    passes through for non-rotating ones.

    Bug discovery: 2026-04-22 T3v2 Mistral log has 35 occurrences of
    `'ChatOpenAI' object has no attribute 'get'` — silent browser_use
    failures from the same root cause.
    """

    # T7.1
    def test_unwrap_key_rotating_returns_instance_0(self):
        from src.tools.auto_browser import _unwrap_for_browser_use

        class Rotating:
            def __init__(self):
                self._instances = ["first", "second", "third"]

        wrapper = Rotating()
        assert _unwrap_for_browser_use(wrapper) == "first"

    # T7.2
    def test_unwrap_non_rotating_passes_through(self):
        from src.tools.auto_browser import _unwrap_for_browser_use
        plain = types.SimpleNamespace(model_name="mistral-small")
        assert _unwrap_for_browser_use(plain) is plain

    # T7.3 — empty _instances list should pass through (defensive)
    def test_unwrap_empty_instances_passes_through(self):
        from src.tools.auto_browser import _unwrap_for_browser_use
        stub = types.SimpleNamespace(_instances=[])
        assert _unwrap_for_browser_use(stub) is stub


# ---------------------------------------------------------------------------
# T6 — TestEndToEndFixture (2)
# ---------------------------------------------------------------------------


class _FakeHistory:
    def __init__(self, contents, urls):
        self._contents = contents
        self._urls = urls
        self.history = list(range(max(1, len(contents))))  # non-empty

    def extracted_content(self):
        return self._contents

    def urls(self):
        return self._urls


class _FakeAgent:
    """Mocks `browser_use.Agent` for end-to-end fixture tests. Records
    construction kwargs so tests can assert the model-specific branch
    was taken, and returns a pre-canned `_FakeHistory`.
    """

    last_kwargs: dict | None = None
    fake_history: _FakeHistory | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)

    async def run(self, max_steps):  # noqa: ARG002 — match upstream sig
        if type(self).fake_history is None:
            raise AssertionError("fake_history must be set before run()")
        return type(self).fake_history

    async def close(self):
        return None


class TestEndToEndFixture:
    """Exercise `AutoBrowserUseTool.forward` with `browser_use.Agent`
    mocked out. Confirms both the kwarg wiring AND that extracted
    content round-trips through the tool unchanged.
    """

    @pytest.fixture
    def tool(self, stub_pdf_server, monkeypatch):
        from src.tools import auto_browser as ab
        monkeypatch.setattr(ab, "Agent", _FakeAgent)
        _FakeAgent.last_kwargs = None
        _FakeAgent.fake_history = None
        yield ab

    # T6.1
    def test_qwen_model_takes_raw_mode_and_extracts_content(self, tool, monkeypatch):
        """End-to-end: a registered Qwen LangChain model triggers raw
        mode + extracts content successfully.
        """
        # Register a minimal Qwen stub under the alias the tool resolves.
        qwen_stub = types.SimpleNamespace(
            model_id="qwen/qwen3.6-plus",
            model_name="qwen/qwen3.6-plus",
        )
        from src.models import model_manager
        saved = dict(model_manager.registed_models)
        model_manager.registed_models["langchain-or-qwen3.6-plus"] = qwen_stub

        try:
            # Pre-canned extraction: navigation confirmation + a real
            # content chunk (the T3 v2 raw-mode probe showed CONTENT_LEN
            # stuck at 54 chars on the navigation-only failure; >200
            # here proves the fix end-to-end).
            _FakeAgent.fake_history = _FakeHistory(
                contents=[
                    "🔗 Navigated to https://en.wikipedia.org/wiki/Kangaroo",
                    "Kangaroos are marsupials from the subfamily Macropodinae "
                    "(macropods, meaning 'large foot'). In common use the term "
                    "is used to describe the largest species from this family, "
                    "the red kangaroo, the antilopine kangaroo, the eastern grey "
                    "kangaroo, and the western grey kangaroo.",
                ],
                urls=["https://en.wikipedia.org/wiki/Kangaroo"],
            )

            t = tool.AutoBrowserUseTool(
                model_id="langchain-or-qwen3.6-plus",
                max_steps=6,
            )
            result = asyncio.run(t.forward(task="stub"))
            assert result.error is None, f"unexpected error: {result.error}"
            assert result.output is not None
            assert len(result.output) >= 100, (
                f"extracted content too short for Qwen fix: {len(result.output)} chars"
            )
            assert "Kangaroo" in result.output

            # Critical: raw mode was picked because the wire id starts with "qwen/".
            assert _FakeAgent.last_kwargs["tool_calling_method"] == "raw"

            # Critical: the tolerant parser patch was installed as part
            # of the Qwen code path (L2).
            from src.tools._browser_json_extractor import is_patched
            assert is_patched() is True
        finally:
            model_manager.registed_models.clear()
            model_manager.registed_models.update(saved)

    # T6.2
    def test_mistral_model_takes_default_mode_and_extracts_content(self, tool):
        """Mistral's path must NOT pass `tool_calling_method` at all —
        that preserves browser_use's auto detection (→ 'function_calling'
        for ChatOpenAI). Also asserts the tolerant-parser patch is NOT
        installed for Mistral (R6 fairness)."""
        mistral_stub = types.SimpleNamespace(
            model_id="mistral-small-latest",
            model_name="mistral-small-latest",
        )
        from src.models import model_manager
        saved = dict(model_manager.registed_models)
        model_manager.registed_models["langchain-mistral-small-latest"] = mistral_stub

        try:
            _FakeAgent.fake_history = _FakeHistory(
                contents=[
                    "Mistral extracted content about a topic with more than two hundred "
                    "characters to make sure the threshold check is satisfied here. "
                    "This is test filler to ensure sufficient length for the regression "
                    "assertion below."
                ],
                urls=["https://example.com"],
            )
            t = tool.AutoBrowserUseTool(
                model_id="langchain-mistral-small-latest",
                max_steps=6,
            )
            result = asyncio.run(t.forward(task="stub"))
            assert result.error is None
            assert result.output is not None
            assert len(result.output) >= 100

            # Mistral path: no `tool_calling_method` kwarg was passed.
            assert "tool_calling_method" not in _FakeAgent.last_kwargs

            # Mistral path: the tolerant parser patch was NOT installed.
            from src.tools._browser_json_extractor import is_patched
            assert is_patched() is False
        finally:
            model_manager.registed_models.clear()
            model_manager.registed_models.update(saved)
