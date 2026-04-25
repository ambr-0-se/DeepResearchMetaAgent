"""Tolerant JSON extractor for ``browser_use`` used by the Qwen raw-mode path.

Context
-------
``browser_use.agent.service.get_next_action`` (line ~707 in the 0.1.48
wheel) calls ``extract_json_from_model_output(content)`` at module scope
to parse the LLM's structured response when ``tool_calling_method='raw'``
is active. The shipped implementation
(``browser_use.agent.message_manager.utils.extract_json_from_model_output``)
does ``content.split('```')[1]`` and then ``content.split('\\n', 1)[1]``,
which assumes the output is wrapped in a complete pair of triple-backtick
fences. Qwen via OpenRouter+Alibaba regularly violates that contract:

- Emits only the closing fence: ``"json\\n{...}\\n```"`` → split returns
  ``['json\\n{...}\\n', '']`` → ``[1]`` is empty → ``json.JSONDecodeError``.
- Emits only the opening fence: ``"```json\\n{...}"`` (no close) → same
  parser succeeds on the slice but a later step may see the inverse.
- Emits prose prefix + JSON body with no fences.
- Emits the inner ``"current_state": {...}`` object without the outer
  braces.
- Returns ``content=''`` when the reasoning token budget is consumed.

Root cause on the Qwen side is traced in F6 of
``docs/handoffs/HANDOFF_E1_E2_RESULTS.md`` and the 2026-04-23 raw-mode
live probe captured in this session.

Design
------
We install a **stateless fallback** replacement for the module-scope
function ``_bu_utils.extract_json_from_model_output``. The replacement:

1. Tries the original browser_use parser first. If it succeeds, return
   its result unchanged. This keeps Mistral / Kimi / future-model paths
   identical byte-for-byte with pre-patch behaviour (R4, R6, R12 in the
   plan).
2. On ``ValueError``/``json.JSONDecodeError``, applies ordered tolerant
   strategies (fence strip, balanced-brace extraction, Anthropic
   content-block flatten) — all mirror the precedent at
   ``src/skills/_extractor.py:_parse_json_response``.
3. On hard failure, raises ``ValueError('Could not parse response.')``
   — the exact exception browser_use's retry path catches, preserving
   the 3-strike retry semantics (R4 in the plan).

Installation is idempotent and restores cleanly for tests
(``_reset_for_tests`` mirrors the guard in
``src/models/tool_choice.py``).
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Module-level state: references to the replacement state so tests and
# concurrent callers can probe / reset without digging through attribute
# lookups on the vendored module.
_ORIGINAL: Callable[[str], dict] | None = None
_PATCHED: bool = False

# Synchronises install/reset so two concurrent coroutines can't race
# through the "capture _ORIGINAL → swap module attr → set _PATCHED"
# critical section. Code-review HIGH-1 (2026-04-23): under concurrent
# Qwen tasks (P2 streaming worker pool, concurrency=8) both callers
# could pass the `if _PATCHED: return` guard, the second would capture
# the already-installed wrapper as `_ORIGINAL`, and a subsequent
# `_reset_for_tests()` would restore the wrapper (not the true
# browser_use function). `threading.Lock` is sufficient because
# assignment is brief and the lock is never held across I/O.
_INSTALL_LOCK = threading.Lock()


def _flatten_anthropic_blocks(raw: Any) -> str:
    """Flatten an Anthropic-style content list ``[{"type":"text","text":"..."}]``
    into a single string. Passes non-list inputs through ``str(...)``.

    Mirrors ``src/skills/_extractor.py::_parse_json_response`` — the one
    tolerant-parser precedent already shipping in this repo.
    """
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(raw) if raw is not None else ""


def _strip_fences(text: str) -> str:
    """Strip optional leading/trailing markdown code fences, the leading
    ``json\\n`` prefix that Qwen emits when the opening fence is missing,
    and the ``<tool_call>...</tool_call>`` XML wrapper that
    qwen3-next-80b-a3b-instruct emits on OpenRouter (probed 2026-04-26).

    Loops until the input stops shrinking so nested wrappers like
    ``"```json\\n<tool_call>{...}</tool_call>\\n```"`` are handled in any
    order. Bounded by max-iter to prevent pathological loops on
    adversarial input.

    Order within each pass:
      - ``"<tool_call>...</tool_call>"`` XML wrapper
      - ```` ```json ``` / ``` ``` ```` markdown fences
      - bare ``"json\\n"`` language hint
      - trailing ```` ``` ```` fence
    """
    t = text.strip()
    for _ in range(4):  # bounded — wrappers seen so far nest at most 2 deep
        before = t
        # qwen3-next-80b XML wrapper. The model emits the OpenAI-style
        # tool-call payload `{"name": "AgentOutput", "arguments": {...}}`
        # inside; the unwrapping in `_unwrap_openai_toolcall_payload` below
        # converts it to the bare AgentOutput shape browser_use expects.
        if t.startswith("<tool_call>"):
            t = t[len("<tool_call>"):].lstrip("\n").lstrip("\r\n").strip()
        if t.endswith("</tool_call>"):
            t = t[: -len("</tool_call>")].rstrip()
        # Leading fence variants (full or partial)
        if t.startswith("```json"):
            t = t[len("```json"):].lstrip("\n").strip()
        elif t.startswith("```"):
            t = t[len("```"):].lstrip("\n").strip()
        # Leading bare language hint (Qwen quirk — see 2026-04-23 probe)
        if t.startswith("json\n"):
            t = t[len("json\n"):].strip()
        elif t.startswith("json\r\n"):
            t = t[len("json\r\n"):].strip()
        # Trailing fence
        if t.endswith("```"):
            t = t[: -len("```")].rstrip()
        if t == before:
            break
    return t


def _unwrap_openai_toolcall_payload(d: dict) -> dict:
    """Unwrap an OpenAI-style tool-call envelope to the inner AgentOutput.

    qwen3-next-80b on OR emits content like::

        <tool_call>
        {"name": "AgentOutput", "arguments": {"current_state": {...}, "action": [...]}}
        </tool_call>

    After fence stripping + JSON parse we get the envelope dict. browser_use
    expects the inner ``arguments`` shape (``{"current_state": ..., "action":
    ...}``). Unwrap when the envelope is unambiguous; otherwise return the
    input unchanged so well-formed direct outputs still pass through.
    """
    if (
        isinstance(d, dict)
        and set(d.keys()) == {"name", "arguments"}
        and isinstance(d.get("arguments"), dict)
    ):
        return d["arguments"]
    return d


def _balanced_brace_span(text: str) -> str | None:
    """Return the substring from the first ``{`` to the last ``}``,
    inclusive, or ``None`` if either is absent.

    Not a real parser — we rely on ``json.loads`` to validate. Good
    enough for the Qwen quirk where prose surrounds a valid JSON body.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def tolerant_extract_json_from_model_output(content: Any) -> dict:
    """Replacement for
    ``browser_use.agent.message_manager.utils.extract_json_from_model_output``.

    Args:
        content: raw LLM output — string, Anthropic content-block list,
            or anything with ``str(...)`` semantics.

    Returns:
        A ``dict`` equivalent to what the original parser would return
        for well-formed input.

    Raises:
        ValueError: same message as the original (``'Could not parse
        response.'``) so browser_use's retry path is unchanged.
    """
    # Flatten upstream shapes before trying any parser.
    text_input = _flatten_anthropic_blocks(content)

    # Fast path: defer to the original parser when possible. Keeps
    # Mistral / Kimi / future-model paths byte-identical with pre-patch
    # (R6 regression guard). We catch `AssertionError` too because the
    # upstream parser uses `assert isinstance(result, dict)` to reject
    # non-dict JSON — our fallback translates that to the `ValueError`
    # browser_use's retry path expects.
    if _ORIGINAL is not None:
        try:
            return _ORIGINAL(text_input)
        except (ValueError, AssertionError):
            # Fall through to tolerant strategies below.
            pass

    # Strategy 1 — direct parse after stripping fences + Qwen prefix +
    # `<tool_call>` wrapper. Then unwrap OpenAI-style envelope if present.
    stripped = _strip_fences(text_input)
    if stripped:
        try:
            result = json.loads(stripped)
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
                result = result[0]  # browser_use quirk — upstream pattern at utils.py:44
            if isinstance(result, dict):
                return _unwrap_openai_toolcall_payload(result)
        except json.JSONDecodeError:
            pass

    # Strategy 2 — balanced-brace span from prose.
    span = _balanced_brace_span(text_input)
    if span:
        try:
            result = json.loads(span)
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
                result = result[0]
            if isinstance(result, dict):
                return _unwrap_openai_toolcall_payload(result)
        except json.JSONDecodeError:
            pass

    # Strategy 3 — same balanced-brace extraction on the fence-stripped
    # text (handles ``"```json\\nprose {{json}} more\\n```"``).
    if stripped:
        span = _balanced_brace_span(stripped)
        if span:
            try:
                result = json.loads(span)
                if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
                    result = result[0]
                if isinstance(result, dict):
                    return _unwrap_openai_toolcall_payload(result)
            except json.JSONDecodeError:
                pass

    # All strategies exhausted — log + raise the same ValueError
    # browser_use expects, so its 3-strike retry path is preserved
    # verbatim.
    preview = (text_input or "")[:200]
    logger.warning(
        "[browser_json_extractor] Failed to parse model output after "
        "tolerant strategies; raising to preserve browser_use retry "
        "contract. prefix=%r",
        preview,
    )
    raise ValueError("Could not parse response.")


def install_tolerant_extractor() -> None:
    """Replace
    ``browser_use.agent.message_manager.utils.extract_json_from_model_output``
    with :func:`tolerant_extract_json_from_model_output`.

    Idempotent: calling twice is a no-op on the second call (avoids
    stacking the wrapper on top of itself). Raises ``RuntimeError`` if
    the browser_use API has changed and the function is missing — same
    failure mode as P5's
    ``make_tool_choice_downgrading_chat_openai`` guard at
    ``src/models/openaillm.py:893``, so a library upgrade fails loudly
    instead of silently disabling the patch (R1 in the plan).
    """
    global _ORIGINAL, _PATCHED
    # Fast path: no lock needed for the common "already installed" case.
    if _PATCHED:
        return

    with _INSTALL_LOCK:
        # Re-check under the lock (double-checked locking). Protects against
        # two concurrent callers both passing the unlocked guard above.
        if _PATCHED:
            return

        try:
            from browser_use.agent.message_manager import utils as _bu_utils
        except ImportError as e:  # pragma: no cover — import-level sanity
            raise RuntimeError(
                "install_tolerant_extractor requires browser_use to be "
                "importable; failed with: " + str(e)
            ) from e

        if not hasattr(_bu_utils, "extract_json_from_model_output"):
            raise RuntimeError(
                "browser_use.agent.message_manager.utils no longer exposes "
                "`extract_json_from_model_output`. The Qwen raw-mode parser "
                "patch depends on this symbol. Pin browser_use==0.1.48 in "
                "the environment or port the patch to the new location. "
                "See docs/handoffs/HANDOFF_QWEN_BROWSER_RAW_MODE.md."
            )

        _ORIGINAL = _bu_utils.extract_json_from_model_output
        _bu_utils.extract_json_from_model_output = tolerant_extract_json_from_model_output
        _PATCHED = True
        logger.info(
            "[browser_json_extractor] installed tolerant extractor "
            "(wrapping %s.%s)",
            _bu_utils.__name__,
            "extract_json_from_model_output",
        )


def _reset_for_tests() -> None:
    """Restore the original browser_use function and clear internal
    state. Only intended for unit tests so the autouse fixture can
    guarantee no cross-test leakage (R8 in the plan).
    """
    global _ORIGINAL, _PATCHED
    # Hold the install lock so a concurrent installer can't observe
    # mid-reset state.
    with _INSTALL_LOCK:
        if _ORIGINAL is not None:
            try:
                from browser_use.agent.message_manager import utils as _bu_utils
                _bu_utils.extract_json_from_model_output = _ORIGINAL
            except ImportError:
                pass
        _ORIGINAL = None
        _PATCHED = False


def is_patched() -> bool:
    """Test hook: report whether the module patch is currently live."""
    return _PATCHED
