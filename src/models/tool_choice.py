"""Hybrid `tool_choice` dispatch for OpenAI-compatible model wrappers.

Agents default to ``tool_choice="required"`` so every step emits a tool call.
Some providers reject that value (e.g. OpenRouter returns 404 "No endpoints
found that support the provided 'tool_choice' value" for the Qwen family; some
Gemma 4 routings are similarly unreliable). For those, we downgrade to
``"auto"`` and let the caller's retry guard re-prompt plain-text replies back
into tool calls.

The lookup keys on the **wire id** stored as ``Model.model_id`` at the agent
call site (e.g. ``qwen/qwen3.6-plus``, ``google/gemma-4-31b-it``,
``moonshotai/kimi-k2.5``), *not* the registration alias (``or-qwen*`` /
``or-gemma-4-31b-it`` / ``or-kimi-k2.5``). See
``docs/handoffs/HANDOFF_PENDING_KIMI_AND_TOOL_CHOICE.md`` for the full rationale.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Named wire ids that must be downgraded to ``"auto"``. Kept separate from the
# prefix rule below so each entry can be audited and removed independently once
# a live probe confirms ``"required"`` works.
MODELS_REJECTING_REQUIRED: set[str] = {
    # D5 live smoke probe (2026-04-18) against OR `google/gemma-4-31b-it` with
    # the DeepInfra/Together provider pin + `reasoning.enabled=false` passed
    # `tool_choice="required"` cleanly: HTTP 200, finish_reason="tool_calls",
    # proper chat-template rendering (no `<|tool_call>` leak into `content`).
    # Gemma 4 31B is therefore NOT in the defensive set. Keep the set declared
    # as an extensibility hook for future named entries that don't fit the
    # Qwen prefix rule below.
}


# Wire-id prefixes subject to blanket downgrade. D3 (2026-04-18) applies this
# to the whole Qwen ecosystem: OpenRouter's backend routing for Qwen is
# inconsistent across providers and a slug that honours ``"required"`` today
# can silently fail tomorrow when OR shifts backends.
_AUTO_WIRE_PREFIXES: tuple[str, ...] = (
    "qwen/",  # all OpenRouter Qwen wire slugs we register or might register
)


# Dedup guard for the one-per-(run, model) INFO log emitted by dispatch
# consumers. Bounded by the number of registered models, so unbounded growth
# is not a concern.
_LOGGED_DOWNGRADES: set[str] = set()


def pick_tool_choice(
    model_id: str | None,
    default: str | dict | None = "required",
) -> str | dict | None:
    """Resolve ``tool_choice`` for a given wire-id model.

    Args:
        model_id: the wire id stored on ``Model.model_id`` at the call site.
        default: the caller's desired value when no rule matches (almost
            always the agent default ``"required"``).

    Returns:
        ``"auto"`` when the model is in :data:`MODELS_REJECTING_REQUIRED` or
        matches one of :data:`_AUTO_WIRE_PREFIXES`, else ``default`` unchanged.
    """
    if not model_id:
        return default
    if model_id in MODELS_REJECTING_REQUIRED:
        return "auto"
    if model_id.startswith(_AUTO_WIRE_PREFIXES):
        return "auto"
    return default


def log_downgrade_once(model_id: str) -> None:
    """Emit a single INFO log line the first time a model is downgraded.

    Consumers should call this only when the dispatcher actually changed the
    caller's value (``"required"`` -> ``"auto"``), not on every tool call.
    """
    if model_id in _LOGGED_DOWNGRADES:
        return
    _LOGGED_DOWNGRADES.add(model_id)
    logger.info(
        "[tool_choice] %s -> auto (model in auto-dispatch set)", model_id,
    )


def _reset_logged_downgrades_for_tests() -> None:
    """Reset the one-shot log guard. Only intended for unit tests."""
    _LOGGED_DOWNGRADES.clear()
