"""Pure filter for the premature / duplicate `final_answer_tool` guard.

Separated into its own module so tests can import the logic without
pulling in the full `GeneralAgent` stack (models, tools, registry, prompts).
"""

from __future__ import annotations

from typing import Literal

GuardStatus = Literal["none", "premature", "duplicate"]


def apply_final_answer_guard(tool_calls: list) -> tuple[list, GuardStatus]:
    """Partition a model turn's tool calls and decide what to keep.

    Protects against the small-model failure mode where `final_answer_tool`
    is emitted in the same turn as other tool calls (often with a fabricated
    `answer` argument) — and against the rarer case where multiple
    `final_answer_tool` calls appear with no siblings.

    Args:
        tool_calls: raw `ChatMessage.tool_calls` list (any order, any count).
            Each entry is a duck-typed object exposing
            `.function.name` and `.function.arguments`.

    Returns:
        `(effective_tool_calls, status)` where:
            * `effective_tool_calls` is the subset of `tool_calls` to execute.
              For `"none"` status the returned list is identity-equal to the
              input (caller can use `is` to skip mutation work).
            * `status` is one of `"none"`, `"premature"`, `"duplicate"`.

    The caller is responsible for (a) logging the guard event,
    (b) syncing `chat_message.tool_calls` so downstream memory-replay sees
    only the kept calls, and (c) processing `effective_tool_calls` as usual.
    """
    final_answer_tcs = [tc for tc in tool_calls if tc.function.name == "final_answer_tool"]
    other_tcs = [tc for tc in tool_calls if tc.function.name != "final_answer_tool"]

    if other_tcs and final_answer_tcs:
        return other_tcs, "premature"
    if len(final_answer_tcs) > 1:
        return [], "duplicate"
    return tool_calls, "none"
