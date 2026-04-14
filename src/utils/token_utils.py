"""
Token estimation and context pruning helpers for chat messages.

Defaults:
- ``token_estimation_mode`` ``tiktoken``: count text with tiktoken (approximate for non-OpenAI models).
- ``token_estimation_mode`` ``heuristic``: chars / 3.5 (legacy tests).
- Unknown model IDs fall back to ``cl100k_base`` encoding (documented approximation).
"""

from __future__ import annotations

import json
from typing import Any, Literal

import tiktoken

# Intentionally avoid importing src.models here (breaks circular import: utils -> models -> logger -> utils).

TokenEstimationMode = Literal["tiktoken", "heuristic"]

# Defaults aligned with GeneralAgent config getattr fallbacks
DEFAULT_CONTEXT_PRUNE_THRESHOLD_RATIO = 0.85
DEFAULT_CONTEXT_PRUNE_RESERVE_TOKENS = 4096
DEFAULT_CONTEXT_PRUNE_TAIL_SEGMENTS = 4
DEFAULT_CONTEXT_IMAGE_TOKEN_ESTIMATE = 1024
DEFAULT_TOKEN_ESTIMATION_MODE: TokenEstimationMode = "tiktoken"


def normalize_model_id_for_tiktoken(model_id: str | None) -> str:
    """Strip vendor prefixes so ``tiktoken.encoding_for_model`` can resolve known OpenAI names."""
    if not model_id:
        return "gpt-4o"
    s = model_id.strip()
    for prefix in ("anthropic/", "openai/", "google/", "gemini/", "azure/", "vertex_ai/"):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :]
            break
    return s or "gpt-4o"


def _get_tiktoken_encoding(model_id: str | None) -> tiktoken.Encoding:
    name = normalize_model_id_for_tiktoken(model_id)
    try:
        return tiktoken.encoding_for_model(name)
    except KeyError:
        try:
            return tiktoken.encoding_for_model(model_id.strip() if model_id else name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


def get_token_count(prompt: str, model: str = "gpt-4o") -> int:
    """
    Get the number of tokens in a prompt.
    :param prompt: The prompt to count tokens for.
    :param model: The model to use for tokenization. Default is "gpt-4o".
    :return: The number of tokens in the prompt.
    """
    enc = _get_tiktoken_encoding(model)
    return len(enc.encode(prompt))


def extract_text_from_chat_message(msg: Any) -> str:
    """Flatten textual content from a ChatMessage (excludes image bytes)."""
    parts: list[str] = []
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, dict):
                parts.append(str(part))
            else:
                parts.append(str(part))
    else:
        parts.append(str(content or ""))
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            parts.append(_tool_call_to_approximate_string(tc))
    return "\n".join(parts)


def _tool_call_to_approximate_string(tc: Any) -> str:
    fn = getattr(tc, "function", tc)
    name = getattr(fn, "name", "?")
    args = getattr(fn, "arguments", None)
    if isinstance(args, (dict, list)):
        args_s = json.dumps(args)
    else:
        args_s = str(args) if args is not None else ""
    return f"{name}({args_s})"


def count_image_blocks_in_message(msg: Any) -> int:
    n = 0
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                n += 1
    return n


def estimate_messages_tokens(
    messages: list[Any],
    model_id: str | None,
    *,
    mode: TokenEstimationMode = DEFAULT_TOKEN_ESTIMATION_MODE,
    context_image_token_estimate: int = DEFAULT_CONTEXT_IMAGE_TOKEN_ESTIMATE,
) -> int:
    """Estimate total tokens for a message list (content + tool-call strings + image surcharge)."""
    if mode not in ("tiktoken", "heuristic"):
        mode = "tiktoken"
    if mode == "heuristic":
        total_chars = 0
        image_extra = 0
        for msg in messages:
            total_chars += len(extract_text_from_chat_message(msg))
            image_extra += count_image_blocks_in_message(msg) * context_image_token_estimate
        return int(total_chars / 3.5) + image_extra

    enc = _get_tiktoken_encoding(model_id)
    total = 0
    for msg in messages:
        text = extract_text_from_chat_message(msg)
        total += len(enc.encode(text))
        total += count_image_blocks_in_message(msg) * context_image_token_estimate
    return total


def group_messages_for_pruning(messages: list[Any]) -> list[list[Any]]:
    """
    Split messages into segments so Tier B chains (assistant + tool_calls + tool results)
    are never split across segments.
    """
    out: list[list[Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        tcalls = getattr(msg, "tool_calls", None)
        if tcalls:
            ids = {tc.id for tc in tcalls}
            group = [msg]
            i += 1
            while i < len(messages):
                m = messages[i]
                role = m.role if isinstance(m.role, str) else getattr(m.role, "value", str(m.role))
                if role != "tool":
                    break
                tid = getattr(m, "tool_call_id", None)
                if not tid or tid not in ids:
                    break
                group.append(m)
                i += 1
            out.append(group)
        else:
            out.append([msg])
            i += 1
    return out


def _flatten_segments(segments: list[list[Any]]) -> list[Any]:
    flat: list[Any] = []
    for seg in segments:
        flat.extend(seg)
    return flat


def _segment_starts_with_system(segments: list[list[Any]]) -> bool:
    if not segments or not segments[0]:
        return False
    r = segments[0][0].role
    rs = r if isinstance(r, str) else getattr(r, "value", str(r))
    return rs == "system"


def prune_messages_to_budget(
    messages: list[Any],
    model_id: str | None,
    *,
    max_model_len: int = 32768,
    context_prune_threshold_ratio: float = DEFAULT_CONTEXT_PRUNE_THRESHOLD_RATIO,
    context_prune_reserve_tokens: int = DEFAULT_CONTEXT_PRUNE_RESERVE_TOKENS,
    context_prune_tail_segments: int = DEFAULT_CONTEXT_PRUNE_TAIL_SEGMENTS,
    token_estimation_mode: TokenEstimationMode = DEFAULT_TOKEN_ESTIMATION_MODE,
    context_image_token_estimate: int = DEFAULT_CONTEXT_IMAGE_TOKEN_ESTIMATE,
) -> list[Any]:
    """
    If estimated tokens exceed the effective budget, drop middle conversation segments (never splitting
    assistant+tool chains), inserting a single user placeholder. Returns ``messages`` unchanged if under budget.
    """
    if token_estimation_mode not in ("tiktoken", "heuristic"):
        token_estimation_mode = "tiktoken"
    effective_budget = int(max_model_len * context_prune_threshold_ratio) - context_prune_reserve_tokens
    if effective_budget <= 0:
        effective_budget = max(1024, int(max_model_len * 0.5))

    est = estimate_messages_tokens(
        messages,
        model_id,
        mode=token_estimation_mode,
        context_image_token_estimate=context_image_token_estimate,
    )
    if est <= effective_budget:
        return messages

    segments = group_messages_for_pruning(messages)
    n = len(segments)
    if n == 0:
        return messages

    head_len = 1 if _segment_starts_with_system(segments) else 0
    tail_k = min(context_prune_tail_segments, n)

    # Middle spans segment indices [head_len, n - tail_k)
    middle_start = head_len
    middle_end = n - tail_k
    if middle_end <= middle_start:
        # Nothing to drop between head and tail; try shrinking tail segments
        return _aggressive_prune_tail(
            messages,
            model_id,
            effective_budget,
            token_estimation_mode,
            context_image_token_estimate,
        )

    omitted = middle_end - middle_start
    head_segs = segments[:middle_start]
    tail_segs = segments[middle_end:]
    # Late import: only when building placeholder (avoids circular import at module load).
    from src.models.base import ChatMessage, MessageRole

    placeholder = ChatMessage(
        role=MessageRole.USER,
        content=[
            {
                "type": "text",
                "text": f"[Earlier conversation truncated for context length: {omitted} segment(s) omitted.]",
            }
        ],
    )
    result = _flatten_segments(head_segs) + [placeholder] + _flatten_segments(tail_segs)

    new_est = estimate_messages_tokens(
        result,
        model_id,
        mode=token_estimation_mode,
        context_image_token_estimate=context_image_token_estimate,
    )
    if new_est > effective_budget:
        return _aggressive_prune_tail(
            result,
            model_id,
            effective_budget,
            token_estimation_mode,
            context_image_token_estimate,
        )
    return result


def _aggressive_prune_tail(
    messages: list[Any],
    model_id: str | None,
    effective_budget: int,
    token_estimation_mode: TokenEstimationMode,
    context_image_token_estimate: int,
) -> list[Any]:
    """If still over budget, drop segments after the head until under budget, then truncate long text."""
    if estimate_messages_tokens(
        messages,
        model_id,
        mode=token_estimation_mode,
        context_image_token_estimate=context_image_token_estimate,
    ) <= effective_budget:
        return messages

    segments = group_messages_for_pruning(messages)
    if not segments:
        return messages

    head_len = 1 if _segment_starts_with_system(segments) else 0
    cur = [list(s) for s in segments]
    while len(cur) > head_len + 1:
        est = estimate_messages_tokens(
            _flatten_segments(cur),
            model_id,
            mode=token_estimation_mode,
            context_image_token_estimate=context_image_token_estimate,
        )
        if est <= effective_budget:
            return _flatten_segments(cur)
        # Drop the first segment after the preserved head (oldest middle/tail chunk)
        cur = cur[:head_len] + cur[head_len + 1 :]

    flat = _flatten_segments(cur)
    if estimate_messages_tokens(
        flat,
        model_id,
        mode=token_estimation_mode,
        context_image_token_estimate=context_image_token_estimate,
    ) <= effective_budget:
        return flat
    return _truncate_longest_messages(
        flat,
        model_id,
        effective_budget,
        token_estimation_mode,
        context_image_token_estimate,
    )


def _truncate_longest_messages(
    messages: list[Any],
    model_id: str | None,
    effective_budget: int,
    token_estimation_mode: TokenEstimationMode,
    context_image_token_estimate: int,
) -> list[Any]:
    """Truncate the longest text-bearing messages until estimate fits budget."""
    from src.models.base import ChatMessage

    msgs = list(messages)
    for _ in range(len(msgs) * 2):
        est = estimate_messages_tokens(
            msgs,
            model_id,
            mode=token_estimation_mode,
            context_image_token_estimate=context_image_token_estimate,
        )
        if est <= effective_budget:
            return msgs
        best_i = -1
        best_len = 0
        for i, m in enumerate(msgs):
            t = extract_text_from_chat_message(m)
            if len(t) > best_len:
                best_len = len(t)
                best_i = i
        if best_i < 0 or best_len < 100:
            return msgs
        m = msgs[best_i]
        text = extract_text_from_chat_message(m)
        truncated = text[:250] + " ... [truncated] ... " + text[-200:]
        msgs[best_i] = ChatMessage(
            role=m.role,
            content=[{"type": "text", "text": truncated}],
            tool_calls=None,
            tool_call_id=getattr(m, "tool_call_id", None),
            raw=getattr(m, "raw", None),
        )
    return msgs
