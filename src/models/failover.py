"""FailoverModel — transparent primary → backup switch for OpenAI-compatible models.

Designed for the Qwen DashScope-native → OpenRouter fallback: the primary client
hits DashScope (free tier), and on a quota-exhaustion error we flip permanently
(within the process) to the OpenRouter backup. The model behaves like a regular
`OpenAIServerModel` to the agent layer — same `generate`, `generate_stream`,
`model_id`, `_last_input_token_count` surface.

Switch semantics:
- One-way: once switched, we do not try the primary again in this process.
- Permanent per process: each eval run is a fresh process, so the next invocation
  will retry the primary. This is intentional — quotas reset daily on DashScope.
- Detection: HTTP 429 with common free-tier quota phrases, or HTTP 402/403 with
  billing language. Transient 429s from short rate-limit windows are NOT treated
  as exhaustion; only explicit quota-exceeded responses trigger the switch.
- The underlying `OpenAIServerModel`'s own retry loop (5 attempts with backoff)
  still runs first; failover only fires when those retries are themselves exhausted
  AND the terminal error matches the quota pattern.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.models.base import ChatMessage, ChatMessageStreamDelta
    from src.models.openaillm import OpenAIServerModel

logger = logging.getLogger(__name__)


# Case-insensitive substrings that indicate the primary's free tier is exhausted
# and falling through to retry will not help. Observed patterns as of 2026-04:
#   DashScope: "quota exceeded", "free tier", "insufficient balance"
#   Generic:   "billing", "payment required"
_QUOTA_PATTERNS = [
    r"quota\s+exceeded",
    r"quota\s+has\s+been\s+exceeded",
    r"free\s+tier\s+(quota|limit)",
    r"insufficient\s+(balance|quota)",
    r"billing",
    r"payment\s+required",
    r"throttling\.freetier",
    r"exceeded\s+your\s+current\s+quota",
]

_QUOTA_REGEX = re.compile("|".join(_QUOTA_PATTERNS), re.IGNORECASE)


def _looks_like_quota_exhaustion(exc: BaseException) -> bool:
    """Return True if the exception matches a provider quota-exhaustion signal.

    Intentionally conservative: only known strings trigger the switch. A plain
    429 rate-limit (handled by the retry loop upstream) falls through.
    """
    # Check HTTP status first where available
    status = getattr(exc, "status_code", None)
    if status in (402, 403):  # billing / forbidden
        return True
    # Inspect message / response body
    body = getattr(exc, "message", None) or str(exc)
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = f"{body} {resp.text}"  # type: ignore[attr-defined]
        except Exception:
            pass
    return bool(_QUOTA_REGEX.search(body or ""))


class FailoverModel:
    """Two-model failover wrapper. Duck-types `OpenAIServerModel`.

    Parameters:
        primary: The preferred model (e.g. DashScope native).
        backup:  The fallback model (e.g. OpenRouter mirror).
        alias:   Public `model_id` exposed to the agent layer / logger.
    """

    def __init__(self, primary: OpenAIServerModel, backup: OpenAIServerModel, alias: str):
        self._primary = primary
        self._backup = backup
        self._switched = False
        self.model_id = alias
        # The two children maintain their own token counters; we expose whichever
        # was active on the last call. Initialized here so attribute reads before
        # any call still work.
        self._last_input_token_count: int = 0
        self._last_output_token_count: int = 0

    # ------------------------------------------------------------------
    # Active-child selection
    # ------------------------------------------------------------------
    @property
    def _active(self) -> OpenAIServerModel:
        return self._backup if self._switched else self._primary

    def _maybe_switch(self, exc: BaseException) -> bool:
        """Return True if we just switched and the caller should retry on backup."""
        if self._switched:
            return False
        if not _looks_like_quota_exhaustion(exc):
            return False
        logger.warning(
            "[FailoverModel:%s] primary (%s) quota exhausted — switching to backup (%s). "
            "Reason: %s",
            self.model_id,
            self._primary.model_id,
            self._backup.model_id,
            exc,
        )
        self._switched = True
        return True

    # ------------------------------------------------------------------
    # Proxied API surface
    # ------------------------------------------------------------------
    async def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        try:
            result = await self._active.generate(
                messages=messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )
        except Exception as exc:
            if self._maybe_switch(exc):
                # Retry once on backup
                result = await self._active.generate(
                    messages=messages,
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
            else:
                raise
        # Bubble token counts up for the agent logger.
        self._last_input_token_count = getattr(self._active, "_last_input_token_count", 0)
        self._last_output_token_count = getattr(self._active, "_last_output_token_count", 0)
        return result

    def generate_stream(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        """Streaming failover. Detection happens on the FIRST emitted chunk —
        if the primary raises before yielding anything, we switch and re-open
        the stream on the backup. Mid-stream failures are not recoverable and
        propagate unchanged (same guarantee as the underlying SDK).
        """
        def _stream_from(model: OpenAIServerModel):
            for delta in model.generate_stream(
                messages=messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            ):
                yield delta

        gen = _stream_from(self._active)
        try:
            first = next(gen)
        except StopIteration:
            return
        except Exception as exc:
            if self._maybe_switch(exc):
                yield from _stream_from(self._active)
                return
            raise
        yield first
        yield from gen

    async def __call__(self, *args, **kwargs) -> ChatMessage:
        return await self.generate(*args, **kwargs)

    # Allow read-through for any other attribute an agent might peek at
    # (flatten_messages_as_text, message_manager, kwargs, etc.). We route to
    # the active child so telemetry reflects the currently-serving backend.
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._active, name)
