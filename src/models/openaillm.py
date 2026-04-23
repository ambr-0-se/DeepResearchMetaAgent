import asyncio
import logging
import random
from typing import Any
from collections.abc import Generator

from src.models.base import (ApiModel,
                             ChatMessage,
                             tool_role_conversions,
                             MessageRole,
                             TokenUsage,
                             ChatMessageStreamDelta,
                             ChatMessageToolCallStreamDelta)
from src.models.message_manager import MessageManager
from src.models.tool_choice import log_downgrade_once, pick_tool_choice

logger = logging.getLogger(__name__)

_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0   # seconds

# Single-request read timeout enforced on every async chat-completion call.
# Guards against silent SSE / HTTP-read stalls on OpenAI-compatible providers
# (observed on Moonshot AI via OpenRouter for Kimi K2.5: an otherwise-valid
# completion can wedge mid-stream with no bytes, no TCP close, no error
# from the SDK. Previously this consumed the outer per-question 1200s
# timeout in run_gaia.py — I2 2026-04-19 saw 11/12 Kimi Qs hit that wall
# with 12+ min of log silence.)
#
# 120s is generous enough for long legitimate generations (GAIA Qs with
# large context) while still turning a true hang into a retryable error
# ~10x faster than the per-Q timeout. Surfaces as asyncio.TimeoutError
# which the tenacity retry loop below converts to a new attempt.
_CHAT_COMPLETION_TIMEOUT_S = 120.0
_RETRY_MAX_DELAY = 60.0   # seconds


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter: uniform(0, min(cap, base * 2^attempt))."""
    ceiling = min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * (2 ** attempt))
    return random.uniform(0, ceiling)


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After value (seconds) from an API error response, if present."""
    try:
        headers = getattr(exc, "response", None) and exc.response.headers  # type: ignore[union-attr]
        if headers:
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value:
                return float(value)
    except Exception:
        pass
    return None


class OpenAIServerModel(ApiModel):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        http_client: Any = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs,
        ):
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )

        self.http_client = http_client

        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }

        self.message_manager = MessageManager(model_id=model_id)

        # Non-OpenAI vendor-specific request fields (e.g. DashScope `enable_thinking`,
        # Moonshot `thinking={"type":"disabled"}`). Injected into every completion call
        # via self.kwargs so `_prepare_completion_kwargs` forwards it to the SDK, which
        # passes it through as the `extra_body` parameter on chat.completions.create.
        if extra_body:
            kwargs = {**kwargs, "extra_body": dict(extra_body)}

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):

        if self.http_client:
            return self.http_client
        else:
            try:
                import openai
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
                ) from e

            return openai.OpenAI(
                **self.client_kwargs
            )

    def _prepare_completion_kwargs(
            self,
            messages: list[ChatMessage],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Any] | None = None,
            custom_role_conversions: dict[str, str] | None = None,
            convert_images_to_image_urls: bool = False,
            tool_choice: str | dict | None = "required",  # Configurable tool_choice parameter
            **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, response_format, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        flatten_messages_as_text = kwargs.pop("flatten_messages_as_text", self.flatten_messages_as_text)
        messages_as_dicts = self.message_manager.get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages_as_dicts,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        # Handle tools parameter
        if tools_to_call_from:
            tools_config = {
                "tools": [self.message_manager.get_tool_json_schema(tool, model_id=self.model_id) for tool in
                          tools_to_call_from],
            }
            # Hybrid tool_choice dispatch (D3/D5): downgrade "required" to
            # "auto" for models whose OR backends reject the forced value.
            # Wire-id keyed; see src/models/tool_choice.py. The retry guard in
            # GeneralAgent._step_stream / ToolCallingAgent._step_stream catches
            # plain-text replies when we end up on the "auto" path.
            resolved_tool_choice = pick_tool_choice(self.model_id, tool_choice)
            if resolved_tool_choice != tool_choice and resolved_tool_choice == "auto":
                log_downgrade_once(self.model_id or "<unknown>")
            if resolved_tool_choice is not None:
                tools_config["tool_choice"] = resolved_tool_choice
            completion_kwargs.update(tools_config)

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        completion_kwargs = self.message_manager.get_clean_completion_kwargs(completion_kwargs)

        return completion_kwargs

    def generate_stream(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            http_client=self.http_client,
            **kwargs,
        )
        for event in self.client.chat.completions.create(
            **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if event.usage:
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    # Some providers stream reasoning on a parallel `reasoning_content`
                    # delta (DeepSeek-reasoner, Qwen3-thinking). Capture it alongside
                    # the content delta so downstream agglomeration keeps them distinct.
                    reasoning_delta = getattr(choice.delta, "reasoning_content", None)
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        reasoning_content=reasoning_delta,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")

    async def generate(
            self,
            messages: list[ChatMessage],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Any] | None = None,
            **kwargs,
    ) -> ChatMessage:
        import openai

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        last_exc: Exception | None = None
        for attempt in range(_RETRY_MAX_ATTEMPTS + 1):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**completion_kwargs),
                    timeout=_CHAT_COMPLETION_TIMEOUT_S,
                )
                # Some providers via OpenRouter (observed: Gemma-4 31B via
                # DeepInfra) return `usage=None` intermittently even though the
                # completion is valid. Previously we crashed here with
                # `'NoneType' object has no attribute 'prompt_tokens'`, which
                # surfaced as `agent_error` at question level (I2 2026-04-19:
                # 1/48 smoke Qs). Treat a missing usage block as zero tokens
                # rather than a hard error — accounting is approximate but the
                # completion is still usable.
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                self._last_input_token_count = prompt_tokens
                self._last_output_token_count = completion_tokens
                # Defensive guard on `response.choices`. Observed 2026-04-19
                # during E0 training: some OR providers (notably Qwen/Gemma
                # on certain DeepInfra routes) return `choices=None` or
                # `choices=[]` intermittently on otherwise-valid completions,
                # which crashed here with `'NoneType' object is not
                # subscriptable` (or `'list' object has no attribute` when
                # downstream code expected a dict at [0]). That bubbled up
                # as `AgentGenerationError` at question level and lost the
                # whole Q. Converting this to a retryable `APIStatusError`
                # routes it through the existing retry loop instead.
                choices = getattr(response, "choices", None)
                if not choices:
                    # Raise as APIConnectionError so the existing retry
                    # branch (see except below) picks it up and retries
                    # with backoff rather than surfacing as a Q-level error.
                    raise openai.APIConnectionError(
                        message=(
                            f"Provider returned no choices (response.choices="
                            f"{choices!r}) for model '{self.model_id}'"
                        ),
                        request=None,  # type: ignore[arg-type]
                    )
                # Capture reasoning_content directly from the SDK message object before
                # dumping: include={"role","content","tool_calls"} would silently drop
                # it, which breaks DeepSeek-reasoner (and other thinking models) on the
                # next tool-loop turn since the provider requires it to be echoed back.
                msg_obj = choices[0].message
                msg_dict = msg_obj.model_dump(include={"role", "content", "tool_calls"})
                reasoning = getattr(msg_obj, "reasoning_content", None)
                if reasoning:
                    msg_dict["reasoning_content"] = reasoning
                return ChatMessage.from_dict(
                    msg_dict,
                    raw=response,
                    token_usage=TokenUsage(
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                    ),
                )
            except openai.RateLimitError as e:
                last_exc = e
                if attempt == _RETRY_MAX_ATTEMPTS:
                    break
                # Honour Retry-After header when present, else exponential backoff
                retry_after = _parse_retry_after(e)
                delay = retry_after if retry_after else _backoff(attempt)
                logger.warning(
                    f"Rate limit hit (429) for model '{self.model_id}', "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS})"
                )
                await asyncio.sleep(delay)
            except openai.APIStatusError as e:
                last_exc = e
                if e.status_code < 500 or attempt == _RETRY_MAX_ATTEMPTS:
                    raise
                delay = _backoff(attempt)
                logger.warning(
                    f"Server error {e.status_code} for model '{self.model_id}', "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS})"
                )
                await asyncio.sleep(delay)
            except openai.APIConnectionError as e:
                last_exc = e
                if attempt == _RETRY_MAX_ATTEMPTS:
                    break
                delay = _backoff(attempt)
                logger.warning(
                    f"Connection error for model '{self.model_id}', "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS})"
                )
                await asyncio.sleep(delay)
            except asyncio.TimeoutError as e:
                # The `asyncio.wait_for` wrapper bounded a single chat
                # completion at _CHAT_COMPLETION_TIMEOUT_S. This catches silent
                # SSE/HTTP-read stalls (Moonshot AI via OpenRouter, observed
                # I2 2026-04-19) and converts them into a normal retry — one
                # hung request no longer wedges an entire question.
                last_exc = e
                if attempt == _RETRY_MAX_ATTEMPTS:
                    break
                delay = _backoff(attempt)
                logger.warning(
                    f"Chat completion timed out after {_CHAT_COMPLETION_TIMEOUT_S:.0f}s "
                    f"for model '{self.model_id}', retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS})"
                )
                await asyncio.sleep(delay)

        raise last_exc

    async def __call__(self, *args, **kwargs) -> ChatMessage:
        """
        Call the model with the given arguments.
        This is a convenience method that calls `generate` with the same arguments.
        """
        return await self.generate(*args, **kwargs)


# ===========================================================================
# Multi-key round-robin — §P4 of HANDOFF_THROUGHPUT_REFACTOR.md
# ===========================================================================
#
# Mistral's free tier is 5 req/min per API key. E0 v3 evidence: 9,286 SSE
# streaming stalls (`Chat completion timed out after 120s`) + 7 explicit
# 429s across 80 training questions against a single key. With two keys
# and round-robin dispatch the effective ceiling doubles to 10 req/min.
#
# Design
# ------
# The existing retry loop in `OpenAIServerModel.generate()` already handles
# `RateLimitError` with Retry-After, exponential backoff, and bounded
# attempts. This refactor reuses that loop — rotation simply changes which
# underlying AsyncOpenAI client each attempt targets. We do NOT add a second
# retry loop.
#
# Two injection points:
#   - native path: `KeyRotatingOpenAIServerModel.create_client()` returns
#     a `_RotatingAsyncClientProxy` that duck-types the subset of
#     `AsyncOpenAI` the `generate()` / `generate_stream()` paths touch
#     (`.chat.completions.create`). Each call picks the next ready key;
#     on `RateLimitError` the picked key's cooldown is set from
#     Retry-After (or a default) before the exception re-raises into the
#     existing retry loop.
#
#   - LangChain path: `KeyRotatingChatOpenAI` — `auto_browser_use_tool`
#     uses LangChain's `ChatOpenAI` which browser-use invokes via
#     `.ainvoke` / `.bind_tools` / etc. We wrap N `ChatOpenAI` instances
#     and delegate the common methods; anything else falls through via
#     `__getattr__` onto the first instance (documented caveat).

# Cooldown on a 429 when no Retry-After header is present. Mistral free
# tier's rate-limit window is one minute / 5 requests = 12s per slot;
# +1s slack for clock skew.
_DEFAULT_COOLDOWN_SECS = 13.0

# Max number of suffixed env vars to scan in `_load_suffix_keys`.
# `MISTRAL_API_KEY`, `MISTRAL_API_KEY_2`, … `MISTRAL_API_KEY_{_MAX_KEY_SUFFIX}`.
# Bounded so a typo-forever-loop can't eat startup time.
_MAX_KEY_SUFFIX = 8


def _load_suffix_keys(base_env_name: str, placeholder: str = "") -> list[str]:
    """Load N API keys from env vars with the `BASE`, `BASE_2`, `BASE_3`, …
    suffix convention. Stops at the first unset / placeholder.

    Returns an empty list when none are set (caller is responsible for
    the "no keys → skip registration" warning).

    Consistency note: the base name is **unsuffixed** (`MISTRAL_API_KEY`),
    then `_2`, `_3`, … Matches how most providers document a "primary +
    additional keys" pattern without forcing the user to rename the key
    they already had.
    """
    import os

    keys: list[str] = []

    primary = os.getenv(base_env_name, placeholder) or ""
    if primary and primary != placeholder:
        keys.append(primary)

    # Stop at the first unset / placeholder suffix so an operator who
    # sets MISTRAL_API_KEY + MISTRAL_API_KEY_2 + MISTRAL_API_KEY_4
    # doesn't accidentally rotate through a key they forgot to drop at
    # _3 — safer to be contiguous-only.
    for i in range(2, _MAX_KEY_SUFFIX + 1):
        k = os.getenv(f"{base_env_name}_{i}", placeholder) or ""
        if not k or k == placeholder:
            break
        keys.append(k)

    return keys


class _KeyPoolState:
    """Shared rotation + cooldown bookkeeping. One instance per logical
    model (native + LangChain wrappers share the SAME pool so cooldowns
    set on one dispatch path are honoured by the other — they are the
    same upstream keys).
    """

    def __init__(self, api_keys: list[str]) -> None:
        assert len(api_keys) >= 1, "_KeyPoolState requires >= 1 key"
        self._n = len(api_keys)
        # Don't store the keys in a named attribute to minimise accidental
        # log exposure — hand them out only via index to _create_client.
        self._keys: tuple[str, ...] = tuple(api_keys)
        self._cool_until: list[float] = [0.0] * self._n
        self._next_idx = 0

    @property
    def n(self) -> int:
        return self._n

    def _key(self, idx: int) -> str:
        """Return the API key at `idx`. Private — do not log the result.
        Underscore-prefixed to match the rest of the internal state
        (`_keys`, `_cool_until`, `_next_idx`); external callers that need
        to distinguish between keys should pass indices, not values.
        """
        return self._keys[idx]

    def pick_index(self) -> int:
        """Round-robin pick of the next ready key. If all keys are cooling,
        return the one with the earliest cool_until so the caller's wait
        is at most the shortest remaining cooldown.
        """
        now = _monotonic_now()
        # Try up to n positions starting from the round-robin cursor.
        for _ in range(self._n):
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % self._n
            if self._cool_until[idx] <= now:
                return idx
        # All cooling — return earliest-ready.
        return min(range(self._n), key=lambda i: self._cool_until[i])

    def mark_cooldown(self, idx: int, seconds: float) -> None:
        """Mark `idx` as unavailable for the next `seconds` seconds."""
        self._cool_until[idx] = _monotonic_now() + max(0.0, float(seconds))

    def cool_until_raw(self, idx: int) -> float:
        """Test hook — monotonic timestamp at which `idx` becomes usable."""
        return self._cool_until[idx]


def _monotonic_now() -> float:
    """Indirection for test patchability."""
    import time
    return time.monotonic()


def _extract_cooldown_from_rate_limit(exc: Exception) -> float:
    """Pick a cooldown duration from a `RateLimitError`. Prefer the
    Retry-After header when the provider returns one; else use the
    `_DEFAULT_COOLDOWN_SECS` constant.
    """
    retry_after = _parse_retry_after(exc)
    return retry_after if retry_after is not None else _DEFAULT_COOLDOWN_SECS


class _RotatingAsyncCompletions:
    """Duck-types `AsyncOpenAI.chat.completions`: exposes `.create` as an
    async method that picks a client, delegates, and records cooldown on
    `RateLimitError`.
    """

    def __init__(self, owner: "KeyRotatingOpenAIServerModel") -> None:
        self._owner = owner

    async def create(self, **kwargs):  # noqa: ANN202 — matches SDK shape
        import openai
        idx = self._owner._pool.pick_index()
        client = self._owner._get_async_client(idx)
        try:
            return await client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            self._owner._pool.mark_cooldown(
                idx, _extract_cooldown_from_rate_limit(e)
            )
            raise


class _RotatingAsyncChat:
    def __init__(self, owner: "KeyRotatingOpenAIServerModel") -> None:
        self.completions = _RotatingAsyncCompletions(owner)


class _RotatingAsyncClientProxy:
    """Duck-types the subset of `openai.AsyncOpenAI` that the parent
    model's `generate()` / `generate_stream()` actually use.

    Limitations (documented, not a bug):
      - No sync `openai.OpenAI` proxy. If a caller expects a sync client
        it should use the plain `OpenAIServerModel` with a single key.
        The Mistral registration path exclusively uses AsyncOpenAI so
        this is moot.
      - Attributes other than `.chat` are not exposed; accessing them
        raises `AttributeError` loudly so a future use of a different
        SDK surface (say `.embeddings`) is caught rather than silently
        defaulting to one key.
    """

    def __init__(self, owner: "KeyRotatingOpenAIServerModel") -> None:
        self.chat = _RotatingAsyncChat(owner)


class KeyRotatingOpenAIServerModel(OpenAIServerModel):
    """`OpenAIServerModel` variant that rotates across N API keys.

    Drop-in usage:

        m = KeyRotatingOpenAIServerModel(
            model_id="mistral-small-2603",
            api_base="https://api.mistral.ai/v1",
            api_keys=["KEY1", "KEY2"],
            custom_role_conversions=...,
        )
        self.registed_models["mistral-small"] = m

    Semantics
    ---------
    - Each call to `self.client.chat.completions.create(...)` picks a
      ready key via round-robin; keys are marked cooling on
      `RateLimitError` (Retry-After header if present, else 13 s).
    - The existing retry loop in `OpenAIServerModel.generate()` is
      reused — each retry attempt naturally lands on a potentially
      different key.
    - When only one key is available this class still works, but
      registration should instead use plain `OpenAIServerModel` to
      keep logs + telemetry honest; `_register_mistral_models` enforces
      this branching.
    """

    def __init__(
        self,
        model_id: str,
        api_keys: list[str],
        api_base: str | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        extra_body: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if not api_keys:
            raise ValueError("KeyRotatingOpenAIServerModel requires >= 1 api_key")

        # Parent expects a single `api_key` for bookkeeping + `http_client`
        # or `client_kwargs` to build its client. We pass key[0] for
        # compat (never used for real requests because create_client()
        # below returns a rotating proxy) and build the pool + async
        # clients ourselves.
        self._pool = _KeyPoolState(api_keys)

        # Lazily-built AsyncOpenAI clients, one per key. Built on first
        # access so module import doesn't network / auth.
        self._async_clients: dict[int, Any] = {}
        self._api_base = api_base

        super().__init__(
            model_id=model_id,
            api_base=api_base,
            api_key=api_keys[0],
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            extra_body=extra_body,
            **kwargs,
        )

        logger.info(
            f"[KeyRotatingOpenAIServerModel] model={model_id!r} "
            f"registered with {self._pool.n} keys"
        )

    # ---- client seam --------------------------------------------------

    def create_client(self):  # noqa: D401 — inherited name
        """Return the rotating proxy. Parent's `__init__` assigns the
        result to `self.client`; every call site in `generate()` /
        `generate_stream()` goes through the proxy transparently.
        """
        return _RotatingAsyncClientProxy(self)

    def _get_async_client(self, idx: int):
        """Lazily build + cache an `openai.AsyncOpenAI` for key `idx`.
        Never logs the key material.
        """
        if idx in self._async_clients:
            return self._async_clients[idx]
        import openai
        client = openai.AsyncOpenAI(
            api_key=self._pool._key(idx),
            base_url=self._api_base,
        )
        self._async_clients[idx] = client
        return client


class KeyRotatingChatOpenAI:
    """LangChain wrapper that rotates across N `ChatOpenAI` instances.

    Why not subclass `ChatOpenAI` directly: `ChatOpenAI` uses Pydantic
    for its model fields, which makes attribute overrides brittle. A
    thin delegator is easier to test and easier to reason about.

    Methods commonly called by `browser_use.Agent` are implemented
    explicitly so rotation happens per-call. Anything else falls
    through via `__getattr__` onto a stable "primary" instance, which
    means calls that reach through `__getattr__` are NOT rotating —
    documented caveat. If a future user of this wrapper exercises a
    method that should rotate, add it to this class explicitly.

    Rotation scope: `ainvoke`, `invoke`, `astream`, `stream`. Cooldown
    shares the same pool semantics as the native wrapper — on a
    `RateLimitError` the picked index is marked cooling.
    """

    def __init__(
        self,
        model: str,
        api_keys: list[str],
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        if not api_keys:
            raise ValueError("KeyRotatingChatOpenAI requires >= 1 api_key")

        from langchain_openai import ChatOpenAI

        self._pool = _KeyPoolState(api_keys)
        self._instances = [
            ChatOpenAI(model=model, api_key=k, base_url=base_url, **kwargs)
            for k in api_keys
        ]
        self._model = model
        # Mirror the model_name / model_id surface a few callers read.
        self.model_name = model
        self.model = model

        logger.info(
            f"[KeyRotatingChatOpenAI] model={model!r} "
            f"registered with {self._pool.n} keys"
        )

    # ---- rotation helpers ---------------------------------------------

    def _pick(self):
        idx = self._pool.pick_index()
        return idx, self._instances[idx]

    def _handle_exc(self, idx: int, exc: BaseException) -> None:
        """Mark cooldown on a rate-limit-shaped error; otherwise no-op."""
        try:
            import openai
            if isinstance(exc, openai.RateLimitError):
                self._pool.mark_cooldown(
                    idx, _extract_cooldown_from_rate_limit(exc)
                )
        except Exception:  # openai not importable? ignore silently
            pass

    # ---- dispatching methods ------------------------------------------

    async def ainvoke(self, *args, **kwargs):  # noqa: ANN202
        idx, inst = self._pick()
        try:
            return await inst.ainvoke(*args, **kwargs)
        except BaseException as e:
            self._handle_exc(idx, e)
            raise

    def invoke(self, *args, **kwargs):  # noqa: ANN202
        idx, inst = self._pick()
        try:
            return inst.invoke(*args, **kwargs)
        except BaseException as e:
            self._handle_exc(idx, e)
            raise

    async def astream(self, *args, **kwargs):  # noqa: ANN202
        idx, inst = self._pick()
        try:
            async for chunk in inst.astream(*args, **kwargs):
                yield chunk
        except BaseException as e:
            self._handle_exc(idx, e)
            raise

    def stream(self, *args, **kwargs):  # noqa: ANN202
        idx, inst = self._pick()
        try:
            yield from inst.stream(*args, **kwargs)
        except BaseException as e:
            self._handle_exc(idx, e)
            raise

    # ---- Pydantic v1 validator shim --------------------------------------
    #
    # Some callers (browser_use `AgentSettings.__init__` → LangChain
    # `raise_deprecation` pre_init validator) receive this wrapper as the
    # `values` dict and call `values.get("callback_manager")`. That path
    # can't be fulfilled by `__getattr__` delegation because `ChatOpenAI`
    # has no `.get()`, so the AttributeError surfaces as
    # ``Error: 'ChatOpenAI' object has no attribute 'get'`` and silently
    # kills every browser_use session on multi-key Mistral / any other
    # non-Pydantic-BaseChatModel wrapper. The clean fix is to provide an
    # explicit dict-shaped `.get(key, default=None)` so validators that
    # treat us as dict-like see a well-defined "unset" answer for every
    # field. See HANDOFF_QWEN_BROWSER_RAW_MODE.md §KeyRotatingChatOpenAI
    # compatibility fix.
    def get(self, key: str, default=None):
        """Dict-like shim for LangChain's pre-init validators that walk
        `values.get("...")`. Returns the attribute if it exists, else
        `default` — never raises.
        """
        # Serve the caller from the first instance when possible. If the
        # attribute is missing (common: `callback_manager` in langchain
        # >=0.3), fall back to `default` instead of raising — matches
        # the dict.get() contract.
        instances = self.__dict__.get("_instances")
        if not instances:
            return default
        return getattr(instances[0], key, default)

    # ---- pass-through for the long tail -------------------------------

    def __getattr__(self, name: str):  # noqa: D401
        """Fallback: any attribute not explicitly wrapped above goes to
        `_instances[0]`. This keeps `bind_tools`, `with_structured_output`,
        property reads, etc. working without per-attribute boilerplate —
        at the cost of NOT rotating on whatever path they open up (the
        returned bound/structured model is locked to instance 0).

        Accepting this tradeoff: for the current browser-use integration
        the primary hot path is `ainvoke`, which IS rotated. Anything
        chained off `bind_tools` is a setup step that runs once per
        browser-use agent construction, not per question.
        """
        # `__getattr__` is only called when normal lookup failed. Avoid
        # infinite recursion via `__dict__`.
        instances = self.__dict__.get("_instances")
        if instances is None:  # during __init__ before _instances set
            raise AttributeError(name)
        return getattr(instances[0], name)


# ===========================================================================
# LangChain-path tool_choice downgrade — Qwen OR regression (2026-04-22)
# ===========================================================================
#
# Problem
# -------
# The native `OpenAIServerModel.generate()` path passes every outgoing
# completion through `pick_tool_choice()` in `src/models/tool_choice.py`,
# which downgrades `tool_choice="required"` → `"auto"` for wire-ids that
# OpenRouter's Qwen providers reject (prefix rule `qwen/*`).
#
# The LangChain `ChatOpenAI` path — used by `auto_browser_use_tool` via
# `browser_use.Agent(llm=..., page_extraction_llm=...)` — bypasses that
# dispatch entirely. `browser_use` calls `bind_tools` internally and emits
# `tool_choice` values that Alibaba (the sole OR Qwen provider as of
# 2026-04-22) rejects with HTTP 404 "No endpoints found that support the
# provided 'tool_choice' value".
#
# The Qwen LangChain path never triggered this bug before because Qwen's
# planner did not invoke `auto_browser_use_tool` in E0 v3 training
# (verified 2026-04-22: 0 browser calls / 80 Qwen questions). The
# 2026-04-22 T3 smoke was the first real exposure.
#
# Fix
# ---
# Extend the downgrade to the LangChain path by subclassing `ChatOpenAI`
# and overriding `_get_request_payload` — the lowest-level hook that
# builds the dict sent to `openai.AsyncOpenAI.chat.completions.create`.
# The override re-uses the canonical `pick_tool_choice` rule so behaviour
# is identical to the native path (one source of truth for the downgrade
# policy).


class ToolChoiceDowngradingChatOpenAI:
    """LangChain `ChatOpenAI` subclass-in-name-only that applies the
    project's hybrid `tool_choice` dispatch to outgoing payloads.

    Runtime-built subclass (see factory ``make_tool_choice_downgrading_chat_openai``
    below) so that `ChatOpenAI` is imported lazily — this module is
    imported at project start and `langchain_openai` is heavy.

    Semantics
    ---------
    - For every outgoing chat completion, inspect the payload's
      `tool_choice` key. If `pick_tool_choice(model_id, current_value)`
      returns a different value, swap it in. Emit `log_downgrade_once`
      for the first downgrade per model.
    - When `tool_choice` is absent from the payload, pass through
      unchanged — LangChain's default behaviour is retained.
    - The `model_id` used for the dispatch decision is `self.model_name`
      (which LangChain populates from the `model=` constructor arg),
      not a separate wire-id. For OR Qwen wrappers this is
      `"qwen/qwen3.6-plus"` which matches the `qwen/` prefix rule.

    Test hook
    ---------
    The factory exposes the class so unit tests can instantiate and
    invoke `_get_request_payload` directly without hitting the network.
    """


def make_tool_choice_downgrading_chat_openai():
    """Build the `ChatOpenAI` subclass. Lazy — only imports
    `langchain_openai` on first call.

    Returns a class object; callers construct instances exactly like
    `ChatOpenAI`:

        cls = make_tool_choice_downgrading_chat_openai()
        model = cls(model="qwen/qwen3.6-plus", api_key=..., base_url=...)

    Implementation-stability guard
    ------------------------------
    Overrides `BaseChatOpenAI._get_request_payload` — a method with a
    leading underscore in the LangChain public API, which means
    LangChain is free to rename or restructure it across releases. We
    assert it exists at factory time so a langchain-openai upgrade
    that moves the method fails loudly at first call instead of
    silently disabling the downgrade. Pinned + tested against
    langchain-openai==0.3.11 (install date 2026-04-18); before
    upgrading, confirm the method is still present via
    `inspect.getsourcefile(ChatOpenAI._get_request_payload)`.
    """
    from langchain_openai import ChatOpenAI

    if not hasattr(ChatOpenAI, "_get_request_payload"):
        raise RuntimeError(
            "langchain-openai removed or renamed "
            "`ChatOpenAI._get_request_payload`. The Qwen tool_choice "
            "downgrade path (P5) depends on this internal hook. Pin "
            "the previous working version (0.3.11) in the environment "
            "or port the override to the new method name. See "
            "docs/handoffs/HANDOFF_THROUGHPUT_REFACTOR.md §P5 for the "
            "seam rationale."
        )

    class _Impl(ChatOpenAI):
        """Concrete subclass. See `ToolChoiceDowngradingChatOpenAI` for
        the design doc. Name kept private because LangChain's Pydantic
        machinery is brittle with class-name introspection in some
        code paths."""

        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)

            model_id = payload.get("model") or getattr(self, "model_name", None)

            # Raw-mode guard for Qwen (2026-04-23). When browser_use runs
            # `tool_calling_method='raw'` it calls `llm.invoke(messages)`
            # WITHOUT `bind_tools` — so the payload has no `tools` key.
            # Alibaba's Qwen backend nonetheless returns
            # `finish_reason='tool_calls'` with `content=""` (observed in
            # the 2026-04-23 verbose probe: 243–298 completion tokens,
            # 0 reasoning tokens, `content=''`, no tool_calls surfaced
            # through LangChain's AIMessage). The library tags the
            # response as tool-calls even though nothing was bound to
            # call, which makes `extract_json_from_model_output` fail on
            # empty content.
            # Explicitly setting `tool_choice="none"` (supported by
            # Alibaba per DashScope docs — the opposite knob to the
            # `"required"` that was rejected) forces Qwen into plain
            # chat-completion mode → `finish_reason="stop"` with the
            # JSON in `content`. This is only applied when no tools are
            # bound, so the standard `bind_tools`/`with_structured_output`
            # paths are untouched.
            tools = payload.get("tools")
            tool_choice_unset = "tool_choice" not in payload
            if (
                tool_choice_unset
                and not tools
                and model_id
                and model_id.startswith("qwen/")
            ):
                payload["tool_choice"] = "none"
                logger.info(
                    "[tool_choice] %s raw-mode (no tools) -> injected tool_choice='none'",
                    model_id,
                )
                return payload

            requested = payload.get("tool_choice")
            if requested is None:
                # Caller didn't set one; nothing to downgrade. Pass through.
                return payload

            resolved = pick_tool_choice(model_id, default=requested)
            if resolved != requested and resolved == "auto":
                payload["tool_choice"] = resolved
                log_downgrade_once(model_id or "<unknown>")
            elif resolved != requested:
                # Some non-"auto" rewrite (currently impossible per
                # pick_tool_choice rules, but defensive if policy changes).
                payload["tool_choice"] = resolved

            return payload

    return _Impl