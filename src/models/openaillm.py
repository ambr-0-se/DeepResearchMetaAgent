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

logger = logging.getLogger(__name__)

_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_DELAY = 1.0   # seconds
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
            if tool_choice is not None:
                tools_config["tool_choice"] = tool_choice
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
                response = await self.client.chat.completions.create(**completion_kwargs)
                self._last_input_token_count = response.usage.prompt_tokens
                self._last_output_token_count = response.usage.completion_tokens
                # Capture reasoning_content directly from the SDK message object before
                # dumping: include={"role","content","tool_calls"} would silently drop
                # it, which breaks DeepSeek-reasoner (and other thinking models) on the
                # next tool-loop turn since the provider requires it to be echoed back.
                msg_obj = response.choices[0].message
                msg_dict = msg_obj.model_dump(include={"role", "content", "tool_calls"})
                reasoning = getattr(msg_obj, "reasoning_content", None)
                if reasoning:
                    msg_dict["reasoning_content"] = reasoning
                return ChatMessage.from_dict(
                    msg_dict,
                    raw=response,
                    token_usage=TokenUsage(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
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

        raise last_exc

    async def __call__(self, *args, **kwargs) -> ChatMessage:
        """
        Call the model with the given arguments.
        This is a convenience method that calls `generate` with the same arguments.
        """
        return await self.generate(*args, **kwargs)