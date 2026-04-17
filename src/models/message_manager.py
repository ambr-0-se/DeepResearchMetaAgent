import json
from typing import Dict, List, Optional, Any
from copy import deepcopy

from src.models.base import MessageRole, ChatMessage, ChatMessageToolCall
from src.utils import encode_image_base64, make_image_url


def _tool_calls_to_openai_api_format(tool_calls: list) -> list[dict[str, Any]]:
    """Serialize tool_calls for OpenAI Chat Completions (arguments must be a JSON string)."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        assert isinstance(tc, ChatMessageToolCall)
        args = tc.function.arguments
        if isinstance(args, (dict, list)):
            args_str = json.dumps(args)
        elif args is None:
            args_str = "{}"
        else:
            args_str = str(args)
        out.append(
            {
                "id": tc.id,
                "type": tc.type or "function",
                "function": {"name": tc.function.name, "arguments": args_str},
            }
        )
    return out


def _tool_message_content_to_str(message: ChatMessage) -> str:
    c = message.content
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        el = c[0]
        if isinstance(el, dict) and el.get("type") == "text":
            return el.get("text", "")
    return str(c)

DEFAULT_ANTHROPIC_MODELS = [
    'claude37-sonnet',
    "claude37-sonnet-thinking",
]
UNSUPPORTED_STOP_MODELS = [
    'claude37-sonnet',
    'o4-mini',
    'o3',
    'langchain-o3'
]
UNSUPPORTED_TOOL_CHOICE_MODELS = [
    'claude37-sonnet',
]

# Sampling params to drop for Moonshot Kimi (temperature/top_p locked by provider;
# n=1 only; presence/frequency penalty ignored — stripping avoids 400s).
# Applied AFTER caller-kwargs merge so provider constraint always wins.
_KIMI_BANNED_SAMPLING_PARAMS = (
    "temperature", "top_p", "n", "presence_penalty", "frequency_penalty", "logprobs", "logit_bias"
)

# MiniMax M2.7: temperature must be in (0, 1.0]; n must be 1; presence/frequency
# penalty silently ignored upstream but cleaner to drop.
_MINIMAX_BANNED_SAMPLING_PARAMS = (
    "n", "presence_penalty", "frequency_penalty", "logprobs", "logit_bias"
)


def _model_id_tail(model_id: str) -> str:
    """Return the terminal segment of a model id ('moonshotai/kimi-k2.5' -> 'kimi-k2.5')."""
    return (model_id or "").split("/")[-1].lower()


def is_moonshot_kimi(model_id: str) -> bool:
    tail = _model_id_tail(model_id)
    return tail.startswith("kimi-") or "moonshot" in tail


def is_minimax(model_id: str) -> bool:
    return "minimax" in _model_id_tail(model_id)


def needs_reasoning_echo(model_id: str) -> bool:
    """Providers that require the assistant's previous `reasoning_content` to be
    echoed back in the next request or they 400 on turn 2+ of a tool loop.

    As of 2026-04: DeepSeek V3.2 reasoner is the primary offender. Qwen3 thinking
    mode on DashScope returns reasoning in `reasoning_content` and also expects it
    echoed when `enable_thinking=True` is persisted across turns.
    """
    tail = _model_id_tail(model_id)
    return (
        "deepseek-reasoner" in tail
        or tail.startswith("deepseek-v3.2")
        or ("qwen3" in tail and "thinking" in tail)
    )

class MessageManager():
    def __init__(self, model_id: str, api_type: str = "chat/completions"):
        self.model_id = model_id
        self.api_type = api_type

    def get_clean_message_list(self,
            message_list: list[ChatMessage],
            role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
            convert_images_to_image_urls: bool = False,
            flatten_messages_as_text: bool = False,
            api_type: str = "chat/completions",
    ) -> list[dict[str, Any]]:
        """
        Creates a list of messages to give as input to the LLM. These messages are dictionaries and chat template compatible with transformers LLM chat template.
        Subsequent messages with the same role will be concatenated to a single message.

        Args:
            message_list (`list[dict[str, str]]`): List of chat messages.
            role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
            convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
            flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
        """
        api_type = api_type or self.api_type
        if api_type == "responses":
            return self._get_responses_message_list(
                message_list, role_conversions, convert_images_to_image_urls, flatten_messages_as_text
            )
        else:
            return self._get_chat_completions_message_list(
                message_list, role_conversions, convert_images_to_image_urls, flatten_messages_as_text
            )

    def _get_chat_completions_message_list(self,
            message_list: list[ChatMessage],
            role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
            convert_images_to_image_urls: bool = False,
            flatten_messages_as_text: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Creates a list of messages in chat completions format.
        Supports OpenAI-native tool turns: assistant + tool_calls, then role=tool with tool_call_id per result.
        """
        output_message_list: list[dict[str, Any]] = []
        message_list = deepcopy(message_list)  # Avoid modifying the original list
        for message in message_list:
            role = message.role
            if role not in MessageRole.roles():
                raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

            if role in role_conversions:
                message.role = role_conversions[role]  # type: ignore

            # Native tool result (Tier B): one message per tool_call_id — never merge with adjacent messages
            if message.role == MessageRole.TOOL:
                if not message.tool_call_id:
                    raise ValueError("ChatMessage with role 'tool' requires tool_call_id for Chat Completions API.")
                output_message_list.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": _tool_message_content_to_str(message),
                    }
                )
                continue

            # Assistant message that issued tool_calls (must include tool_calls in the API payload)
            if message.role == MessageRole.ASSISTANT and message.tool_calls:
                assistant_payload: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": _tool_calls_to_openai_api_format(message.tool_calls),
                }
                # Echo reasoning_content when the provider requires it for tool-loop
                # continuity (DeepSeek-reasoner / Qwen3-thinking). Sending it to
                # providers that don't recognize the field is harmless (silently dropped),
                # but we gate on the predicate to avoid surprising OpenAI/Anthropic payloads.
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning and needs_reasoning_echo(self.model_id):
                    assistant_payload["reasoning_content"] = reasoning
                output_message_list.append(assistant_payload)
                continue

            # encode images if needed
            if isinstance(message.content, list):
                for element in message.content:
                    assert isinstance(element, dict), "Error: this element should be a dict:" + str(element)
                    if element["type"] == "image":
                        assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                        if convert_images_to_image_urls:
                            element.update(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                                }
                            )
                        else:
                            element["image"] = encode_image_base64(element["image"])

            # Don't merge assistant messages that carry reasoning_content — merging
            # into the previous turn would fabricate reasoning attribution and risk
            # double-billing the provider's reasoning-token count.
            has_reasoning = bool(getattr(message, "reasoning_content", None))
            can_merge = (
                len(output_message_list) > 0
                and message.role == output_message_list[-1]["role"]
                and message.role != MessageRole.TOOL
                and not has_reasoning
            )
            if can_merge:
                assert isinstance(message.content, list), "Error: wrong content:" + str(message.content)
                if flatten_messages_as_text:
                    output_message_list[-1]["content"] += "\n" + message.content[0]["text"]
                else:
                    for el in message.content:
                        if el["type"] == "text" and output_message_list[-1]["content"][-1]["type"] == "text":
                            # Merge consecutive text messages rather than creating new ones
                            output_message_list[-1]["content"][-1]["text"] += "\n" + el["text"]
                        else:
                            output_message_list[-1]["content"].append(el)
            else:
                if flatten_messages_as_text:
                    if isinstance(message.content, list) and message.content:
                        content = message.content[0]["text"]
                    else:
                        content = message.content or ""
                else:
                    content = message.content
                payload: Dict[str, Any] = {
                    "role": message.role,
                    "content": content,
                }
                if has_reasoning and needs_reasoning_echo(self.model_id):
                    payload["reasoning_content"] = message.reasoning_content
                output_message_list.append(payload)
        return output_message_list

    def _get_responses_message_list(self,
            message_list: list[ChatMessage],
            role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
            convert_images_to_image_urls: bool = False,
            flatten_messages_as_text: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Creates a list of messages in responses format (OpenAI responses API).
        """
        output_message_list: list[dict[str, Any]] = []
        message_list = deepcopy(message_list)  # Avoid modifying the original list
        
        for message in message_list:
            role = message.role
            if role not in MessageRole.roles():
                raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

            if role in role_conversions:
                message.role = role_conversions[role]  # type: ignore
            
            # Handle content processing
            if isinstance(message.content, list):
                # Process each content element
                processed_content = []
                for element in message.content:
                    assert isinstance(element, dict), "Error: this element should be a dict:" + str(element)
                    
                    if element["type"] == "image":
                        assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                        if convert_images_to_image_urls:
                            processed_content.append({
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            })
                        else:
                            processed_content.append({
                                "type": "image",
                                "image": encode_image_base64(element["image"])
                            })
                    elif element["type"] == "text":
                        processed_content.append(element)
                    else:
                        processed_content.append(element)
                
                content = processed_content
            else:
                # Handle string content
                if flatten_messages_as_text:
                    content = message.content
                else:
                    content = [{"type": "text", "text": message.content}] if message.content else []

            # Handle tool calls for responses format
            tool_calls = None
            if message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                            "description": tool_call.function.description
                        }
                    })

            # Create message in responses format
            message_dict = {
                "role": message.role,
                "content": content,
            }
            
            if tool_calls:
                message_dict["tool_calls"] = tool_calls

            # Merge consecutive messages with same role
            if len(output_message_list) > 0 and message.role == output_message_list[-1]["role"]:
                if flatten_messages_as_text:
                    if isinstance(content, list) and content and content[0]["type"] == "text":
                        output_message_list[-1]["content"] += "\n" + content[0]["text"]
                    else:
                        output_message_list[-1]["content"] += "\n" + str(content)
                else:
                    # Merge content lists
                    if isinstance(output_message_list[-1]["content"], list) and isinstance(content, list):
                        output_message_list[-1]["content"].extend(content)
                    else:
                        output_message_list[-1]["content"] = content
                
                # Merge tool calls
                if tool_calls and "tool_calls" in output_message_list[-1]:
                    output_message_list[-1]["tool_calls"].extend(tool_calls)
                elif tool_calls:
                    output_message_list[-1]["tool_calls"] = tool_calls
            else:
                output_message_list.append(message_dict)

        return output_message_list

    def get_tool_json_schema(self,
                             tool: Any,
                             model_id: Optional[str] = None
                             ) -> Dict:
        properties = deepcopy(tool.parameters['properties'])

        required = []
        for key, value in properties.items():
            if value["type"] == "any":
                value["type"] = "string"
            if not ("nullable" in value and value["nullable"]):
                required.append(key)

        model_id = model_id.split("/")[-1]

        if model_id in DEFAULT_ANTHROPIC_MODELS:
            return {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        else:
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

    def get_clean_completion_kwargs(self, completion_kwargs: Dict[str, Any]):
        """Final sanitizer applied AFTER caller kwargs have merged into completion_kwargs.

        Order matters: provider hard-constraints must win over caller-injected overrides,
        so this runs at the tail of `_prepare_completion_kwargs` in OpenAIServerModel.
        """

        model_id = self.model_id.split("/")[-1]

        if model_id in UNSUPPORTED_TOOL_CHOICE_MODELS:
            completion_kwargs.pop("tool_choice", None)
        if model_id in UNSUPPORTED_STOP_MODELS:
            completion_kwargs.pop("stop", None)

        # Moonshot Kimi locks sampling params (temperature/top_p), allows n=1 only,
        # ignores penalty/logit params. Strip them regardless of what the caller sent —
        # the provider will 400 otherwise. Applied after caller merge on purpose.
        if is_moonshot_kimi(self.model_id):
            for param in _KIMI_BANNED_SAMPLING_PARAMS:
                completion_kwargs.pop(param, None)

        # MiniMax: clamp temperature to (0, 1.0], force n=1. presence/frequency
        # penalty silently ignored by provider — drop to keep requests clean.
        if is_minimax(self.model_id):
            for param in _MINIMAX_BANNED_SAMPLING_PARAMS:
                completion_kwargs.pop(param, None)
            temp = completion_kwargs.get("temperature")
            if temp is not None:
                # Provider rejects temperature=0 and values > 1.0.
                if temp <= 0:
                    completion_kwargs["temperature"] = 0.01
                elif temp > 1.0:
                    completion_kwargs["temperature"] = 1.0

        return completion_kwargs