from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict, Union, Optional

from src.models import ChatMessage, MessageRole
from src.exception import (
    AgentError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
)
from src.utils import make_json_serializable
from src.logger import LogLevel, AgentLogger, Timing, TokenUsage


_NO_REFUSAL_DIRECTIVE = (
    "Do NOT reply 'Unable to determine', 'I don't know', or any refusal. "
    "If evidence is incomplete, commit to your best-guess answer using "
    "what you have plus general knowledge.\n"
)


def _render_error_message(
    error: AgentError | BaseException,
    action_output: Any = None,
    tool_call_id: str | None = None,
) -> str:
    """Build a differentiated, actionable error message for the planner.

    Dispatches on error type so the planner sees guidance tailored to the
    failure mode (max_steps vs tool-call malformed vs tool runtime failure
    vs parsing vs generation). Every branch ends with the no-refusal
    directive so the planner cannot fall back to 'Unable to determine'.
    """
    err_str = str(error)
    progress_block = ""
    # Use `is not None` rather than truthy check so falsy-but-meaningful
    # outputs (0, False, "", []) are still surfaced as progress.
    if action_output is not None:
        progress_block = (
            "\nProgress synthesized from work so far:\n"
            + str(action_output)
            + "\n"
        )

    if isinstance(error, AgentMaxStepsError):
        guidance = (
            "The sub-agent hit its max_steps budget. Do NOT re-delegate the "
            "same task verbatim. Synthesize what was gathered and either "
            "(a) call final_answer_tool with a concrete best guess, or "
            "(b) try ONE genuinely different decomposition (different "
            "sub-agent, different sub-question). "
        )
    elif isinstance(error, AgentToolCallError):
        guidance = (
            "The tool call was malformed (bad arguments / schema). Fix the "
            "arguments and retry the same tool ONCE. If it fails again, "
            "switch to a different tool or commit a best-guess via "
            "final_answer_tool. "
        )
    elif isinstance(error, AgentToolExecutionError):
        guidance = (
            "The tool ran but failed at execution (network error, parse "
            "failure, upstream service issue). Do NOT immediately retry the "
            "same call. Try ONE alternative (different tool, different "
            "source, different query), then commit a best-guess answer "
            "via final_answer_tool. "
        )
    elif isinstance(error, AgentParsingError):
        guidance = (
            "Your previous output failed to parse as a valid action. "
            "Re-emit a single, well-formed tool call (or final_answer_tool) "
            "in the required format. Do not repeat the malformed structure. "
        )
    # Note: AgentGenerationError is re-raised in async_multistep_agent.py and
    # never stored in `action_step.error`, so an explicit branch here would
    # be dead code. Generation failures fall through to the generic
    # AgentError branch below if they ever reach this helper through a
    # different path.
    elif isinstance(error, AgentError):
        guidance = (
            "Use information gathered to commit to a concrete best-guess "
            "answer now via final_answer_tool. Only try ONE genuinely "
            "different approach (different tool, source, or search terms) "
            "before giving up. Do not repeat the same call. "
        )
    else:
        guidance = (
            "An unexpected error occurred. Commit to a concrete best-guess "
            "answer now via final_answer_tool, or try ONE genuinely "
            "different approach before giving up. "
        )

    body = "Error:\n" + err_str + "\n" + progress_block + "\n" + guidance + _NO_REFUSAL_DIRECTIVE
    if tool_call_id:
        return f"Call id: {tool_call_id}\n" + body
    return body



if TYPE_CHECKING:
    import PIL.Image


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    step_number: int
    timing: Timing
    model_input_messages: list[ChatMessage] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | None = None
    observations: str | None = None
    """Per tool_call_id results for OpenAI Chat Completions (Tier B). Order matches parallel tool execution."""
    tool_results: list[dict[str, str]] | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": self.model_output_message.dict() if self.model_output_message else None,
            "model_output": self.model_output,
            "observations": self.observations,
            "tool_results": self.tool_results,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        messages = []
        # Tier B: OpenAI-native assistant + tool_calls, then one role=tool message per tool_call_id
        if (
            self.tool_results
            and self.model_output_message
            and self.model_output_message.tool_calls
            and not summary_mode
        ):
            mos = self.model_output_message
            messages.append(
                ChatMessage(
                    role=mos.role,
                    content=mos.content,
                    tool_calls=mos.tool_calls,
                )
            )
            for tr in self.tool_results:
                messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        tool_call_id=tr["id"],
                        content=tr["content"],
                    )
                )
            if self.observations_images:
                messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[
                            {
                                "type": "image",
                                "image": image,
                            }
                            for image in self.observations_images
                        ],
                    )
                )
            if self.error is not None:
                # Phase 1 (2026-04-26): differentiated error rendering via
                # _render_error_message dispatch. Replaces the previous
                # one-size-fits-all Fix E block.
                tc_id = self.tool_calls[0].id if self.tool_calls else None
                message_content = _render_error_message(
                    self.error,
                    action_output=getattr(self, "action_output", None),
                    tool_call_id=tc_id,
                )
                messages.append(
                    ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
                )
            return messages

        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            # Phase 1 (2026-04-26): see Tier B branch above for rationale.
            tc_id = self.tool_calls[0].id if self.tool_calls else None
            message_content = _render_error_message(
                self.error,
                action_output=getattr(self, "action_output", None),
                tool_call_id=tc_id,
            )
            messages.append(
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages

@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: list[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]
            ),
            # This second message creates a role change to prevent models models from simply continuing the plan message
        ]

@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any
    
@dataclass
class UserPromptStep(MemoryStep):
    user_prompt: str

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": self.user_prompt}])]


class AgentMemory:
    def __init__(self, system_prompt: str, user_prompt: Optional[str] = None):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        if user_prompt is not None:
            self.user_prompt = UserPromptStep(user_prompt=user_prompt)
        else:
            self.user_prompt = None
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        if self.user_prompt is not None:
            logger.log_markdown(title="User prompt", content=self.user_prompt.user_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)


__all__ = ["AgentMemory"]
