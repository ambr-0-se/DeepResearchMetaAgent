import asyncio
from typing import (
    Any,
    Callable,
    Optional
)
import json
import yaml
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown
from collections.abc import AsyncGenerator

from src.tools import AsyncTool
from src.exception import (
    AgentGenerationError,
    AgentParsingError,
    AgentToolExecutionError,
    AgentToolCallError
)
from src.base.async_multistep_agent import (PromptTemplates,
                                            populate_template,
                                            AsyncMultiStepAgent,
                                            )
from src.base import (ToolOutput,
                      ActionOutput,
                      StreamEvent)

from src.memory import (ActionStep,
                        ToolCall,
                        AgentMemory)
from src.logger import (LogLevel,
                        YELLOW_HEX,
                        logger)
from src.models import (Model,
                        parse_json_if_needed,
                        agglomerate_stream_deltas,
                        ChatMessage,
                        ChatMessageStreamDelta)
from src.models.tool_choice import pick_tool_choice

# Max number of corrective re-prompts when tool_choice is dispatched to "auto"
# and the model returns plain text instead of a tool call. Each retry injects a
# user message reminding the model that every step requires a tool call.
# See docs/handoffs/HANDOFF_PENDING_KIMI_AND_TOOL_CHOICE.md §Change 2b.
MAX_TOOL_RETRIES = 2

_TOOL_CHOICE_RETRY_PROMPT = (
    "You replied in plain text, but every step REQUIRES a tool call. "
    "Choose exactly one tool from the list above and call it now. "
    "If you intended to give a final answer, call `final_answer` with your answer."
)
from src.utils.agent_types import (
    AgentAudio,
    AgentImage,
)
from src.registry import AGENT
from src.utils import assemble_project_path
from src.utils.token_utils import (
    DEFAULT_CONTEXT_IMAGE_TOKEN_ESTIMATE,
    DEFAULT_CONTEXT_PRUNE_RESERVE_TOKENS,
    DEFAULT_CONTEXT_PRUNE_TAIL_SEGMENTS,
    DEFAULT_CONTEXT_PRUNE_THRESHOLD_RATIO,
    estimate_messages_tokens,
    prune_messages_to_budget,
)
from src.agent.general_agent._tool_call_guard import apply_final_answer_guard


@AGENT.register_module(name="general_agent", force=True)
class GeneralAgent(AsyncMultiStepAgent):
    def __init__(
            self,
            config,
            tools: list[Any],
            model: Model,
            prompt_templates: PromptTemplates | None = None,
            planning_interval: int | None = None,
            stream_outputs: bool = False,
            max_tool_threads: int | None = None,
            **kwargs,
    ):
        self.config = config

        super(GeneralAgent, self).__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )

        template_path = assemble_project_path(self.config.template_path)
        with open(template_path, "r") as f:
            self.prompt_templates = yaml.safe_load(f)

        self.system_prompt = self.initialize_system_prompt()
        self.user_prompt = self.initialize_user_prompt()

        # Streaming setup
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        # Retry-guard advisory: the guard only runs on the non-streaming path
        # (see `_retry_on_missing_tool_calls`). If a model in the auto-dispatch
        # set ever runs with streaming enabled, plain-text replies will fall
        # straight through to the `parse_tool_calls` fallback with no retry.
        # Warn loudly at construction so the misconfiguration is visible.
        if self.stream_outputs:
            model_id = getattr(self.model, "model_id", None)
            if pick_tool_choice(model_id, default="required") == "auto":
                logger.warning(
                    "[tool_choice] %s resolves to \"auto\" but stream_outputs=True "
                    "— the plain-text retry guard is inactive on the streaming "
                    "path. Expect empty-tool-call failures if the model replies "
                    "in text. Disable stream_outputs for this model or extend "
                    "the retry guard to the streaming branch.",
                    model_id or "<unknown>",
                )
        # Tool calling setup
        self.max_tool_threads = max_tool_threads

        self.memory = AgentMemory(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )

        logger.info(
            f"[{self.__class__.__name__} GeneralAgent.__init__ done] "
            f"tools={list(self.tools.keys())}, "
            f"managed_agents={list(self.managed_agents.keys())} "
            f"(id={id(self.managed_agents):#x})"
        )

    def initialize_system_prompt(self) -> str:
        """Initialize the system prompt for the agent."""
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools, "managed_agents": self.managed_agents},
        )
        return system_prompt

    def initialize_user_prompt(self) -> str:

        user_prompt = populate_template(
            self.prompt_templates["user_prompt"],
            variables={},
        )

        return user_prompt

    def initialize_task_instruction(self) -> str:
        """
        Initialize the task instruction for the agent.

        Variables passed to the Jinja template:
            task                  — the user-provided task text
            skill_registry_block  — optional skill-registry injection
                                    (C4). Defaults to empty string so
                                    templates can safely conditionalise
                                    with `{% if skill_registry_block %}`
                                    even when skills are not in use.
            plus any additional key/value pairs the owner placed in
            `self._extra_task_variables` (a plain dict attribute) before
            calling `run()`. AdaptivePlanningAgent uses this to pass the
            per-sub-agent skill registry block without modifying the
            base `AsyncMultiStepAgent.__call__` path.

        Unknown variables in `_extra_task_variables` are forwarded verbatim;
        it is the template author's responsibility to only reference known
        names.
        """
        variables = {
            "task": self.task,
            # Always provided so Jinja's StrictUndefined mode does not raise
            # when a template uses `{% if skill_registry_block %}` even in
            # C0/C2/C3 where no skill registry is active. Empty string is
            # falsy, so the block is effectively omitted.
            "skill_registry_block": "",
        }
        extras = getattr(self, "_extra_task_variables", None) or {}
        variables.update(extras)
        task_instruction = populate_template(
            self.prompt_templates["task_instruction"],
            variables=variables,
        )
        return task_instruction

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    @property
    def tools_and_managed_agents(self):
        """Returns a combined list of tools and managed agents."""
        return list(self.tools.values()) + list(self.managed_agents.values())

    def _estimate_token_count(self, messages: list) -> int:
        """Token estimate via tiktoken (or heuristic chars/3.5 when configured)."""
        model_id = getattr(self.model, "model_id", None)
        mode = getattr(self.config, "token_estimation_mode", "tiktoken")
        image_est = getattr(
            self.config, "context_image_token_estimate", DEFAULT_CONTEXT_IMAGE_TOKEN_ESTIMATE
        )
        return estimate_messages_tokens(
            messages,
            model_id,
            mode=mode,
            context_image_token_estimate=image_est,
        )

    def _prune_messages_if_needed(self, messages: list) -> list:
        """Prune conversation history if approaching the model's context limit.

        Drops whole Tier B segments (assistant + tool results) from the middle before the tail,
        never splitting tool chains. Inserts a placeholder when middle content is removed.
        """
        max_model_len = getattr(self.config, "max_model_len", 32768)
        ratio = getattr(
            self.config, "context_prune_threshold_ratio", DEFAULT_CONTEXT_PRUNE_THRESHOLD_RATIO
        )
        reserve = getattr(
            self.config, "context_prune_reserve_tokens", DEFAULT_CONTEXT_PRUNE_RESERVE_TOKENS
        )
        tail_segments = getattr(
            self.config, "context_prune_tail_segments", DEFAULT_CONTEXT_PRUNE_TAIL_SEGMENTS
        )
        mode = getattr(self.config, "token_estimation_mode", "tiktoken")
        image_est = getattr(
            self.config, "context_image_token_estimate", DEFAULT_CONTEXT_IMAGE_TOKEN_ESTIMATE
        )
        model_id = getattr(self.model, "model_id", None)

        effective_budget = int(max_model_len * ratio) - reserve
        estimated_tokens = self._estimate_token_count(messages)
        if estimated_tokens <= effective_budget:
            return messages

        logger.warning(
            f"[Context Pruning] Estimated {estimated_tokens} tokens exceeds "
            f"effective budget {effective_budget} (max_model_len={max_model_len}, "
            f"ratio={ratio}, reserve={reserve}). Pruning..."
        )

        result = prune_messages_to_budget(
            messages,
            model_id,
            max_model_len=max_model_len,
            context_prune_threshold_ratio=ratio,
            context_prune_reserve_tokens=reserve,
            context_prune_tail_segments=tail_segments,
            token_estimation_mode=mode,
            context_image_token_estimate=image_est,
        )
        new_est = estimate_messages_tokens(
            result,
            model_id,
            mode=mode,
            context_image_token_estimate=image_est,
        )
        logger.warning(
            f"[Context Pruning] Pruned {len(messages)} messages -> {len(result)} messages, "
            f"estimated tokens: {estimated_tokens} -> {new_est}"
        )
        return result

    async def _retry_on_missing_tool_calls(
        self,
        chat_message: ChatMessage,
        input_messages: list[ChatMessage],
    ) -> ChatMessage:
        """Re-prompt the model when a dispatched-to-``auto`` turn returns no tool calls.

        Only fires when :func:`pick_tool_choice` resolves the current model to
        ``"auto"``. Returns the most recent ``ChatMessage`` (either the
        original, or one produced by a successful retry). On exhaustion falls
        through so the caller's existing ``parse_tool_calls`` fallback + error
        path handles the failure.

        Known caveats:
        - **reasoning_content is dropped on the retry turn.** The injected
          assistant echo carries only ``role`` + ``content``. For any model
          matched by ``needs_reasoning_echo()`` (DeepSeek-reasoner,
          Qwen3-thinking, etc.), the provider will 400 on the retry turn
          because it requires prior ``reasoning_content`` to be echoed. The
          intersection between the auto-dispatch set and
          ``needs_reasoning_echo`` is currently empty (no Qwen3-thinking
          slugs in the matrix and none auto-dispatched), so this is latent
          rather than active. Widen this helper to copy reasoning_content
          before adding any such model.
        - **Streaming path is not covered.** If ``stream_outputs=True`` and
          the model is in the auto-dispatch set, plain-text replies fall
          through to ``parse_tool_calls`` with no retry — see the warning
          emitted from ``GeneralAgent.__init__``.
        """
        model_id = getattr(self.model, "model_id", None)
        if pick_tool_choice(model_id, default="required") != "auto":
            return chat_message

        retries = 0
        conversation = list(input_messages)
        while (
            (chat_message.tool_calls is None or len(chat_message.tool_calls) == 0)
            and retries < MAX_TOOL_RETRIES
        ):
            retries += 1
            logger.warning(
                "[tool_choice retry %d/%d] %s replied in plain text; "
                "injecting corrective prompt.",
                retries, MAX_TOOL_RETRIES, model_id or "<unknown>",
            )
            conversation = conversation + [
                ChatMessage(role="assistant", content=chat_message.content or ""),
                ChatMessage(role="user", content=_TOOL_CHOICE_RETRY_PROMPT),
            ]
            chat_message = await self.model(
                conversation,
                stop_sequences=["Observation:", "Calling tools:"],
                tools_to_call_from=self.tools_and_managed_agents,
            )
        return chat_message

    async def _step_stream(self, memory_step: ActionStep) -> AsyncGenerator[ChatMessageStreamDelta | ToolOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = await self.write_memory_to_messages()
        memory_messages = self._prune_messages_if_needed(memory_messages)

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
            else:
                chat_message: ChatMessage = await self.model(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                self.logger.log_markdown(
                    content=chat_message.content if chat_message.content else str(chat_message.raw),
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

                # Retry guard: when tool_choice was dispatched to "auto" (see
                # src/models/tool_choice.py) the model can legally reply in
                # plain text with no tool_calls. Re-prompt up to
                # MAX_TOOL_RETRIES times so the model converts its text into
                # a tool call. Streaming path is skipped — retry semantics on
                # a partially-consumed stream are not worth the complexity.
                chat_message = await self._retry_on_missing_tool_calls(
                    chat_message, input_messages,
                )

            # Record model output
            memory_step.model_output_message = chat_message
            memory_step.model_output = chat_message.content
            memory_step.token_usage = chat_message.token_usage
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)

        async for event in self.process_tool_calls(chat_message, memory_step):
            yield event

    async def process_tool_calls(self, chat_message: ChatMessage, memory_step: ActionStep) -> AsyncGenerator[StreamEvent]:
        """Process tool calls from the model output and update agent memory.

        Args:
            chat_message (`ChatMessage`): Chat message containing tool calls from the model.
            memory_step (`ActionStep)`: Memory ActionStep to update with results.

        Yields:
            `ActionOutput`: The final output of tool execution.
        """
        model_outputs = []
        tool_calls = []
        observations = []
        tool_results: list[dict[str, str]] = []

        final_answer_call = None
        parallel_calls = []
        parallel_tool_calls_meta = []
        assert chat_message.tool_calls is not None

        # --- Up-front premature-final-answer guard ---
        # Small models (e.g. Qwen-4B) autoregressively generate a `final_answer_tool`
        # call in the same turn as research/analysis calls, with a fabricated answer
        # argument. Inspect the raw tool-call list and decide what to keep before
        # we start yielding or recording anything. Ordering-independent; covers
        # duplicate finals too. Pure-function logic lives in `_tool_call_guard`
        # so it can be unit-tested without instantiating a full GeneralAgent.
        raw_tool_calls = list(chat_message.tool_calls)
        effective_tool_calls, guard_status = apply_final_answer_guard(raw_tool_calls)

        if guard_status == "premature":
            dropped = [tc for tc in raw_tool_calls if tc.function.name == "final_answer_tool"]
            self.logger.log(
                Text(
                    f"[premature-final-answer guard] Rejected {len(dropped)} "
                    f"final_answer_tool call(s) emitted alongside {len(effective_tool_calls)} other "
                    f"tool call(s). Dropped arguments: "
                    f"{[tc.function.arguments for tc in dropped]!r}. "
                    f"Other calls will run; model must regenerate final answer next turn.",
                    style=f"bold {YELLOW_HEX}",
                ),
                level=LogLevel.WARNING,
            )
        elif guard_status == "duplicate":
            dropped = [tc for tc in raw_tool_calls if tc.function.name == "final_answer_tool"]
            self.logger.log(
                Text(
                    f"[duplicate-final-answer guard] Received {len(dropped)} "
                    f"final_answer_tool calls with no other tools; dropping all and "
                    f"forcing regeneration. Arguments: "
                    f"{[tc.function.arguments for tc in dropped]!r}",
                    style=f"bold {YELLOW_HEX}",
                ),
                level=LogLevel.WARNING,
            )

        # Sync the chat_message so memory/message-replay sees only the kept calls.
        # `memory_step.model_output_message` holds a reference to this ChatMessage;
        # without this sync, next turn's API request would carry assistant tool_calls
        # with no matching role=tool messages, which vLLM/OpenAI reject as 400.
        if guard_status != "none":
            chat_message.tool_calls = effective_tool_calls or None

        for tool_call in effective_tool_calls:
            yield tool_call
            tool_name = tool_call.function.name
            tool_arguments = tool_call.function.arguments
            model_outputs.append(str(f"Called Tool: '{tool_name}' with arguments: {tool_arguments}"))
            tool_calls.append(ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call.id))
            # Track final_answer separately, add others to parallel processing list
            if tool_name == "final_answer_tool":
                final_answer_call = (tool_name, tool_arguments)
                break  # Stop: final answer reached, no further tool calls
            else:
                parallel_calls.append((tool_name, tool_arguments))
                parallel_tool_calls_meta.append(tool_call)

        # Helper function to process a single tool call
        async def process_single_tool_call(call_info):
            tool_name, tool_arguments = call_info
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            if tool_arguments is None:
                tool_arguments = {}
            tool_call_result = await self.execute_tool_call(tool_name, tool_arguments)
            tool_call_result_type = type(tool_call_result)
            if tool_call_result_type in [AgentImage, AgentAudio]:
                if tool_call_result_type == AgentImage:
                    observation_name = "image.png"
                elif tool_call_result_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: tool_call_result naming could allow for different names of same type
                self.state[observation_name] = tool_call_result
                observation = f"Stored '{observation_name}' in memory."
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            return observation

        # Process tool calls in parallel (Tier B: preserve tool_call order for tool_results)
        if parallel_calls:
            if len(parallel_calls) == 1:
                observation = await process_single_tool_call(parallel_calls[0])
                observations.append(observation)
                tool_results.append({"id": parallel_tool_calls_meta[0].id, "content": observation})
                yield ToolOutput(output=None, is_final_answer=False)
            else:
                tasks = [process_single_tool_call(call_info) for call_info in parallel_calls]
                results = await asyncio.gather(*tasks)
                observations.extend(results)
                for tc, obs in zip(parallel_tool_calls_meta, results):
                    tool_results.append({"id": tc.id, "content": obs})
                for _ in results:
                    yield ToolOutput(output=None, is_final_answer=False)

        # Process final_answer call if present
        if final_answer_call:
            tool_name, tool_arguments = final_answer_call
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            answer = (
                tool_arguments["answer"]
                if isinstance(tool_arguments, dict) and "answer" in tool_arguments
                else tool_arguments
            )
            if isinstance(answer, str) and answer in self.state.keys():
                # if the answer is a state variable, return the value
                # State variables are not JSON-serializable (AgentImage, AgentAudio) so can't be passed as arguments to execute_tool_call
                final_answer = self.state[answer]
                self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                # Allow arbitrary keywords
                final_answer = await self.execute_tool_call("final_answer_tool", tool_arguments)
                self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )
            memory_step.action_output = final_answer
            yield ToolOutput(output=final_answer, is_final_answer=True)

        # Update memory step with all results
        if model_outputs:
            memory_step.model_output = "\n".join(model_outputs)
        if tool_calls:
            memory_step.tool_calls = tool_calls
        if tool_results:
            memory_step.tool_results = tool_results
            memory_step.observations = "\n\n".join(tr["content"] for tr in tool_results)
        elif observations:
            memory_step.observations = "\n".join(observations)

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                return await tool(**arguments) if is_managed_agent else await tool(**arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, str):
                return await tool(arguments) if is_managed_agent else await tool(arguments, sanitize_inputs_outputs=True)
            else:
                raise TypeError(f"Unsupported arguments type: {type(arguments)}")

        except TypeError as e:
            # Handle invalid arguments
            description = getattr(tool, "description", "No description")
            if is_managed_agent:
                error_msg = (
                    f"Invalid request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this team member with a valid request.\n"
                    f"Team member description: {description}"
                )
            else:
                error_msg = (
                    f"Invalid call to tool '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this tool with correct input arguments.\n"
                    f"Expected parameters: {json.dumps(tool.parameters)}\n"
                    f"Returns output type: {tool.output_type}\n"
                    f"Tool description: '{description}'"
                )
            raise AgentToolCallError(error_msg, self.logger) from e

        except Exception as e:
            # Pass 3.1 RC2 diagnostics: when the inner exception is a Python
            # scope error (NameError / UnboundLocalError), log the FULL
            # exception chain with tracebacks before wrapping. The wrap below
            # reduces the inner exception to a single-line `str(e)`, which has
            # made it impossible to trace the reported `cannot access local
            # variable 'final_answer'` RC2 bug. One-line diagnostic behavior;
            # does not change error-propagation on the happy path.
            if isinstance(e, (NameError, UnboundLocalError)):
                import traceback
                chain_parts: list[str] = []
                cur: BaseException | None = e
                label = "root"
                depth = 0
                while cur is not None and depth < 10:
                    tb_str = "".join(traceback.format_exception(type(cur), cur, cur.__traceback__))
                    chain_parts.append(
                        f"--- [{depth}] {label}: {type(cur).__name__}: {cur} ---\n{tb_str}"
                    )
                    # Determine next link AND its label together (the label
                    # describes HOW we got from the current exception to the
                    # next, which is what makes the chain readable).
                    cause = getattr(cur, "__cause__", None)
                    context = getattr(cur, "__context__", None)
                    if cause is not None and cause is not cur:
                        nxt, label = cause, "__cause__"
                    elif context is not None and context is not cur:
                        nxt, label = context, "__context__"
                    else:
                        break
                    cur = nxt
                    depth += 1
                self.logger.log(
                    f"[RC2 diagnostic] Scope error in sub-agent '{tool_name}' — "
                    f"full exception chain follows:\n" + "\n".join(chain_parts),
                    level=LogLevel.ERROR,
                )

            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {json.dumps(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e