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
from src.utils.agent_types import (
    AgentAudio,
    AgentImage,
)
from src.registry import AGENT
from src.utils import assemble_project_path


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
        """Initialize the task instruction for the agent."""
        task_instruction = populate_template(
            self.prompt_templates["task_instruction"],
            variables={"task": self.task},
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
        """Rough token estimate from message character count (chars / 3.5)."""
        total_chars = 0
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total_chars += len(part["text"])
                    else:
                        total_chars += len(str(part))
            else:
                total_chars += len(str(content))
        return int(total_chars / 3.5)

    def _prune_messages_if_needed(self, messages: list) -> list:
        """Prune conversation history if approaching the model's context limit.

        Keeps the system prompt (first message) and the most recent messages,
        truncating middle messages to tool-call summaries.
        """
        max_model_len = getattr(self.config, 'max_model_len', 32768)
        threshold = int(max_model_len * 0.85)

        estimated_tokens = self._estimate_token_count(messages)
        if estimated_tokens <= threshold:
            return messages

        logger.warning(
            f"[Context Pruning] Estimated {estimated_tokens} tokens exceeds "
            f"threshold {threshold} (max_model_len={max_model_len}). Pruning..."
        )

        if len(messages) <= 4:
            return messages

        keep_tail = min(4, len(messages) - 1)
        head = messages[:1]
        tail = messages[-keep_tail:]
        middle = messages[1:-keep_tail]

        pruned_middle = []
        for msg in middle:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if isinstance(content, list):
                text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                full_text = " ".join(text_parts)
            else:
                full_text = str(content)

            if len(full_text) > 500:
                truncated = full_text[:250] + " ... [truncated] ... " + full_text[-200:]
                init_kwargs = {"role": msg.role, "content": [{"type": "text", "text": truncated}]}
                for attr in ("tool_calls", "tool_call_id", "name", "raw"):
                    val = getattr(msg, attr, None)
                    if val is not None:
                        init_kwargs[attr] = val
                try:
                    pruned_msg = ChatMessage(**init_kwargs)
                except TypeError:
                    pruned_msg = ChatMessage(role=msg.role, content=[{"type": "text", "text": truncated}])
                pruned_middle.append(pruned_msg)
            else:
                pruned_middle.append(msg)

        result = head + pruned_middle + tail
        new_est = self._estimate_token_count(result)
        logger.warning(
            f"[Context Pruning] Pruned {len(messages)} messages -> {len(result)} messages, "
            f"estimated tokens: {estimated_tokens} -> {new_est}"
        )
        return result

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
        for tool_call in chat_message.tool_calls:
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