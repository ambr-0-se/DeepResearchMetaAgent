"""
AdaptivePlanningAgent: Planning agent with runtime self-modification capabilities.

This agent extends PlanningAgent with the ability to:
- Diagnose sub-agent failures by examining their execution history
- Modify sub-agents' tools, instructions, and capabilities at runtime
- Create new specialized sub-agents dynamically
- All modifications are task-scoped and reset after each task
"""

from typing import Any, Optional
import yaml

from src.agent.planning_agent import PlanningAgent
from src.base.async_multistep_agent import PromptTemplates
from src.memory import AgentMemory
from src.models import Model
from src.registry import AGENT
from src.utils import assemble_project_path
from src.logger import logger, LogLevel

# Import meta-agent components
from src.meta.adaptive_mixin import AdaptiveMixin
from src.meta.diagnose_tool import DiagnoseSubAgentTool
from src.meta.modify_tool import ModifySubAgentTool


@AGENT.register_module(name="adaptive_planning_agent", force=True)
class AdaptivePlanningAgent(AdaptiveMixin, PlanningAgent):
    """
    Planning agent with THINK-ACT-OBSERVE loop and reactive self-modification capabilities.

    Extends PlanningAgent with:
    - AdaptiveMixin for runtime modifications to sub-agents
    - diagnose_subagent tool for investigating sub-agent failures (reactive; agent-invoked)
    - modify_subagent tool for modifying sub-agent capabilities (reactive; agent-invoked)
    - Enhanced prompt template with reactive diagnose/modify guidance
    - State management for task-scoped changes

    Note: a structural REVIEW step (automatic post-delegation assessment) is added by the
    C3 variant via the optional `review_step` component; see `src/meta/review_step.py`.

    All modifications made during a task are automatically reset after task completion,
    ensuring isolation between tasks.
    """
    
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
        """
        Initialize the AdaptivePlanningAgent.
        
        Args:
            config: Agent configuration
            tools: List of tools for the agent
            model: LLM model to use
            prompt_templates: Optional custom prompt templates
            planning_interval: Steps between planning phases
            stream_outputs: Whether to stream outputs
            max_tool_threads: Max threads for parallel tool execution
            **kwargs: Additional arguments passed to parent
        """
        # Call parent constructor
        super().__init__(
            config=config,
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            stream_outputs=stream_outputs,
            max_tool_threads=max_tool_threads,
            **kwargs,
        )
        
        # Add adaptive tools (diagnose and modify)
        self._add_adaptive_tools()
        
        # Refresh system prompt to include the new tools
        self._refresh_system_prompt()
        
        logger.log(
            f"[AdaptivePlanningAgent] Initialized with {len(self.tools)} tools, "
            f"{len(self.managed_agents)} managed agents",
            level=LogLevel.INFO
        )
    
    def _add_adaptive_tools(self) -> None:
        """Add the diagnostic and modification tools."""
        # Add diagnose_subagent tool
        diagnose_tool = DiagnoseSubAgentTool(self)
        self.tools[diagnose_tool.name] = diagnose_tool
        
        # Add modify_subagent tool
        modify_tool = ModifySubAgentTool(self)
        self.tools[modify_tool.name] = modify_tool
        
        logger.log(
            "[AdaptivePlanningAgent] Added adaptive tools: diagnose_subagent, modify_subagent",
            level=LogLevel.DEBUG
        )
    
    async def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        Run the agent on a task with state management.
        
        Stores original state before execution and restores it after,
        ensuring all modifications are task-scoped.
        
        Args:
            task: The task to execute
            stream: Whether to stream outputs
            reset: Whether to reset memory before running
            images: Optional images for the task
            additional_args: Additional arguments
            max_steps: Override for max steps
            
        Returns:
            Task result
        """
        # Store original state before task execution
        self._store_original_state()
        
        logger.log(
            f"[AdaptivePlanningAgent] Starting task with adaptive capabilities",
            level=LogLevel.INFO
        )
        
        try:
            # Execute the task using parent's run method
            result = await super().run(
                task=task,
                stream=stream,
                reset=reset,
                images=images,
                additional_args=additional_args,
                max_steps=max_steps,
            )
            return result
            
        finally:
            # Always reset state after task, even on error
            self._reset_to_original_state()
            logger.log(
                "[AdaptivePlanningAgent] Task complete, state reset",
                level=LogLevel.DEBUG
            )
    
    def initialize_system_prompt(self) -> str:
        """
        Initialize the system prompt for the agent.
        
        Overrides parent to ensure adaptive tools are properly included.
        
        Returns:
            The initialized system prompt string
        """
        from src.base.async_multistep_agent import populate_template
        
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents
            },
        )
        return system_prompt
    
    def get_adaptive_status(self) -> dict:
        """
        Get the current adaptive state for debugging/monitoring.
        
        Returns:
            Dictionary with current adaptive state information
        """
        return {
            "tools": list(self.tools.keys()),
            "managed_agents": list(self.managed_agents.keys()),
            "has_original_state": hasattr(self, '_original_state'),
            "managed_agent_tools": {
                name: list(agent.tools.keys()) if hasattr(agent, 'tools') else []
                for name, agent in self.managed_agents.items()
            }
        }
