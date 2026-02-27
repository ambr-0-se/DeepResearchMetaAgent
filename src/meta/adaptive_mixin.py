"""
AdaptiveMixin: Provides runtime modification capabilities for agents.

This mixin enables agents to modify their managed sub-agents' tools, 
instructions, and create/remove agents dynamically during task execution.
All modifications are task-scoped and reset after each task.
"""

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.memory import SystemPromptStep
from src.tools import AsyncTool, Tool
from src.logger import logger, LogLevel

if TYPE_CHECKING:
    from src.base.async_multistep_agent import AsyncMultiStepAgent


class AdaptiveMixin:
    """
    Mixin that adds runtime tool/agent/prompt modification capabilities.
    
    Must be mixed into AsyncMultiStepAgent or its subclasses.
    All modifications are task-scoped and reset after task completion.
    """
    
    # ========== STATE MANAGEMENT ==========
    
    def _store_original_state(self: "AsyncMultiStepAgent") -> None:
        """
        Store original state before task execution.
        Called at the start of run() to enable reset after task.
        """
        self._original_state = {
            "tools": dict(self.tools),  # Shallow copy of tools dict
            "managed_agents": dict(self.managed_agents),  # Shallow copy of managed agents dict
            "agent_states": {},  # Deep state for each managed agent
        }
        
        # Store each managed agent's modifiable state
        for agent_name, agent in self.managed_agents.items():
            self._original_state["agent_states"][agent_name] = {
                "tools": dict(agent.tools) if hasattr(agent, 'tools') else {},
                "task_instruction_template": copy.deepcopy(
                    agent.prompt_templates.get("task_instruction", "")
                ) if hasattr(agent, 'prompt_templates') else "",
                "max_steps": getattr(agent, 'max_steps', None),
            }
        
        logger.log(
            f"[AdaptiveMixin] Stored original state: {len(self.tools)} tools, "
            f"{len(self.managed_agents)} managed agents",
            level=LogLevel.DEBUG
        )
    
    def _reset_to_original_state(self: "AsyncMultiStepAgent") -> None:
        """
        Reset to original state after task execution.
        Called in finally block of run() to ensure cleanup.
        """
        if not hasattr(self, '_original_state'):
            logger.log(
                "[AdaptiveMixin] No original state to restore",
                level=LogLevel.WARNING
            )
            return
        
        # Restore parent agent's tools and managed agents
        self.tools = self._original_state["tools"]
        self.managed_agents = self._original_state["managed_agents"]
        
        # Restore each managed agent's internal state
        for agent_name, original in self._original_state["agent_states"].items():
            if agent_name in self.managed_agents:
                agent = self.managed_agents[agent_name]
                
                # Restore tools
                if hasattr(agent, 'tools'):
                    agent.tools = original["tools"]
                
                # Restore task_instruction template
                if hasattr(agent, 'prompt_templates') and original["task_instruction_template"]:
                    agent.prompt_templates["task_instruction"] = original["task_instruction_template"]
                
                # Restore max_steps
                if original["max_steps"] is not None and hasattr(agent, 'max_steps'):
                    agent.max_steps = original["max_steps"]
        
        # Refresh system prompt to reflect restored state
        self._refresh_system_prompt()
        
        # Clean up
        del self._original_state
        
        logger.log(
            "[AdaptiveMixin] Reset to original state",
            level=LogLevel.DEBUG
        )
    
    def _refresh_system_prompt(self: "AsyncMultiStepAgent") -> None:
        """Refresh system prompt after modifications."""
        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
    
    # ========== TOOL MANAGEMENT FOR SUB-AGENTS ==========
    
    def add_tool_to_agent(
        self: "AsyncMultiStepAgent",
        agent_name: str,
        tool: AsyncTool
    ) -> bool:
        """
        Add an existing tool to a managed sub-agent.
        
        Args:
            agent_name: Name of the managed agent to modify
            tool: Tool instance to add
            
        Returns:
            True if successful, False otherwise
        """
        if agent_name not in self.managed_agents:
            logger.log(
                f"[AdaptiveMixin] Agent '{agent_name}' not found in managed agents",
                level=LogLevel.WARNING
            )
            return False
        
        agent = self.managed_agents[agent_name]
        
        if not hasattr(agent, 'tools'):
            logger.log(
                f"[AdaptiveMixin] Agent '{agent_name}' does not have tools attribute",
                level=LogLevel.WARNING
            )
            return False
        
        # Add the tool
        agent.tools[tool.name] = tool
        
        # Refresh agent's system prompt if possible
        if hasattr(agent, 'initialize_system_prompt') and hasattr(agent, 'memory'):
            agent.system_prompt = agent.initialize_system_prompt()
            agent.memory.system_prompt = SystemPromptStep(system_prompt=agent.system_prompt)
        
        logger.log(
            f"[AdaptiveMixin] Added tool '{tool.name}' to agent '{agent_name}'",
            level=LogLevel.INFO
        )
        return True
    
    def add_new_tool_to_agent(
        self: "AsyncMultiStepAgent",
        agent_name: str,
        tool_code: str
    ) -> bool:
        """
        Create a new tool from code and add it to a managed sub-agent.
        
        Args:
            agent_name: Name of the managed agent to modify
            tool_code: Python code defining a Tool subclass
            
        Returns:
            True if successful, False otherwise
        """
        if agent_name not in self.managed_agents:
            logger.log(
                f"[AdaptiveMixin] Agent '{agent_name}' not found",
                level=LogLevel.WARNING
            )
            return False
        
        try:
            # Create tool from code
            tool = Tool.from_code(tool_code)
            
            # Wrap as async if needed
            if not isinstance(tool, AsyncTool):
                tool = self._wrap_as_async_tool(tool)
            
            return self.add_tool_to_agent(agent_name, tool)
            
        except Exception as e:
            logger.log(
                f"[AdaptiveMixin] Failed to create tool from code: {e}",
                level=LogLevel.ERROR
            )
            return False
    
    def remove_tool_from_agent(
        self: "AsyncMultiStepAgent",
        agent_name: str,
        tool_name: str
    ) -> bool:
        """
        Remove a tool from a managed sub-agent.
        
        Args:
            agent_name: Name of the managed agent to modify
            tool_name: Name of the tool to remove
            
        Returns:
            True if successful, False otherwise
        """
        if agent_name not in self.managed_agents:
            logger.log(
                f"[AdaptiveMixin] Agent '{agent_name}' not found",
                level=LogLevel.WARNING
            )
            return False
        
        agent = self.managed_agents[agent_name]
        
        if not hasattr(agent, 'tools') or tool_name not in agent.tools:
            logger.log(
                f"[AdaptiveMixin] Tool '{tool_name}' not found in agent '{agent_name}'",
                level=LogLevel.WARNING
            )
            return False
        
        # Don't allow removing final_answer_tool
        if tool_name in ["final_answer_tool", "final_answer"]:
            logger.log(
                f"[AdaptiveMixin] Cannot remove essential tool '{tool_name}'",
                level=LogLevel.WARNING
            )
            return False
        
        del agent.tools[tool_name]
        
        # Refresh agent's system prompt
        if hasattr(agent, 'initialize_system_prompt') and hasattr(agent, 'memory'):
            agent.system_prompt = agent.initialize_system_prompt()
            agent.memory.system_prompt = SystemPromptStep(system_prompt=agent.system_prompt)
        
        logger.log(
            f"[AdaptiveMixin] Removed tool '{tool_name}' from agent '{agent_name}'",
            level=LogLevel.INFO
        )
        return True
    
    # ========== INSTRUCTION MANAGEMENT FOR SUB-AGENTS ==========
    
    def modify_agent_instructions(
        self: "AsyncMultiStepAgent",
        agent_name: str,
        additional_instructions: str
    ) -> bool:
        """
        Append additional instructions to a managed sub-agent's task_instruction.
        
        Args:
            agent_name: Name of the managed agent to modify
            additional_instructions: Instructions to append
            
        Returns:
            True if successful, False otherwise
        """
        if agent_name not in self.managed_agents:
            logger.log(
                f"[AdaptiveMixin] Agent '{agent_name}' not found",
                level=LogLevel.WARNING
            )
            return False
        
        agent = self.managed_agents[agent_name]
        
        if not hasattr(agent, 'prompt_templates'):
            logger.log(
                f"[AdaptiveMixin] Agent '{agent_name}' does not have prompt_templates",
                level=LogLevel.WARNING
            )
            return False
        
        # Append to task_instruction
        current = agent.prompt_templates.get("task_instruction", "")
        agent.prompt_templates["task_instruction"] = f"""{current}

Additional Instructions:
{additional_instructions}"""
        
        logger.log(
            f"[AdaptiveMixin] Modified instructions for agent '{agent_name}'",
            level=LogLevel.INFO
        )
        return True
    
    def set_agent_max_steps(
        self: "AsyncMultiStepAgent",
        agent_name: str,
        max_steps: int
    ) -> bool:
        """
        Set the maximum steps for a managed sub-agent.
        
        Args:
            agent_name: Name of the managed agent to modify
            max_steps: New max_steps value (bounded 1-50)
            
        Returns:
            True if successful, False otherwise
            
        Note:
            max_steps is bounded between 1 and 50 to prevent:
            - Values too low (< 1): agent cannot make progress
            - Values too high (> 50): excessive API costs and latency
        """
        if agent_name not in self.managed_agents:
            logger.log(
                f"[AdaptiveMixin] Agent '{agent_name}' not found",
                level=LogLevel.WARNING
            )
            return False
        
        # Bound the value (see docstring for rationale)
        max_steps = max(1, min(50, max_steps))
        
        agent = self.managed_agents[agent_name]
        agent.max_steps = max_steps
        
        logger.log(
            f"[AdaptiveMixin] Set max_steps={max_steps} for agent '{agent_name}'",
            level=LogLevel.INFO
        )
        return True
    
    # ========== MANAGED AGENT MANAGEMENT ==========
    
    def add_managed_agent(
        self: "AsyncMultiStepAgent",
        agent: "AsyncMultiStepAgent"
    ) -> bool:
        """
        Add a new managed agent to this agent.
        
        Args:
            agent: Agent instance to add as managed agent
            
        Returns:
            True if successful, False otherwise
        """
        if not agent.name or not agent.description:
            logger.log(
                "[AdaptiveMixin] Managed agent must have name and description",
                level=LogLevel.WARNING
            )
            return False
        
        if agent.name in self.managed_agents:
            logger.log(
                f"[AdaptiveMixin] Agent '{agent.name}' already exists",
                level=LogLevel.WARNING
            )
            return False
        
        # Set up agent interface for tool-like calling
        agent.inputs = {
            "task": {"type": "string", "description": "Task description for the agent."},
            "additional_args": {"type": "object", "description": "Extra context/data."},
        }
        agent.parameters = {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task description for the agent."},
                "additional_args": {"type": "object", "description": "Extra context/data."},
            },
            "required": ["task"],
        }
        agent.output_type = "string"
        
        # Add to managed agents
        self.managed_agents[agent.name] = agent
        
        # Refresh system prompt to include new agent
        self._refresh_system_prompt()
        
        logger.log(
            f"[AdaptiveMixin] Added managed agent '{agent.name}'",
            level=LogLevel.INFO
        )
        return True
    
    def remove_managed_agent(
        self: "AsyncMultiStepAgent",
        agent_name: str
    ) -> bool:
        """
        Remove a managed agent from this agent.
        
        Args:
            agent_name: Name of the agent to remove
            
        Returns:
            True if successful, False otherwise
        """
        if agent_name not in self.managed_agents:
            logger.log(
                f"[AdaptiveMixin] Agent '{agent_name}' not found",
                level=LogLevel.WARNING
            )
            return False
        
        del self.managed_agents[agent_name]
        
        # Refresh system prompt
        self._refresh_system_prompt()
        
        logger.log(
            f"[AdaptiveMixin] Removed managed agent '{agent_name}'",
            level=LogLevel.INFO
        )
        return True
    
    # ========== UTILITY METHODS ==========
    
    def _wrap_as_async_tool(
        self: "AsyncMultiStepAgent",
        tool: Tool
    ) -> AsyncTool:
        """
        Wrap a synchronous Tool as an AsyncTool.
        
        Args:
            tool: Synchronous tool to wrap
            
        Returns:
            AsyncTool wrapper
            
        Raises:
            ValueError: If tool lacks required attributes
        """
        # Validate tool has required attributes
        if not hasattr(tool, 'name') or not tool.name:
            raise ValueError(f"Tool must have a valid 'name' attribute")
        if not hasattr(tool, 'description'):
            raise ValueError(f"Tool '{tool.name}' must have a 'description' attribute")
        if not hasattr(tool, 'inputs'):
            raise ValueError(f"Tool '{tool.name}' must have an 'inputs' attribute")
        if not hasattr(tool, 'output_type'):
            raise ValueError(f"Tool '{tool.name}' must have an 'output_type' attribute")
        
        class WrappedAsyncTool(AsyncTool):
            name = tool.name
            description = tool.description
            inputs = tool.inputs
            output_type = tool.output_type
            
            def __init__(self):
                super().__init__()
                self._sync_tool = tool
            
            async def forward(self, **kwargs):
                return self._sync_tool.forward(**kwargs)
        
        return WrappedAsyncTool()
    
    def get_managed_agent_memory(
        self: "AsyncMultiStepAgent",
        agent_name: str
    ) -> Optional[Any]:
        """
        Get the memory of a managed sub-agent.
        
        Args:
            agent_name: Name of the managed agent
            
        Returns:
            Agent's memory object or None if not found
        """
        if agent_name not in self.managed_agents:
            return None
        
        agent = self.managed_agents[agent_name]
        return getattr(agent, 'memory', None)
    
    def get_managed_agent_tools(
        self: "AsyncMultiStepAgent",
        agent_name: str
    ) -> Dict[str, AsyncTool]:
        """
        Get the tools of a managed sub-agent.
        
        Args:
            agent_name: Name of the managed agent
            
        Returns:
            Dict of tool name to tool instance
        """
        if agent_name not in self.managed_agents:
            return {}
        
        agent = self.managed_agents[agent_name]
        return getattr(agent, 'tools', {})
