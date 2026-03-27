"""
ModifySubAgentTool: Modifies sub-agent capabilities at runtime.

This tool allows the parent agent to modify managed sub-agents by:
- Adding/removing tools
- Modifying instructions
- Creating new specialized agents
- Removing agents
"""

from typing import TYPE_CHECKING, Any, Optional

from src.tools import AsyncTool, Tool
from src.registry import TOOL
from src.logger import logger, LogLevel

if TYPE_CHECKING:
    from src.base.async_multistep_agent import AsyncMultiStepAgent


class ModifySubAgentTool(AsyncTool):
    """
    Modifies sub-agent capabilities to better handle tasks.
    
    Use this tool when:
    - A sub-agent needs additional tools to complete tasks
    - A sub-agent's instructions need clarification
    - A new specialized sub-agent is needed
    - An existing sub-agent should be removed or replaced
    """
    
    name = "modify_subagent"
    description = """Modify sub-agent capabilities to better handle tasks.

Available actions:
1. "add_agent": Create a new specialized sub-agent
2. "remove_agent": Remove an existing sub-agent
3. "add_existing_tool_to_agent": Add a registered tool to a sub-agent
4. "add_new_tool_to_agent": Generate and add a new tool to a sub-agent
5. "remove_tool_from_agent": Remove a tool from a sub-agent
6. "modify_agent_instructions": Update a sub-agent's instructions
7. "set_agent_max_steps": Change a sub-agent's max execution steps

All modifications are task-scoped (reset after task completion).

Examples:
- Add web search capability: action="add_existing_tool_to_agent", agent_name="deep_analyzer_agent", specification="web_searcher_tool"
- Create calculation specialist: action="add_agent", agent_name="math_agent", specification="A specialized agent for mathematical calculations using python"
- Add instructions: action="modify_agent_instructions", agent_name="browser_use_agent", specification="Focus on extracting specific data points, not general summaries" """

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: 'add_agent', 'remove_agent', 'add_existing_tool_to_agent', 'add_new_tool_to_agent', 'remove_tool_from_agent', 'modify_agent_instructions', 'set_agent_max_steps'"
            },
            "agent_name": {
                "type": "string",
                "description": "Name of the sub-agent to modify (or name for new agent)"
            },
            "specification": {
                "type": "string",
                "description": "Depends on action: add_agent (agent description), add_existing_tool_to_agent (tool name), add_new_tool_to_agent (tool description), modify_agent_instructions (additional instructions), set_agent_max_steps (new max_steps value), remove_* (optional reason)",
                "nullable": True
            }
        },
        "required": ["action", "agent_name"]
    }
    output_type = "string"
    
    # Actions that require specification
    ACTIONS_REQUIRING_SPEC = {
        "add_agent",
        "add_existing_tool_to_agent", 
        "add_new_tool_to_agent",
        "modify_agent_instructions",
        "set_agent_max_steps"
    }
    
    def __init__(self, parent_agent: "AsyncMultiStepAgent"):
        """
        Initialize the modification tool.
        
        Args:
            parent_agent: Reference to the parent agent that owns this tool
        """
        super().__init__()
        self.parent_agent = parent_agent
        
        # Lazy import to avoid circular dependencies
        self._tool_generator = None
        self._agent_generator = None
    
    def _find_agent(self, agent_name: str):
        """Look up agent in managed_agents first, then tools as fallback."""
        if agent_name in self.parent_agent.managed_agents:
            return self.parent_agent.managed_agents[agent_name]
        if hasattr(self.parent_agent, 'tools') and agent_name in self.parent_agent.tools:
            obj = self.parent_agent.tools[agent_name]
            if hasattr(obj, 'memory') or hasattr(obj, 'managed_agents'):
                return obj
        return None

    def _get_available_agent_names(self) -> list[str]:
        """List all agent names from both managed_agents and tools."""
        names = list(self.parent_agent.managed_agents.keys())
        if hasattr(self.parent_agent, 'tools'):
            for name, obj in self.parent_agent.tools.items():
                if hasattr(obj, 'memory') or hasattr(obj, 'managed_agents'):
                    if name not in names:
                        names.append(name)
        return names

    @property
    def tool_generator(self):
        """Lazy load ToolGenerator."""
        if self._tool_generator is None:
            from src.meta.tool_generator import ToolGenerator
            self._tool_generator = ToolGenerator(self.parent_agent.model)
        return self._tool_generator
    
    @property
    def agent_generator(self):
        """Lazy load AgentGenerator."""
        if self._agent_generator is None:
            from src.meta.agent_generator import AgentGenerator
            self._agent_generator = AgentGenerator()
        return self._agent_generator
    
    async def forward(
        self,
        action: str,
        agent_name: str,
        specification: Optional[str] = None
    ) -> str:
        """
        Execute a modification action.
        
        Args:
            action: The modification action to perform
            agent_name: Target agent name
            specification: Action-specific details
            
        Returns:
            Result message indicating success or failure
        """
        # Validate action
        valid_actions = [
            "add_agent", "remove_agent",
            "add_existing_tool_to_agent", "add_new_tool_to_agent", 
            "remove_tool_from_agent", "modify_agent_instructions",
            "set_agent_max_steps"
        ]
        
        if action not in valid_actions:
            return f"Error: Unknown action '{action}'. Valid actions: {valid_actions}"
        
        # Check if specification is required
        if action in self.ACTIONS_REQUIRING_SPEC and not specification:
            return f"Error: Action '{action}' requires a specification parameter"
        
        # Dispatch to appropriate handler
        handlers = {
            "add_agent": self._add_agent,
            "remove_agent": self._remove_agent,
            "add_existing_tool_to_agent": self._add_existing_tool,
            "add_new_tool_to_agent": self._add_new_tool,
            "remove_tool_from_agent": self._remove_tool,
            "modify_agent_instructions": self._modify_instructions,
            "set_agent_max_steps": self._set_max_steps,
        }
        
        try:
            result = await handlers[action](agent_name, specification)
            return result
        except Exception as e:
            logger.log(
                f"[ModifySubAgentTool] Error executing {action}: {e}",
                level=LogLevel.ERROR
            )
            return f"Error: Failed to execute {action}: {str(e)}"
    
    async def _add_agent(self, agent_name: str, specification: str) -> str:
        """Create and add a new specialized agent."""
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', agent_name):
            return f"Error: Invalid agent name '{agent_name}'. Must be a valid Python identifier (letters, numbers, underscores; cannot start with number)."
        
        if self._find_agent(agent_name) is not None:
            return f"Error: Agent '{agent_name}' already exists"
        
        try:
            # Determine tools based on specification
            tools_to_use = self._infer_tools_from_spec(specification)
            
            # Generate the agent
            new_agent = await self.agent_generator.generate_agent(
                name=agent_name,
                description=specification[:500],  # Use spec as description
                task_guidance=specification,
                tools=tools_to_use,
                model=self.parent_agent.model,
                max_steps=10
            )
            
            # Add to parent's managed agents
            success = self.parent_agent.add_managed_agent(new_agent)
            
            if success:
                return f"Successfully created and added agent '{agent_name}' with tools: {[t.name for t in tools_to_use]}"
            else:
                return f"Error: Failed to add agent '{agent_name}' to managed agents"
                
        except Exception as e:
            return f"Error: Failed to create agent: {str(e)}"
    
    async def _remove_agent(self, agent_name: str, specification: Optional[str]) -> str:
        """Remove an existing agent."""
        success = self.parent_agent.remove_managed_agent(agent_name)
        if not success and hasattr(self.parent_agent, 'tools') and agent_name in self.parent_agent.tools:
            obj = self.parent_agent.tools[agent_name]
            if hasattr(obj, 'memory') or hasattr(obj, 'managed_agents'):
                del self.parent_agent.tools[agent_name]
                success = True
        
        if success:
            return f"Successfully removed agent '{agent_name}'"
        else:
            available = self._get_available_agent_names()
            return f"Error: Agent '{agent_name}' not found. Available: {available}"
    
    async def _add_existing_tool(self, agent_name: str, tool_name: str) -> str:
        """Add an existing registered tool to an agent."""
        if tool_name not in TOOL.module_dict:
            available_tools = list(TOOL.module_dict.keys())[:20]
            return f"Error: Tool '{tool_name}' not found in registry. Some available: {available_tools}"
        
        agent = self._find_agent(agent_name)
        if agent is None:
            available = self._get_available_agent_names()
            return f"Error: Agent '{agent_name}' not found. Available: {available}"
        
        try:
            tool = TOOL.build({"type": tool_name})
            
            success = self.parent_agent.add_tool_to_agent(agent_name, tool)
            if not success and hasattr(agent, 'tools'):
                agent.tools[tool.name] = tool
                success = True
            
            if success:
                return f"Successfully added tool '{tool_name}' to agent '{agent_name}'"
            else:
                return f"Error: Failed to add tool to agent"
                
        except Exception as e:
            return f"Error: Failed to build/add tool: {str(e)}"
    
    async def _add_new_tool(self, agent_name: str, specification: str) -> str:
        """Generate a new tool and add it to an agent."""
        agent = self._find_agent(agent_name)
        if agent is None:
            available = self._get_available_agent_names()
            return f"Error: Agent '{agent_name}' not found. Available: {available}"
        
        tool_name = self._generate_tool_name(specification)
        
        try:
            tool_code = await self.tool_generator.generate_tool_code(
                requirement=specification,
                tool_name=tool_name
            )
            
            success = self.parent_agent.add_new_tool_to_agent(agent_name, tool_code)
            
            if success:
                return f"Successfully created and added new tool '{tool_name}' to agent '{agent_name}'"
            else:
                return f"Error: Failed to add generated tool to agent"
                
        except Exception as e:
            return f"Error: Failed to generate/add tool: {str(e)}"
    
    async def _remove_tool(self, agent_name: str, tool_name: str) -> str:
        """Remove a tool from an agent."""
        if not tool_name:
            return "Error: Must specify tool name to remove"
        
        agent = self._find_agent(agent_name)
        if agent is None:
            available = self._get_available_agent_names()
            return f"Error: Agent '{agent_name}' not found. Available: {available}"
        
        success = self.parent_agent.remove_tool_from_agent(agent_name, tool_name)
        if not success and hasattr(agent, 'tools') and tool_name in agent.tools:
            if tool_name not in ["final_answer_tool", "final_answer"]:
                del agent.tools[tool_name]
                success = True
        
        if success:
            return f"Successfully removed tool '{tool_name}' from agent '{agent_name}'"
        else:
            return f"Error: Failed to remove tool. Check that agent and tool exist."
    
    async def _modify_instructions(self, agent_name: str, instructions: str) -> str:
        """Modify an agent's instructions."""
        success = self.parent_agent.modify_agent_instructions(agent_name, instructions)
        if not success:
            agent = self._find_agent(agent_name)
            if agent is not None and hasattr(agent, 'prompt_templates'):
                current = agent.prompt_templates.get("task_instruction", "")
                agent.prompt_templates["task_instruction"] = (
                    f"{current}\n\nAdditional Instructions:\n{instructions}"
                )
                success = True
        
        if success:
            return f"Successfully updated instructions for agent '{agent_name}'"
        else:
            available = self._get_available_agent_names()
            return f"Error: Agent '{agent_name}' not found. Available: {available}"
    
    async def _set_max_steps(self, agent_name: str, max_steps_str: str) -> str:
        """Set an agent's max_steps."""
        try:
            max_steps = int(max_steps_str)
        except ValueError:
            return f"Error: max_steps must be a number, got '{max_steps_str}'"
        
        success = self.parent_agent.set_agent_max_steps(agent_name, max_steps)
        if not success:
            agent = self._find_agent(agent_name)
            if agent is not None and hasattr(agent, 'max_steps'):
                max_steps = max(1, min(50, max_steps))
                agent.max_steps = max_steps
                success = True
        
        if success:
            return f"Successfully set max_steps={max_steps} for agent '{agent_name}'"
        else:
            available = self._get_available_agent_names()
            return f"Error: Agent '{agent_name}' not found. Available: {available}"
    
    def _infer_tools_from_spec(self, specification: str) -> list:
        """
        Infer which tools to give a new agent based on specification.
        
        Args:
            specification: Natural language description
            
        Returns:
            List of tool instances
        """
        tools = []
        spec_lower = specification.lower()
        
        # Always include python_interpreter if math/calculation mentioned
        if any(kw in spec_lower for kw in ['math', 'calcul', 'compute', 'statistic', 'data']):
            try:
                tools.append(TOOL.build({"type": "python_interpreter_tool"}))
            except Exception as e:
                logger.log(f"[ModifySubAgentTool] Failed to build python_interpreter_tool: {e}", level=LogLevel.WARNING)
        
        # Include web tools if search/research mentioned
        if any(kw in spec_lower for kw in ['search', 'web', 'research', 'find', 'look up']):
            try:
                tools.append(TOOL.build({"type": "web_searcher_tool"}))
            except Exception as e:
                logger.log(f"[ModifySubAgentTool] Failed to build web_searcher_tool: {e}", level=LogLevel.WARNING)
        
        # Include browser if browsing/interaction mentioned
        if any(kw in spec_lower for kw in ['browse', 'website', 'interact', 'page']):
            try:
                tools.append(TOOL.build({"type": "auto_browser_use_tool"}))
            except Exception as e:
                logger.log(f"[ModifySubAgentTool] Failed to build auto_browser_use_tool: {e}", level=LogLevel.WARNING)
        
        # Default to python_interpreter if nothing matched
        if not tools:
            try:
                tools.append(TOOL.build({"type": "python_interpreter_tool"}))
            except Exception as e:
                logger.log(f"[ModifySubAgentTool] Failed to build default python_interpreter_tool: {e}", level=LogLevel.WARNING)
        
        return tools
    
    def _generate_tool_name(self, specification: str) -> str:
        """
        Generate a tool name from specification.
        
        Args:
            specification: Natural language description
            
        Returns:
            Snake_case tool name
        """
        import re
        
        # Extract key words from specification
        words = specification.lower().split()[:5]  # First 5 words
        # Remove common words
        stopwords = {'a', 'an', 'the', 'to', 'for', 'and', 'or', 'that', 'which', 'can', 'will'}
        words = [w for w in words if w not in stopwords]
        
        # Clean and join
        clean_words = [re.sub(r'[^a-z0-9]', '', w) for w in words]
        clean_words = [w for w in clean_words if w]
        
        if clean_words:
            name = '_'.join(clean_words[:3])  # Max 3 words
        else:
            name = "custom"
        
        return f"{name}_tool"
