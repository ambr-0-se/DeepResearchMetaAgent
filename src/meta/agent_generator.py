"""
AgentGenerator: Creates new agents with in-memory templates.

This module allows dynamic creation of GeneralAgent instances using
modified copies of existing templates, without requiring file I/O.
"""

import copy
import yaml
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.utils import assemble_project_path
from src.logger import logger, LogLevel

if TYPE_CHECKING:
    from src.tools import AsyncTool
    from src.models import Model
    from src.agent.general_agent import GeneralAgent


class AgentGenerator:
    """
    Generates new agents dynamically using in-memory templates.
    
    Uses the general_agent.yaml template as a base and modifies it
    for specialized agents without requiring file modifications.
    """
    
    # Cache for base template
    _base_template: Optional[Dict] = None
    
    @classmethod
    def _get_base_template(cls) -> Dict:
        """
        Load and cache the general_agent template.
        
        Returns:
            Dictionary containing the base prompt templates
        """
        if cls._base_template is None:
            template_path = assemble_project_path(
                "src/agent/general_agent/prompts/general_agent.yaml"
            )
            with open(template_path, "r") as f:
                cls._base_template = yaml.safe_load(f)
            logger.log(
                "[AgentGenerator] Loaded base template from general_agent.yaml",
                level=LogLevel.DEBUG
            )
        return cls._base_template
    
    async def generate_agent(
        self,
        name: str,
        description: str,
        task_guidance: str,
        tools: List["AsyncTool"],
        model: "Model",
        max_steps: int = 10,
    ) -> "GeneralAgent":
        """
        Generate a new agent with custom task instructions.
        
        Args:
            name: Agent name (e.g., "math_specialist_agent")
            description: What the agent does (shown to parent agent)
            task_guidance: Custom task_instruction content
            tools: List of tool instances to give the agent
            model: The model to use
            max_steps: Maximum execution steps
            
        Returns:
            A new GeneralAgent instance with in-memory templates
        """
        # Deep copy the base template
        prompt_templates = copy.deepcopy(self._get_base_template())
        
        # Customize the task_instruction with the provided guidance
        prompt_templates["task_instruction"] = f"""{task_guidance}

Here is the task:
{{{{task}}}}"""
        
        # Create the agent using the factory method
        agent = self._create_agent_instance(
            name=name,
            description=description,
            prompt_templates=prompt_templates,
            tools=tools,
            model=model,
            max_steps=max_steps,
        )
        
        logger.log(
            f"[AgentGenerator] Created agent '{name}' with {len(tools)} tools",
            level=LogLevel.INFO
        )
        
        return agent
    
    def _create_agent_instance(
        self,
        name: str,
        description: str,
        prompt_templates: Dict,
        tools: List["AsyncTool"],
        model: "Model",
        max_steps: int,
    ) -> "GeneralAgent":
        """
        Create a GeneralAgent instance with custom templates.
        
        This bypasses the file-loading in GeneralAgent.__init__ by
        manually initializing all necessary attributes.
        
        Args:
            name: Agent name
            description: Agent description
            prompt_templates: Custom prompt templates dict
            tools: Tools for the agent
            model: Model to use
            max_steps: Max steps
            
        Returns:
            Configured GeneralAgent instance
        """
        from src.agent.general_agent import GeneralAgent
        from src.memory import AgentMemory
        from src.logger import AgentLogger, LogLevel as LL
        from src.base.async_multistep_agent import populate_template
        
        # Create a minimal config object that won't cause file loading issues
        class DynamicConfig:
            """Minimal config for dynamically created agents."""
            def __init__(self, name, description, max_steps):
                self.name = name
                self.description = description
                self.max_steps = max_steps
                self.template_path = None  # Signals to skip file loading
                self.provide_run_summary = True
                
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config = DynamicConfig(name, description, max_steps)
        
        # Create agent without calling __init__ (we'll initialize manually)
        agent = object.__new__(GeneralAgent)
        
        # Initialize core attributes
        agent.config = config
        agent.model = model
        agent.name = name
        agent.description = description
        agent.max_steps = max_steps
        agent.provide_run_summary = True
        
        # Initialize tools
        agent.tools = {tool.name: tool for tool in tools}
        # Add final_answer_tool if not present
        if "final_answer_tool" not in agent.tools and "final_answer" not in agent.tools:
            from src.tools.final_answer import FinalAnswerTool
            agent.tools["final_answer_tool"] = FinalAnswerTool()
        
        # Initialize managed agents (empty for generated agents)
        agent.managed_agents = {}
        
        # Set prompt templates
        agent.prompt_templates = prompt_templates
        
        # Initialize prompts
        agent.system_prompt = populate_template(
            prompt_templates["system_prompt"],
            variables={"tools": agent.tools, "managed_agents": agent.managed_agents}
        )
        agent.user_prompt = populate_template(
            prompt_templates.get("user_prompt", ""),
            variables={}
        )
        
        # Initialize memory
        agent.memory = AgentMemory(
            system_prompt=agent.system_prompt,
            user_prompt=agent.user_prompt,
        )
        
        # Initialize other required attributes
        agent.state = {}
        agent.step_number = 0
        agent.stream_outputs = False
        agent.max_tool_threads = None
        agent.logger = AgentLogger(level=LL.INFO)
        agent.final_answer_checks = []
        agent.return_full_result = False
        agent.step_callbacks = []
        agent.planning_interval = None
        agent.task = None
        agent.interrupt_switch = False
        agent.grammar = None
        agent.instructions = None
        
        # Set up interface for tool-like calling (when used as managed agent)
        agent.inputs = {
            "task": {"type": "string", "description": "Task description for the agent."},
            "additional_args": {"type": "object", "description": "Extra context/data."},
        }
        agent.output_type = "string"
        
        # Initialize monitor
        from src.logger import Monitor
        agent.monitor = Monitor(agent.model, agent.logger)
        agent.step_callbacks.append(agent.monitor.update_metrics)
        
        return agent
    
    async def generate_specialized_agent(
        self,
        name: str,
        specialization: str,
        model: "Model",
        include_python: bool = True,
    ) -> "GeneralAgent":
        """
        Generate a specialized agent with preset configuration.
        
        Convenience method for common specializations.
        
        Args:
            name: Agent name
            specialization: One of 'math', 'research', 'analysis', 'general'
            model: Model to use
            include_python: Whether to include python_interpreter_tool
            
        Returns:
            Configured agent for the specialization
        """
        from src.registry import TOOL
        
        # Specialization presets
        presets = {
            "math": {
                "description": "Specialized agent for mathematical calculations and data analysis",
                "guidance": """You are a mathematics and calculation specialist.
* Use python_interpreter_tool for all calculations
* Show your work step by step
* Verify calculations when possible
* Be precise with numerical answers""",
                "tools": ["python_interpreter_tool"]
            },
            "research": {
                "description": "Specialized agent for web research and information gathering",
                "guidance": """You are a research specialist.
* Search for information from reliable sources
* Verify facts when possible
* Cite sources in your findings
* Provide comprehensive summaries""",
                "tools": ["web_searcher_tool", "python_interpreter_tool"]
            },
            "analysis": {
                "description": "Specialized agent for systematic analysis and reasoning",
                "guidance": """You are an analysis specialist.
* Break down problems systematically
* Consider multiple perspectives
* Use structured reasoning
* Support conclusions with evidence""",
                "tools": ["python_interpreter_tool"]
            },
            "general": {
                "description": "General-purpose agent for various tasks",
                "guidance": """You are a helpful assistant.
* Approach tasks methodically
* Use available tools effectively
* Ask for clarification if needed""",
                "tools": ["python_interpreter_tool"]
            }
        }
        
        spec = presets.get(specialization, presets["general"])
        
        # Build tools
        tools = []
        for tool_name in spec["tools"]:
            try:
                tools.append(TOOL.build({"type": tool_name}))
            except Exception as e:
                logger.log(
                    f"[AgentGenerator] Could not build tool {tool_name}: {e}",
                    level=LogLevel.WARNING
                )
        
        return await self.generate_agent(
            name=name,
            description=spec["description"],
            task_guidance=spec["guidance"],
            tools=tools,
            model=model,
            max_steps=10
        )
