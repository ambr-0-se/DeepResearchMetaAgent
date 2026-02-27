"""
DiagnoseSubAgentTool: Investigates sub-agent execution failures.

This tool allows the parent agent to examine a sub-agent's full execution
history (reasoning, tool calls, observations) to understand why it failed
to meet the objective.
"""

from typing import TYPE_CHECKING, Any, Optional

from src.tools import AsyncTool
from src.models import ChatMessage, MessageRole
from src.logger import logger, LogLevel

if TYPE_CHECKING:
    from src.base.async_multistep_agent import AsyncMultiStepAgent


class DiagnoseSubAgentTool(AsyncTool):
    """
    Investigates a sub-agent's execution to understand failure reasons.
    
    Use this tool when:
    - Sub-agent returned unsatisfactory result
    - The reason for failure is NOT obvious from the response
    - You need to see the sub-agent's reasoning, tool calls, and observations
    """
    
    name = "diagnose_subagent"
    description = """Investigate why a sub-agent failed to meet the objective.

This tool retrieves and analyzes the sub-agent's full execution history:
- What the sub-agent was thinking at each step
- What tools it called and with what arguments
- What observations it received
- Where its reasoning went wrong

Use this when the sub-agent's response is unsatisfactory but the reason is not 
obvious from the response alone. This helps you understand the gap between 
expected and actual behavior.

Returns a diagnosis explaining:
- Where the reasoning went wrong
- What capability was missing
- Suggested remediation actions"""

    parameters = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Name of the sub-agent to diagnose (e.g., 'deep_researcher_agent')"
            },
            "task_given": {
                "type": "string",
                "description": "The task you assigned to the sub-agent"
            },
            "expected_outcome": {
                "type": "string",
                "description": "What you expected the sub-agent to achieve"
            },
            "actual_response": {
                "type": "string",
                "description": "The actual response you received from the sub-agent"
            }
        },
        "required": ["agent_name", "task_given", "expected_outcome", "actual_response"]
    }
    output_type = "string"
    
    def __init__(self, parent_agent: "AsyncMultiStepAgent"):
        """
        Initialize the diagnostic tool.
        
        Args:
            parent_agent: Reference to the parent agent that owns this tool
        """
        super().__init__()
        self.parent_agent = parent_agent
    
    async def forward(
        self,
        agent_name: str,
        task_given: str,
        expected_outcome: str,
        actual_response: str
    ) -> str:
        """
        Diagnose a sub-agent's execution.
        
        Args:
            agent_name: Name of the sub-agent to diagnose
            task_given: The task assigned to the agent
            expected_outcome: What was expected
            actual_response: What was received
            
        Returns:
            Diagnosis string with analysis and recommendations
        """
        # Check if agent exists
        if agent_name not in self.parent_agent.managed_agents:
            available = list(self.parent_agent.managed_agents.keys())
            return f"Error: Agent '{agent_name}' not found. Available agents: {available}"
        
        agent = self.parent_agent.managed_agents[agent_name]
        
        # Get agent's memory/execution history
        execution_history = self._format_execution_history(agent)
        
        # Get agent's available tools
        agent_tools = self._format_agent_tools(agent)
        
        # Construct diagnosis prompt
        diagnosis_prompt = self._build_diagnosis_prompt(
            agent_name=agent_name,
            task_given=task_given,
            expected_outcome=expected_outcome,
            actual_response=actual_response,
            execution_history=execution_history,
            agent_tools=agent_tools
        )
        
        # Use parent agent's model for analysis
        try:
            response = await self.parent_agent.model(
                [ChatMessage(role=MessageRole.USER, content=diagnosis_prompt)],
                stop_sequences=None,
            )
            diagnosis = response.content
        except Exception as e:
            logger.log(
                f"[DiagnoseSubAgentTool] Error during LLM analysis: {e}",
                level=LogLevel.ERROR
            )
            # Fall back to basic diagnosis
            diagnosis = self._basic_diagnosis(
                agent_name, execution_history, agent_tools
            )
        
        return diagnosis
    
    def _format_execution_history(self, agent: "AsyncMultiStepAgent") -> str:
        """
        Format the agent's execution history from memory.
        
        Args:
            agent: The agent to get history from
            
        Returns:
            Formatted string of execution history
        """
        if not hasattr(agent, 'memory') or agent.memory is None:
            return "No execution history available (memory is empty)."
        
        history_parts = []
        
        # Get memory steps
        steps = getattr(agent.memory, 'steps', [])
        
        if not steps:
            return "No execution steps recorded."
        
        for i, step in enumerate(steps):
            step_info = [f"=== Step {i + 1} ==="]
            
            # Handle different step types
            step_type = type(step).__name__
            step_info.append(f"Type: {step_type}")
            
            # Extract relevant information based on step type
            if hasattr(step, 'task'):
                step_info.append(f"Task: {step.task[:500]}..." if len(str(step.task)) > 500 else f"Task: {step.task}")
            
            if hasattr(step, 'model_output') and step.model_output:
                output = str(step.model_output)
                step_info.append(f"Agent Reasoning: {output[:800]}..." if len(output) > 800 else f"Agent Reasoning: {output}")
            
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    if hasattr(tc, 'name') and hasattr(tc, 'arguments'):
                        args_str = str(tc.arguments)[:300]
                        step_info.append(f"Tool Called: {tc.name}({args_str})")
            
            if hasattr(step, 'observations') and step.observations:
                obs = str(step.observations)
                step_info.append(f"Observation: {obs[:500]}..." if len(obs) > 500 else f"Observation: {obs}")
            
            if hasattr(step, 'error') and step.error:
                step_info.append(f"ERROR: {step.error}")
            
            history_parts.append("\n".join(step_info))
        
        return "\n\n".join(history_parts)
    
    def _format_agent_tools(self, agent: "AsyncMultiStepAgent") -> str:
        """
        Format the agent's available tools.
        
        Args:
            agent: The agent to get tools from
            
        Returns:
            Formatted string of available tools
        """
        if not hasattr(agent, 'tools') or not agent.tools:
            return "No tools available."
        
        tool_info = []
        for name, tool in agent.tools.items():
            desc = getattr(tool, 'description', 'No description')[:200]
            tool_info.append(f"- {name}: {desc}")
        
        return "\n".join(tool_info)
    
    def _build_diagnosis_prompt(
        self,
        agent_name: str,
        task_given: str,
        expected_outcome: str,
        actual_response: str,
        execution_history: str,
        agent_tools: str
    ) -> str:
        """Build the prompt for LLM-based diagnosis."""
        return f"""Analyze the following sub-agent execution to understand why it failed to meet the objective.

## Sub-Agent Information
Agent Name: {agent_name}

Available Tools:
{agent_tools}

## Task Analysis
Task Given: {task_given}

Expected Outcome: {expected_outcome}

Actual Response: {actual_response}

## Execution History
{execution_history}

## Analysis Required
Please analyze the execution and provide:

1. **Root Cause Analysis**: Where did the agent's reasoning go wrong? What was the critical failure point?

2. **Capability Gap**: What capability (tool, knowledge, instruction) was the agent missing?

3. **Remediation Suggestions**: What specific changes would help? Consider:
   - Additional tools that could be added
   - Modified instructions that could guide better behavior
   - Whether a different agent type would be more suitable

4. **Confidence Assessment**: How confident are you in this diagnosis? (High/Medium/Low)

Provide a concise but thorough analysis."""
    
    def _basic_diagnosis(
        self,
        agent_name: str,
        execution_history: str,
        agent_tools: str
    ) -> str:
        """
        Provide basic diagnosis without LLM when analysis fails.
        
        Args:
            agent_name: Name of the agent
            execution_history: Formatted history
            agent_tools: Formatted tools
            
        Returns:
            Basic diagnosis string
        """
        return f"""## Diagnostic Report for {agent_name}

### Available Tools
{agent_tools}

### Execution History
{execution_history}

### Basic Analysis
Unable to perform LLM-based analysis. Please review the execution history above to identify:
1. What tools were called and their results
2. Any errors that occurred
3. Where the reasoning diverged from expected behavior

Consider:
- Adding tools if the agent lacked necessary capabilities
- Modifying instructions if the agent misunderstood the task
- Using a different agent if this one is not suited for the task type"""
