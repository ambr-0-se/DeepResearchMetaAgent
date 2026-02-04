"""
ToolGenerator: Generates tool code from natural language descriptions.

This module allows dynamic creation of tools during runtime by having
an LLM generate Python code for Tool subclasses.
"""

from typing import TYPE_CHECKING, Optional
import re

from src.models import ChatMessage, MessageRole
from src.logger import logger, LogLevel

if TYPE_CHECKING:
    from src.models import Model


class ToolGenerator:
    """
    Generates tool code from natural language descriptions.
    
    Uses an LLM to generate Python code for Tool subclasses that can
    then be instantiated via Tool.from_code().
    """
    
    GENERATION_PROMPT = '''Generate a Python tool class for the following requirement.

## Requirement
{requirement}

## Tool Name
{tool_name}

## Instructions
Create a Tool class that:
1. Inherits from the Tool base class
2. Has clear name, description, inputs, and output_type attributes
3. Implements a forward() method that performs the tool's function

## Template
```python
from src.tools import Tool

class {class_name}(Tool):
    """
    {description}
    """
    name = "{tool_name_snake}"
    description = """{description}"""
    inputs = {{
        "param1": {{"type": "string", "description": "Description of param1"}},
    }}
    output_type = "string"
    
    def forward(self, param1: str) -> str:
        # Implementation
        result = "..."
        return result
```

## Constraints
- Only use these imports: requests, json, re, os, datetime, math, typing
- Do NOT use external API keys or credentials
- Keep the implementation simple and focused
- Handle errors gracefully
- Return meaningful error messages if something fails

## Output
Return ONLY the Python code for the Tool class. No explanations, no markdown code blocks.
Start directly with "from src.tools import Tool" or the import statement.'''

    def __init__(self, model: "Model"):
        """
        Initialize the tool generator.
        
        Args:
            model: LLM model to use for code generation
        """
        self.model = model
    
    async def generate_tool_code(
        self,
        requirement: str,
        tool_name: str
    ) -> str:
        """
        Generate tool code from a requirement description.
        
        Args:
            requirement: Natural language description of what the tool should do
            tool_name: Desired name for the tool
            
        Returns:
            Python code string defining the Tool class
            
        Raises:
            ValueError: If code generation fails or produces invalid code
        """
        # Normalize tool name
        tool_name_snake = self._to_snake_case(tool_name)
        class_name = self._to_class_name(tool_name)
        
        # Build prompt
        prompt = self.GENERATION_PROMPT.format(
            requirement=requirement,
            tool_name=tool_name,
            tool_name_snake=tool_name_snake,
            class_name=class_name,
            description=requirement[:200]
        )
        
        try:
            response = await self.model(
                [ChatMessage(role=MessageRole.USER, content=prompt)],
                stop_sequences=None,
            )
            code = response.content
        except Exception as e:
            logger.log(
                f"[ToolGenerator] LLM generation failed: {e}",
                level=LogLevel.ERROR
            )
            raise ValueError(f"Failed to generate tool code: {e}")
        
        # Clean up the code
        code = self._clean_code(code)
        
        # Validate the code
        self._validate_code(code)
        
        return code
    
    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        # Replace spaces and hyphens with underscores
        name = re.sub(r'[\s\-]+', '_', name)
        # Insert underscore before capitals
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        # Convert to lowercase
        return name.lower()
    
    def _to_class_name(self, name: str) -> str:
        """Convert name to PascalCase class name."""
        # Split by spaces, underscores, hyphens
        words = re.split(r'[\s_\-]+', name)
        # Capitalize each word and join
        class_name = ''.join(word.capitalize() for word in words)
        # Ensure it ends with Tool
        if not class_name.endswith('Tool'):
            class_name += 'Tool'
        return class_name
    
    def _clean_code(self, code: str) -> str:
        """Clean up generated code."""
        # Remove markdown code blocks if present
        if '```python' in code:
            code = code.split('```python')[1]
            if '```' in code:
                code = code.split('```')[0]
        elif '```' in code:
            parts = code.split('```')
            if len(parts) >= 2:
                code = parts[1]
        
        # Strip whitespace
        code = code.strip()
        
        # Ensure it starts with an import
        if not code.startswith('from') and not code.startswith('import'):
            # Try to find where the actual code starts
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('from') or line.strip().startswith('import'):
                    code = '\n'.join(lines[i:])
                    break
        
        return code
    
    def _validate_code(self, code: str) -> None:
        """
        Validate that the generated code is safe and well-formed.
        
        Args:
            code: Python code to validate
            
        Raises:
            ValueError: If code is invalid or unsafe
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            r'os\.system',
            r'subprocess\.',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'open\s*\([^)]*,\s*["\']w',  # Writing to files
            r'shutil\.rmtree',
            r'os\.remove',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Generated code contains potentially dangerous pattern: {pattern}")
        
        # Check that it defines a Tool subclass
        if 'class ' not in code or '(Tool)' not in code:
            raise ValueError("Generated code must define a class that inherits from Tool")
        
        # Check for required attributes
        required = ['name', 'description', 'inputs', 'output_type', 'def forward']
        for attr in required:
            if attr not in code:
                raise ValueError(f"Generated code missing required attribute/method: {attr}")
        
        # Try to compile the code (syntax check)
        try:
            compile(code, '<generated_tool>', 'exec')
        except SyntaxError as e:
            raise ValueError(f"Generated code has syntax error: {e}")
        
        logger.log(
            "[ToolGenerator] Code validation passed",
            level=LogLevel.DEBUG
        )


# Convenience function for simple tool generation
async def generate_simple_tool(
    model: "Model",
    requirement: str,
    tool_name: str
) -> str:
    """
    Generate tool code using a model.
    
    Args:
        model: LLM model to use
        requirement: What the tool should do
        tool_name: Name for the tool
        
    Returns:
        Python code string
    """
    generator = ToolGenerator(model)
    return await generator.generate_tool_code(requirement, tool_name)
