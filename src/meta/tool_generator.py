"""
ToolGenerator: Generates tool code from natural language descriptions.

This module allows dynamic creation of tools during runtime by having
an LLM generate Python code for Tool subclasses.
"""

from __future__ import annotations

import ast
import os
import re
from typing import TYPE_CHECKING, Optional

from src.models import ChatMessage, MessageRole
from src.logger import logger, LogLevel

if TYPE_CHECKING:
    from src.models import Model

# Single source of truth for imports allowed in generated tools (plus src.tools; validated in AST).
_BASE_ALLOWED_TOP_LEVEL_MODULES: frozenset[str] = frozenset(
    {
        "base64",
        "collections",
        "csv",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "math",
        "operator",
        "os",
        "pathlib",
        "re",
        "requests",
        "typing",
        "urllib",
    }
)

_DATA_SCIENCE_MODULES: frozenset[str] = frozenset({"numpy", "pandas"})

# Optional second tier (same flag as data science for simplicity).
_EXTENDED_FILE_MODULES: frozenset[str] = frozenset({"openpyxl", "yaml"})

_DISALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {"subprocess", "socket", "pickle", "ctypes", "multiprocessing", "importlib"}
)

# Input types agents may advertise to the LLM / tool schema layer (see src.tools.tools.AUTHORIZED_TYPES).
_TOOL_INPUT_TYPES_GUIDE = (
    "string, boolean, integer, number, image, audio, array, object, any, null"
)

_REPAIR_PROMPT = """The following generated Tool code failed validation.

Error:
{error}

## Original requirement
{requirement}

## Expected identifiers
- Tool `name` (snake_case string): {tool_name_snake}
- Class name (PascalCase, ends with Tool): {class_name}

## Previous code (rewrite completely; do not append a patch)
```python
{previous_code}
```

Return ONLY the corrected full Python source for one Tool subclass (imports + class), with no markdown fences and no explanation."""


def allowed_top_level_modules(allow_data_science: bool = False) -> frozenset[str]:
    """Return the effective allowlist (for prompts and external docs)."""
    modules = set(_BASE_ALLOWED_TOP_LEVEL_MODULES)
    if allow_data_science:
        modules.update(_DATA_SCIENCE_MODULES)
        modules.update(_EXTENDED_FILE_MODULES)
    return frozenset(modules)


def format_allowlist_for_prompt(allow_data_science: bool = False) -> str:
    """Comma-separated sorted list for GENERATION_PROMPT."""
    return ", ".join(sorted(allowed_top_level_modules(allow_data_science)))


def _is_src_tools_import(module: str | None) -> bool:
    if not module:
        return False
    return module == "src.tools" or module.startswith("src.tools.")


def _validate_imports_ast(code: str, allowed_roots: frozenset[str]) -> None:
    """Ensure all imports use an allowed root package or src.tools."""
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DISALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"Disallowed import: {alias.name!r}")
                if root == "src":
                    if not (alias.name == "src.tools" or alias.name.startswith("src.tools.")):
                        raise ValueError(
                            f"Import from 'src' only allowed under src.tools, got {alias.name!r}"
                        )
                    continue
                if root not in allowed_roots:
                    raise ValueError(
                        f"Import root {root!r} from {alias.name!r} is not in the allowlist"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise ValueError("Relative imports are not allowed in generated tools")
            mod = node.module
            if _is_src_tools_import(mod or ""):
                continue
            if mod is None:
                raise ValueError("Invalid import-from with no module")
            if mod.split(".")[0] == "src":
                raise ValueError(
                    f"Only 'from src.tools import ...' is allowed under src.*, got module {mod!r}"
                )
            root = mod.split(".")[0]
            if root in _DISALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Disallowed import from module {mod!r}")
            if root not in allowed_roots:
                raise ValueError(
                    f"Import from module {mod!r} (root {root!r}) is not in the allowlist"
                )


class ToolGenerator:
    """
    Generates tool code from natural language descriptions.

    Uses an LLM to generate Python code for Tool subclasses that can
    then be instantiated via Tool.from_code().
    """

    def __init__(self, model: "Model", allow_data_science: bool = False):
        """
        Initialize the tool generator.

        Args:
            model: LLM model to use for code generation
            allow_data_science: If True, allow pandas/numpy/openpyxl/yaml in generated
                code. Defaults to False so generated tools stay on a smaller, easier-to
                audit import surface; CSV-style tasks can use pathlib/csv/json, or set
                environment variable ``TOOL_GENERATOR_ALLOW_DATA_SCIENCE`` to ``1`` /
                ``true`` / ``yes`` (or pass True here) when pandas-backed codegen is
                acceptable for your deployment.
        """
        self.model = model
        self.allow_data_science = allow_data_science or (
            os.environ.get("TOOL_GENERATOR_ALLOW_DATA_SCIENCE", "").lower()
            in ("1", "true", "yes")
        )

    @property
    def _allowed_roots(self) -> frozenset[str]:
        return allowed_top_level_modules(self.allow_data_science)

    def _build_generation_prompt(
        self,
        requirement: str,
        tool_name: str,
        tool_name_snake: str,
        class_name: str,
        description: str,
    ) -> str:
        allowlist = format_allowlist_for_prompt(self.allow_data_science)
        example_b = _EXAMPLE_B_PANDAS if self.allow_data_science else _EXAMPLE_B_TWO_STRINGS
        return f"""Generate a Python tool class for the following requirement.

## Requirement
{requirement}

## Tool Name
{tool_name}

## Instructions
Create a Tool class that:
1. Inherits from the Tool base class (`from src.tools import Tool`)
2. Sets class attributes: `name` (must equal "{tool_name_snake}"), `description`, `inputs`, `output_type`
3. Implements `forward(self, **kwargs)` or `forward(self, <names>...)` such that every key in `inputs` is a parameter name with matching type entries
4. Use only these top-level import roots (and `from src.tools import Tool`): {allowlist}
5. `inputs` values must be dicts with a "type" field; use only these type strings: {_TOOL_INPUT_TYPES_GUIDE}
6. Do NOT use external API keys or credentials
7. Handle errors gracefully and return meaningful error strings on failure

Implement the Requirement section. Reference examples illustrate structure only — do not copy unrelated logic or names unless they fit your requirement.

## Template
```python
from src.tools import Tool

class {class_name}(Tool):
    \"\"\"{description}\"\"\"
    name = "{tool_name_snake}"
    description = \"\"\"{description}\"\"\"
    inputs = {{
        "param1": {{"type": "string", "description": "Description of param1"}},
    }}
    output_type = "string"

    def forward(self, param1: str) -> str:
        result = "..."
        return result
```

## Reference example A (stdlib — pathlib + json)
{_EXAMPLE_A_PATHLIB_JSON}

## Reference example B
{example_b}

## Constraints
- Only use imports allowed above; `from src.tools import Tool` is always permitted
- Do NOT use external API keys or credentials
- Keep the implementation simple and focused

## Output
Return ONLY the Python code for the Tool class. No explanations, no markdown code blocks.
Start directly with "from src.tools import Tool" or another allowed import line."""

    async def generate_tool_code(
        self,
        requirement: str,
        tool_name: str,
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
        tool_name_snake = self._to_snake_case(tool_name)
        class_name = self._to_class_name(tool_name)
        description = requirement[:200]

        prompt = self._build_generation_prompt(
            requirement=requirement,
            tool_name=tool_name,
            tool_name_snake=tool_name_snake,
            class_name=class_name,
            description=description,
        )

        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = await self.model(
                messages,
                stop_sequences=None,
            )
            code = response.content
        except Exception as e:
            logger.log(
                f"[ToolGenerator] LLM generation failed: {e}",
                level=LogLevel.ERROR,
            )
            raise ValueError(f"Failed to generate tool code: {e}") from e

        code = self._clean_code(code)

        try:
            self._validate_code(code)
        except ValueError as first_err:
            logger.log(
                f"[ToolGenerator] First codegen failed validation: {first_err}",
                level=LogLevel.INFO,
            )
            # Preserve user/assistant alternation for providers that require it.
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=response.content),
            )
            repair = _REPAIR_PROMPT.format(
                error=first_err,
                requirement=requirement,
                tool_name_snake=tool_name_snake,
                class_name=class_name,
                previous_code=code[:4000],
            )
            messages.append(
                ChatMessage(role=MessageRole.USER, content=repair),
            )
            try:
                response2 = await self.model(
                    messages,
                    stop_sequences=None,
                )
                code2 = self._clean_code(response2.content)
                self._validate_code(code2)
                return code2
            except Exception as e2:
                logger.log(
                    f"[ToolGenerator] Repair generation/validation failed: {e2}",
                    level=LogLevel.ERROR,
                )
                raise ValueError(
                    f"Failed to generate valid tool code after repair: {e2}"
                ) from e2

        return code

    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        name = re.sub(r"[\s\-]+", "_", name)
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
        return name.lower()

    def _to_class_name(self, name: str) -> str:
        """Convert name to PascalCase class name."""
        words = re.split(r"[\s_\-]+", name)
        class_name = "".join(word.capitalize() for word in words)
        if not class_name.endswith("Tool"):
            class_name += "Tool"
        return class_name

    def _clean_code(self, code: str) -> str:
        """Clean up generated code."""
        if "```python" in code:
            code = code.split("```python", 1)[1]
            if "```" in code:
                code = code.split("```", 1)[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 2:
                code = parts[1]

        code = code.strip()

        if not code.startswith("from") and not code.startswith("import"):
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("from") or line.strip().startswith("import"):
                    code = "\n".join(lines[i:])
                    break

        return code

    def _validate_code(self, code: str) -> None:
        """
        Validate that the generated code is safe and well-formed.

        Raises:
            ValueError: If code is invalid or unsafe
        """
        dangerous_patterns = [
            r"os\.system",
            r"subprocess\.",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
            r"open\s*\([^)]*,\s*[\"']w",  # Writing to files
            r"shutil\.rmtree",
            r"os\.remove",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                raise ValueError(
                    f"Generated code contains potentially dangerous pattern: {pattern}"
                )

        if "class " not in code or "(Tool)" not in code:
            raise ValueError("Generated code must define a class that inherits from Tool")

        required = ["name", "description", "inputs", "output_type", "def forward"]
        for attr in required:
            if attr not in code:
                raise ValueError(
                    f"Generated code missing required attribute/method: {attr}"
                )

        try:
            compile(code, "<generated_tool>", "exec")
        except SyntaxError as e:
            raise ValueError(f"Generated code has syntax error: {e}") from e

        _validate_imports_ast(code, self._allowed_roots)

        logger.log(
            "[ToolGenerator] Code validation passed",
            level=LogLevel.DEBUG,
        )


_EXAMPLE_A_PATHLIB_JSON = '''```python
from pathlib import Path
import json
from src.tools import Tool

class ReadTextJsonMetaTool(Tool):
    """Example: read a UTF-8 text file and return JSON metadata."""
    name = "read_text_json_meta_tool"
    description = "Return JSON with line_count and char_count for a UTF-8 text file path."
    inputs = {
        "file_path": {"type": "string", "description": "Path to a readable text file"},
    }
    output_type = "string"

    def forward(self, file_path: str) -> str:
        try:
            p = Path(file_path)
            text = p.read_text(encoding="utf-8")
            payload = {"line_count": text.count(chr(10)) + (1 if text else 0), "char_count": len(text)}
            return json.dumps(payload)
        except Exception as e:
            return json.dumps({"error": str(e)})
```'''

_EXAMPLE_B_TWO_STRINGS = '''```python
import re
from src.tools import Tool

class NormalizeWhitespaceTool(Tool):
    """Example: collapse internal whitespace between two markers."""
    name = "normalize_whitespace_tool"
    description = "Trim ends and collapse repeated spaces between prefix and suffix substrings."
    inputs = {
        "text": {"type": "string", "description": "Input text"},
        "mode": {"type": "string", "description": "'strict' or 'lenient'"},
    }
    output_type = "string"

    def forward(self, text: str, mode: str) -> str:
        try:
            t = text.strip()
            if mode == "strict":
                return re.sub(r"\s+", " ", t)
            return re.sub(r"\s{2,}", " ", t)
        except Exception as e:
            return f"error: {e}"
```'''

_EXAMPLE_B_PANDAS = '''```python
import json
from pathlib import Path
import pandas as pd
from src.tools import Tool

class CsvHeadSummaryTool(Tool):
    """Example: first N rows summary for a small CSV (pandas)."""
    name = "csv_head_summary_tool"
    description = "Read CSV path; return JSON list of column dtypes for first 5 rows."
    inputs = {
        "file_path": {"type": "string", "description": "Path to a CSV file"},
    }
    output_type = "string"

    def forward(self, file_path: str) -> str:
        try:
            df = pd.read_csv(Path(file_path), nrows=5)
            meta = {col: str(dtype) for col, dtype in df.dtypes.items()}
            return json.dumps({"columns": meta})
        except Exception as e:
            return json.dumps({"error": str(e)})
```'''


async def generate_simple_tool(
    model: "Model",
    requirement: str,
    tool_name: str,
    *,
    allow_data_science: bool = False,
) -> str:
    """
    Generate tool code using a model.

    Args:
        model: LLM model to use
        requirement: What the tool should do
        tool_name: Name for the tool
        allow_data_science: Forwarded to ToolGenerator

    Returns:
        Python code string
    """
    generator = ToolGenerator(model, allow_data_science=allow_data_science)
    return await generator.generate_tool_code(requirement, tool_name)
