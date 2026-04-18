import os
import sys

from fastmcp import FastMCP
from dotenv import load_dotenv
import asyncio
from pathlib import Path

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from src.utils import assemble_project_path
from src.logger import logger

# Load environment variables
load_dotenv(override=True)

# Initialize FastMCP
mcp = FastMCP("LocalMCP")
_mcp_tools_namespace = {}

async def register_tool_from_script(script_info):
    """
    Register a tool from a script content.
    """

    name = script_info.get("name", "UnnamedTool")
    description = script_info.get("description", "No description provided.")
    script_content = script_info.get("script_content", "")

    # Registry entries are produced by an LLM and frequently ship with one or
    # more trailing ``` fences, example invocations, or follow-up prose glued
    # after the implementation. Old logic stripped fences globally, which left
    # the junk lines (e.g. `[{"call": ...}]`) in the compiled source and
    # produced SyntaxError at runtime. Extract strictly the FIRST fenced
    # ``` ```python ... ``` ``` block (or, if none, use the raw string).
    stripped = script_content.lstrip()
    if stripped.startswith("```python") or stripped.startswith("```"):
        fence = "```python" if stripped.startswith("```python") else "```"
        body = stripped[len(fence):]
        end = body.find("```")
        if end == -1:
            # Opening fence without a matching close: the body likely trails into
            # example invocations or prose that isn't valid Python. Reject the
            # registration rather than exec() the junk — the existing
            # `tool_function is None` guard below would still fire, but emitting
            # a clearer error here surfaces the real cause (malformed registry).
            logger.error(
                f"Tool '{name}' skipped — fenced script missing closing ``` marker."
            )
            return
        script_content = body[:end]
    # Else: raw, non-fenced source — exec as-is.

    try:
        exec(script_content, _mcp_tools_namespace)
    except Exception as e:
        logger.error(f"Error executing script for tool '{name}': {e}")
        return

    tool_function = _mcp_tools_namespace.get(name, None)
    if tool_function is None:
        logger.error(f"Tool function '{name}' not found in script content.")
        return
    else:
        mcp.tool(
            tool_function,
            name=name,
            description=description,
        )
        logger.info(f"Tool '{name}' registered successfully.")

async def register_tools(script_info_path):
    """
    Register tools from a JSON file containing script information.
    """
    import json

    try:
        with open(script_info_path, 'r') as f:
            script_info_list = json.load(f)

        for script_info in script_info_list:
            await register_tool_from_script(script_info)

    except FileNotFoundError:
        logger.info(f"Script info file not found: {script_info_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from script info file: {script_info_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while registering tools: {e}")

    logger.info("All tools registered successfully.")

    mcp_tools = await mcp.get_tools()
    logger.info(f"Registered tools: {', '.join([tool for tool in mcp_tools])}")

if __name__ == "__main__":
    script_info_path = assemble_project_path(os.path.join("src", "mcp", "local", "mcp_tools_registry.json"))
    asyncio.run(register_tools(script_info_path))
    mcp.run()