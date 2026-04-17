"""
Meta-agent module for adaptive agent capabilities.

This module provides tools and utilities for runtime modification of agents,
including diagnostic tools, modification tools, and generators for creating
new tools and agents dynamically.
"""

from src.meta.adaptive_mixin import AdaptiveMixin
from src.meta.diagnose_tool import DiagnoseSubAgentTool
from src.meta.modify_tool import ModifySubAgentTool
from src.meta.tool_generator import (
    ToolGenerator,
    allowed_top_level_modules,
    format_allowlist_for_prompt,
)
from src.meta.agent_generator import AgentGenerator

__all__ = [
    "AdaptiveMixin",
    "DiagnoseSubAgentTool",
    "ModifySubAgentTool",
    "ToolGenerator",
    "AgentGenerator",
    "allowed_top_level_modules",
    "format_allowlist_for_prompt",
]
