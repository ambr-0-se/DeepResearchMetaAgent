# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepResearchAgent is a hierarchical multi-agent system for deep research and general-purpose task solving. It uses a two-layer architecture: a top-level planning agent coordinates specialized lower-level agents (Deep Analyzer, Deep Researcher, Browser Use, MCP Manager, General Tool Calling Agent).

## Common Commands

### Installation
```bash
# Using Poetry (recommended)
conda create -n dra python=3.11
conda activate dra
make install

# Using requirements.txt
make install-requirements
```

### Running the Agent
```bash
# Main hierarchical agent
python main.py

# Single agent example
python examples/run_general.py

# GAIA evaluation (baseline)
python examples/run_gaia.py --config configs/config_gaia.py

# GAIA evaluation (adaptive agent)
python examples/run_gaia.py --config configs/config_gaia_adaptive.py

# Compare baseline vs adaptive results
python scripts/compare_results.py workdir/gaia/dra.jsonl workdir/gaia_adaptive/dra.jsonl

# Override config options
python main.py --config configs/config_main.py --cfg-options model_id=gpt-4.1
```

### Analyzing Evaluation Results
```bash
# Terminal summary
python scripts/analyze_results.py workdir/<run_dir>/dra.jsonl

# Interactive HTML report
python scripts/analyze_results.py workdir/<run_dir>/dra.jsonl --html
```

### Running Tests
```bash
# Tests are in tests/ directory
python tests/test_models.py
python tests/test_analyzer.py
python tests/test_eval_fixes.py
python tests/test_tier_b_tool_messages.py  # requires pytest
```

## Architecture

### Registry System
Uses mmengine's Registry pattern (`src/registry.py`) to register and build components:
- `AGENT`: Registers agent types (`src/agent/`)
- `TOOL`: Registers tool types (`src/tools/`)
- `DATASET`: Registers dataset loaders (`src/dataset/`)

### Configuration
Config files use mmengine format (Python-based, not TOML) in `configs/`:
- `base.py`: Tool configurations (web_fetcher, web_searcher, deep_researcher, etc.)
- `config_main.py`: Main agent setup with hierarchical agent structure
- `config_general.py`, `config_gaia.py`, etc.: Task-specific configurations

Agent configs reference tool configs by name convention: `{tool_name}` → `{tool_name}_config`

### Agent Types
**Top-level:** `PlanningAgent` - coordinates lower-level agents

**Adaptive top-level:** `AdaptivePlanningAgent` - extends PlanningAgent with self-modification capabilities
- Uses THINK-ACT-OBSERVE-REFLECT loop
- Can diagnose sub-agent failures via `diagnose_subagent` tool
- Can modify sub-agents at runtime via `modify_subagent` tool
- All modifications are task-scoped (reset after each task)

**Specialized lower-level agents:**
- `DeepAnalyzerAgent` - in-depth analysis of input information
- `DeepResearcherAgent` - web research and knowledge synthesis
- `BrowserUseAgent` - automated browser operations
- `GeneralAgent` - general-purpose tool calling interface
- MCP Manager - manages MCP tools/services (integrated via `MCPAdapt`, not a separate agent class)

Default `config_main.py` uses: DeepAnalyzerAgent, DeepResearcherAgent, BrowserUseAgent

### Meta-Agent Module (`src/meta/`)
Provides runtime modification capabilities for adaptive agents:
- `AdaptiveMixin` - Mixin class for state management and modification methods
- `DiagnoseSubAgentTool` - Investigates sub-agent execution failures
- `ModifySubAgentTool` - Modifies sub-agent tools, instructions, and capabilities
- `ToolGenerator` - Generates new tools from natural language descriptions
- `AgentGenerator` - Creates new agents with in-memory templates

Agents are created via `create_agent()` in `src/agent/agent.py`. The function:
1. Loads MCP tools from config
2. Builds managed agents recursively
3. Builds the main agent with tools and managed agents

### Core Base Classes
- `MultiStepAgent` (`src/base/multistep_agent.py`): Base class using ReAct framework
- `AsyncMultiStepAgent` (`src/base/async_multistep_agent.py`): Async variant
- `Tool` / `AsyncTool` (`src/tools/tools.py`): Base tool classes

### Model Integration
Models are managed through `model_manager` (`src/models/`). Supported:
- OpenAI (gpt-4.1, o3)
- Anthropic (claude-3.7-sonnet-thinking, claude-4-sonnet)
- Google (gemini-2.5-pro, imagen, veo3)
- Local models via vLLM (qwen2.5-7b/14b/32b-instruct, qwen3-vl-4b-instruct)

### MCP (Model Context Protocol)
MCP tools are integrated via `src/mcp/`:
- Configure in `mcp_tools_config` in base config
- Supports both local stdio servers and remote HTTP servers
- Uses `MCPAdapt` class to load and convert MCP tools

## Environment Variables
Copy `.env.template` to `.env` and configure API keys:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `FIRECRAWL_API_KEY` for web search
- `QWEN_API_BASE`, `QWEN_API_KEY` for local vLLM models

## Key Files
- `main.py`: Entry point for hierarchical agent
- `src/agent/agent.py`: Agent creation and building logic (passes managed agents as real objects, not tool wrappers)
- `src/tools/default_tools.py`: Default tool mappings
- `src/memory/memory.py`: Agent memory management (supports Tier B tool_results)
- `src/logger/logger.py`: Logging and visualization

### Tier B Tool Message Protocol
OpenAI-native tool message support throughout the stack:
- `src/models/base.py`: `ChatMessage.tool_call_id` field, `MessageRole.TOOL` role
- `src/models/message_manager.py`: Serializes assistant+tool_calls and role=tool messages per tool_call_id
- `src/memory/memory.py`: `ActionStep.tool_results` stores per-tool_call_id results
- `src/base/tool_calling_agent.py` / `src/agent/general_agent/general_agent.py`: Tracks tool_call_id through parallel execution

### Context Pruning
`src/agent/general_agent/general_agent.py` has `_prune_messages_if_needed()`:
- Triggers at 85% of `max_model_len` (default 32768)
- Keeps system prompt + last 4 messages, truncates middle messages >500 chars
- Preserves `tool_call_id` and other ChatMessage fields

### Adaptive Agent Files
- `src/meta/adaptive_mixin.py`: Runtime modification mixin (uses `_find_managed_agent` fallback to check both `managed_agents` and `tools`)
- `src/meta/diagnose_tool.py`: Sub-agent diagnostic tool
- `src/meta/modify_tool.py`: Sub-agent modification tool
- `src/meta/tool_generator.py`: Dynamic tool generation
- `src/meta/agent_generator.py`: Dynamic agent creation
- `src/agent/adaptive_planning_agent/`: Adaptive planning agent implementation
- `configs/config_gaia_adaptive.py`: GAIA evaluation config for adaptive agent
- `configs/config_gaia_adaptive_qwen.py`: Qwen/vLLM evaluation config
- `scripts/compare_results.py`: Compare baseline vs adaptive results
- `scripts/analyze_results.py`: Generate terminal/HTML evaluation reports

### Evaluation Infrastructure
- `run_combined_eval.sh`: Full GAIA evaluation SLURM job with vLLM watchdog
- `run_combined_test.sh`: Single-question test SLURM job
- `examples/run_gaia.py`: Evaluation runner with per-question timeout and transient-error retry
- `INSTRUCTIONS_RUN_EVAL.md`: GPU farm evaluation instructions
