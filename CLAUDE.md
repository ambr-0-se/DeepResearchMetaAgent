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

# GAIA evaluation — experimental conditions (see "Experimental Conditions" section)
python examples/run_gaia.py --config configs/config_gaia_c0.py        # C0: baseline PlanningAgent
python examples/run_gaia.py --config configs/config_gaia_adaptive.py  # C2: AdaptivePlanningAgent (reactive)
# C3/C4 configs are added in later implementation phases

# Legacy alias (equivalent to C0)
python examples/run_gaia.py --config configs/config_gaia.py

# ARC-AGI evaluation (API models)
python examples/run_arc.py --config configs/config_arc.py

# ARC-AGI on HKU CS GPU farm (Qwen + vLLM; see INSTRUCTIONS_RUN_EVAL.md)
# sbatch run_arc_test.sh
# sbatch run_arc_eval.sh

# Compare conditions (output dirs are workdir/gaia_<tag>/ where <tag> comes from the config)
python scripts/compare_results.py workdir/gaia_c0/dra.jsonl workdir/gaia_adaptive/dra.jsonl

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
- Uses THINK-ACT-OBSERVE loop (same as `PlanningAgent`) augmented with reactive self-modification tools
- Can diagnose sub-agent failures via `diagnose_subagent` tool (reactive; agent-invoked)
- Can modify sub-agents at runtime via `modify_subagent` tool (reactive; agent-invoked)
- All modifications are task-scoped (reset after each task)
- A structural REVIEW step (automatic post-delegation assessment) is added by C3 via an optional `review_step` component — see `src/meta/review_step.py` and `configs/config_gaia_c3.py`

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
- `src/meta/diagnose_tool.py`: Sub-agent diagnostic tool (delegates formatting to `_memory_format`)
- `src/meta/modify_tool.py`: Sub-agent modification tool (7 actions: add/remove agents, add/remove tools, modify instructions, set max_steps)
- `src/meta/tool_generator.py`: Dynamic tool generation
- `src/meta/agent_generator.py`: Dynamic agent creation
- `src/meta/_memory_format.py`: Shared helpers for rendering agent memory/tools (reused by diagnose tool and future review components)
- `src/agent/adaptive_planning_agent/`: Adaptive planning agent implementation
- `configs/config_gaia_c0.py`: Condition C0 — baseline PlanningAgent (alias over `config_gaia.py`)
- `configs/config_gaia_adaptive.py`: Condition C2 — AdaptivePlanningAgent with reactive diagnose/modify tools
- `configs/config_gaia_adaptive_qwen.py`: Qwen/vLLM evaluation config
- `scripts/compare_results.py`: Compare results across conditions
- `scripts/analyze_results.py`: Generate terminal/HTML evaluation reports

### Experimental Conditions

This codebase is set up to run four experimental conditions for ADAS research on GAIA:

| Condition | Config | Planner | Meta-agent capability |
|-----------|--------|---------|----------------------|
| **C0** | `config_gaia_c0.py` | `PlanningAgent` | None (baseline) |
| **C2** | `config_gaia_adaptive.py` | `AdaptivePlanningAgent` | Reactive diagnose/modify tools (agent-invoked) |
| **C3** | `config_gaia_c3.py` (planned) | `AdaptivePlanningAgent` + `review_step` | C2 + structural REVIEW step (automatic post-delegation assessment) |
| **C4** | `config_gaia_c4.py` (planned) | C3 + `skill_registry` | C3 + cross-task skill library (pre-seeded + learned) |

C1 (reactive diagnose-only, no modify) was dropped because the existing `modify_subagent` action space covers both C1 and C2 uses without meaningful distinction. All conditions are selected via config; the `AdaptivePlanningAgent` class is shared with optional `review_step` / `skill_registry` components. See the plan file for implementation details.

### Evaluation Infrastructure
- `run_combined_eval.sh`: Full GAIA evaluation SLURM job with vLLM watchdog
- `run_combined_test.sh`: Single-question test SLURM job
- `examples/run_gaia.py`: GAIA evaluation runner with per-question timeout and transient-error retry
- `examples/run_arc.py`: ARC-AGI evaluation runner with grid-based scoring
- `INSTRUCTIONS_RUN_EVAL.md`: GPU farm evaluation instructions

### ARC-AGI Evaluation
- `src/dataset/arc.py`: `ARCDataset` — loads ARC-AGI JSON task files, flattens test cases, formats grid questions
- `src/metric/arc_scorer.py`: `arc_question_scorer` — exact 2D grid match with robust grid extraction from text
- `src/agent/arc_reformulator.py`: `prepare_arc_response` — extracts grid answers from agent conversation
- `configs/config_arc.py`: ARC evaluation config (API models; deep_analyzer + general_agent)
- `configs/config_arc_qwen.py`: ARC on local vLLM (Qwen), for SLURM scripts
- `run_arc_test.sh` / `run_arc_eval.sh`: GPU farm jobs (same lifecycle as `run_combined_*.sh`)
- ARC data expected at `data/arc-agi/` with `training/` and `evaluation/` subdirectories of JSON files
