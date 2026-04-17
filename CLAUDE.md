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
python examples/run_gaia.py --config configs/config_gaia_c0.py         # C0: baseline PlanningAgent
python examples/run_gaia.py --config configs/config_gaia_adaptive.py   # C2: AdaptivePlanningAgent (reactive diagnose/modify)
python examples/run_gaia.py --config configs/config_gaia_c3.py         # C3: + structural REVIEW step
python examples/run_gaia.py --config configs/config_gaia_c4.py         # C4: + cross-task skill library (training mode)

# Legacy alias (equivalent to C0)
python examples/run_gaia.py --config configs/config_gaia.py

# ARC-AGI evaluation (API models)
python examples/run_arc.py --config configs/config_arc.py

# ARC-AGI on HKU CS GPU farm (Qwen + vLLM; see INSTRUCTIONS_RUN_EVAL.md)
# sbatch run_arc_test.sh
# sbatch run_arc_eval.sh

# Compare conditions (output dirs are workdir/gaia_<tag>/ where <tag> comes from the config)
python scripts/compare_results.py workdir/gaia_c0/dra.jsonl workdir/gaia_adaptive/dra.jsonl
python scripts/compare_results.py workdir/gaia_c3/dra.jsonl workdir/gaia_c4/dra.jsonl

# Validate seed / learned SKILL.md files
python -m src.skills.validate src/skills

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
- All architectural modifications are task-scoped (reset after each task)
- **C3**: adds a structural REVIEW step (automatic post-delegation assessment) via optional `review_step` component — see `src/meta/review_step.py` and `configs/config_gaia_c3.py`. Review findings are injected into `action_step.observations` with a `[REVIEW]` marker so the next THINK sees them.
- **C4**: adds a cross-task skill library via optional `skill_registry` component — see `src/skills/` and `configs/config_gaia_c4.py`. Skills persist across tasks (unlike architectural modifications). Each agent (planner and sub-agents) gets an `activate_skill` tool scoped to the skills visible to it.


**Specialized lower-level agents:**
- `DeepAnalyzerAgent` - in-depth analysis of input information
- `DeepResearcherAgent` - web research and knowledge synthesis
- `BrowserUseAgent` - automated browser operations
- `GeneralAgent` - general-purpose tool calling interface
- MCP Manager - manages MCP tools/services (integrated via `MCPAdapt`, not a separate agent class)

Default `config_main.py` uses: DeepAnalyzerAgent, DeepResearcherAgent, BrowserUseAgent

### Meta-Agent Module (`src/meta/`)
Provides runtime modification, review, and skill-activation capabilities for adaptive agents:
- `AdaptiveMixin` — Mixin for state management (`_store_original_state` / `_reset_to_original_state`) and modification methods (add/remove agents + tools, modify instructions, set max_steps)
- `DiagnoseSubAgentTool` — Reactive sub-agent failure investigation (agent-invoked). Reused internally by `ReviewAgent`.
- `ModifySubAgentTool` — 7-action modification tool (add/remove agents, add existing/new tool, remove tool, modify instructions, set max_steps)
- `ToolGenerator` — Generates new tools from natural language descriptions (used by `modify_subagent`'s `add_new_tool_to_agent` action)
- `AgentGenerator` — Creates new agents with in-memory templates (used by `modify_subagent`'s `add_agent` action)
- `_memory_format.py` — Shared helpers (`format_execution_history`, `format_agent_tools`) used by both DiagnoseSubAgentTool and the sealed ReviewAgent.

**Structural REVIEW (`enable_review=True` — C3 and C4):**
- `review_schema.py` — Pydantic models: `ReviewResult`, `RootCauseCategory` (8-item taxonomy), polymorphic `NextAction` discriminated union (`ProceedSpec`, `RetrySpec`, `ModifyAgentSpec`, `EscalateSpec`).
- `review_agent.py` — Internal sealed `ReviewAgent`. Subclass of `GeneralAgent` but NOT registered via `@AGENT.register_module`. Built via `object.__new__` + manual attribute init so no YAML file I/O is needed.
- `review_step.py` — `ReviewStep` orchestrator. Invoked by `AdaptivePlanningAgent._post_action_hook` after every sub-agent delegation. Extracts `DelegationContext`, fast-paths skip for non-delegations / final answers / near max_steps, invokes the sealed ReviewAgent, parses JSON, validates `EscalateSpec.to_agent` against real managed_agents, falls back safely on any error.

**C4 components (skill activation):**
- `activate_skill_tool.py` — `ActivateSkillTool` (AsyncTool). Each instance is bound to a fixed `consumer` scope so skills scoped to one agent cannot leak to another. Returns skill body or error string; read-only over the registry.

Agents are created via `create_agent()` in `src/agent/agent.py`. The function:
1. Loads MCP tools from config
2. Builds managed agents recursively
3. Builds the main agent with tools and managed agents

For `AdaptivePlanningAgent`, review and skill components are constructed inside `__init__` based on `config.enable_review` / `config.enable_skills` / `config.enable_skill_extraction` flags — the `build_agent()` factory path is unchanged.

### Skill Library (`src/skills/` — C4)
Filesystem-backed catalogue following the [agentskills.io](https://agentskills.io/specification) specification. Each skill is a directory containing a `SKILL.md` file with YAML frontmatter + Markdown body.

Python package (internal):
- `_model.py` — `Skill` + `SkillMetadata` dataclasses. Parses frontmatter, validates names against the agentskills.io regex, enforces name-equals-directory-name.
- `_registry.py` — `SkillRegistry`. Scans `skills_dir` at startup. `metadata_for(consumer)` filters skills visible to a given consumer (planner, sub-agent name, or `all`). `load_body()` reads body on demand. `render_registry_block()` produces the injection for system prompts. `add()` and `increment_verified_uses()` persist via atomic temp+rename writes.
- `_extractor.py` — `SkillExtractor`. Six-stage pipeline at task end: worthiness heuristic → LLM propose JSON → structural validation → entity blocklist (years / currency / URLs / long numerics) → LLM-as-judge dedup → `registry.add`. Contracted never to raise.
- `_seed.py` — `seed_skills_dir(dst, src)`. Copies the 7 canonical seed skills from `src/skills/` into a fresh per-run `skills_dir` on first construction. Writes a `.seeded` marker last so resumed runs do not re-copy and clobber learned skills. No-op when `dst == src` (legacy standalone behaviour). Only directories containing `SKILL.md` are copied — package files (`_registry.py`, `README.md`, etc.) are skipped by construction.
- `validate.py` — CLI validator (`python -m src.skills.validate <path>`).

**Per-run output layout (all conditions).** Every matrix config — C0, C2, C3, and C4 — derives `tag = f"gaia_<cond>_<model>_{_RUN_ID}"` from the `DRA_RUN_ID` env var (fresh `YYYYMMDD_HHMMSS` fallback at config load). Each invocation therefore writes `dra.jsonl` into its own `workdir/gaia_<cond>_<model>_<run_id>/` directory. For C4 specifically, `skills_dir=f"workdir/{tag}/skills"` co-locates the evolved skill library with the results that produced it. This guarantees:
- Parallel runs (matrix runner launches mistral/kimi/qwen in parallel) cannot race on any shared directory.
- Repeated runs of the same `(model, condition)` never overwrite history — each run stays inspectable forever (`diff -r workdir/gaia_c4_mistral_<A>/skills/ workdir/gaia_c4_mistral_<B>/skills/`).
- `scripts/run_eval_matrix.sh` exports one `DRA_RUN_ID` for the whole invocation and maintains a `workdir/gaia_<cond>_<model>_latest` symlink per cell pointing at the most recent run for quick inspection.
- To resume a prior run (e.g. after Ctrl-C): `DRA_RUN_ID=<prior_id> bash scripts/run_eval_matrix.sh full` — the existing `dra.jsonl` is appended to, and for C4 the `.seeded` marker causes seed-copy to be skipped so prior learned skills survive.

Pre-seeded skills (committed to the repo, 7 total covering all 4 consumer scopes):
- Planner scope: `handling-file-attachments`, `task-decomposition-complex-queries`, `delegation-failure-recovery`
- `deep_analyzer_agent` scope: `pdf-table-extraction`, `multi-hop-math-verification`
- `browser_use_agent` scope: `browser-paywall-recovery`
- `deep_researcher_agent` scope: `research-fallback-sources`

Metadata schema extensions beyond agentskills.io (under `metadata:` block):
- `consumer`: routing scope (`planner` | sub-agent name | `all`)
- `skill_type`: canonical taxonomy (delegation_pattern, task_decomposition, failure_avoidance, modification_pattern, verification_pattern, tool_usage, domain_workflow)
- `source`: seeded / success / failure
- `verified_uses`, `confidence`, `created_at`, `learned_from_task_type`: telemetry & provenance.

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
- **2026-04 multi-provider integration:** DeepSeek V3.2 (`deepseek-chat`, `deepseek-reasoner`), Mistral Small 4 (`mistral-small`), Qwen3 family (`qwen3-max`, `qwen3.6-plus`, `qwen3-coder-plus`, plus `*-thinking` variants via DashScope `enable_thinking`), Moonshot Kimi K2.5 (`kimi-k2.5`, `kimi-k2.5-no-thinking`), MiniMax M2.7 (`minimax-m2.7`). Each is registered both natively and via OpenRouter (`or-*` prefix).
- **Failover wrapper:** `qwen3.6-plus-failover` (in `src/models/failover.py`) routes to DashScope first (free tier) and switches one-way to OpenRouter on quota-exhaustion errors. Detection is conservative — only known free-tier quota strings trigger the switch; transient 429s do not.

**Provider quirks handled:**
- Reasoning content (DeepSeek-reasoner, Qwen3-thinking) — `ChatMessage.reasoning_content` round-trips through memory and is echoed on assistant messages when the model id matches `needs_reasoning_echo()`.
- Kimi locked sampling — `MessageManager.get_clean_completion_kwargs` strips `temperature`/`top_p`/`n`/penalty/`logprobs` after caller merge.
- MiniMax temperature clamp — clamped to (0, 1.0]; `n` forced to 1.
- DashScope `enable_thinking` and Moonshot `thinking={"type":"disabled"}` — passed through `OpenAIServerModel(extra_body=...)`.

**GAIA evaluation matrix (3 models × 4 conditions = 12 configs):**
- Generated by `scripts/gen_eval_configs.py` from a single template — do not hand-edit; regenerate.
- Files: `configs/config_gaia_<c0|c2|c3|c4>_<mistral|kimi|qwen>.py`.
- Single-model constraint: every agent + tool in a given config uses the same `model_id`. Kimi configs use `kimi-k2.5-no-thinking` because structural REVIEW (C3/C4) and the C4 `SkillExtractor` need JSON output. Qwen configs use `qwen3.6-plus-failover` for the DashScope→OpenRouter auto-switch.
- Parallel runner: `scripts/run_eval_matrix.sh smoke|full [model] [condition]` — runs the 3 model streams in parallel, conditions sequential within each stream.

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

Meta-agent module (`src/meta/`):
- `adaptive_mixin.py` — Runtime modification mixin (uses `_find_managed_agent` fallback over `managed_agents` + `tools`)
- `diagnose_tool.py` — Reactive sub-agent diagnostic tool (delegates formatting to `_memory_format`)
- `modify_tool.py` — 7-action modification tool (add/remove agents, add/remove tools, modify instructions, set max_steps)
- `tool_generator.py` / `agent_generator.py` — Dynamic tool and agent creation
- `_memory_format.py` — Shared history/tools formatting helpers
- **`review_schema.py`** — Pydantic `ReviewResult` + `NextAction` discriminated union (structural REVIEW; C3 and C4 when `enable_review=True`)
- **`review_agent.py`** — Sealed internal `ReviewAgent`; not registered, not in `managed_agents` (same)
- **`review_step.py`** — Orchestrator called from `_post_action_hook` after each delegation (same)
- **`activate_skill_tool.py`** — Consumer-scoped `ActivateSkillTool` AsyncTool (C4)

Skill library (`src/skills/`):
- **`_model.py`** — `Skill` + `SkillMetadata` + frontmatter parser (C4)
- **`_registry.py`** — `SkillRegistry` with atomic persistence (C4)
- **`_extractor.py`** — End-of-task `SkillExtractor` (C4)
- **`validate.py`** — CLI validator (C4)
- **7 seed `SKILL.md` files** covering all 4 consumer scopes (C4)

Planning agent and prompts (`src/agent/adaptive_planning_agent/`):
- `adaptive_planning_agent.py` — Single `AdaptivePlanningAgent` class; optional `review_step` / `skill_registry` / `skill_extractor` components selected by config flags.
- `prompts/adaptive_planning_agent.yaml` — C2 prompt template (reactive tools only)
- **`prompts/adaptive_planning_agent_c3.yaml`** — C3 prompt template (documents `[REVIEW]` block)
- **`prompts/adaptive_planning_agent_c4.yaml`** — C4 prompt template (same `[REVIEW]` guidance as C3, plus skill registry + `activate_skill`)

Sub-agent prompt templates (`src/agent/{deep_analyzer,browser_use,deep_researcher}_agent/prompts/*.yaml`):
- Contain a `{%- if skill_registry_block %}` conditional for the skill registry injection. Renders empty in C0/C2/C3; populated in C4.

Configs:
- `config_gaia_c0.py` — Condition C0 (baseline `PlanningAgent`; alias over `config_gaia.py`)
- `config_gaia_adaptive.py` — Condition C2 (reactive diagnose/modify tools)
- **`config_gaia_c3.py`** — Condition C3 (C2 + `enable_review=True`)
- **`config_gaia_c4.py`** — Condition C4 (C3 + `enable_skills=True` + `enable_skill_extraction=True`)
- `config_gaia_adaptive_qwen.py` — Qwen/vLLM evaluation config

Scripts:
- `scripts/compare_results.py` — Compare results across conditions
- `scripts/analyze_results.py` — Generate terminal/HTML evaluation reports

Tests:
- **`tests/test_review_schema.py`** — Pydantic round-trip + validation tests (Phase 1)
- **`tests/test_skill_registry.py`** — Skill parsing + registry behavior tests (Phase 2)

### Experimental Conditions

This codebase is set up to run four experimental conditions for ADAS research on GAIA. **All four are fully implemented** (as of 2026-04).

| Condition | Config | Planner components | Meta-agent capability |
|-----------|--------|-------------------|----------------------|
| **C0** | `config_gaia_c0.py` | `PlanningAgent` | None (baseline) |
| **C2** | `config_gaia_adaptive.py` | `AdaptivePlanningAgent` | Reactive `diagnose_subagent` + `modify_subagent` (agent-invoked) |
| **C3** | `config_gaia_c3.py` | C2 + `review_step` | C2 + structural REVIEW step (automatic post-delegation assessment) |
| **C4** | `config_gaia_c4.py` | C3 + `skill_registry` + `skill_extractor` | C3 + cross-task skill library (pre-seeded + learned via task-end extractor) |

**Design notes:**

- C1 (reactive diagnose-only, no modify) was dropped because the existing `modify_subagent` action space covers both C1 and C2 uses without meaningful distinction.
- All four conditions use the same `AdaptivePlanningAgent` class (no subclassing). Components are composed via constructor kwargs + config flags (`enable_review`, `enable_skills`, `enable_skill_extraction`). **C4 is not “skills-only”:** with `enable_review=True` (as in `config_gaia_c4.py`), the sealed `ReviewAgent` / `ReviewStep` path is identical to C3; C4 only adds `skill_registry` and optional `SkillExtractor`.
- Architectural modifications (tools, managed agents, instructions, max_steps) are **task-scoped** and reset after each task via `AdaptiveMixin._reset_to_original_state`.
- The skill library is **cross-task** — newly-extracted skills persist across tasks and are visible to the next run. Set `enable_skill_extraction=False` to freeze the library for evaluation (recommended when the goal is to measure the contribution of a pre-trained skill library rather than online learning).
- The REVIEW apparatus (`ReviewAgent`, `SkillRegistry`, `SkillExtractor`) is **sealed** from `modify_subagent`: none of these objects appear in `managed_agents` or `tools`, so `_find_managed_agent` cannot reach them. This is deliberate — allowing the planner to modify its own reviewer would enable reward hacking.

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
