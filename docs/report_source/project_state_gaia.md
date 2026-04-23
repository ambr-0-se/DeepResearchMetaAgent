# Project State — DeepResearchMetaAgent (GAIA Track)

_Last updated: 2026-04-19_

## 1. Summary

DeepResearchMetaAgent is a research implementation of embedded, online meta-agent optimisation on the GAIA benchmark. It extends the open-source DeepResearchAgent hierarchical framework (planning agent + three specialised sub-agents: Deep Analyser, Browser User, Deep Researcher) with four increasingly sophisticated experimental conditions that test reactive adaptation, automatic review, and cross-task skill learning. The codebase runs a 4-model × 4-condition = 16-cell evaluation matrix (Mistral Small, Qwen 3.6, Gemini 4-31B, Kimi K2.5) across conditions C0 (baseline), C2 (reactive diagnose/modify), C3 (sealed structural review), and C4 (cross-task skill library). Conditions C2 through C4 all use `AdaptivePlanningAgent` with optional compositional components; all architectural modifications are task-scoped and reset after each question.

## 2. Architecture Inherited from DeepResearchAgent

DeepResearchMetaAgent is a downstream research fork of DeepResearchAgent (arXiv 2506.12508). The upstream architecture provides:

- **Two-layer hierarchical design**: a top-level `PlanningAgent` (THINK-ACT-OBSERVE loop) that coordinates three specialised lower-level agents.
- **Base agent classes** (`AsyncMultiStepAgent`, `MultiStepAgent`) using ReAct framework with memory management, Tier B tool-message protocol (per-tool_call_id results), and token-budget-aware context pruning.
- **Tool system**: base `Tool` / `AsyncTool` classes with code introspection, registry-based instantiation, and safe Python sandbox.
- **Configuration system**: mmengine-based Python configs with inheritance (not TOML) for reproducibility and programmatic override.
- **Model integration**: routing layer supporting OpenAI (GPT-4.1), Anthropic (Claude 3.7 Sonnet), Google Gemini, Qwen (local vLLM + DashScope), Mistral (OpenRouter), Kimi (OpenRouter).
- **GAIA evaluation infrastructure**: per-question timeout, transient-error retry, vLLM health watchdog with auto-restart, per-tool and per-question result tracking.

This half of the project adds only the meta-agent layer (src/meta/, src/skills/) on top of the base hierarchy and introduces no structural changes to the three managed sub-agents themselves.

## 3. Conditions C0 / C2 / C3 / C4

### C0 — Baseline

**Configuration file:** `configs/config_gaia_c0.py` (thin alias over `config_gaia.py`)

**What it tests:** Vanilla `PlanningAgent` without any adaptive capabilities. The planner coordinates the three sub-agents but cannot diagnose failures or modify their behaviour at runtime.

**Key design choice:** C0 serves as the lower bound to measure the value of each meta-agent increment (C2, C3, C4). Results are written to `workdir/gaia_c0_<RUN_ID>/dra.jsonl` so per-model runs (e.g. C0 on Mistral, C0 on Qwen) never collide.

**Implementation:** No new code; uses inherited `PlanningAgent` class from DeepResearchAgent.

### C2 — Reactive Runtime Modification

**Configuration file:** `configs/config_gaia_adaptive.py`

**What it tests:** Agent-invoked reactive adaptation. The planner gets two new tools:
- `diagnose_subagent` — inspect a sub-agent's execution history (memory steps, tool calls, reasoning) after a failure.
- `modify_subagent` — apply one of seven modifications to a sub-agent (add/remove agents, add/remove/modify tools, modify instructions, adjust max_steps).

All architectural changes are task-scoped; `AdaptiveMixin._store_original_state()` snapshots the original config at task start, and `_reset_to_original_state()` restores it in the finally block after the task completes.

**Key files:**
- `src/meta/adaptive_mixin.py` — state management + modification methods
- `src/meta/diagnose_tool.py` — reactive failure diagnosis
- `src/meta/modify_tool.py` — seven-action modification tool
- `src/meta/tool_generator.py` — dynamic tool creation (used by `modify_subagent`'s `add_new_tool_to_agent` action)
- `src/meta/agent_generator.py` — dynamic agent creation
- `src/meta/_memory_format.py` — shared helpers for formatting execution history

**Design choice defended in report:** Task-scoping ensures clean isolation between GAIA questions and no cross-question pollution from learned modifications. The seven actions cover the common failure patterns (missing capability, wrong tool, unclear instructions); agent creates tools dynamically from natural-language specifications.

### C3 — Sealed Structural Review

**Configuration file:** `configs/config_gaia_c3.py`

**What it tests:** Automatic post-delegation assessment. After the planner delegates to any sub-agent, an internal sealed `ReviewAgent` runs automatically (at most 3 steps, not counting against planner budget). The reviewer produces a structured `ReviewResult` with:
- **Verdict** (satisfactory / partial / unsatisfactory)
- **Root-cause taxonomy** (8-category: missing_tool, wrong_tool, bad_instruction, misread_task, external, model_limit, unclear_goal, incomplete)
- **Polymorphic `next_action`** (proceed, retry, modify_agent, escalate) that maps directly onto `modify_subagent` if remediation is needed.

The review verdict and recommendation are injected into the planner's next THINK observation with a `[REVIEW]` marker. The planner may override the review and invoke `modify_subagent` manually if it disagrees.

**Key architectural property — sealing:** The `ReviewAgent` is NOT registered via the AGENT registry, NOT added to `managed_agents`, and NOT added to `tools`. It is constructed via `object.__new__()` + manual attribute init to avoid YAML file I/O. This prevents reward hacking where the agent learns to make the reviewer lenient instead of solving the task.

**Key files:**
- `src/meta/review_schema.py` — Pydantic models: `ReviewResult`, `RootCauseCategory` (8-item enum), `NextAction` (discriminated union of 4 action specs)
- `src/meta/review_agent.py` — the sealed internal ReviewAgent (in-source prompt, no YAML file, no registry entry)
- `src/meta/review_step.py` — orchestrator that fires from `_post_action_hook` after each sub-agent delegation; includes fast-path skips (non-delegation steps, final answer, near max_steps) and error handling
- `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c3.yaml` — planner prompt with guidance on the automatic `[REVIEW]` block

**Design choice:** Automatic review decouples detection of failure from remediation — the planner sees the verdict without needing to diagnose manually. Sealed apparatus prevents reward hacking. The 8-category taxonomy enables aggregate analysis across runs (fixed set, not free text).

### C4 — Cross-Task Skill Library

**Configuration file:** `configs/config_gaia_c4.py`

**What it tests:** Persistent, cross-task skill library following the agentskills.io specification. Seven pre-seeded skills (covering all four consumer scopes) are available from the first task. At task end, a `SkillExtractor` proposes new skills via a six-stage pipeline (worthiness heuristic → LLM propose JSON → structural validation → entity blocklist → LLM-as-judge dedup → disk persist). Extracted skills remain in the per-run `workdir/gaia_c4_<RUN_ID>/skills/` directory and are visible to subsequent tasks.

**Skill structure:** Each skill is a directory containing a `SKILL.md` file with YAML frontmatter + Markdown body. Frontmatter includes name, description, and metadata (consumer scope, skill_type, source, verified_uses). Each agent (planner + sub-agents) gets an `activate_skill` tool scoped to skills visible to it.

**Per-run isolation:** Every C4 invocation writes its `dra.jsonl` AND its evolved skill library to a fresh `workdir/gaia_c4_<model>_<RUN_ID>/skills/` directory. Parallel runs cannot race; repeated runs preserve prior learned skills (via a `.seeded` marker that prevents re-seeding). Resume is supported: `DRA_RUN_ID=<prior_id> python examples/run_gaia.py --config configs/config_gaia_c4.py`.

**Frozen-library evaluation:** To measure a pre-trained library (without online learning), create a second config inheriting from C4 and set `enable_skill_extraction=False`. This requires `--cfg-options agent_config.enable_skill_extraction=False` override (the pair `agent_config` and `planning_agent_config` are separate ConfigDicts after `mmengine.Config.fromfile` materialises them).

**Key files:**
- `src/meta/activate_skill_tool.py` — consumer-scoped read-only tool to retrieve full skill bodies
- `src/skills/_model.py` — `Skill` + `SkillMetadata` dataclasses with YAML frontmatter parser
- `src/skills/_registry.py` — `SkillRegistry` with atomic disk persistence and consumer-based filtering
- `src/skills/_extractor.py` — `SkillExtractor` with six-stage pipeline (worthiness, propose, validate, blocklist, deduplicate, persist)
- `src/skills/_seed.py` — pre-seed 7 canonical skills on first run (no-op on resume)
- `src/skills/validate.py` — CLI validator (`python -m src.skills.validate <dir>`)
- 7 seed skill directories: `handling-file-attachments`, `task-decomposition-complex-queries`, `delegation-failure-recovery` (planner scope); `pdf-table-extraction`, `multi-hop-math-verification` (deep_analyzer_agent scope); `browser-paywall-recovery` (browser_use_agent scope); `research-fallback-sources` (deep_researcher_agent scope)
- `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c4.yaml` — planner prompt with skill registry block + activate_skill guidance

**Design choice:** Consumer-scoped skills (routed to planner, specific sub-agents, or all) prevent information leakage and enable specialisation. Cross-task persistence tests whether agents can build and compound learned knowledge. Entity blocklist (years, currency, URLs, long numerics) guards against extracting overly-specific solutions.

## 4. Key Design Decisions with Citation Hooks

1. **Task-scoped state reset** (`src/meta/adaptive_mixin.py:_store_original_state()` / `_reset_to_original_state()`) — ensures clean isolation between GAIA questions so no accumulated drift pollutes later tasks.

2. **Sealed review apparatus** (`src/meta/review_agent.py`, `src/meta/review_step.py`) — ReviewAgent not in any registry, not in managed_agents, constructed via `object.__new__()` to prevent the agent from learning to make the reviewer lenient (reward hacking).

3. **8-category root-cause taxonomy** (`src/meta/review_schema.py:RootCauseCategory`) — fixed enum (not free text) to enable aggregate analysis and quantify which failure modes benefit most from which remediation.

4. **Polymorphic `NextAction` discriminated union** (`src/meta/review_schema.py:NextAction`) — each of four action variants (proceed, retry, modify_agent, escalate) is a Pydantic model so the planner can dispatch without manual type-checking; `ModifyAgentSpec` fields mirror `ModifySubAgentTool.forward()` so specs pass through unchanged.

5. **agentskills.io conformance** (`src/skills/_model.py`) — follows upstream specification for future ecosystem interop; extended with DRMA-specific metadata (consumer, skill_type, source, verified_uses).

6. **Consumer-scoped skill injection** (`src/skills/_registry.py:metadata_for()`) — skills visible to a given agent depend on consumer scope (planner, named sub-agent, or all), preventing over-generalisation and information leakage.

7. **Per-run skill directory + seeding strategy** (`src/skills/_seed.py`) — every C4 run gets its own `workdir/gaia_c4_<RUN_ID>/skills/` with a `.seeded` marker; on resume, the marker prevents re-seeding so learned skills survive; parallel runs never race.

8. **Six-stage skill extraction pipeline** (`src/skills/_extractor.py`) — worthiness heuristic → LLM propose → structure validation → entity blocklist → LLM dedup → persist; contracts never to raise, so extraction failure is silent and graceful.

9. **Config inheritance with per-run tagging** (`configs/config_gaia_c0.py` through `config_gaia_c4.py`) — each condition derives a unique `tag` from `DRA_RUN_ID` env var so results never collide; mmengine inheritance (`_base_ = ...`) avoids duplication.

10. **Tier B tool-message protocol** (inherited from DeepResearchAgent, used in memory + review diagnostics) — per-tool_call_id result tracking enables parallelisation and precise reconstruction of what each tool returned.

## 5. Evaluation Pipeline

**Entry point:** `examples/run_gaia.py` — GAIA dataset loader, per-question evaluation loop, transient-error retry, timeout handling.

**Run configuration:** Generated by `scripts/gen_eval_configs.py` (creates all 16 cell configs from a template) and launched in parallel by `scripts/run_eval_matrix.sh`. Matrix runner:
- Smoke mode: 3 questions × 16 cells = 48 total (default; override with LIMIT=5 for 80 q)
- Full mode: all GAIA test split × 16 cells

**Model routing:** Four slots registered in `src/models/models.py`:
- Mistral Small (`mistral-small`) via Mistral La Plateforme
- Kimi K2.5 (`or-kimi-k2.5`) via OpenRouter (Moonshot provider pinned; thinking disabled)
- Qwen 3.6 Plus (`or-qwen3.6-plus`) via OpenRouter with failover wrapper (DashScope → OpenRouter on quota exhaustion)
- Gemini 4 31B (`or-gemma-4-31b-it`) via OpenRouter (DeepInfra + Together providers; concurrency capped to 4 to avoid vLLM parser pad-bug under parallel load)

**Result output:** Each run writes JSON Lines to `workdir/gaia_<cond>_<model>_<RUN_ID>/dra.jsonl` with per-question structure: question ID, prediction, true answer, steps (serialised from memory objects with binary fields stripped), agent error (if any).

**Reliability hacks:** vLLM health watchdog (auto-restart if unresponsive); transient-error retry (429, 503, 504); token-budget-aware context pruning (85% threshold, keeps system + last 4 messages, truncates middle).

**Analysis:** `scripts/analyze_results.py` produces terminal summaries and optional HTML reports; `scripts/compare_results.py` diffs two JSONL files (e.g. C0 vs C2 on the same subset).

## 6. Test Coverage

**Unit tests (4889 lines across 11 files):**
- `test_review_schema.py` — Pydantic round-trip (all 4 next_action variants), discriminated union dispatch, validation of action enum alignment with `ModifySubAgentTool`
- `test_skill_registry.py` — skill parsing (valid + malformed), metadata filtering by consumer, atomic disk persistence, registry block rendering
- `test_tool_generator.py`, `test_skill_seed.py` — tool code introspection, in-memory templates, skill library seeding logic
- Additional files cover failover routing, tool message protocol, context pruning, reasoning content preservation across models

**What is NOT tested:**
- Integration tests of end-to-end GAIA runs (too expensive; covered by smoke/full matrix runners)
- Reward-hacking scenarios (the sealed ReviewAgent design is the guard; no unit test can verify an agent won't try to hack it under adversarial pressure)
- Long-running session behaviour — GAIA tasks are short (median < 1 minute); no evaluation of whether session-length drift accumulates
- Cross-task skill library evaluation against external baselines (no reproduction of Reflexion, ADAS, or DSPy on GAIA)

## 7. Experiments Status

**Runs completed (visible in workdir):**
- **2026-04-18 (phc = prior hand-crafted):** C0/C2/C3/C4 × Mistral/Kimi/Qwen, smoke size (~10 q per cell)
- **2026-04-19 (latest):** C0 × Mistral/Kimi/Gemma; C2/C3 × Kimi; individual test runs for validation

**Run layout:** Each directory `workdir/gaia_<cond>_<model>_<timestamp>/` contains:
- `dra.jsonl` — per-question results (prediction, steps, errors)
- `log.txt` — agent logs and config dump
- `skills/` subdirectory (C4 only) — evolved skill library with `.seeded` marker

**Missing / incomplete:**
- No full matrix run on all 16 cells × full test split (resource-intensive; smoke has been run)
- No frozen-library evaluation (`enable_skill_extraction=False` variant) — C4 results so far are training-mode (online learning enabled)
- No per-model ablation (e.g. C2 on Mistral, then swap Mistral for Kimi within same run to isolate model effect)

**Resumption:** Prior runs can be resumed: `DRA_RUN_ID=<prior_timestamp> bash scripts/run_eval_matrix.sh full`. The existing `dra.jsonl` is appended to; for C4, the `.seeded` marker causes seed-copy to be skipped so prior learned skills survive.

## 8. Open Gaps / Known Weaknesses

1. **No external baseline comparison** — only internal ablation (C0 → C2 → C3 → C4). No direct comparison to published ADAS systems (Reflexion, DSPy, LangChain RAG), Reflexion, or multi-round LLM-as-optimizer work; this limits claims about SOTA.

2. **GAIA is short-horizon** — median task length < 1 minute. Does not evaluate "long-running autonomous sessions" mentioned in the literature review as underdeveloped; cannot observe whether architectural drift accumulates over hours of multi-task sessions.

3. **C4 training-mode results conflate learning from skill extraction with the value of a pre-seeded library** — no frozen-library evaluation (`enable_skill_extraction=False`) has been run at scale. Cannot isolate the contribution of seeded skills vs. learned ones.

4. **Sealed ReviewAgent is not truly sealed against sophisticated agents** — the agent could theoretically learn to ignore [REVIEW] blocks or frame requests so they always trigger `proceed`. Only adversarial evaluation (not done) would catch this.

5. **Entity blocklist for skill extraction is heuristic** — blocklist guards against years/currency/URLs but a sophisticated agent could find other ways to extract overfitting (e.g. long sequences of numbers formatted as strings). No formal evaluation of the blocklist's efficacy.

6. **Tool generation (`ToolGenerator`) relies on LLM introspection** — if the code generation fails silently or produces invalid imports, the agent sees a cryptic error. No fallback grammar-based generation; all failures surface to the planner as tool-generation errors.

7. **Review verdict is LLM-generated without ground truth** — the ReviewAgent's judgement (e.g. "wrong_tool" vs. "misread_task") is never validated against a human oracle. Aggregate statistics on verdict distribution are descriptive, not prescriptive.

8. **Skill extractor has no human curation loop** — all extracted skills persist automatically (after dedup). No way for a human (or the meta-agent) to flag low-quality skills; once persisted, they accumulate across runs indefinitely.

9. **Matrix runner uses 4 models but no statistical significance testing** — differences in accuracy (e.g. C0 66% vs. C2 71% on Mistral) are reported as point estimates; no confidence intervals or cross-validation.

10. **Resume protocol assumes config immutability** — if a config changes between runs with the same `DRA_RUN_ID`, the appended JSONL will have inconsistent metadata. No validation of this assumption.

## 9. File Index

| Path | Role |
|------|------|
| `src/meta/__init__.py` | Public exports of adaptive components |
| `src/meta/adaptive_mixin.py` | Mixin for task-scoped state management + modification methods |
| `src/meta/diagnose_tool.py` | Reactive sub-agent failure diagnosis tool |
| `src/meta/modify_tool.py` | Seven-action modification tool (add/remove agents, add/remove tools, modify instructions, set max_steps) |
| `src/meta/tool_generator.py` | Dynamic tool creation from natural-language specs (used by modify_subagent) |
| `src/meta/agent_generator.py` | Dynamic agent creation with in-memory templates |
| `src/meta/_memory_format.py` | Shared helpers for formatting execution history and tool lists |
| `src/meta/review_schema.py` | Pydantic models: ReviewResult, RootCauseCategory (8 enum), NextAction discriminated union |
| `src/meta/review_agent.py` | Sealed internal ReviewAgent (not registered, not in managed_agents) |
| `src/meta/review_step.py` | Orchestrator for automatic post-delegation review (invoked from _post_action_hook) |
| `src/meta/activate_skill_tool.py` | Consumer-scoped tool to retrieve skill bodies from registry |
| `src/skills/__init__.py` | Skill module exports |
| `src/skills/_model.py` | Skill + SkillMetadata dataclasses; YAML frontmatter parser |
| `src/skills/_registry.py` | SkillRegistry with scan, filtering, atomic persistence, and rendering |
| `src/skills/_extractor.py` | SkillExtractor: six-stage task-end pipeline for new skill proposal and persist |
| `src/skills/_seed.py` | Pre-seed 7 canonical skills on first run; `.seeded` marker prevents re-seed on resume |
| `src/skills/validate.py` | CLI validator for SKILL.md files |
| `src/skills/{7 skill dirs}` | Canonical pre-seeded skills (handling-file-attachments, pdf-table-extraction, browser-paywall-recovery, etc.) |
| `src/agent/adaptive_planning_agent/adaptive_planning_agent.py` | AdaptivePlanningAgent class: extends PlanningAgent with review_step, skill_registry, skill_extractor (optional) |
| `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml` | C2 prompt template (reactive tools guidance) |
| `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c3.yaml` | C3 prompt template (documents [REVIEW] block) |
| `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c4.yaml` | C4 prompt template (documents [REVIEW] + activate_skill) |
| `configs/config_gaia_c0.py` | C0 baseline (vanilla PlanningAgent); retags output to gaia_c0_<RUN_ID> |
| `configs/config_gaia_adaptive.py` | C2 configuration (reactive diagnose/modify tools) |
| `configs/config_gaia_c3.py` | C3 configuration (C2 + enable_review=True) |
| `configs/config_gaia_c4.py` | C4 configuration (C3 + enable_skills=True + enable_skill_extraction=True) |
| `configs/config_gaia.py` | Base GAIA config (inherited by C0, C2, C3, C4) |
| `examples/run_gaia.py` | GAIA evaluation entry point; dataset loader, per-question loop, timeout/retry handling |
| `scripts/analyze_results.py` | Result analysis: terminal summary + optional HTML report, per-tool stats, adaptive-tool usage patterns |
| `scripts/compare_results.py` | Diff two JSONL result files (e.g. C0 vs C2) with accuracy breakdown by difficulty |
| `scripts/gen_eval_configs.py` | Generate all 16 cell configs (4 models × 4 conditions) from template |
| `scripts/run_eval_matrix.sh` | Parallel matrix runner (smoke / full modes, per-model model slot routing, vLLM watchdog) |
| `tests/test_review_schema.py` | Unit tests for ReviewResult Pydantic models and next_action variants |
| `tests/test_skill_registry.py` | Unit tests for skill parsing, registry filtering, persistence |
| `tests/test_tool_generator.py` | Tool code introspection and dynamic creation tests |
| `tests/test_skill_seed.py` | Pre-seed logic and .seeded marker tests |
