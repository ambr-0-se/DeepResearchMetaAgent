# Handoff: Core C3 / C4 Implementation (REVIEW step + skill library)

**Session date:** 2026-04-17
**Branch:** `main` (pushed — all four commits live on `origin/main`)
**Commits:** `60065a8` → `433c30e` → `0643089` → `d247605`
**Scope:** Implements the four GAIA experimental conditions for the ADAS research project.
- **C0** (baseline `PlanningAgent`)
- **C2** (existing `AdaptivePlanningAgent`: reactive `diagnose_subagent` + `modify_subagent`)
- **C3** (C2 + structural REVIEW step)
- **C4** (C3 + cross-task skill library, with optional task-end `SkillExtractor`)

**Relationship to other handoffs:**
- `HANDOFF_SILENT_FAILURES.md` (`ba28f21`) — tool-level fixes, orthogonal.
- `HANDOFF_PROVIDER_MATRIX.md` (`7632470`/`9883a3a`) — extends C0–C4 to 3 additional providers. Validates C3/C4 implicitly via the 12-cell matrix; this handoff adds validation focused on the **implementation correctness** itself (sealing, schema alignment, extractor pipeline, backward compat) independent of model provider.

---

## TL;DR Checklist

### Completed (this session)

- [x] **Phase 0** (`60065a8`): corrected misleading THINK-ACT-OBSERVE-REFLECT naming; extracted `src/meta/_memory_format.py` shared helpers; added `configs/config_gaia_c0.py` (baseline alias).
- [x] **Phase 1** (`433c30e`): implemented structural REVIEW step for condition C3 — Pydantic schemas, sealed internal `ReviewAgent`, `ReviewStep` orchestrator, `_post_action_hook` extension point in `AsyncMultiStepAgent._run_stream`, compositional `AdaptivePlanningAgent.review_step` kwarg, C3 prompt template, `configs/config_gaia_c3.py`.
- [x] **Phase 2** (`0643089`): implemented cross-task skill library for condition C4 — agentskills.io-compliant `Skill`/`SkillMetadata` parser, filesystem-backed `SkillRegistry`, `ActivateSkillTool` (consumer-scoped), 6-stage `SkillExtractor` pipeline, 7 pre-seeded SKILL.md files, sub-agent YAML conditional injection, C4 prompt template, `configs/config_gaia_c4.py`.
- [x] **Phase 3** (`d247605`): docs — README Updates, CLAUDE.md Experimental Conditions table + Skill Library section, `src/skills/README.md`.
- [x] Code reviewed by two independent Haiku subagents (Phase 1 and Phase 2 separately), plus a final holistic review across all four commits. 13/13 requirements PASS. 0 CRITICAL, 0 HIGH.
- [x] AST parse + YAML lint + Jinja render (empty + populated cases) all green.
- [x] Schema round-trip smoke tests (all 4 `NextAction` variants) pass.
- [x] Seed skills validated end-to-end (`Skill.from_skill_md` parses all 7; all 4 consumer scopes covered).
- [x] All four commits pushed to `origin/main` before later work landed on top.

### To Do Next Session (GPU Farm)

- [ ] `git pull origin main` on GPU farm; confirm HEAD includes `d247605` (plus any later commits, which don't affect this handoff's scope).
- [ ] **Pytest pass** — `pytest tests/test_review_schema.py tests/test_skill_registry.py -v`. Both suites must be GREEN (they couldn't run in the dev shell due to missing `huggingface_hub` dep on the local machine).
- [ ] **Pre-flight validators** — `python -m src.skills.validate src/skills` should report 7/7 OK with no WARNING for body length or description length on the seed skills.
- [ ] **Smoke — C0** (3 questions on the GAIA validation split) — sanity that baseline is unaffected by Phase 0's renames / helper extraction / C0 alias.
- [ ] **Smoke — C2** (3 questions) — confirms reactive-only behaviour is identical to pre-Phase-1. Existing `HANDOFF_SILENT_FAILURES.md` reprographs already validate C2-ish baselines with Qwen — if a recent C2 baseline exists, skip this.
- [ ] **Smoke — C3** (3 questions) — validates the REVIEW hook fires and mutates observations (the new thing).
- [ ] **Smoke — C4 in training mode** (5 questions) — validates skill activation AND extraction end-to-end.
- [ ] **Sealing audit** — verify in a real run that `modify_subagent` cannot target the sealed apparatus (explicit negative test via log grep).
- [ ] **Move to Completed** in `HANDOFF_INDEX.md` once the four validation criteria below pass.

---

## Original Problem

The APAI4799 MetaAgent research project studies Automated Design of Agentic Systems (ADAS) on GAIA. The experimental protocol requires four cleanly-attributable conditions that each add one marginal capability on top of the previous:

| Condition | What it adds over the previous |
|-----------|-------------------------------|
| C0        | — (vanilla `PlanningAgent` baseline) |
| C2        | Reactive `diagnose_subagent` + `modify_subagent` (agent-invoked when the agent notices failure) |
| C3        | Structural REVIEW step (automatic post-delegation assessment — the agent does not control when it fires) |
| C4        | Cross-task skill library (persistent across tasks; a task-end `SkillExtractor` can add new skills during training) |

Before this session, only C0 (as `config_gaia.py`) and C2 (as `config_gaia_adaptive.py`) existed. C1 was dropped (per the plan, the existing modify_subagent action space covered both C1 and C2 without meaningful distinction). C3 and C4 did not exist.

Additionally, the existing `AdaptivePlanningAgent` docstring, prompt template, and README claimed a **THINK-ACT-OBSERVE-REFLECT** loop. Tracing the code confirmed the actual loop is plain **THINK-ACT-OBSERVE** (no REFLECT step). The naming was corrected in Phase 0 to avoid methodological confusion.

**Research-methodology constraint:** the review apparatus (ReviewAgent, SkillRegistry, SkillExtractor) MUST be unreachable via `modify_subagent`. If the planner could modify its own reviewer, it would learn to make the reviewer lenient (classic reward hacking), and the C3/C4 contributions would be measuring a different thing than intended. This constraint drove the "sealed" design.

---

## Changes & Commits

### Commit `60065a8` — Phase 0 (housekeeping)

| File | Change |
|------|--------|
| `README.md`, `CLAUDE.md`, `src/agent/adaptive_planning_agent/adaptive_planning_agent.py` (docstring), `.yaml` prompt template | Replace "THINK-ACT-OBSERVE-REFLECT" with "THINK-ACT-OBSERVE" (reality) |
| `src/meta/_memory_format.py` (NEW) | Extract `format_execution_history()` and `format_agent_tools()` from `DiagnoseSubAgentTool` as module-level functions |
| `src/meta/diagnose_tool.py` | Former methods become thin wrappers calling the new module functions (backward compatible) |
| `configs/config_gaia_c0.py` (NEW) | Thin alias over `config_gaia.py` with `tag="gaia_c0"` so baseline results land in their own workdir |

Diff: 8 files, +206 / −98.

### Commit `433c30e` — Phase 1 (C3 structural REVIEW)

| File | Change |
|------|--------|
| `src/meta/review_schema.py` (NEW) | Pydantic `ReviewResult` + `RootCauseCategory` (8-item taxonomy) + polymorphic `NextAction` discriminated union (`ProceedSpec`, `RetrySpec`, `ModifyAgentSpec`, `EscalateSpec`). `ModifyAgentSpec` fields match `ModifySubAgentTool.forward()` exactly for pass-through. |
| `src/meta/review_agent.py` (NEW) | Sealed internal `ReviewAgent`. Subclass of `GeneralAgent` but NO `@AGENT.register_module` decorator. Built via `object.__new__` + manual attribute init. |
| `src/meta/review_step.py` (NEW) | Orchestrator. `run_if_applicable(action_step)` — fast-path skips for non-delegations / final answers / near max_steps, then invokes ReviewAgent, parses JSON, validates `EscalateSpec.to_agent` against real managed_agents, falls back to `ProceedSpec` on any error. Never raises. |
| `src/base/async_multistep_agent.py` | New `async def _post_action_hook(memory_step)` extension point (default no-op); wired into `_run_stream` finally block between `_finalize_step` and `memory.steps.append + yield`. |
| `src/agent/adaptive_planning_agent/adaptive_planning_agent.py` | Optional `review_step` kwarg; `_build_review_step_from_config()` reads `config.enable_review`; `_post_action_hook` override appends `[REVIEW]\n{rendered_result}` to `action_step.observations` and attaches raw `review_result` as a dataclass field for downstream use. |
| `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c3.yaml` (NEW) | C3-specific prompt template explaining the `[REVIEW]` block structure and `next_action` → `modify_subagent` arg mapping. |
| `configs/config_gaia_c3.py` (NEW) | Inherits from `config_gaia_adaptive.py`; `tag="gaia_c3"`; `enable_review=True`. |
| `tests/test_review_schema.py` (NEW) | Round-trip for all 4 `NextAction` variants; parametric test over all 7 `modify_action` values; validation bounds. |

Diff: 8 files, +1576 / −7.

### Commit `0643089` — Phase 2 (C4 skill library)

| File | Change |
|------|--------|
| `src/skills/__init__.py`, `_model.py`, `_registry.py`, `_extractor.py`, `validate.py` (ALL NEW) | agentskills.io-compliant Skill dataclass + SkillRegistry with atomic writes + SkillExtractor 6-stage pipeline (worthiness heuristic → LLM propose → structural validation → entity blocklist → LLM-judge dedup → persist) + CLI validator. |
| `src/meta/activate_skill_tool.py` (NEW) | `ActivateSkillTool` (AsyncTool). Per-instance `consumer` binding so scoped skills cannot leak. Read-only over the registry. |
| `src/agent/general_agent/general_agent.py` | `initialize_task_instruction` now merges `_extra_task_variables` dict into Jinja variables with `skill_registry_block=""` default (so conditionals don't break C0/C2/C3 under StrictUndefined). |
| `src/agent/deep_analyzer_agent/prompts/*.yaml`, `browser_use_agent/prompts/*.yaml`, `deep_researcher_agent/prompts/*.yaml` | Each gains `{%- if skill_registry_block %}...{%- endif %}` conditional block above `Here is the task:`. |
| `src/agent/adaptive_planning_agent/adaptive_planning_agent.py` | Optional `skill_registry` kwarg; `_build_skill_registry_from_config()`; `_install_skill_tools()` installs `ActivateSkillTool` on planner + every managed sub-agent; `_refresh_skill_registry_blocks()` runs before every `super().run()`; `SkillExtractor` wired to fire between `super().run()` and `_reset_to_original_state()`. |
| `src/skills/*/SKILL.md` × 7 (NEW) | Pre-seeded skills covering all 4 consumer scopes: `handling-file-attachments`, `task-decomposition-complex-queries`, `delegation-failure-recovery` (planner); `pdf-table-extraction`, `multi-hop-math-verification` (deep_analyzer_agent); `browser-paywall-recovery` (browser_use_agent); `research-fallback-sources` (deep_researcher_agent). |
| `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c4.yaml` (NEW) | C4 planner template — C3 content plus Jinja-conditional skill registry section. |
| `configs/config_gaia_c4.py` (NEW) | Inherits from `config_gaia_c3.py`; `tag="gaia_c4"`; `enable_skills=True` + `enable_skill_extraction=True`. |
| `tests/test_skill_registry.py` (NEW) | Skill parsing (valid + 7 rejection cases); metadata round-trip; registry scan / consumer filtering / load_body / render_registry_block / add / increment_verified_uses. |

Diff: 21 files, +2653 / −9.

### Commit `d247605` — docs

| File | Change |
|------|--------|
| `README.md` | New 2026.04 Updates entries summarising C3 + C4. |
| `CLAUDE.md` | Running commands now include C3 + C4 + validator; Experimental Conditions table marks all four as implemented; new "Skill Library" section; "Adaptive Agent Files" listing includes every new artifact. |
| `src/skills/README.md` (NEW) | Explains the dual-purpose `src/skills/` layout (Python package + skill catalogue), metadata schema, consumer routing, validation, seed vs. learned skills, sealing. |

Diff: 3 files, +183 / −35.

### Not in these four commits (intentional scope exclusions)

- Multi-provider model registrations (`7632470`)
- GAIA eval matrix for 3 models × C0-C4 (`9883a3a`)
- `final_answer_tool` premature-emission fixes (`54e7707`, `a9a6985`, `c52cf91`, `912685f`, `d36f4d4`)
- Expanded `modify_subagent` action coverage in prompts (`764c6bf`)
- Tool-level silent-failure fixes (`ba28f21` — covered by `HANDOFF_SILENT_FAILURES.md`)

These landed on top of our work, extend it, and are validated by their own handoffs or haven't been handed off yet. They are listed here only to make it clear they are NOT in scope for this handoff's validation.

---

## How to Test on GPU Farm

### Prerequisites

```bash
ssh <gpu-farm-host>
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
git pull origin main
git log -1 --oneline  # Should show something at or after d247605
conda activate dra
# Playwright browsers + any provider keys per HANDOFF_PROVIDER_MATRIX.md
bash scripts/ensure_playwright_browsers.sh
```

### Step 1 — Unit tests (no GPU needed)

```bash
pytest tests/test_review_schema.py tests/test_skill_registry.py -v
```

**Pass criterion:** all tests green. Expect ~25+ passing tests combined. If any fail, do NOT proceed to smoke tests until triaged.

### Step 2 — Skill validator

```bash
python -m src.skills.validate src/skills
```

**Pass criterion:** all 7 seed skills report `[OK]`. Any `WARNING` lines are informational but should be reviewed (description length, body length > 500 lines).

### Step 3 — Per-condition smoke tests

Use whatever small-run harness the farm already supports (e.g. `sbatch run_combined_test.sh` with an appropriate config override, or a custom 3-question `run_gaia.py` invocation). Run 3-5 questions per condition on the GAIA validation split so you can inspect traces manually.

```bash
# Each of these writes to workdir/gaia_<tag>/dra.jsonl where <tag> matches the config
python examples/run_gaia.py --config configs/config_gaia_c0.py  # expect: workdir/gaia_c0/
python examples/run_gaia.py --config configs/config_gaia_adaptive.py   # workdir/gaia_adaptive/
python examples/run_gaia.py --config configs/config_gaia_c3.py         # workdir/gaia_c3/
python examples/run_gaia.py --config configs/config_gaia_c4.py         # workdir/gaia_c4/
```

Limit to 3-5 questions via whatever the farm supports (`--max-samples 3` if that flag exists, or temporary task-ID filter per `HANDOFF_SILENT_FAILURES.md` §"Option B"). Do NOT commit the filter.

---

## How to Validate

### Validation 1 — C3 REVIEW hook actually fires

Run the C3 smoke above, then:

```bash
# Every sub-agent delegation should be followed by a [REVIEW] block in observations.
# Count the delegations vs the review blocks; they should match (modulo fast-path skips).

NEW_RUN=workdir/gaia_c3/dra.jsonl  # or $(ls -td workdir/gaia_c3_* | head -1)/dra.jsonl
DELEG_COUNT=$(grep -oE '"tool_calls":\s*\[\{"id":"[^"]*","type":"function","function":\{"name":"(deep_analyzer_agent|browser_use_agent|deep_researcher_agent)"' "$NEW_RUN" | wc -l | tr -d ' ')
REVIEW_COUNT=$(grep -c '\[REVIEW\]' "$NEW_RUN")

echo "delegations: $DELEG_COUNT"
echo "review blocks: $REVIEW_COUNT"
```

**Pass criterion:** `REVIEW_COUNT` >= `DELEG_COUNT - 2` (allow 1-2 fast-path skips for final-answer / near-max-steps). A ratio lower than that means the hook is not firing correctly.

Also confirm the REVIEW block parses as a well-formed block:

```bash
grep -A 6 '\[REVIEW\]' "$NEW_RUN" | grep -E "^verdict:|^summary:|^next_action:" | sort | uniq -c
```

Every REVIEW block must have `verdict:`, `summary:`, and `next_action:` lines (root_cause is conditional, detail is optional).

### Validation 2 — C4 skill activation actually happens

Run C4 smoke, then:

```bash
NEW_RUN=workdir/gaia_c4/dra.jsonl

# activate_skill appears as a tool call when a seed skill matches a task
grep -c '"name":"activate_skill"' "$NEW_RUN"
```

**Pass criterion:** `>= 1` activation across a 5-question run, assuming at least one question matches a seed skill (tasks with file attachments → `handling-file-attachments`; web-research tasks → `research-fallback-sources`; PDF table tasks → `pdf-table-extraction`; etc.). If zero activations across multiple runs, either (a) the activate_skill tool isn't installed, or (b) the registry block isn't reaching the planner's system prompt.

Cross-check by inspecting the planner's first THINK message in the log:

```bash
grep -A 50 'skill-registry consumer' workdir/gaia_c4/log.txt | head -100
```

You should see the `<skill-registry consumer='planner'>` block with at least 3 skills listed (the 3 planner-consumer seeds + any `all`-scoped seeds).

### Validation 3 — C4 skill extraction writes new SKILL.md files

C4 runs with `enable_skill_extraction=True` by default. After a ≥5-question training run:

```bash
ls -la src/skills/
# Count directories; each is a skill
find src/skills -name SKILL.md | wc -l
```

**Pass criterion:** the count should be ≥ 7 (the seeds). If extractor proposed new skills, it will be `> 7`. Newly-extracted skills carry `metadata.source: success` or `metadata.source: failure` — inspect via:

```bash
grep -l "source: success\|source: failure" src/skills/*/SKILL.md
```

It is acceptable (and common on a 5-question smoke) for the extractor to propose ZERO new skills — the worthiness heuristic and entity blocklist are deliberately conservative. What must NOT happen: the extractor raising an exception that breaks the planner's run. Check:

```bash
grep -c "SkillExtractor.*pipeline failed" workdir/gaia_c4/log.txt
```

Must be `0`. If `> 0`, the extractor is catching errors correctly (it never raises out of the planner), but the errors need triaging.

### Validation 4 — Sealing invariant (most important for research methodology)

The planner must NEVER be able to target the ReviewAgent, SkillRegistry, or SkillExtractor via `modify_subagent`. Verify negatively:

```bash
# modify_subagent call targets, across a full C4 run
grep -oE '"agent_name":"[^"]*"' workdir/gaia_c4/dra.jsonl | sort -u
```

**Pass criterion:** every `agent_name` in this list must be one of `deep_analyzer_agent`, `browser_use_agent`, `deep_researcher_agent`, or a newly-generated agent from an `add_agent` call (e.g., `math_expert_agent`). None may be `review_agent`, `skill_registry`, `skill_extractor`, or any name resembling the sealed components. If any such name appears, the sealing invariant is violated and C3/C4 results are methodologically compromised.

Also audit the `_find_managed_agent` path — if the farm has the full Python env, run:

```python
from src.agent.agent import create_agent
# ... build a C4 agent ...
assert "review_agent" not in planner.managed_agents
assert "review_agent" not in planner.tools
assert "skill_registry" not in planner.managed_agents
assert "skill_registry" not in planner.tools
assert "skill_extractor" not in planner.managed_agents
assert "skill_extractor" not in planner.tools
assert planner.review_step is not None  # C3+ only
assert planner.skill_registry is not None  # C4 only
assert planner.review_step._review_agent is not None  # sealed, reachable only via review_step
```

---

## Known Unknowns / Caveats

- **Local dev env could not run pytest.** The repository's `src/` transitively imports `huggingface_hub`, which isn't installed in the local shell that wrote Phases 0-2. All verification in that shell was AST-only + isolated module load + Jinja render. Pytest **must** run green on the farm before these commits are considered validated.
- **Sub-agent skill block requires AdaptivePlanningAgent to install the tool.** If you construct a sub-agent directly (without going through AdaptivePlanningAgent), the sub-agent's YAML will reference `{{skill_registry_block}}` but the default `""` from `GeneralAgent.initialize_task_instruction` kicks in, rendering nothing. This is correct behaviour, just something to be aware of during isolated sub-agent testing.
- **Later commits modified prompt templates.** Commits `54e7707`, `a9a6985`, `c52cf91`, `912685f` added rules about never emitting `final_answer_tool` in the same turn as other tools, and renamed the template `final_answer` → `final_answer_tool` in examples. These changes are orthogonal to C3/C4 correctness but the YAMLs you'll see on the farm differ from what's in `433c30e` / `0643089`. The additional rule does NOT break the [REVIEW] block or the skill registry block — both are Jinja-conditional on data the LLM cannot set.
- **Extractor model quality matters.** The SkillExtractor LLM proposal uses `self.parent.model` — the same model as the planner. If running C4 with a weak model, expect many "skip" responses (that is healthy — the entity blocklist and dedup filters are intentionally strict). If running with a strong model on a contamination-sensitive benchmark like GAIA, the blocklist should catch leaked proper nouns / specific numbers; verify by spot-checking any newly-extracted SKILL.md before promoting to a shared branch.
- **Provider matrix (`HANDOFF_PROVIDER_MATRIX.md`) covers more ground.** If the 12-cell matrix completes successfully with non-zero C3 and C4 cells, that implicitly validates most of Validations 1-3 above. This handoff's validation steps are the MINIMUM required for the C3/C4 implementation itself; the provider matrix is the broader matrix that confirms the implementation survives different providers.

---

## Useful Commands

```bash
# Show all four of our commits with file stats
git log --stat 60065a8^..d247605 | head -80

# Show just the diff of a specific commit
git show 433c30e --stat  # Phase 1
git show 0643089 --stat  # Phase 2

# Count REVIEW blocks in all recent C3/C4 runs
for f in workdir/gaia_c3*/log.txt workdir/gaia_c4*/log.txt; do
  echo "$f : $(grep -c '\[REVIEW\]' "$f") reviews"
done

# Show which seed skills are visible to which consumer
for skill in src/skills/*/SKILL.md; do
  printf '%-50s -> consumer: ' "$skill"
  grep -A1 '^metadata:' "$skill" | grep 'consumer:' | head -1
done
```
