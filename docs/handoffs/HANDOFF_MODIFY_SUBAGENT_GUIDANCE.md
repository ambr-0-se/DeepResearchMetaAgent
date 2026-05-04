# Handoff: `modify_subagent` Prompt + Tool-Description Guidance Expansion

**Session date:** 2026-04-17
**Commit:** `764c6bf` — `feat(meta): expand modify_subagent action coverage in prompts and tool description`
**Branch:** `main` — **already pushed** to `origin/main`
**Scope:** Adaptive planning agent prompt guidance (C1/C2/C3) + `ModifySubAgentTool` description. No handler logic changed.

---

## TL;DR Checklist

### Completed

- [x] Audited the `modify_subagent` prompt/tool-description surface — found 3/7 actions had worked examples; `add_new_tool_to_agent` param wording was ambiguous; no failure→action mapping; no anti-patterns.
- [x] Rewrote `ModifySubAgentTool.description` to include all 7 worked examples; tightened `parameters.specification.description` with per-action guidance and the real `ToolGenerator` import allowlist + disallowed-patterns list.
- [x] Added a failure-mode → action table + condition-scoped anti-patterns + 4 new example blobs to **all three** adaptive YAMLs: C1 (`adaptive_planning_agent.yaml`), C2 (`adaptive_planning_agent_c2.yaml`), C3 (`adaptive_planning_agent_c3.yaml`).
- [x] Independent code review by Haiku code-reviewer: raised 1 HIGH + 1 MEDIUM + 2 LOW findings; all 4 resolved in the committed version (REVIEW examples gap closed; anti-pattern scoped differently per condition to avoid conflicting with partial-verdict REVIEW recommendations; "Allowed imports" not "stdlib"; clamp wording consistent between Python docstring and YAML table).
- [x] Static verification locally: all 3 YAMLs parse, `modify_tool.py` compiles, AST-based check confirms all 7 `action="..."` example strings appear in the description and that `specification` description contains `Allowed imports`, `requests, json`, `Disallowed`, `hard clamp [1, 50]`.
- [x] Committed as `764c6bf` (4 files, +192/−3) and pushed to `origin/main`.

### To Do Next Session (on GPU farm)

- [ ] Pull on GPU farm: `ssh <gpu-farm> && cd DeepResearchMetaAgent && git pull` — confirm `git log -1 --oneline` shows `764c6bf`.
- [ ] Run the **rendered-prompt check** (Validation 1 below) against all 9 adaptive matrix configs to confirm the new guidance actually reaches the model via YAML→Jinja rendering.
- [ ] Run a **1-question smoke test** for each (model, condition) cell that exercises `modify_subagent` — 9 runs total: `{mistral, kimi, qwen} × {c1, c2, c3}`.
- [ ] **Qwen token-budget check** specifically: the new guidance adds ~60 lines to the planner's `task_instruction`. If `qwen3.6-plus-failover` instruction-following degrades on the longer prompt, plan a compact fallback (collapse anti-pattern to 2 lines).
- [ ] **Full eval matrix** comparison to the pre-`764c6bf` baseline once smoke passes: expected signal is non-zero usage of the 4 previously-unused actions (`add_new_tool_to_agent`, `remove_tool_from_agent`, `set_agent_max_steps`, `remove_agent`), accuracy and recovery-rate non-decreasing.

---

## Original Problems

All four problems are motivational (no failing test, no specific eval log): the hypothesis is that under-specified prompt/tool-description guidance causes the planner to leave 4 of the 7 `modify_subagent` actions effectively unused, hurting recovery on C1/C2/C3.

### Problem 1 — Only 3 of 7 actions had worked examples

**Where:** `src/meta/modify_tool.py:33-49` and `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent{,_c2,_c3}.yaml`

**Symptom (at decision time):** Both the tool description string and every adaptive YAML showed identical 3 example JSON blobs — `modify_agent_instructions`, `add_existing_tool_to_agent`, `add_agent`. The remaining 4 actions (`add_new_tool_to_agent`, `remove_tool_from_agent`, `set_agent_max_steps`, `remove_agent`) had zero worked examples anywhere visible to the LLM.

**Expected impact on eval:** The planner biases toward whichever action shapes it has seen. The 4 unused actions — including `set_agent_max_steps` (cheap unblocking on max-step timeout) and `add_new_tool_to_agent` (the LLM-powered novel-capability path) — are effectively dormant.

### Problem 2 — `add_new_tool_to_agent` spec argument ambiguous

**Where:** `src/meta/modify_tool.py:64` (old `specification` description)

**Root cause:** The description said `"add_new_tool_to_agent (tool description)"` — it did not clarify that (a) the planner passes a natural-language requirement, NOT Python code; (b) `ToolGenerator.GENERATION_PROMPT` (`src/meta/tool_generator.py:26-70`) constrains allowed imports to `requests, json, re, os, datetime, math, typing`; (c) `ToolGenerator._validate_code` (`:190-213`) rejects `os.system`, `subprocess`, `eval`, `exec`, `__import__`, file writes, `shutil.rmtree`, `os.remove`.

**Expected impact:** Planner either avoids the action entirely or generates specs that require external API keys — the validator rejects those, wasting a call.

### Problem 3 — No failure-mode → action mapping in the prompt

**Where:** `task_instruction` "After each team member response" block in all three YAMLs.

**Root cause:** The old bullet said only *"use modify_subagent (tools, instructions, or a new sub-agent per the tool description)"* — no ordering signal for which action to pick given an observed failure mode.

### Problem 4 — No anti-pattern guardrails

**Root cause:** Nothing in the prompt discouraged premature modification, stacked near-identical modifications, using `add_new_tool_to_agent` when `python_interpreter_tool` would suffice, or creating new agents when an instruction edit would fix the issue.

---

## Changes & Commits

### Commit `764c6bf` — four files

| File | Change |
|------|--------|
| [src/meta/modify_tool.py](src/meta/modify_tool.py) | `description`: +4 action example lines (`add_new_tool_to_agent`, `remove_tool_from_agent`, `set_agent_max_steps`, `remove_agent`). `parameters.specification.description`: rewritten from one-line "Depends on action:…" to per-action multi-line guidance including ToolGenerator's allowed-imports list, disallowed-patterns list, and `set_agent_max_steps` clamp `[1, 50]` (recommended practical cap 20). No handler-logic changes; `ACTIONS_REQUIRING_SPEC` unchanged. |
| [src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml](src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml) | **C1** (reactive). Replaced single-line `modify_subagent` hint with failure-mode→action table covering all 7 actions + preference-order statement. Added C1-scoped "Do NOT" block (first bullet: "before at least one delegation has been attempted"). Appended 4 new example blobs after the existing 3. |
| [src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c2.yaml](src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c2.yaml) | **C2** (+ sealed REVIEW). Same table/preference-order inserted just before "Example call shapes" inside the "How to act on REVIEW" block. Anti-pattern first bullet differs from C1 (**"Override REVIEW's `next_action: proceed` reflexively"**) to align with REVIEW's legitimate `verdict=partial` recommendations. 4 new example blobs appended inside the REVIEW-action block (the primary modification trigger in C2). |
| [src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c3.yaml](src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c3.yaml) | **C3** (+ skills). Identical edits to C2 (same table, same REVIEW-scoped anti-pattern, same 4 new example blobs). Skill-library section untouched. |

Diff stats: **+192 / −3** across four files. No action enum changes; existing tests (`tests/test_review_schema.py:93-99`, `tests/test_eval_fixes.py:229-242`) are unaffected by design.

### Not in the commit

- No config files touched. The 9 adaptive matrix configs (`configs/config_gaia_{c1,c2,c3}_{mistral,kimi,qwen}.py`) reference the three base YAMLs via `template_path`, so my edits propagate to all of them automatically — no regeneration via `scripts/gen_eval_configs.py` needed.
- No sub-agent YAMLs touched. Sub-agents don't have `modify_subagent` bound; the tool description auto-renders into the planner's system prompt via Jinja `{%- for tool in tools.values() %}` at `adaptive_planning_agent.yaml:93-97`.

---

## How to Test on GPU Farm

### Prerequisites

```bash
ssh <gpu-farm-host>
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
git pull origin main
git log -1 --oneline   # Expect: 764c6bf feat(meta): expand modify_subagent action coverage ...

conda activate dra
```

### Smoke-test matrix

The new guidance affects only the 3 adaptive conditions (C1, C2, C3) — **not C0** (baseline `PlanningAgent` has no `modify_subagent`). Tests are needed for 9 configs:

```
configs/config_gaia_c1_{mistral,kimi,qwen}.py
configs/config_gaia_c2_{mistral,kimi,qwen}.py
configs/config_gaia_c3_{mistral,kimi,qwen}.py
```

Kick off one-question smoke runs via the matrix runner in smoke mode:

```bash
# From the matrix runner (parallel per model, sequential per condition within a model)
bash scripts/run_eval_matrix.sh smoke
```

Or target a single (model, condition) cell with the matrix runner's filter arguments:

```bash
bash scripts/run_eval_matrix.sh smoke qwen c1
bash scripts/run_eval_matrix.sh smoke kimi c2
bash scripts/run_eval_matrix.sh smoke mistral c3
```

If `run_eval_matrix.sh`'s arg form differs, fall back to direct calls:

```bash
sbatch run_combined_test.sh   # adjust config inside script to the target adaptive config
# or for a direct 1-question run (local, no SLURM):
python examples/run_gaia.py --config configs/config_gaia_c1_qwen.py --limit 1
```

- Matrix results land in: `workdir/gaia_<tag>_<JOBID>_<timestamp>/`
- Monitor: `squeue -u $USER && tail -f logs/combined_test_<JOBID>.out`

### Full eval (after smoke passes)

```bash
bash scripts/run_eval_matrix.sh full
```

Wall-clock depends on the matrix and GAIA subset size — plan for multi-hour, per CLAUDE.md evaluation matrix guidance.

---

## How to Validate the Fixes

### Validation 1 — Rendered prompt reaches the model correctly

The YAML→Jinja rendering path is the main risk for a prompt-only change. Before running any eval, confirm the new guidance actually lands in the rendered system/task prompt for each adaptive config. Pick any one config per condition (the template_path is shared within a condition):

```bash
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
conda activate dra

python - <<'PY'
from mmengine.config import Config
from src.agent.agent import create_agent
import asyncio

for cfg_path in [
    "configs/config_gaia_c1_qwen.py",
    "configs/config_gaia_c2_qwen.py",
    "configs/config_gaia_c3_qwen.py",
]:
    cfg = Config.fromfile(cfg_path)
    agent = asyncio.run(create_agent(cfg))   # create_agent is async in this repo
    ti = agent.prompt_templates["task_instruction"]
    # Core new-guidance markers:
    assert "Preference order" in ti, f"{cfg_path}: preference-order sentence missing"
    assert "| Failure mode" in ti, f"{cfg_path}: failure-mode table missing"
    assert "Do NOT" in ti, f"{cfg_path}: anti-pattern block missing"
    # All 4 previously-unused action example strings must appear somewhere:
    for a in ["add_new_tool_to_agent", "remove_tool_from_agent",
              "set_agent_max_steps", "remove_agent"]:
        assert a in ti, f"{cfg_path}: missing action example for {a}"
    # Condition-scoped anti-pattern sanity:
    if "_c2" in cfg_path or "_c3" in cfg_path:
        assert "Override REVIEW" in ti, f"{cfg_path}: REVIEW-scoped anti-pattern missing"
    else:
        assert "before at least one delegation has been attempted" in ti
    print(f"OK {cfg_path}")
PY
```

If `create_agent` is synchronous in the current branch, drop the `asyncio.run` wrap. Adjust the import if it has moved.

**Pass criterion:** All three configs print `OK`. Failure means the Jinja template isn't rendering what we wrote — investigate before running any eval.

### Validation 2 — Planner actually uses the previously-dormant actions

After smoke runs land, grep the 9 smoke-test log files for each of the 4 previously-unused actions:

```bash
for f in workdir/gaia_c1_*_SMOKE_*/dra.jsonl workdir/gaia_c2_*_SMOKE_*/dra.jsonl workdir/gaia_c3_*_SMOKE_*/dra.jsonl; do
  echo "=== $f ==="
  for a in add_new_tool_to_agent remove_tool_from_agent set_agent_max_steps remove_agent; do
    c=$(grep -c "\"action\": \"$a\"" "$f" 2>/dev/null || echo 0)
    echo "  $a: $c"
  done
done
```

**Pass criterion:** This is a 9-question sample so zero counts are expected for most cells. Look for **any** non-zero count on at least one of the 4 actions across the 9 smoke runs. Full-eval Validation 5 tightens this.

### Validation 3 — Tool calls still parse

The prompt grew ~60 lines. Kimi has locked sampling; Qwen3-VL-4B is the smallest model we evaluate. Check for tool-call-parse errors that weren't present in the baseline:

```bash
for f in workdir/gaia_{c1,c2,c3}_{mistral,kimi,qwen}_SMOKE_*/dra.jsonl; do
  bad=$(grep -cE "Tool call parse (error|failure)|could not parse action" "$f" 2>/dev/null || echo 0)
  echo "$(basename $(dirname $f)): $bad parse failures"
done
```

**Pass criterion:** No new parse-failure patterns that weren't already in the pre-`764c6bf` baseline. If Kimi or Qwen regress here, the prompt is too long for that model — prepare a compact variant (collapse anti-pattern block to 2 lines; drop the preference-order sentence).

### Validation 4 — REVIEW → `modify_subagent` mapping still works in C2/C3

C2/C3's REVIEW step can emit `next_action: modify_agent <agent> [<modify_action>]: <specification>`. Confirm the planner still maps this onto a valid `modify_subagent` call (this path is unchanged by my edits, but the new anti-pattern's first bullet was rewritten specifically to not conflict with REVIEW — sanity check):

```bash
# How many REVIEW-driven modify_subagent calls fire? (REVIEW exists on C2/C3 only.)
grep -c "\[REVIEW\]" workdir/gaia_c2_*_SMOKE_*/log.txt workdir/gaia_c3_*_SMOKE_*/log.txt
grep -c "modify_subagent" workdir/gaia_c2_*_SMOKE_*/log.txt workdir/gaia_c3_*_SMOKE_*/log.txt
```

**Pass criterion:** The ratio of `modify_subagent` to `[REVIEW]` blocks in C2/C3 should not regress vs. the pre-`764c6bf` baseline.

### Validation 5 — Full-eval signal (after smoke passes)

Once the 9 smoke runs pass Validations 1–4, run the full matrix and compare to the pre-`764c6bf` baseline (if a pre-change run exists) or the most recent archived `gaia_c{1,2,3}_*` run:

```bash
python scripts/compare_results.py \
  workdir/gaia_c1_qwen_<OLD>_*/dra.jsonl \
  workdir/gaia_c1_qwen_<NEW>_*/dra.jsonl
# Repeat for the other 8 cells.

# If compare_results.py does not track per-action modify counts, drop to raw grep:
for a in add_new_tool_to_agent remove_tool_from_agent set_agent_max_steps remove_agent \
         add_existing_tool_to_agent modify_agent_instructions add_agent; do
  old=$(grep -c "\"action\": \"$a\"" workdir/gaia_c1_qwen_<OLD>_*/dra.jsonl)
  new=$(grep -c "\"action\": \"$a\"" workdir/gaia_c1_qwen_<NEW>_*/dra.jsonl)
  printf "%s: old=%s new=%s\n" "$a" "$old" "$new"
done
```

**Pass criterion:**
- Non-zero new-count for at least one of `{add_new_tool_to_agent, remove_tool_from_agent, set_agent_max_steps, remove_agent}` on at least one (model, condition) cell. Most likely to appear: `set_agent_max_steps` on long-running tasks; `add_new_tool_to_agent` on domain-specific tasks with no existing tool fit.
- Accuracy (answered / total) non-decreasing per cell.
- Recovery rate (answered after initial sub-agent failure / total failed attempts) non-decreasing per cell.
- No spike in stacked near-identical `modify_agent_instructions` calls on the same sub-agent (anti-pattern working).

---

## Context the Next Session Will Need

### Key file paths

- **Approved plan:** `~/.claude/plans/plan-the-implementation-fancy-falcon.md` (local machine only — includes full design rationale and all 4 resolved review findings with severity).
- **Tool source:** [src/meta/modify_tool.py:33-79](src/meta/modify_tool.py:33) — description + parameters.
- **ToolGenerator constraints (truth source for allow/disallow lists):** [src/meta/tool_generator.py:26-70](src/meta/tool_generator.py:26) (GENERATION_PROMPT), [:190-213](src/meta/tool_generator.py:190) (_validate_code).
- **Prompts:** `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent{,_c2,_c3}.yaml` — `task_instruction` sections.

### Design decisions (so next session can judge trade-offs)

- **C2/C3 anti-pattern first bullet differs from C1 deliberately.** C2/C3's REVIEW step can legitimately recommend `modify_agent` on `verdict=partial` (not just outright failure). C1's "before at least one delegation has been attempted" would conflict with that; the C2/C3 version is "Override REVIEW's `next_action: proceed` reflexively…".
- **`set_agent_max_steps` documented as clamp `[1, 50]` but recommended practical cap 20.** Hard clamp is in [src/meta/modify_tool.py:335](src/meta/modify_tool.py:335); the lower recommended cap is a prompt-level nudge.
- **`ToolGenerator` allowed imports list includes `requests` (third-party).** Worded as "Allowed imports" not "stdlib only" after review caught that `requests` is not stdlib — `requests` is installed in the project env so the phrasing matters only for future readers / contributors.
- **Sub-agent YAMLs intentionally NOT touched.** The tool's `description` auto-renders into the planner's system prompt via Jinja; sub-agents don't bind `modify_subagent`.

### Known unknowns / risks

- **Token-budget degradation on Qwen3-VL-4B and Kimi.** The prompt grew ~60 lines. If Qwen's instruction-following regresses, the plan has a compact fallback ready (collapse anti-pattern to 2 lines, drop preference-order sentence).
- **No dedicated test exercises `add_new_tool_to_agent` or `set_agent_max_steps` through `forward()`.** Not a regression risk (behavior unchanged) but means the only signal we have is eval-behavioral.
- **`compare_results.py` granularity.** Per the original plan this may not track per-action counts; the raw `grep` fallback in Validation 5 is authoritative.

### Useful commands

```bash
# Show the commit diff
git show 764c6bf

# Count modify_subagent calls by action in a given dra.jsonl
for a in add_agent remove_agent add_existing_tool_to_agent add_new_tool_to_agent \
         remove_tool_from_agent modify_agent_instructions set_agent_max_steps; do
  c=$(grep -c "\"action\": \"$a\"" workdir/<run_dir>/dra.jsonl)
  printf "%-32s %s\n" "$a" "$c"
done

# Compare two runs (falls back to manual grep if script lacks per-action granularity)
python scripts/compare_results.py workdir/<old_run>/dra.jsonl workdir/<new_run>/dra.jsonl
```
