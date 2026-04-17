# Handoff: ToolGenerator Hardening + Dynamic `Tool.from_code` Selection

**Session date:** 2026-04-18  
**Commit:** `0161321` — `feat(meta): harden ToolGenerator with allowlist, AST imports, and repair retry`  
**Branch:** `main` (**`git pull origin main`** on farm after this hash)  
**Scope:** `ToolGenerator` (allowlist, AST import validation, repair retry, richer prompt), `Tool.from_code` disambiguation, `modify_subagent` collision-safe names, adaptive YAML copy, unit tests.

---

## TL;DR Checklist

### Completed (implementation)

- [x] Single **allowlisted import** source (`allowed_top_level_modules` / `format_allowlist_for_prompt`) shared by prompt text and AST validator.
- [x] **AST import validation** (`_validate_imports_ast`): no relative imports; only `src.tools` under `src.*`; roots must be in allowlist; disallowed roots rejected.
- [x] **Extended base allowlist** (stdlib-style helpers + `requests`, `urllib`) and optional **pandas / numpy / openpyxl / yaml** when `ToolGenerator(..., allow_data_science=True)` **or** `TOOL_GENERATOR_ALLOW_DATA_SCIENCE` env is `1` / `true` / `yes`.
- [x] **One repair retry** in `generate_tool_code` after `_validate_code` fails (assistant turn with first codegen, then user repair message with error + truncated prior code).
- [x] **Prompt enrichment:** `inputs` vs `forward` alignment note, input types line aligned with framework vocabulary, two reference examples (stdlib A + stdlib B or pandas B when data-science on).
- [x] **`Tool.from_code(..., expected_tool_name=...)`** — deterministic subclass pick when multiple `Tool` subclasses exist in one module string.
- [x] **`add_new_tool_to_agent`** passes `expected_tool_name`; **`_generate_tool_name`** appends SHA-256 prefix when `{base}_tool` already exists on target agent.
- [x] **`ModifySubAgentTool.parameters`** + adaptive **C2/C3/C4 YAML** wording aligned with allowlist (no stale “stdlib-only” list).
- [x] **`tests/test_tool_generator.py`** (+ mock `add_new_tool_to_agent(**kwargs)` in `tests/test_eval_fixes.py`).
- [x] Re-exports in `src/meta/__init__.py`: `allowed_top_level_modules`, `format_allowlist_for_prompt`.

### To do next session

- [x] **Commit + push** — hash `0161321`; index row updated.
- [ ] **GPU farm / CI env** — `git pull`, `conda activate dra` (or project venv), confirm **`crawl4ai` is importable** (declared in `pyproject.toml` / `requirements.txt`; see [HANDOFF_PROVIDER_MATRIX.md](HANDOFF_PROVIDER_MATRIX.md) if import fails).
- [ ] Run **`pytest tests/test_tool_generator.py -q`** (full `pytest` if farm policy requires whole suite).
- [ ] Optional smoke: one adaptive run that calls `modify_subagent` with `add_new_tool_to_agent` and confirm `[ToolGenerator] Code validation passed` / no `Multiple Tool subclasses` errors in `log.txt`.
- [ ] If eval uses generated CSV-style tools, decide whether farm jobs should set **`TOOL_GENERATOR_ALLOW_DATA_SCIENCE=1`** for pandas-backed codegen (trade-off: wider import surface).

---

## Original Problems

### Problem 1 — Allowlist drift and under-specified codegen

**Where:** [`src/meta/tool_generator.py`](src/meta/tool_generator.py)

**Symptoms:** Prompt listed a short import set while `pyproject.toml` already ships **pandas**, **numpy**, etc.; planner YAMLs said “stdlib-ish” while **`requests`** was allowed. Models produced **`inputs` / `forward` mismatches** or invalid imports; failures wasted turns.

**Root cause:** No single source of truth between prompt and validation; no AST-level import gate; minimal skeleton example only.

### Problem 2 — Fragile `Tool.from_code` selection

**Where:** [`src/tools/tools.py`](src/tools/tools.py) `Tool.from_code`

**Symptom:** `next(... issubclass ...)` picked an arbitrary `Tool` subclass when the LLM emitted helpers or multiple classes.

**Root cause:** No `expected_tool_name` hook tied to `modify_subagent`’s precomputed snake name.

### Problem 3 — Tool name collisions on repeat `add_new_tool_to_agent`

**Where:** [`src/meta/modify_tool.py`](src/meta/modify_tool.py) `_generate_tool_name`

**Symptom:** Same NL spec could yield the same `*_tool` name while an earlier dynamic tool still lived on the agent.

**Root cause:** No collision check against `agent.tools`.

---

## Changes (file-level)

| File | Change |
|------|--------|
| [`src/meta/tool_generator.py`](src/meta/tool_generator.py) | Allowlist constants + `format_allowlist_for_prompt` / `allowed_top_level_modules`; AST import validator; `ToolGenerator(model, allow_data_science=...)` + env; `_build_generation_prompt` with examples + repair retry in `generate_tool_code`; stricter `_validate_code` pipeline. |
| [`src/tools/tools.py`](src/tools/tools.py) | `from_code(..., expected_tool_name=None)`; collect and sort `Tool` subclasses; disambiguate or raise. |
| [`src/meta/adaptive_mixin.py`](src/meta/adaptive_mixin.py) | `add_new_tool_to_agent(..., *, expected_tool_name=None)` forwarded to `from_code`. |
| [`src/meta/modify_tool.py`](src/meta/modify_tool.py) | `_generate_tool_name(spec, agent)` collision suffix; `_add_new_tool` passes `expected_tool_name`; `specification` description updated. |
| [`src/meta/__init__.py`](src/meta/__init__.py) | Export allowlist helpers. |
| [`src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml`](src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent.yaml) (+ `_c3`, `_c4`) | Replace stale “stdlib-ish” import sentence with allowlist + `TOOL_GENERATOR_ALLOW_DATA_SCIENCE`; tweak CSV example spec line. |
| [`tests/test_tool_generator.py`](tests/test_tool_generator.py) | New unit tests (allowlist, AST, clean, validate, `from_code`, retry mock, collision naming). |
| [`tests/test_eval_fixes.py`](tests/test_eval_fixes.py) | Mock `add_new_tool_to_agent` accepts `**kwargs`. |

**Not changed:** `Model` temperature / max_tokens (out of scope per plan). No sandboxed `exec`.

---

## How to Test on GPU Farm

### Prerequisites

```bash
ssh <gpu-farm-host>
cd /path/to/DeepResearchMetaAgent
git pull origin main
git log -1 --oneline   # Confirm ToolGenerator handoff commit is present

conda activate dra   # or the env your SLURM jobs use
python -c "import crawl4ai; print('crawl4ai ok')" || pip install "crawl4ai>=0.6.3"
```

**Why `crawl4ai`:** Importing `src` pulls `src.utils` → `url_utils` → `crawl4ai`. A bare Python without project deps fails **test collection** before any test logic runs. This matches declared dependencies in [`requirements.txt`](requirements.txt) / [`pyproject.toml`](pyproject.toml); see also [HANDOFF_PROVIDER_MATRIX.md](HANDOFF_PROVIDER_MATRIX.md) (crawl4ai import error row).

### Unit tests (fast gate)

```bash
cd /path/to/DeepResearchMetaAgent
pytest tests/test_tool_generator.py -q --tb=short
```

**Pass criterion:** exit code 0; at least one run exercises the **repair retry** path (`_MockModel` with two responses).

### Optional eval smoke (slow)

Pick one C2/C3/C4 job that previously exercised `add_new_tool_to_agent`, or a 1-question harness. Grep logs:

```bash
grep -E "ToolGenerator|add_new_tool_to_agent|Multiple Tool subclasses" workdir/<run>/log.txt | tail -50
```

**Pass criterion:** no unhandled `Multiple Tool subclasses`; validation errors either disappear after retry or surface as a single planner-visible tool error (not a crash).

---

## Validation Criteria

| # | Check | Pass |
|---|--------|------|
| 1 | `pytest tests/test_tool_generator.py` | Exit 0 on farm env with full deps |
| 2 | `python -c "from src.meta.tool_generator import format_allowlist_for_prompt; print(format_allowlist_for_prompt(False))"` | Includes `pathlib`, `requests`; **no** `pandas` |
| 3 | Same with `True` or env set | Includes `pandas`, `numpy` |
| 4 | `grep -r "stdlib-ish imports are permitted" src/agent/adaptive_planning_agent/prompts/*.yaml` | **No hits** (old wording removed) |
| 5 | Optional: `modify_subagent` JSON in logs uses `add_new_tool_to_agent` | Still valid shapes (unchanged contract) |

---

## Default `allow_data_science`

**Default remains `False`** (constructor and `ModifySubAgentTool` lazy `ToolGenerator`). Rationale: generated code is `exec`’d; a smaller allowlist is easier to audit and avoids pulling heavy / sensitive numerical stacks into every codegen path. CSV-style tools can use **pathlib / csv / json** on the base allowlist; set **`TOOL_GENERATOR_ALLOW_DATA_SCIENCE=1`** (or pass `ToolGenerator(..., allow_data_science=True)`) when pandas-backed codegen is explicitly desired.

## Known Unknowns / Caveats

- **Repair transcript** is now **user → assistant (first codegen) → user (repair)** before the second model call, for APIs that require strict role alternation.
- **`exec` on generated code** remains; AST allowlist + regex bans reduce risk but do not sandbox execution.
- **`TOOL_GENERATOR_ALLOW_DATA_SCIENCE` overrides narrow codegen** — document in run scripts if farm jobs should opt in.
- **`tests/test_tool_generator.py`** does not hit the real LLM; farm validation is still needed for **live** codegen quality.

---

## For `HANDOFF_INDEX.md`

Row **#7** in [HANDOFF_INDEX.md](HANDOFF_INDEX.md) tracks this handoff. The **Commit(s)** column lists **`0161321`**, the implementation-only commit (code, tests, planner YAML). This handoff file and the index row were added in a separate docs commit. Move row #7 to **Completed / Archived** after farm validation is signed off.
