# Migration: GAIA experimental conditions (C2/C3/C4 → C1/C2/C3)

This note applies to clones and automation updated before **May 2026**, when condition labels and config names were **renumbered** so the matrix is contiguous **C0–C3** with a clear progression.

## Why it changed

Older docs and configs used **C2** for the reactive adaptive planner, **C3** for the sealed REVIEW step, and **C4** for the cross-task skill library. That matched an internal sequence where **C1** had been dropped from the evaluated grid. The repo now uses **C1/C2/C3** for those three capability levels so naming matches the paper/report convention (C0 baseline + three adaptive tiers).

**C0 is unchanged** (vanilla `PlanningAgent` baseline).

## Condition mapping

| Old label | New label | Capability |
|-----------|-----------|------------|
| C0 | C0 | Baseline planning agent |
| C2 (reactive adaptive) | **C1** | `diagnose_subagent` + `modify_subagent`; no structural REVIEW |
| C3 (+ REVIEW) | **C2** | C1 + automatic sealed `ReviewStep` after each delegation |
| C4 (+ skills) | **C3** | C2 + `SkillRegistry`, `activate_skill`, optional `SkillExtractor` |

Config flags on the adaptive planner:

| Flag | Old “main” condition | New condition |
|------|----------------------|---------------|
| Reactive tools only (no `enable_review`) | C2 | **C1** |
| `enable_review=True` | C3 | **C2** |
| `enable_review=True` and `enable_skills=True` | C4 | **C3** |

## Config files

| Old | New |
|-----|-----|
| `configs/config_gaia_adaptive.py` | **Stub:** inherits `config_gaia_c1.py` only for backward-compatible `--config configs/config_gaia_adaptive.py` — **prefer `config_gaia_c1.py`** in new work |
| `configs/config_gaia_adaptive_qwen.py` | `configs/config_gaia_c1_qwen_local.py` (local/Qwen-focused C1) |
| `configs/config_gaia_c4.py` | **Removed** — use `config_gaia_c3.py` |
| `configs/config_gaia_c4_{gemma,kimi,mistral,qwen}.py` | `configs/config_gaia_c1_{gemma,kimi,mistral,qwen}.py` (same variants, **C1** slot) |
| *N/A* | **`configs/config_gaia_c2.py`** added — C2 entry (REVIEW), model-specific files still `config_gaia_c2_*.py` |
| `configs/config_gaia_c3.py` (old: REVIEW only) | **`configs/config_gaia_c3.py`** now extends **`config_gaia_c2.py`** and adds **skills** + extractor options |

Always open the module docstring at the top of each `config_gaia_*.py` for the exact `tag` / `workdir` pattern.

## Prompt templates

| Old path | New path |
|----------|----------|
| `src/agent/adaptive_planning_agent/prompts/adaptive_planning_agent_c4.yaml` | `.../adaptive_planning_agent_c2.yaml` (REVIEW-era planner copy) |
| `adaptive_planning_agent_c3.yaml` | Still **C3** prompt, updated for **skill library** (not “review-only”) |

## Scripts and pipelines

| Old | New |
|-----|-----|
| `scripts/integration_i3_c4_pipeline.sh` | `scripts/integration_i3_c3_pipeline.sh` |

Regenerate Slurm/matrix artifacts if you still have job scripts or `gen_eval_configs` output pointing at **`gaia_c4_*`** or old **`config_gaia_c3`** as “review-only”; use **`config_gaia_c2`** / **`gaia_c2_*`** for REVIEW and **`config_gaia_c3`** / **`gaia_c3_*`** for skills.

## `workdir/` tags

Update any resume/`TAG=` variables or analysis paths:

- `gaia_c4_<...>` → **`gaia_c3_<...>`** (skill-library runs)
- If you stored **review-only** runs under `gaia_c3_<...>`, those correspond to **C2** now — prefer **`gaia_c2_<...>`** for new runs; old folders can stay on disk but **labels in papers/scripts** should say **C2**.

## What you should do

1. **Search** your notes and cluster scripts for `config_gaia_c4`, `gaia_c4`, `integration_i3_c4`, and “C4” skill-library wording; update to **C3**.
2. **Search** for old “C3” meaning *review only*; update references to **C2** and `config_gaia_c2.py`.
3. **Re-run** `scripts/gen_eval_configs.py` / matrix helpers if you keep generated configs alongside the repo.
4. **Paper/report**: align figure and table captions with **C1/C2/C3** as defined above.

## Root-level download clutter (GAIA / browser)

During GAIA, the browser tool may save PDFs (and occasionally other files) **at the repository root**. Those files are **not part of the codebase**; `.gitignore` includes `/*.pdf` and `/*[?]*` so they are not committed. Safe cleanup is `rm` for those paths or moving evaluations to a dedicated download directory if you change tool defaults. See maintainer notes in the README.
