# `src/skills/` — Skill library (condition C4)

This directory serves two purposes:

1. **Python package** for the skill infrastructure (`_model.py`, `_registry.py`, `_extractor.py`, `validate.py`, `__init__.py`). These are internal modules; prefer `from src.skills import Skill, SkillMetadata, SkillRegistry`.

2. **On-disk catalogue of individual skills**. Each skill is a subdirectory (one per skill) containing a `SKILL.md` file that follows the [agentskills.io specification](https://agentskills.io/specification):

```
src/skills/
├── __init__.py               # package
├── _model.py                 # package
├── _registry.py              # package
├── _extractor.py             # package
├── validate.py               # package
├── README.md                 # this file
│
├── handling-file-attachments/
│   └── SKILL.md              # skill
├── pdf-table-extraction/
│   └── SKILL.md              # skill
└── …                         # (7 seed skills total)
```

The registry (`SkillRegistry`) is careful to skip reserved file names (`__init__.py`, `_*.py`, etc.) when scanning, so the coexistence does not cause false positives.

## Skill format

Every `SKILL.md` has:

- **YAML frontmatter** with required `name` (lowercase-hyphen, ≤64 chars) and `description` (≤1024 chars), plus optional `metadata` block containing DRMA-specific fields: `consumer`, `skill_type`, `source`, `verified_uses`, `confidence`, `created_at`, `learned_from_task_type`.
- **Markdown body** with recommended sections: `## When to activate`, `## Workflow`, `## Avoid`.

See any seed file for a complete example.

## Consumer routing

Each skill declares a `consumer` in its metadata that decides which agent(s) see it in their `<skill-registry>` system-prompt block:

| `consumer` value | Who sees this skill |
|------------------|---------------------|
| `planner` | The top-level `AdaptivePlanningAgent` |
| `deep_analyzer_agent` | Only when the deep analyzer is running a delegated task |
| `browser_use_agent` | Only the browser sub-agent |
| `deep_researcher_agent` | Only the deep researcher sub-agent |
| `all` | Every agent (use sparingly; applies when the skill is truly general) |

## Validation

Validate every SKILL.md in this directory:

```bash
python -m src.skills.validate src/skills
```

The validator checks agentskills.io compliance (name regex, description length, name-equals-directory-name) and DRMA conventions (canonical `skill_type` values, body length threshold, description minimum length).

## Seed vs. learned skills

The 7 SKILL.md files shipped in this repo are **seeded** — a **small curated starter corpus** for condition C4, drafted with model assistance then **edited against the real tool and interpreter surfaces** in this repo (agent tool lists, `python_interpreter_tool` import allowlist, GAIA defaults). They all have `metadata.source: seeded`. They are not guaranteed to match every custom deployment; extend or replace seeds when your config adds tools (e.g. `archive_searcher_tool`) or extra `authorized_imports`.

This directory is the **canonical seed source**. It is read-only at run time — C4 runs never write back here. Instead, on first construction each C4 invocation copies every SKILL.md-bearing subdirectory of `src/skills/` into its own per-run `skills_dir=workdir/gaia_c4_<model>_<run_id>/skills/` (see `_seed.py`), writes a `.seeded` marker last, and subsequent `SkillExtractor` writes land in that per-run dir. This guarantees:

- Cross-model runs in the 3×4 matrix cannot contaminate each other — each `(model, run_id)` cell has a disjoint skill library.
- Parallel runs of the matrix never race on a shared seed dir.
- Every historical C4 run remains inspectable after the fact: the `skills/` that produced a given `dra.jsonl` lives next to it under the same timestamped directory.

Newly-extracted skills carry `metadata.source: success` or `metadata.source: failure` depending on whether the trajectory succeeded or revealed an actionable failure mode. Because they land under `workdir/` (git-ignored), learned skills never dirty the tracked tree; diff two runs with `diff -r workdir/gaia_c4_<model>_<A>/skills/ workdir/gaia_c4_<model>_<B>/skills/`.

For frozen-library evaluation (measure the contribution of a pre-trained corpus without online learning), set `enable_skill_extraction=False` in the config.

## Sealing

The `SkillRegistry` and `SkillExtractor` live on the planning agent as instance attributes, not inside `managed_agents` or `tools`. This means `modify_subagent`'s `_find_managed_agent` cannot reach them — the planner cannot modify its own learning apparatus during a task. The `ActivateSkillTool` given to each agent is read-only over the registry (returns skill bodies; never calls `add()` or `increment_verified_uses()`).

See `CLAUDE.md` in the repo root for the full experimental-conditions reference.
