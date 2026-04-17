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

The 7 SKILL.md files shipped in this repo are **seeded** — hand-authored from the literature review and committed as part of condition C4's starting corpus. They all have `metadata.source: seeded`.

New skills **learned** during C4 training runs are written to this same directory by `SkillExtractor` at task end. They carry `metadata.source: success` or `metadata.source: failure` depending on whether the trajectory succeeded or revealed an actionable failure mode. Learned skills are not pre-committed — they accumulate on the branch used for the training run.

For frozen-library evaluation (measure the contribution of a pre-trained corpus without online learning), set `enable_skill_extraction=False` in the config.

## Sealing

The `SkillRegistry` and `SkillExtractor` live on the planning agent as instance attributes, not inside `managed_agents` or `tools`. This means `modify_subagent`'s `_find_managed_agent` cannot reach them — the planner cannot modify its own learning apparatus during a task. The `ActivateSkillTool` given to each agent is read-only over the registry (returns skill bodies; never calls `add()` or `increment_verified_uses()`).

See `CLAUDE.md` in the repo root for the full experimental-conditions reference.
