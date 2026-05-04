"""
Skill library for condition C3 (cross-task learning).

Skills follow the agentskills.io specification:
https://agentskills.io/specification

Each skill lives in its own subdirectory of `src/skills/` with a
`SKILL.md` file containing YAML frontmatter and markdown body.

Public API:
    Skill          — a parsed skill (frontmatter + body + path)
    SkillMetadata  — just the frontmatter (cheap, always-in-memory)
    SkillRegistry  — filesystem-backed catalogue + lookup + persist

Internal modules (underscore-prefixed) — import from here, not directly:
    _model     — Skill, SkillMetadata, constants, parser
    _registry  — SkillRegistry
    _extractor — SkillExtractor (added in Phase 2.5)

The `validate` module is a CLI tool: `python -m src.skills.validate <path>`.
"""

from src.skills._model import Skill, SkillMetadata
from src.skills._registry import SkillRegistry

__all__ = [
    "Skill",
    "SkillMetadata",
    "SkillRegistry",
]
