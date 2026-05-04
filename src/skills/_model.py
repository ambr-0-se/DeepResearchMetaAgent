"""
Data model for skills (condition C3 skill library).

A "skill" is a directory containing a `SKILL.md` file whose structure follows
the agentskills.io specification (https://agentskills.io/specification):

    <skill-name>/
        SKILL.md          # required: YAML frontmatter + markdown body
        scripts/          # optional: executable helpers (not used by DRMA yet)
        references/       # optional: additional documentation
        assets/           # optional: templates, images, etc.

This module defines:

- `SkillMetadata` — the parsed frontmatter (always kept in memory for routing
  and rendering the registry block).
- `Skill` — frontmatter + body + on-disk path. The body is loaded lazily so
  the registry scan stays cheap even with hundreds of skills.

Skills are consumed at three points in DRMA:

1. Registry block injected into an agent's system prompt (planner or
   sub-agent) lists the `name` + `description` of all visible skills — this
   is the "metadata always loaded" level of progressive disclosure.
2. `activate_skill` tool returns the full SKILL.md body on demand — this is
   the "full body loaded when activated" level.
3. `SkillExtractor` creates new SKILL.md files at task end (C3 learning loop).

All writes go through `SkillRegistry.add` / `persist_metadata`; nothing in
this module mutates disk directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# --- Constants --------------------------------------------------------------

#: agentskills.io spec: name = lowercase alphanumerics + hyphens, 1-64 chars,
#: no leading/trailing hyphens, no consecutive hyphens.
SKILL_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"^[a-z0-9]+(-[a-z0-9]+)*$"
)
SKILL_NAME_MAX_LEN: int = 64
SKILL_DESCRIPTION_MAX_LEN: int = 1024

#: Scope routing (DRMA-specific, stored under `metadata.consumer`):
#: "planner"       → injected into AdaptivePlanningAgent's system prompt
#: "<agent name>"  → injected into that specific managed agent's task template
#: "all"           → injected into both (rare)
VALID_CONSUMERS_RESERVED: frozenset[str] = frozenset({"planner", "all"})

#: Skill source, stored under `metadata.source`:
#: "seeded"   — pre-committed corpus skill (curated / human-verified for this repo)
#: "success"  — extracted from a successful task trajectory
#: "failure"  — extracted from a failed task (failure-avoidance skill)
VALID_SOURCES: frozenset[str] = frozenset({"seeded", "success", "failure"})

#: Skill type taxonomy, stored under `metadata.skill_type`. Not exhaustive —
#: registries accept arbitrary strings so researchers can extend without
#: code changes, but these are the canonical values used by pre-seeded skills
#: and the extractor.
CANONICAL_SKILL_TYPES: frozenset[str] = frozenset({
    "delegation_pattern",
    "task_decomposition",
    "failure_avoidance",
    "modification_pattern",
    "verification_pattern",
    "tool_usage",
    "domain_workflow",
})


# --- SkillMetadata ----------------------------------------------------------

@dataclass(frozen=True)
class SkillMetadata:
    """
    Parsed YAML frontmatter of a SKILL.md file.

    agentskills.io-required fields:
        name         — lowercase-hyphen, ≤64 chars, matches directory name
        description  — 1-1024 chars, shown in the registry block and used for
                       activation decisions

    DRMA-specific fields (under the spec's optional `metadata` block):
        consumer              — routing scope (see VALID_CONSUMERS_RESERVED)
        skill_type            — taxonomy (see CANONICAL_SKILL_TYPES)
        source                — "seeded" / "success" / "failure"
        verified_uses         — cumulative count of activations that led to
                                task success; used for telemetry and pruning
        confidence            — reviewer/extractor's prior confidence in the
                                skill; used when multiple skills match
        created_at            — ISO8601 timestamp (informational)
        learned_from_task_type — optional provenance tag

    This is the cheap, always-in-memory view. The full SKILL.md body lives
    on disk and is loaded only when a skill is activated.
    """
    name: str
    description: str
    consumer: str = "planner"
    skill_type: str = "delegation_pattern"
    source: str = "seeded"
    verified_uses: int = 0
    confidence: float = 0.5
    created_at: Optional[str] = None
    learned_from_task_type: Optional[str] = None

    def to_yaml_frontmatter(self) -> str:
        """
        Serialise back to YAML frontmatter (for SkillRegistry.persist_metadata
        when telemetry like verified_uses needs to be rewritten).
        """
        # Order fields deterministically so diffs stay readable.
        payload = {
            "name": self.name,
            "description": self.description,
        }
        meta = {
            "consumer": self.consumer,
            "skill_type": self.skill_type,
            "source": self.source,
            "verified_uses": self.verified_uses,
            "confidence": self.confidence,
        }
        if self.created_at is not None:
            meta["created_at"] = self.created_at
        if self.learned_from_task_type is not None:
            meta["learned_from_task_type"] = self.learned_from_task_type
        payload["metadata"] = meta
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip()


# --- Skill ------------------------------------------------------------------

@dataclass
class Skill:
    """
    A skill directory parsed into memory.

    Construct via `Skill.from_skill_md(directory)`; do not instantiate
    directly. The body is intentionally loaded eagerly from disk on first
    construction but CAN be cleared and re-read by the registry if the file
    changes mid-run (not needed for the current workflow).
    """
    path: Path              # directory containing SKILL.md
    metadata: SkillMetadata
    body: str               # markdown body (everything after the frontmatter)

    @classmethod
    def from_skill_md(cls, skill_dir: Path) -> "Skill":
        """
        Parse a skill directory into a Skill instance.

        Raises:
            ValueError  — malformed frontmatter, bad name/description, or
                          name mismatch with directory name.
            FileNotFoundError — no SKILL.md in the directory.
        """
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            raise FileNotFoundError(f"No SKILL.md in {skill_dir}")

        text = skill_md_path.read_text(encoding="utf-8")
        frontmatter_text, body = _split_frontmatter(text)

        try:
            raw = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Malformed YAML frontmatter in {skill_md_path}: {e}") from e

        if not isinstance(raw, dict):
            raise ValueError(f"Frontmatter in {skill_md_path} must be a mapping")

        name = raw.get("name")
        description = raw.get("description")
        metadata_block = raw.get("metadata") or {}
        if not isinstance(metadata_block, dict):
            raise ValueError(
                f"metadata block in {skill_md_path} must be a mapping, got {type(metadata_block).__name__}"
            )

        metadata = SkillMetadata(
            name=_validate_name(name, skill_dir),
            description=_validate_description(description, skill_md_path),
            consumer=str(metadata_block.get("consumer", "planner")),
            skill_type=str(metadata_block.get("skill_type", "delegation_pattern")),
            source=str(metadata_block.get("source", "seeded")),
            verified_uses=int(metadata_block.get("verified_uses", 0)),
            confidence=float(metadata_block.get("confidence", 0.5)),
            created_at=metadata_block.get("created_at"),
            learned_from_task_type=metadata_block.get("learned_from_task_type"),
        )

        return cls(path=skill_dir, metadata=metadata, body=body)


# --- Helpers ----------------------------------------------------------------

def _split_frontmatter(text: str) -> tuple[str, str]:
    """
    Split a SKILL.md file into (yaml_frontmatter, markdown_body).

    Accepts the standard `---`-delimited frontmatter. Returns empty
    frontmatter if none is present, so the caller can raise a more
    specific error (missing required fields).
    """
    stripped = text.lstrip("\ufeff")  # strip BOM if present
    if not stripped.startswith("---"):
        return "", text

    # Find the closing --- on its own line.
    lines = stripped.split("\n")
    # lines[0] == "---". Search from line 1 onward.
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            frontmatter = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1:])
            # Trim exactly one leading newline from body if present
            return frontmatter, body.lstrip("\n")
    # No closing delimiter — treat entire content as body
    return "", text


def _validate_name(name: object, skill_dir: Path) -> str:
    """Validate name against the agentskills.io spec and directory alignment."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{skill_dir / 'SKILL.md'}: `name` is required and must be a non-empty string")
    if len(name) > SKILL_NAME_MAX_LEN:
        raise ValueError(
            f"{skill_dir / 'SKILL.md'}: `name` exceeds {SKILL_NAME_MAX_LEN} chars ({len(name)})"
        )
    if not SKILL_NAME_PATTERN.match(name):
        raise ValueError(
            f"{skill_dir / 'SKILL.md'}: `name`={name!r} must match {SKILL_NAME_PATTERN.pattern} "
            f"(lowercase alphanumerics and hyphens, no leading/trailing/consecutive hyphens)"
        )
    if name != skill_dir.name:
        raise ValueError(
            f"{skill_dir / 'SKILL.md'}: `name`={name!r} does not match parent directory name "
            f"{skill_dir.name!r}"
        )
    return name


def _validate_description(description: object, skill_md_path: Path) -> str:
    """Validate description length per agentskills.io spec."""
    if not isinstance(description, str) or not description:
        raise ValueError(f"{skill_md_path}: `description` is required and must be a non-empty string")
    if len(description) > SKILL_DESCRIPTION_MAX_LEN:
        raise ValueError(
            f"{skill_md_path}: `description` exceeds {SKILL_DESCRIPTION_MAX_LEN} chars ({len(description)})"
        )
    return description
