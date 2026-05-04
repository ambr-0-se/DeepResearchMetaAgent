"""
SkillRegistry — filesystem-backed skill catalogue (condition C3 skill library).

Scans a skills directory at startup, filters by consumer for each prompt
injection point, and persists metadata updates (verified_uses counter) back
to disk so telemetry survives across task runs.

Design notes:

- The registry treats the skills directory as the source of truth. In-memory
  state is a cache rebuilt at startup; any write (skill added by extractor,
  verified_uses incremented after successful activation) is also persisted.
- Skill bodies are NOT pruned from memory after parse — the corpus is small
  enough (tens to low hundreds) that keeping everything resident is cheaper
  than repeated disk I/O. If the library grows past ~1000 skills, switch
  `Skill.body` to a lazy property.
- `render_registry_block` produces a compact Jinja-safe string for
  injection into system prompts. Each skill occupies one line so the
  planner's context budget stays predictable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from src.logger import LogLevel, logger
from src.skills._model import (
    Skill,
    SkillMetadata,
    SKILL_DESCRIPTION_MAX_LEN,
    SKILL_NAME_MAX_LEN,
    SKILL_NAME_PATTERN,
)


# Directories under the skills root that are NOT individual skills — they
# hold the Python package files for the registry itself. These are skipped
# during the scan so the registry doesn't try to parse its own source.
_RESERVED_SUBDIRECTORY_NAMES: frozenset[str] = frozenset({
    "__pycache__",
})

# Files under the skills root that are also reserved (the package files).
_RESERVED_FILE_NAMES: frozenset[str] = frozenset({
    "__init__.py",
    "_model.py",
    "_registry.py",
    "_extractor.py",
    "validate.py",
})


class SkillRegistry:
    """
    Scan + lookup + persist for a skills directory.

    Typical lifecycle:
        registry = SkillRegistry(Path("src/skills"))
        # planner init:
        planner_meta = registry.metadata_for("planner")
        # on sub-agent init:
        sub_meta = registry.metadata_for("deep_analyzer_agent")
        # on activate_skill tool call:
        body = registry.load_body("handling-file-attachments")
        # on successful use:
        registry.increment_verified_uses("handling-file-attachments")
        # on extraction of a new skill:
        registry.add(new_skill)
    """

    #: Truncation cap for descriptions shown in the registry block. The raw
    #: description in metadata can be up to 1024 chars (spec); in the
    #: injected block we keep it shorter to preserve the planner's context.
    REGISTRY_BLOCK_DESCRIPTION_CAP: int = 300

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}
        if self.skills_dir.exists():
            self._scan()
        else:
            logger.log(
                f"[SkillRegistry] skills_dir={self.skills_dir} does not exist; "
                f"registry starts empty",
                level=LogLevel.WARNING,
            )

    # -- scan ----------------------------------------------------------------

    def _scan(self) -> None:
        """
        Populate `self._skills` by scanning every subdirectory of skills_dir.

        A directory is treated as a skill iff it contains a SKILL.md file.
        Malformed SKILL.md files are logged and skipped — one bad skill
        must not prevent the registry from loading the rest.
        """
        for entry in sorted(self.skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name in _RESERVED_SUBDIRECTORY_NAMES:
                continue
            if not (entry / "SKILL.md").exists():
                continue
            try:
                skill = Skill.from_skill_md(entry)
            except (ValueError, FileNotFoundError) as e:
                logger.log(
                    f"[SkillRegistry] skipping malformed skill at {entry}: {e}",
                    level=LogLevel.WARNING,
                )
                continue
            self._skills[skill.metadata.name] = skill

        logger.log(
            f"[SkillRegistry] loaded {len(self._skills)} skills from {self.skills_dir}",
            level=LogLevel.INFO,
        )

    def reload(self) -> None:
        """Re-scan the filesystem. Useful after external edits."""
        self._skills.clear()
        self._scan()

    # -- read API ------------------------------------------------------------

    def names(self) -> list[str]:
        """All skill names, in sorted order."""
        return sorted(self._skills.keys())

    def get(self, skill_name: str) -> Optional[Skill]:
        """Return a Skill by name, or None if absent."""
        return self._skills.get(skill_name)

    def metadata_for(self, consumer: str) -> list[SkillMetadata]:
        """
        Return metadata for skills visible to `consumer`, sorted by name.

        A skill is visible when its `metadata.consumer` equals `consumer`
        OR is the sentinel "all". The planner uses consumer="planner"; each
        sub-agent uses consumer=<its own name>.
        """
        visible: list[SkillMetadata] = []
        for skill in self._skills.values():
            if skill.metadata.consumer in (consumer, "all"):
                visible.append(skill.metadata)
        visible.sort(key=lambda m: m.name)
        return visible

    def load_body(self, skill_name: str) -> str:
        """
        Return the full SKILL.md body for activation.

        Raises:
            KeyError — skill not found.
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            raise KeyError(f"No skill named {skill_name!r}")
        return skill.body

    def render_registry_block(self, consumer: str) -> str:
        """
        Render the always-loaded registry block for a given consumer.

        Returns an empty string when no skills are visible — caller
        templates should `{% if skill_registry_block %}` to omit the block
        entirely rather than rendering a header with no entries.

        Format is one-skill-per-line, capped to
        `REGISTRY_BLOCK_DESCRIPTION_CAP` chars per description so the
        injection cost stays bounded even with 100+ skills.
        """
        visible = self.metadata_for(consumer)
        if not visible:
            return ""

        lines = [
            f"<skill-registry consumer={consumer!r}>",
            "The following skills are available. Call `activate_skill` with a "
            "skill name to load its full workflow into context.",
            "",
        ]
        for meta in visible:
            desc = meta.description
            if len(desc) > self.REGISTRY_BLOCK_DESCRIPTION_CAP:
                desc = desc[: self.REGISTRY_BLOCK_DESCRIPTION_CAP].rstrip() + "..."
            lines.append(f"* {meta.name} ({meta.skill_type}): {desc}")
        lines.append("</skill-registry>")
        return "\n".join(lines)

    # -- write API -----------------------------------------------------------

    def add(self, skill: Skill) -> None:
        """
        Persist a newly-extracted (or seed-authored) skill to disk and to
        the in-memory cache.

        The caller must have already validated the Skill (Skill.from_skill_md
        does this on parse; direct construction is fine for extractor-built
        skills as long as the name matches `skill.path.name`).

        Behaviour:
            - Creates `skill.path` (the directory) if absent.
            - Writes `skill.path / "SKILL.md"` atomically (write + rename).
            - Updates `self._skills[skill.metadata.name]`.

        Raises:
            ValueError — name conflict with an existing skill.
        """
        name = skill.metadata.name
        if name in self._skills and self._skills[name].path != skill.path:
            raise ValueError(
                f"Skill name conflict: {name!r} already registered at "
                f"{self._skills[name].path}"
            )

        skill.path.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            skill.path / "SKILL.md",
            _serialise_skill(skill),
        )
        self._skills[name] = skill
        logger.log(
            f"[SkillRegistry] added skill {name!r} at {skill.path}",
            level=LogLevel.INFO,
        )

    def increment_verified_uses(self, skill_name: str) -> None:
        """
        Bump `verified_uses` and persist to disk.

        Called after a skill activation that is followed by a successful
        task outcome (attribution heuristic — see SkillExtractor docstring).
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            logger.log(
                f"[SkillRegistry] increment_verified_uses: no skill {skill_name!r}",
                level=LogLevel.WARNING,
            )
            return
        new_meta = SkillMetadata(
            name=skill.metadata.name,
            description=skill.metadata.description,
            consumer=skill.metadata.consumer,
            skill_type=skill.metadata.skill_type,
            source=skill.metadata.source,
            verified_uses=skill.metadata.verified_uses + 1,
            confidence=skill.metadata.confidence,
            created_at=skill.metadata.created_at,
            learned_from_task_type=skill.metadata.learned_from_task_type,
        )
        new_skill = Skill(path=skill.path, metadata=new_meta, body=skill.body)
        _atomic_write(
            skill.path / "SKILL.md",
            _serialise_skill(new_skill),
        )
        self._skills[skill_name] = new_skill


# --- Helpers ----------------------------------------------------------------

def _serialise_skill(skill: Skill) -> str:
    """Compose a SKILL.md string from a Skill instance."""
    frontmatter = skill.metadata.to_yaml_frontmatter()
    body = skill.body
    # Ensure the body ends with exactly one trailing newline.
    if not body.endswith("\n"):
        body = body + "\n"
    return f"---\n{frontmatter}\n---\n\n{body}"


def _atomic_write(dst: Path, text: str) -> None:
    """
    Write `text` to `dst` atomically (write to temp + rename).

    This prevents readers from ever seeing a half-written SKILL.md if the
    process is interrupted mid-write.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dst)
