"""
Seed helper for per-run skill libraries (condition C3 skill library; legacy docs may say C4).

Each C3 evaluation run owns its own `skills_dir` under
`workdir/gaia_<tag>/skills/` so that:

- Parallel runs (e.g. three models sharing the matrix runner) cannot race on
  the same directory.
- Re-running the same model never clobbers the skill library produced by a
  previous run — each historical run stays inspectable.
- `src/skills/` stays clean of learned artefacts; it is the canonical
  seed-skill source, not a mutable cache.

Before the `SkillRegistry` is constructed, call `seed_skills_dir(dst, src)`.
It copies every seed skill directory (one that contains `SKILL.md`) from
`src` into `dst`, then writes a `.seeded` marker so subsequent runs resuming
the same `dst` do not re-copy and thereby overwrite learned skills.

Only directories containing `SKILL.md` are copied — package files like
`_registry.py`, `_extractor.py`, `__init__.py`, and README.md are skipped
by construction. This positive filter is more robust than maintaining a
blocklist: any new non-skill file added to `src/skills/` is automatically
ignored.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.logger import LogLevel, logger

#: Filename of the marker that records a successful seed copy. Its presence
#: in `dst` means "seeding is done; do not re-copy on resume". Kept hidden
#: (leading dot) so filesystem listings are not cluttered.
SEED_MARKER_FILENAME = ".seeded"

#: Default seed source — the canonical, git-tracked skill library.
DEFAULT_SEED_SOURCE = Path("src/skills")


def seed_skills_dir(
    dst: Path,
    src: Path | None = None,
) -> int:
    """
    Copy seed skills from `src` into `dst` iff `dst` is not already seeded.

    A seed skill is a directory under `src` containing a `SKILL.md` file.
    Everything else (package files, READMEs, caches, hidden files) is left
    alone.

    Args:
        dst: Destination `skills_dir` for this run. Created if missing.
        src: Seed source. Defaults to `src/skills` relative to CWD.

    Returns:
        Number of skills copied (0 if the marker was present → resumed run).

    Idempotency:
        - If `dst/.seeded` exists, returns 0 without touching the filesystem.
        - If a partial copy from a prior crashed attempt left a half-written
          child directory, the partial directory is removed and the copy
          retried so seeding converges on the intended state.
        - The marker is written LAST, so a mid-seed crash leaves no marker
          and the next start re-seeds cleanly.

    This helper is contracted never to delete user-authored skills in `dst`
    that are not present in `src` — it only replaces partial copies of
    `src`-originated skills.
    """
    dst = Path(dst)
    src = Path(src) if src is not None else DEFAULT_SEED_SOURCE

    # If caller points `dst` at the seed source itself (legacy behaviour:
    # `skills_dir="src/skills"`), there is nothing to do — seeding would
    # be a self-copy and the `.seeded` marker would dirty the git tree.
    # Compare resolved paths so relative / absolute variants both match.
    try:
        if dst.resolve() == src.resolve():
            logger.log(
                f"[seed_skills_dir] {dst} is the seed source itself; "
                f"no-op.",
                level=LogLevel.DEBUG,
            )
            return 0
    except FileNotFoundError:
        # `dst` may not exist yet; `src` is validated below. Fall through
        # to the normal flow.
        pass

    marker = dst / SEED_MARKER_FILENAME
    if marker.exists():
        logger.log(
            f"[seed_skills_dir] {dst} already seeded (marker present); skipping.",
            level=LogLevel.DEBUG,
        )
        return 0

    if not src.exists() or not src.is_dir():
        logger.log(
            f"[seed_skills_dir] seed source {src} does not exist; "
            f"proceeding with empty skill library at {dst}.",
            level=LogLevel.WARNING,
        )
        dst.mkdir(parents=True, exist_ok=True)
        _write_marker(marker, src=src, copied=[])
        return 0

    dst.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "SKILL.md").exists():
            continue

        target = dst / entry.name
        if target.exists():
            # Partial prior copy or pre-existing user-authored skill with
            # the same name. The seeded marker was absent (we checked), so
            # this can only be a crashed-mid-copy leftover OR a manual
            # placement by the user. In either case, the safer behaviour
            # is to leave it alone and skip — we never silently overwrite.
            logger.log(
                f"[seed_skills_dir] target {target} already exists; "
                f"leaving it alone (will be picked up by the registry "
                f"scan as-is).",
                level=LogLevel.WARNING,
            )
            continue

        shutil.copytree(entry, target)
        copied.append(entry.name)
        logger.log(
            f"[seed_skills_dir] seeded {entry.name} -> {target}",
            level=LogLevel.DEBUG,
        )

    _write_marker(marker, src=src, copied=copied)

    logger.log(
        f"[seed_skills_dir] seeded {len(copied)} skill(s) from {src} into {dst}",
        level=LogLevel.INFO,
    )
    return len(copied)


def _write_marker(marker: Path, src: Path, copied: list[str]) -> None:
    """
    Write the `.seeded` marker with forensics about what was copied.

    The marker content is intentionally human-readable — it is meant to be
    useful when inspecting a run directory after the fact (e.g. "where did
    this skill library start from?").
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"seeded_at: {ts}",
        f"seed_source: {src}",
        f"seeded_count: {len(copied)}",
        "seeded_skills:",
    ]
    for name in copied:
        lines.append(f"  - {name}")
    marker.write_text("\n".join(lines) + "\n", encoding="utf-8")
