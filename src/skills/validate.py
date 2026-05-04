"""
Standalone validator for SKILL.md files (condition C3 skill library).

Usage:
    python -m src.skills.validate src/skills
    python -m src.skills.validate src/skills/handling-file-attachments

Exits non-zero if any skill is malformed. Intended for CI and for
double-checking extractor output before committing a new seed skill.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.skills._model import (
    CANONICAL_SKILL_TYPES,
    SKILL_DESCRIPTION_MAX_LEN,
    Skill,
    VALID_CONSUMERS_RESERVED,
    VALID_SOURCES,
)


#: Per Anthropic's skill authoring guidance — bodies over this threshold are
#: harder for LLMs to use efficiently. We warn rather than fail, because
#: research skills may legitimately be longer.
BODY_LINES_WARNING_THRESHOLD: int = 500


def validate_skill(skill_dir: Path) -> tuple[bool, list[str]]:
    """
    Validate a single skill directory.

    Returns:
        (ok, messages) — `ok` is False if any error was found; `messages`
        always contains human-readable lines (errors first, warnings after).
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        skill = Skill.from_skill_md(skill_dir)
    except (ValueError, FileNotFoundError) as e:
        return False, [f"ERROR: {e}"]

    # Canonical taxonomy warnings (non-fatal: researchers may add categories)
    if skill.metadata.skill_type not in CANONICAL_SKILL_TYPES:
        warnings.append(
            f"WARNING: skill_type={skill.metadata.skill_type!r} is not in the "
            f"canonical taxonomy {sorted(CANONICAL_SKILL_TYPES)}"
        )

    if skill.metadata.source not in VALID_SOURCES:
        errors.append(
            f"ERROR: source={skill.metadata.source!r} must be one of "
            f"{sorted(VALID_SOURCES)}"
        )

    # Body-length warning
    body_lines = skill.body.count("\n") + 1
    if body_lines > BODY_LINES_WARNING_THRESHOLD:
        warnings.append(
            f"WARNING: SKILL.md body is {body_lines} lines; Anthropic's "
            f"authoring guidance recommends keeping bodies under "
            f"{BODY_LINES_WARNING_THRESHOLD} lines for optimal LLM usage"
        )

    # Description is length-checked by SkillMetadata validation, but we also
    # flag descriptions that are too terse to be useful for activation.
    if len(skill.metadata.description) < 40:
        warnings.append(
            f"WARNING: description is {len(skill.metadata.description)} chars; "
            f"consider lengthening so the LLM can tell when to activate "
            f"(target: >= 40 chars, <= {SKILL_DESCRIPTION_MAX_LEN} chars)"
        )

    # Body should actually contain guidance, not just a stub.
    if len(skill.body.strip()) < 50:
        warnings.append(
            "WARNING: SKILL.md body is very short (<50 chars); "
            "may not provide enough actionable workflow for the consumer"
        )

    return len(errors) == 0, errors + warnings


def validate_directory(root: Path) -> bool:
    """
    Validate every skill directory under `root`. Returns True iff all skills
    validated without errors (warnings are reported but do not fail).
    """
    all_ok = True
    n_checked = 0

    # Treat `root` itself as a skill dir if it contains a SKILL.md.
    if (root / "SKILL.md").exists():
        ok, messages = validate_skill(root)
        _print_result(root, ok, messages)
        return ok

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "SKILL.md").exists():
            continue
        ok, messages = validate_skill(entry)
        _print_result(entry, ok, messages)
        all_ok = all_ok and ok
        n_checked += 1

    print(f"\nChecked {n_checked} skill(s). {'All valid.' if all_ok else 'Errors found.'}")
    return all_ok


def _print_result(skill_dir: Path, ok: bool, messages: list[str]) -> None:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {skill_dir}")
    for m in messages:
        print(f"    {m}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate SKILL.md files against the agentskills.io spec and DRMA conventions."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a skills root directory OR a single skill directory.",
    )
    args = parser.parse_args()
    if not args.path.exists():
        print(f"ERROR: path does not exist: {args.path}", file=sys.stderr)
        return 2
    ok = validate_directory(args.path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
