"""
Tests for the skill registry (condition C4).

Covers:
- Skill.from_skill_md parsing (valid + malformed)
- SkillMetadata frontmatter round-trip
- SkillRegistry scan, metadata_for filtering by consumer, load_body,
  render_registry_block, add (with disk persistence), increment_verified_uses
- Malformed SKILL.md files are skipped (not fatal to the registry)
- Reserved package files (__init__.py, _model.py, etc.) are not treated as skills
"""

import sys
from pathlib import Path

import pytest

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.skills._model import Skill, SkillMetadata  # noqa: E402
from src.skills._registry import SkillRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_skill(
    parent: Path,
    name: str,
    description: str,
    *,
    consumer: str = "planner",
    skill_type: str = "delegation_pattern",
    source: str = "seeded",
    verified_uses: int = 0,
    body: str = "# Body\n\nSome content.\n",
) -> Path:
    """Create a skill directory with a SKILL.md. Returns the directory path."""
    skill_dir = parent / name
    skill_dir.mkdir()
    frontmatter = (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"metadata:\n"
        f"  consumer: {consumer}\n"
        f"  skill_type: {skill_type}\n"
        f"  source: {source}\n"
        f"  verified_uses: {verified_uses}\n"
        f"---\n\n"
    )
    (skill_dir / "SKILL.md").write_text(frontmatter + body)
    return skill_dir


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a skills directory with three sample skills of different consumers."""
    root = tmp_path / "skills"
    root.mkdir()

    _write_skill(
        root,
        name="handling-file-attachments",
        description="Workflow for tasks with local file attachments. Use when task references a file path.",
        consumer="planner",
        skill_type="delegation_pattern",
    )
    _write_skill(
        root,
        name="pdf-table-extraction",
        description="Extract tables from PDFs using pdfplumber then reason over CSV.",
        consumer="deep_analyzer_agent",
        skill_type="tool_usage",
    )
    _write_skill(
        root,
        name="universal-verification",
        description="Verify outputs before finalizing. Applies to all agents.",
        consumer="all",
        skill_type="verification_pattern",
    )
    return root


# ---------------------------------------------------------------------------
# Skill parsing
# ---------------------------------------------------------------------------

class TestSkillFromSkillMd:
    def test_valid_skill(self, tmp_path: Path):
        skill_dir = _write_skill(
            tmp_path,
            name="my-test-skill",
            description="Valid skill for parse test.",
        )
        skill = Skill.from_skill_md(skill_dir)
        assert skill.metadata.name == "my-test-skill"
        assert skill.metadata.description == "Valid skill for parse test."
        assert skill.metadata.consumer == "planner"
        assert skill.metadata.verified_uses == 0
        assert "# Body" in skill.body

    def test_missing_skill_md(self, tmp_path: Path):
        skill_dir = tmp_path / "no-skill-md"
        skill_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            Skill.from_skill_md(skill_dir)

    def test_name_dir_mismatch_rejected(self, tmp_path: Path):
        skill_dir = tmp_path / "correct-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: wrong-name\ndescription: test\n---\nbody\n"
        )
        with pytest.raises(ValueError, match="does not match parent directory"):
            Skill.from_skill_md(skill_dir)

    def test_uppercase_name_rejected(self, tmp_path: Path):
        skill_dir = tmp_path / "BadName"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: BadName\ndescription: test\n---\nbody\n"
        )
        with pytest.raises(ValueError, match="must match"):
            Skill.from_skill_md(skill_dir)

    def test_consecutive_hyphens_rejected(self, tmp_path: Path):
        skill_dir = tmp_path / "a--b"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: a--b\ndescription: test\n---\nbody\n"
        )
        with pytest.raises(ValueError, match="must match"):
            Skill.from_skill_md(skill_dir)

    def test_missing_description_rejected(self, tmp_path: Path):
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\nbody\n")
        with pytest.raises(ValueError, match="description"):
            Skill.from_skill_md(skill_dir)

    def test_description_too_long_rejected(self, tmp_path: Path):
        skill_dir = tmp_path / "long-desc"
        skill_dir.mkdir()
        long_desc = "x" * 1025  # 1 over the 1024 limit
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: long-desc\ndescription: {long_desc}\n---\nbody\n"
        )
        with pytest.raises(ValueError, match="exceeds"):
            Skill.from_skill_md(skill_dir)

    def test_malformed_yaml_rejected(self, tmp_path: Path):
        skill_dir = tmp_path / "bad-yaml"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: bad-yaml\n  - this: is: bad:::\n---\nbody\n"
        )
        with pytest.raises(ValueError):
            Skill.from_skill_md(skill_dir)

    def test_no_frontmatter_rejected(self, tmp_path: Path):
        skill_dir = tmp_path / "no-fm"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Just a body, no frontmatter.\n")
        with pytest.raises(ValueError):
            Skill.from_skill_md(skill_dir)


# ---------------------------------------------------------------------------
# SkillMetadata round-trip
# ---------------------------------------------------------------------------

class TestSkillMetadataYaml:
    def test_frontmatter_round_trip(self):
        import yaml
        meta = SkillMetadata(
            name="example",
            description="A round-trip test.",
            consumer="planner",
            skill_type="delegation_pattern",
            source="seeded",
            verified_uses=5,
            confidence=0.8,
            created_at="2026-04-17T10:00:00Z",
        )
        fm = meta.to_yaml_frontmatter()
        parsed = yaml.safe_load(fm)
        assert parsed["name"] == "example"
        assert parsed["description"] == "A round-trip test."
        assert parsed["metadata"]["verified_uses"] == 5
        assert parsed["metadata"]["confidence"] == 0.8
        assert parsed["metadata"]["created_at"] == "2026-04-17T10:00:00Z"

    def test_frontmatter_omits_optional_none(self):
        import yaml
        meta = SkillMetadata(
            name="example",
            description="Minimal.",
        )
        fm = meta.to_yaml_frontmatter()
        parsed = yaml.safe_load(fm)
        assert "created_at" not in parsed["metadata"]
        assert "learned_from_task_type" not in parsed["metadata"]


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class TestSkillRegistryScan:
    def test_scan_loads_all_skills(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        assert set(registry.names()) == {
            "handling-file-attachments",
            "pdf-table-extraction",
            "universal-verification",
        }

    def test_missing_skills_dir_is_warning_not_error(self, tmp_path: Path):
        missing = tmp_path / "does-not-exist"
        registry = SkillRegistry(missing)
        assert registry.names() == []

    def test_malformed_skill_skipped(self, skills_dir: Path):
        # Add one malformed skill — registry should load the others.
        bad = skills_dir / "bad-skill"
        bad.mkdir()
        (bad / "SKILL.md").write_text(
            "---\nname: bad-skill\n---\nno description\n"
        )
        registry = SkillRegistry(skills_dir)
        # The malformed one is skipped; the three valid ones remain.
        assert set(registry.names()) == {
            "handling-file-attachments",
            "pdf-table-extraction",
            "universal-verification",
        }

    def test_directory_without_skill_md_ignored(self, skills_dir: Path):
        (skills_dir / "not-a-skill").mkdir()
        # Contains no SKILL.md — should be silently ignored
        registry = SkillRegistry(skills_dir)
        assert "not-a-skill" not in registry.names()


class TestMetadataFor:
    def test_planner_view(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        planner_view = {m.name for m in registry.metadata_for("planner")}
        # Planner sees planner-scoped + all-scoped
        assert planner_view == {"handling-file-attachments", "universal-verification"}

    def test_subagent_view(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        analyzer_view = {m.name for m in registry.metadata_for("deep_analyzer_agent")}
        assert analyzer_view == {"pdf-table-extraction", "universal-verification"}

    def test_unrelated_consumer_only_sees_all_scope(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        view = {m.name for m in registry.metadata_for("browser_use_agent")}
        assert view == {"universal-verification"}

    def test_sorted_order(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        names = [m.name for m in registry.metadata_for("planner")]
        assert names == sorted(names)


class TestLoadBody:
    def test_load_existing(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        body = registry.load_body("handling-file-attachments")
        assert "# Body" in body

    def test_load_missing_raises_key_error(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        with pytest.raises(KeyError, match="no-such-skill"):
            registry.load_body("no-such-skill")


class TestRenderRegistryBlock:
    def test_block_lists_visible_skills(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        block = registry.render_registry_block("planner")
        assert "handling-file-attachments" in block
        assert "universal-verification" in block
        # Non-planner skill NOT in planner's view
        assert "pdf-table-extraction" not in block

    def test_block_empty_for_unseen_consumer(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        registry = SkillRegistry(empty_dir)
        block = registry.render_registry_block("planner")
        assert block == ""

    def test_block_truncates_long_description(self, tmp_path: Path):
        _write_skill(
            tmp_path,
            name="long-desc-skill",
            description="x" * 500,
            consumer="planner",
        )
        registry = SkillRegistry(tmp_path)
        block = registry.render_registry_block("planner")
        # Long descriptions get truncated to REGISTRY_BLOCK_DESCRIPTION_CAP chars
        # (currently 300) followed by "...".
        assert "..." in block


class TestAdd:
    def test_add_new_skill_persists_to_disk(self, tmp_path: Path):
        registry = SkillRegistry(tmp_path)
        new_skill = Skill(
            path=tmp_path / "new-skill",
            metadata=SkillMetadata(
                name="new-skill",
                description="Newly extracted skill.",
                source="success",
            ),
            body="# New Skill\n\nWorkflow steps.\n",
        )
        registry.add(new_skill)
        assert (tmp_path / "new-skill" / "SKILL.md").exists()
        # Re-scan confirms persistence
        fresh = SkillRegistry(tmp_path)
        assert "new-skill" in fresh.names()

    def test_add_conflicting_name_raises(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        conflicting = Skill(
            path=skills_dir / "different-path",  # different dir!
            metadata=SkillMetadata(
                name="handling-file-attachments",  # but same name as existing
                description="Conflict.",
            ),
            body="body\n",
        )
        with pytest.raises(ValueError, match="conflict"):
            registry.add(conflicting)


class TestIncrementVerifiedUses:
    def test_increments_and_persists(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        initial = next(
            m for m in registry.metadata_for("planner")
            if m.name == "handling-file-attachments"
        )
        assert initial.verified_uses == 0

        registry.increment_verified_uses("handling-file-attachments")
        after = next(
            m for m in registry.metadata_for("planner")
            if m.name == "handling-file-attachments"
        )
        assert after.verified_uses == 1

        # Persisted — re-scan sees the new value
        fresh = SkillRegistry(skills_dir)
        persisted = next(
            m for m in fresh.metadata_for("planner")
            if m.name == "handling-file-attachments"
        )
        assert persisted.verified_uses == 1

    def test_unknown_skill_is_warning_not_error(self, skills_dir: Path):
        registry = SkillRegistry(skills_dir)
        # Does not raise — just logs a warning.
        registry.increment_verified_uses("no-such-skill")
