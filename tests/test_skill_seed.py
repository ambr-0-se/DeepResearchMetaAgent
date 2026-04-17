"""Unit tests for `src/skills/_seed.py` — per-run skill library seeding.

These tests avoid `src/__init__.py` side-effects (crawl4ai, huggingface_hub)
by loading `src.skills._seed` directly via importlib. The helper only needs
`src.logger`, which is likewise stubbed so the tests run in any minimal env.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEED_MODULE_PATH = _REPO_ROOT / "src" / "skills" / "_seed.py"


def _install_stub_logger() -> None:
    """Install a minimal `src.logger` stub so _seed.py imports cleanly.

    The real module pulls in heavy package side effects; the helper only
    uses `logger.log(msg, level=...)` so a no-op stub is sufficient.
    """
    if "src" not in sys.modules:
        pkg = types.ModuleType("src")
        pkg.__path__ = [str(_REPO_ROOT / "src")]
        sys.modules["src"] = pkg

    if "src.logger" in sys.modules:
        return

    logger_stub = types.ModuleType("src.logger")

    class _LogLevel:
        DEBUG = "debug"
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"

    class _Logger:
        def log(self, *_args, **_kwargs) -> None:
            return None

    logger_stub.LogLevel = _LogLevel
    logger_stub.logger = _Logger()
    sys.modules["src.logger"] = logger_stub


def _load_seed_module():
    _install_stub_logger()

    if "src.skills" not in sys.modules:
        pkg = types.ModuleType("src.skills")
        pkg.__path__ = [str(_REPO_ROOT / "src" / "skills")]
        sys.modules["src.skills"] = pkg

    if "src.skills._seed" in sys.modules:
        return sys.modules["src.skills._seed"]

    spec = importlib.util.spec_from_file_location(
        "src.skills._seed", _SEED_MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.skills._seed"] = mod
    spec.loader.exec_module(mod)
    return mod


_seed_module = _load_seed_module()
seed_skills_dir = _seed_module.seed_skills_dir
SEED_MARKER_FILENAME = _seed_module.SEED_MARKER_FILENAME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_fake_skill(root: Path, name: str, body: str = "Body") -> Path:
    """Create a minimal skill directory with a SKILL.md inside."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture()
def seed_source(tmp_path: Path) -> Path:
    """A realistic fake `src/skills/` layout: 3 seed skills plus package noise."""
    src = tmp_path / "seed_src"
    src.mkdir()
    _build_fake_skill(src, "alpha-skill")
    _build_fake_skill(src, "beta-skill")
    _build_fake_skill(src, "gamma-skill")

    # Package files that must NOT be copied.
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "_registry.py").write_text("", encoding="utf-8")
    (src / "README.md").write_text("# not a skill\n", encoding="utf-8")

    # Empty dir with no SKILL.md — also must not be copied.
    (src / "not-a-skill").mkdir()
    (src / "not-a-skill" / "notes.txt").write_text("no skill.md here\n")

    return src


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_seeds_fresh_directory(tmp_path: Path, seed_source: Path) -> None:
    dst = tmp_path / "run_dir" / "skills"

    copied = seed_skills_dir(dst, seed_source)

    assert copied == 3
    assert (dst / SEED_MARKER_FILENAME).exists()
    for name in ("alpha-skill", "beta-skill", "gamma-skill"):
        assert (dst / name / "SKILL.md").exists()

    # Package files and non-skill dirs must not leak in.
    assert not (dst / "__init__.py").exists()
    assert not (dst / "_registry.py").exists()
    assert not (dst / "README.md").exists()
    assert not (dst / "not-a-skill").exists()


def test_second_call_is_noop_when_marker_present(
    tmp_path: Path, seed_source: Path
) -> None:
    dst = tmp_path / "run_dir" / "skills"

    first = seed_skills_dir(dst, seed_source)
    assert first == 3

    # Simulate a learned skill added by the extractor between runs.
    _build_fake_skill(dst, "learned-skill", body="Learned after seeding")

    # And a `verified_uses` bump on a seed skill.
    learned_marker_before = (dst / SEED_MARKER_FILENAME).read_text()

    second = seed_skills_dir(dst, seed_source)

    # Second call must not re-copy or touch anything.
    assert second == 0
    assert (dst / "learned-skill" / "SKILL.md").exists()
    assert (dst / SEED_MARKER_FILENAME).read_text() == learned_marker_before


def test_reseed_after_marker_removed_preserves_prior_skills(
    tmp_path: Path, seed_source: Path
) -> None:
    """If the marker is manually deleted, re-seed runs but does not overwrite
    directories that already exist — the pre-existing library is preserved."""
    dst = tmp_path / "run_dir" / "skills"
    seed_skills_dir(dst, seed_source)

    # User nukes the marker and mutates one seed body.
    (dst / SEED_MARKER_FILENAME).unlink()
    (dst / "alpha-skill" / "SKILL.md").write_text(
        "---\nname: alpha-skill\ndescription: user-edited\n---\nEDITED\n",
        encoding="utf-8",
    )

    copied = seed_skills_dir(dst, seed_source)

    # Zero seed skills were copied afresh (all three targets exist, skipped).
    assert copied == 0
    # The user-edited content must be preserved.
    content = (dst / "alpha-skill" / "SKILL.md").read_text()
    assert "EDITED" in content
    # A new marker is still written so the next call is a no-op again.
    assert (dst / SEED_MARKER_FILENAME).exists()


def test_self_copy_is_noop_and_writes_no_marker(
    tmp_path: Path, seed_source: Path
) -> None:
    """When `dst == src` (legacy `skills_dir="src/skills"` behaviour),
    the helper must bail out cleanly without writing a marker into the
    canonical source (which would dirty the git tree)."""
    copied = seed_skills_dir(seed_source, seed_source)

    assert copied == 0
    assert not (seed_source / SEED_MARKER_FILENAME).exists()


def test_missing_seed_source_creates_empty_dst_and_marker(tmp_path: Path) -> None:
    dst = tmp_path / "run_dir" / "skills"
    missing_src = tmp_path / "does_not_exist"

    copied = seed_skills_dir(dst, missing_src)

    assert copied == 0
    assert dst.exists()
    # Marker is written so we don't keep trying to seed from a missing src.
    assert (dst / SEED_MARKER_FILENAME).exists()


def test_marker_records_seeded_skill_names(
    tmp_path: Path, seed_source: Path
) -> None:
    dst = tmp_path / "run_dir" / "skills"
    seed_skills_dir(dst, seed_source)

    marker_text = (dst / SEED_MARKER_FILENAME).read_text()
    for name in ("alpha-skill", "beta-skill", "gamma-skill"):
        assert name in marker_text
    assert "seeded_count: 3" in marker_text
