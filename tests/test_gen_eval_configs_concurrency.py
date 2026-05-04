"""Guardrail for the per-model concurrency defaults baked into the 16
generated matrix configs (§P3 of `docs/handoffs/HANDOFF_THROUGHPUT_REFACTOR.md`).

This test catches two failure modes:
  1. Someone bumps a non-Qwen model's concurrency (or drops Qwen back
     to 4) without regenerating + committing configs.
  2. `scripts/gen_eval_configs.py`'s MODELS tuple drifts out of sync
     with the committed config files — e.g. generator emits c=8 for
     Qwen but the committed Qwen config still says 4 because someone
     forgot to regenerate.

Evidence base for the Qwen=8 decision: E0 v3 Qwen run made 3,006 LLM
calls at concurrency=4 with 0 × 429, 0 × Retry-After, 0 × rate-limit
warnings (all routed to OR→Alibaba). See
`docs/handoffs/HANDOFF_THROUGHPUT_REFACTOR.md` §P3 for the full argument.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

CONDITIONS = ("c0", "c1", "c2", "c3")

# Expected per-model concurrency in the generated configs as of 2026-04-22.
# Update this when the MODELS tuple in `scripts/gen_eval_configs.py` changes.
EXPECTED_CONCURRENCY = {
    "mistral": 4,
    "kimi": 4,
    "qwen": 8,
    "gemma": 4,
}


def _read_concurrency_from_config(path: Path) -> int:
    """Extract the value of the top-level `concurrency = <int>` assignment
    without importing the config (configs pull in mmengine + transitive
    `src.*` deps that are overkill for a plain text-file check).
    """
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped.startswith("concurrency") and "=" in stripped:
            lhs, rhs = stripped.split("=", 1)
            if lhs.strip() == "concurrency":
                value = rhs.split("#", 1)[0].strip()
                return int(value)
    raise AssertionError(
        f"No top-level `concurrency = <int>` assignment found in {path}"
    )


@pytest.mark.parametrize("model,expected", sorted(EXPECTED_CONCURRENCY.items()))
@pytest.mark.parametrize("condition", CONDITIONS)
def test_config_concurrency_matches_expected(
    model: str, expected: int, condition: str
) -> None:
    """Every `config_gaia_<cond>_<model>.py` sets `concurrency` to the
    expected per-model default. A drift here almost always means the
    generator was updated but `python scripts/gen_eval_configs.py` was
    not re-run, leaving generated + tracked configs out of sync.
    """
    path = CONFIGS_DIR / f"config_gaia_{condition}_{model}.py"
    assert path.exists(), f"config file missing: {path}"

    observed = _read_concurrency_from_config(path)
    assert observed == expected, (
        f"{path.name}: concurrency={observed} but expected {expected}. "
        f"If the intent was to change this, update EXPECTED_CONCURRENCY "
        f"AND the MODELS tuple in scripts/gen_eval_configs.py, then "
        f"regenerate configs."
    )


def test_generator_models_tuple_includes_concurrency_field() -> None:
    """Defensive: assert the MODELS tuple in the generator still has
    the concurrency field in the 5th slot. A refactor that removes it
    would silently fall back to a previous default and break the
    c=8-for-Qwen contract without any individual config failing —
    because they'd all be internally consistent at the wrong value.
    """
    from scripts import gen_eval_configs

    assert hasattr(gen_eval_configs, "MODELS")
    for entry in gen_eval_configs.MODELS:
        assert isinstance(entry, tuple) and len(entry) == 5, (
            f"MODELS entry {entry!r} should be a 5-tuple: "
            f"(label, model_id, langchain_alias, comment, concurrency)"
        )
        _, _, _, _, concurrency = entry
        assert isinstance(concurrency, int) and concurrency >= 1, (
            f"concurrency must be a positive int; got {concurrency!r}"
        )
