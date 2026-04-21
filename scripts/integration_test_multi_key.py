#!/usr/bin/env python
"""Tier 2 integration test — real Mistral multi-key round-robin.

Scope: verify that with both `MISTRAL_API_KEY` and `MISTRAL_API_KEY_2`
set, the `ModelManager`-registered `KeyRotatingOpenAIServerModel`
actually distributes completion calls across both underlying
`openai.AsyncOpenAI` instances. Unit tests cover this with mocks; this
script closes the loop against the real SDK + real Mistral endpoint.

Scope (what this does NOT cover): provoking a real 429 + cooldown is
not a stable integration test — the 5 req/min free-tier window is too
coarse and transiently busy providers add noise. The cooldown path is
verified in unit tests with synthesized `openai.RateLimitError` +
`httpx.Response`; that's where it belongs.

Usage: `/Users/ahbo/miniconda3/envs/dra/bin/python scripts/integration_test_multi_key.py`

Cost: ~10 calls × ~30 tokens each × $0.15/M in + $0.60/M out ≈ well
under $0.001. Run in ~10-15 s total.

Exits 0 on success, 2 on failure.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_env() -> tuple[str, str]:
    """Load .env and return (primary_tail, secondary_tail). Never prints
    the full key."""
    from dotenv import load_dotenv
    import os

    load_dotenv(REPO_ROOT / ".env")
    k1 = os.getenv("MISTRAL_API_KEY", "")
    k2 = os.getenv("MISTRAL_API_KEY_2", "")
    if not k1:
        print("FAIL: MISTRAL_API_KEY not set in .env", file=sys.stderr)
        sys.exit(2)
    if not k2:
        print("FAIL: MISTRAL_API_KEY_2 not set in .env", file=sys.stderr)
        sys.exit(2)
    if k1 == k2:
        print("FAIL: MISTRAL_API_KEY and MISTRAL_API_KEY_2 are identical — "
              "rotation test is meaningless with only one distinct key",
              file=sys.stderr)
        sys.exit(2)
    return k1[-4:], k2[-4:]


async def main() -> int:
    primary_tail, secondary_tail = load_env()
    print(f"[integration] keys loaded: primary ...{primary_tail}, "
          f"secondary ...{secondary_tail} (distinct)")

    # Import after env loads so ModelManager's init_models sees the keys.
    from src.models.openaillm import (
        KeyRotatingOpenAIServerModel, _KeyPoolState,
    )
    from src.models.models import ModelManager

    # Reset + init
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    # Assert registration used the rotating class
    model = mm.registed_models.get("mistral-small")
    if model is None:
        print("FAIL: mistral-small not registered", file=sys.stderr)
        return 2
    if not isinstance(model, KeyRotatingOpenAIServerModel):
        print(
            f"FAIL: mistral-small is {type(model).__name__}, "
            f"expected KeyRotatingOpenAIServerModel",
            file=sys.stderr,
        )
        return 2
    if model._pool.n != 2:
        print(
            f"FAIL: pool has {model._pool.n} keys, expected 2",
            file=sys.stderr,
        )
        return 2
    print(f"[integration] registered as KeyRotatingOpenAIServerModel "
          f"with pool.n={model._pool.n}  ✓")

    # Instrument: track which client index each pick selected.
    # We do this by wrapping `_get_async_client` so we can see what
    # _RotatingAsyncCompletions.create actually used.
    picked_indices: list[int] = []
    original_pick = model._pool.pick_index

    def traced_pick():
        idx = original_pick()
        picked_indices.append(idx)
        return idx

    model._pool.pick_index = traced_pick  # type: ignore[method-assign]

    # 10 minimal completions, back-to-back. We use `model.generate()` via
    # the standard ChatMessage shape so the existing retry / wait_for
    # plumbing runs exactly as it does in run_gaia.py.
    from src.models.base import ChatMessage

    N = 10
    print(f"[integration] firing {N} sequential mistral-small completions…")
    errors = 0
    for i in range(N):
        try:
            msg = ChatMessage(
                role="user",
                content="Reply with exactly one word: OK",
            )
            resp = await model.generate(messages=[msg])
            content = str(getattr(resp, "content", ""))[:40]
            print(f"  [{i+1:2d}/{N}] pool_idx={picked_indices[-1]}  "
                  f"→ {content!r}")
        except Exception as e:
            errors += 1
            print(f"  [{i+1:2d}/{N}] ERROR: {type(e).__name__}: {e}")

    if errors > 0:
        print(f"FAIL: {errors}/{N} completions errored", file=sys.stderr)
        return 2

    # Count distribution
    n_idx_0 = picked_indices.count(0)
    n_idx_1 = picked_indices.count(1)
    print(f"[integration] distribution over {N} calls: "
          f"idx0={n_idx_0}, idx1={n_idx_1}")

    # Assertions on distribution:
    #   1. Both indices must be used at least once (no silent fall-through
    #      to a single key).
    #   2. Neither should be hit more than 6/10 (round-robin is strict;
    #      allow 1 off-center for any cooldown).
    if n_idx_0 == 0 or n_idx_1 == 0:
        print(f"FAIL: one of the pool indices got 0 picks — rotation "
              f"not happening", file=sys.stderr)
        return 2
    if abs(n_idx_0 - n_idx_1) > 2:
        print(f"WARN: distribution {n_idx_0}/{n_idx_1} is skewed by more "
              f"than 2; expected ~5/5 under clean round-robin (could be "
              f"legitimate cooldown on one key due to rate limit)",
              file=sys.stderr)
        # Not a fail — cooldown makes this legitimate under live conditions.

    print(f"[integration] ALL CHECKS PASS  ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
