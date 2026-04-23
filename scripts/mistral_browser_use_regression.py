#!/usr/bin/env python
"""Mistral `auto_browser_use_tool` regression smoke.

Companion to `scripts/p5_live_validation.py`: exercises the SAME
`browser_use.Agent` wiring but with a Mistral model instead of Qwen,
to prove the two-layer Qwen fix (raw mode + tolerant JSON extractor)
did not regress Mistral's browser path (R6 fairness guardrail in the
plan).

Pass criteria
-------------
1. `browser_use.Agent` reaches Step 1 + Step 2 (same baseline as
   Mistral E0 v3 — every browser_use session gets multi-step
   trajectories, per `full_mistral.log` count distribution).
2. Extracted content ≥ 100 chars.
3. `tool_calling_method` is NOT passed to `Agent(...)` — Mistral takes
   browser_use's default auto-detection path (function_calling for
   ChatOpenAI).
4. Tolerant JSON extractor patch is NOT installed — Mistral path is
   byte-identical with pre-fix behaviour.

Cost: ~5-20 LLM calls to Mistral La Plateforme at ~150 tokens each
≈ well under $0.01. Runs in ~30-60s.

Exits 0 on success, 2 on any failure.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


PASS_CRITERIA_TASK = (
    # Same task as p5_live_validation.py so results are comparable
    # side-by-side.
    "Navigate to https://en.wikipedia.org/wiki/Kangaroo and report the "
    "scientific genus name for kangaroos as stated in the first sentence."
)


async def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    from src.models.models import ModelManager
    from src.tools.auto_browser import (
        _pick_browser_tool_calling_method,
        _resolve_wire_id,
        _unwrap_for_browser_use,
    )
    from src.tools._browser_json_extractor import is_patched, _reset_for_tests

    # Fresh state — we MUST start with the tolerant-extractor patch
    # uninstalled so we can assert it stays uninstalled on the
    # Mistral path.
    _reset_for_tests()

    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    mistral = mm.registed_models.get("langchain-mistral-small-latest")
    if mistral is None:
        print("FAIL: langchain-mistral-small-latest not registered "
              "(MISTRAL_API_KEY unset?)", file=sys.stderr)
        return 2

    wire_id = _resolve_wire_id(mistral)
    tool_calling_method = _pick_browser_tool_calling_method(wire_id)
    print(f"[mistral-reg] wire_id={wire_id!r} "
          f"tool_calling_method={tool_calling_method!r}")

    if tool_calling_method is not None:
        print(f"FAIL: Mistral path picked tool_calling_method="
              f"{tool_calling_method!r} — regression! (Expected None)",
              file=sys.stderr)
        return 2

    browser_use_buf = io.StringIO()
    handler = logging.StreamHandler(browser_use_buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(name)s %(message)s"))
    bu_logger = logging.getLogger("browser_use")
    bu_logger.addHandler(handler)
    bu_logger.setLevel(logging.INFO)

    print(f"[mistral-reg] task: {PASS_CRITERIA_TASK[:80]}...")

    from browser_use import Agent, Controller
    controller = Controller()

    # Unwrap KeyRotatingChatOpenAI → instance[0] (same path as
    # auto_browser.py). Without this, Pydantic v2 rejects the
    # wrapper because it isn't a BaseChatModel subclass.
    bu_model = _unwrap_for_browser_use(mistral)

    agent_kwargs = dict(
        task=PASS_CRITERIA_TASK,
        llm=bu_model,
        enable_memory=False,
        controller=controller,
        page_extraction_llm=bu_model,
    )
    # Deliberately do NOT set tool_calling_method — verifies the
    # default-path kwarg elision in auto_browser.py.
    browser_agent = Agent(**agent_kwargs)

    try:
        try:
            history = await asyncio.wait_for(
                browser_agent.run(max_steps=8),
                timeout=300,
            )
        except asyncio.TimeoutError:
            print("FAIL: browser_agent.run timed out at 300s",
                  file=sys.stderr)
            return 2
    except Exception as e:
        print(f"FAIL: browser_agent.run raised {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2
    finally:
        try:
            await asyncio.wait_for(browser_agent.close(), timeout=15)
        except Exception:
            pass

    bu_log = browser_use_buf.getvalue()
    content = history.extracted_content() if history else []
    joined = "\n".join(c for c in content if c).strip()

    print("\n[mistral-reg] --- results ---")

    step_lines = [line for line in bu_log.splitlines() if "📍 Step" in line]
    distinct_steps = set()
    for line in step_lines:
        for tok in line.split():
            if tok.isdigit():
                distinct_steps.add(int(tok))
                break
    print(f"[mistral-reg] step numbers observed: {sorted(distinct_steps)}")

    c1 = len(distinct_steps) >= 1
    print(f"[mistral-reg] C1 — reached Step 1: {'PASS' if c1 else 'FAIL'}")

    c2 = max(distinct_steps) >= 2 if distinct_steps else False
    print(f"[mistral-reg] C2 — reached Step 2+: {'PASS' if c2 else 'FAIL'}")

    c3 = len(joined) >= 100
    print(f"[mistral-reg] C3 — extracted ≥100 chars ({len(joined)}): "
          f"{'PASS' if c3 else 'FAIL'}")

    c4 = not is_patched()
    print(f"[mistral-reg] C4 — tolerant extractor NOT patched on Mistral "
          f"path: {'PASS' if c4 else 'FAIL'}")

    if joined:
        print(f"\n[mistral-reg] sample output (first 300 chars):")
        print(f"  {joined[:300]!r}")

    all_pass = c1 and c2 and c3 and c4
    print(f"\n[mistral-reg] overall: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
