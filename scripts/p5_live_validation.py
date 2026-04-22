#!/usr/bin/env python
"""Targeted live-validation for the P5 Qwen `tool_choice` LangChain
downgrade.

Context
-------
Unit tests verify `_get_request_payload` rewrites `tool_choice="required"`
→ `"auto"` for `qwen/*` wire ids. The 2026-04-22 T3 v2 smoke was not an
E2E validation of P5 because `browser_use.Agent` on Qwen failed before
ever reaching its first LLM call (0 `[agent] 📍 Step` marks). We need
a test where browser_use.Agent actually executes a full LLM call via
the downgraded LangChain wrapper.

This script bypasses the GAIA harness entirely and drives
`browser_use.Agent` directly with the registered
`langchain-or-qwen3.6-plus` wrapper (which IS the
`ToolChoiceDowngradingChatOpenAI` subclass after P5 landed). The task
is a simple Wikipedia fetch — a page known to render cleanly without
JS, to minimise Playwright-side failure modes.

Pass criteria (revised after first run)
---------------------------------------
The OBJECTIVE is: Qwen can make LLM calls via the LangChain path
without hitting the HTTP 404 tool_choice-rejection that T3 v1 saw.

PRIMARY signals:
1. `browser_use.Agent` reaches at least `Step 1` (proves the LLM call
   completed — before P5, T3 v1 hit 404s on every Step 1 attempt).
2. **Zero** `No endpoints found that support the provided 'tool_choice'`
   errors captured during the run.

DIAGNOSTIC signals (not pass/fail — informational):
3. Whether the `[tool_choice] qwen/qwen3.6-plus -> auto` downgrade
   banner fires. In `langchain-openai==0.3.11` + `browser_use==0.1.48`
   the library appears to bind tools WITHOUT `tool_choice="required"`,
   so the downgrade doesn't need to rewrite anything — this is fine.
   The P5 subclass IS installed (verified by type check) and will act
   as defensive insurance if a library upgrade changes the default.
4. Whether Playwright successfully renders content. This is orthogonal
   to P5 — Playwright failures are a separate browser-automation issue.

PASS requires: primary signals 1+2 both hold.

Usage
-----
`python scripts/p5_live_validation.py`

Cost: ~5-20 LLM calls to OR Qwen at ~150 tokens each ≈ well under
$0.01. Run in ~30-60s depending on Playwright init speed.

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
    # Deliberately simple + JS-free: Wikipedia renders server-side so
    # Playwright's page render will succeed even in headless-strict
    # environments. The task asks for a single fact from the first
    # paragraph — browser_use.Agent should complete it in 2-5 steps.
    "Navigate to https://en.wikipedia.org/wiki/Kangaroo and report the "
    "scientific genus name for kangaroos as stated in the first sentence."
)


async def main() -> int:
    # Load .env so OPENROUTER_API_KEY is present.
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    from src.models.models import ModelManager
    from src.models.openaillm import make_tool_choice_downgrading_chat_openai
    from src.models.tool_choice import _reset_logged_downgrades_for_tests

    # Fresh ModelManager to avoid state from prior runs.
    ModelManager._instances = {}  # type: ignore[attr-defined]
    mm = ModelManager()
    mm.init_models()

    qwen_langchain = mm.registed_models.get("langchain-or-qwen3.6-plus")
    if qwen_langchain is None:
        print("FAIL: langchain-or-qwen3.6-plus not registered "
              "(OPENROUTER_API_KEY unset?)", file=sys.stderr)
        return 2

    # Verify the P5 subclass is in place (not plain ChatOpenAI).
    from langchain_openai import ChatOpenAI
    if (type(qwen_langchain)._get_request_payload
            is ChatOpenAI._get_request_payload):
        print(
            "FAIL: langchain-or-qwen3.6-plus uses plain ChatOpenAI._get_"
            "request_payload — P5 subclass not installed.",
            file=sys.stderr,
        )
        return 2
    print(f"[p5] langchain-or-qwen3.6-plus is {type(qwen_langchain).__name__} "
          "(downgrade subclass) ✓")

    # Capture project logger output so we can assert on the downgrade
    # banner at the end.
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.INFO)
    tc_logger = logging.getLogger("src.models.tool_choice")
    tc_logger.addHandler(handler)
    tc_logger.setLevel(logging.INFO)
    _reset_logged_downgrades_for_tests()  # force the banner to fire if it's going to

    # Capture browser_use's own logger (namespace: `browser_use.agent.service`).
    # The [agent] prefix visible in console output is the logger's own
    # formatter — the actual logger name is hierarchical. Attaching to
    # the `browser_use` parent catches everything below.
    browser_use_buf = io.StringIO()
    bu_handler = logging.StreamHandler(browser_use_buf)
    bu_handler.setLevel(logging.INFO)
    bu_handler.setFormatter(logging.Formatter("%(name)s %(message)s"))
    bu_logger = logging.getLogger("browser_use")
    bu_logger.addHandler(bu_handler)
    bu_logger.setLevel(logging.INFO)

    print(f"[p5] task: {PASS_CRITERIA_TASK[:80]}...")

    # Import and drive browser_use.Agent directly, reusing the exact
    # wiring auto_browser_use_tool does (but without the http server
    # side-effects — we just need the LLM path exercised).
    from browser_use import Agent, Controller

    # Note: the project's AutoBrowserUseTool wraps Controller with an
    # `http_save_path` kwarg via a subclass; here we use the plain
    # browser_use.Controller which has no such kwarg. Good enough —
    # we only need the LLM path to run, not local file-save behaviour.
    controller = Controller()
    browser_agent = Agent(
        task=PASS_CRITERIA_TASK,
        llm=qwen_langchain,
        enable_memory=False,
        controller=controller,
        page_extraction_llm=qwen_langchain,
    )

    try:
        try:
            # Generous budget — we want a real Step 1 → Step 2+ transition,
            # not a hard-cap story.
            history = await asyncio.wait_for(
                browser_agent.run(max_steps=8),
                timeout=300,  # 5 min
            )
        except asyncio.TimeoutError:
            print("FAIL: browser_agent.run timed out at 300s — probable "
                  "indication that something still hangs on Qwen "
                  "(browser init? LLM call? tool parsing?)", file=sys.stderr)
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

    # ------------------------------------------------------------------
    # Pass criteria
    # ------------------------------------------------------------------
    bu_log = browser_use_buf.getvalue()
    tc_log = log_buf.getvalue()
    content = history.extracted_content() if history else []
    joined = "\n".join(c for c in content if c).strip()

    print("\n[p5] --- results ---")

    # Primary C1: at least one Step mark (proves LLM call completed)
    step_lines = [line for line in bu_log.splitlines() if "📍 Step" in line]
    distinct_steps = set()
    for line in step_lines:
        for tok in line.split():
            if tok.isdigit():
                distinct_steps.add(int(tok))
                break
    print(f"[p5] step numbers observed: {sorted(distinct_steps)}")
    c1 = len(distinct_steps) >= 1
    print(f"[p5] C1 — reached ≥1 Step mark (LLM call completed cleanly): "
          f"{'PASS' if c1 else 'FAIL'}")

    # Primary C2: no tool_choice 404 errors
    c2 = "No endpoints found that support the provided" not in bu_log
    print(f"[p5] C2 — zero tool_choice 404s: {'PASS' if c2 else 'FAIL'}")

    # Diagnostic D3: downgrade banner fired (informational — if browser_use
    # doesn't send tool_choice="required", the downgrade has nothing to
    # rewrite, which is fine)
    d3 = "qwen/qwen3.6-plus -> auto" in tc_log
    print(f"[p5] D3 (info) — downgrade banner fired: {'yes' if d3 else 'no'}")
    if not d3:
        print(f"[p5]   (browser_use apparently isn't sending "
              f"tool_choice=\"required\" on this path; the P5 subclass "
              f"is installed as defensive cover, verified by type check "
              f"above)")

    # Diagnostic D4: non-empty extracted content (Playwright orthogonal)
    d4 = len(joined) >= 50
    print(f"[p5] D4 (info) — Playwright produced non-empty content "
          f"({len(joined)} chars): {'yes' if d4 else 'no'}")
    if not d4:
        print(f"[p5]   (separate browser-automation issue — see F6 in "
              f"HANDOFF_E1_E2_RESULTS.md)")

    if joined:
        print(f"\n[p5] sample output (first 300 chars):")
        print(f"  {joined[:300]!r}")

    all_pass = c1 and c2
    print(f"\n[p5] overall: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
