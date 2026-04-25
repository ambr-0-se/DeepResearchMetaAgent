"""Isolated probe: does browser_use Agent + qwen3.6-plus on OR survive
`tool_calling_method='function_calling'`?

Hypothesis under test: the project's 2026-04-23 finding (raw-mode is
required because function_calling makes Qwen terminate at Step 1 with 0
chars) still holds. If it does, this probe should reproduce that
behaviour. If it no longer holds (provider routing changed, browser_use
upgraded, etc.), function_calling is back on the table.

Procedure:
  1. Build the LangChain wrapper for `or-qwen3.6-plus` exactly like
     `AutoBrowserUseTool` does at runtime.
  2. Start a browser_use Agent with a deliberately trivial task and
     `tool_calling_method='function_calling'` (overriding the auto-pick
     that currently returns 'raw').
  3. Run for at most 6 steps with a tight wall clock.
  4. Report: did the agent take any meaningful action? Did it terminate
     prematurely (Step 1 done with 0 actions)? Did it crash?

Run: `/Users/ahbo/miniconda3/envs/dra/bin/python scripts/probe_qwen_browser_function_calling.py`
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure dotenv is loaded before any model_manager import
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")


async def main() -> int:
    from browser_use import Agent, Browser, BrowserConfig
    from src.models import model_manager
    from src.tools._browser_json_extractor import install_tolerant_extractor
    from src.tools.auto_browser import _unwrap_for_browser_use, _resolve_wire_id

    # Mirror AutoBrowserUseTool's setup so the probe is faithful to runtime.
    model_manager.init_models(use_local_proxy=True)
    install_tolerant_extractor()

    requested = "or-qwen3.6-plus"
    lookup_id = f"langchain-{requested}"
    if lookup_id not in model_manager.registed_models:
        raise SystemExit(f"missing langchain registration: {lookup_id}")
    raw_model = model_manager.registed_models[lookup_id]
    print(f"[probe] raw_model class={type(raw_model).__name__}")
    print(f"[probe] wire_id={_resolve_wire_id(raw_model)!r}")
    llm = _unwrap_for_browser_use(raw_model)
    print(f"[probe] llm class={type(llm).__name__}")

    # Trivial deterministic task — Wikipedia page, no auth, well-formed.
    task = (
        "Go to https://en.wikipedia.org/wiki/Clownfish and report the "
        "scientific name in the first sentence of the article."
    )

    browser = Browser(
        config=BrowserConfig(
            headless=True,
            disable_security=True,
        )
    )

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        # The probe variable: force function_calling mode (overrides
        # `_pick_browser_tool_calling_method`'s 'raw' return).
        tool_calling_method="function_calling",
        max_actions_per_step=1,
        # Hard cap so a runaway loop can't burn budget.
        # browser_use 0.1.48 doesn't honour timeouts directly — we hold
        # it via the outer wait_for below.
    )

    print(f"[probe] agent built, tool_calling_method={agent.tool_calling_method}")
    print("[probe] running task (max 6 steps, 180s wall clock)...")

    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(agent.run(max_steps=6), timeout=180)
    except asyncio.TimeoutError:
        print("[probe] RESULT=wall-clock-timeout (180s)")
        result = None
    except Exception as e:
        print(f"[probe] RESULT=exception {type(e).__name__}: {e}")
        traceback.print_exc()
        result = None
    finally:
        try:
            await browser.close()
        except Exception:
            pass
    elapsed = time.monotonic() - t0
    print(f"[probe] elapsed={elapsed:.1f}s")

    n_steps = getattr(getattr(agent, "state", None), "n_steps", None)
    print(f"[probe] final agent.state.n_steps={n_steps}")
    print(f"[probe] history.history.length={len(agent.state.history.history) if hasattr(agent.state, 'history') else None}")

    # Walk history and check for the early-done degenerate pattern that
    # raw-mode was chosen to avoid.
    history = []
    if hasattr(agent.state, "history") and hasattr(agent.state.history, "history"):
        history = agent.state.history.history
    early_done = False
    for i, h in enumerate(history):
        actions = getattr(h, "model_output", None)
        if actions is None:
            continue
        action_list = getattr(actions, "action", None) or []
        is_done = any(getattr(a, "done", None) is not None for a in action_list)
        print(f"[probe] step {i}: n_actions={len(action_list)} is_done={is_done}")
        if i == 0 and is_done:
            early_done = True

    if early_done:
        print("[probe] CONCLUSION=EARLY-DONE-AT-STEP-1 (matches project's prior finding)")
        return 1
    if n_steps is None or n_steps == 0:
        print("[probe] CONCLUSION=NO-PROGRESS")
        return 2
    print("[probe] CONCLUSION=PROGRESSED — function_calling may be viable; inspect output above")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
