"""Probe: does or-qwen3-next-80b-a3b-instruct survive browser_use raw-mode
without producing the content='' garbage that or-qwen3.6-plus emits?

Verification gate for Option 11/12 — switch only the browser_use_tool +
browser_use_agent slot to qwen3-next-80b, leave the planner/sub-agents
on qwen3.6-plus.

Pass criteria:
  - 0 empty-content responses across the run
  - Agent makes meaningful progress (non-trivial action history)
"""
from __future__ import annotations

import asyncio
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

captured: list[dict] = []


def _install_capture():
    from browser_use.agent import service as _bu_svc

    original = _bu_svc.Agent.get_next_action

    async def wrapped(self, input_messages):
        if self.tool_calling_method == "raw":
            try:
                output = self.llm.invoke(input_messages)
                content = getattr(output, "content", None)
                tool_calls = getattr(output, "tool_calls", None)
                response_metadata = getattr(output, "response_metadata", None) or {}
                captured.append({
                    "content_preview": (str(content) or "")[:200],
                    "content_empty": not str(content or "").strip(),
                    "has_tool_calls": isinstance(tool_calls, list) and len(tool_calls) > 0,
                    "finish_reason": response_metadata.get("finish_reason"),
                    "completion_tokens": (
                        response_metadata.get("token_usage", {}) or {}
                    ).get("completion_tokens"),
                })
            except Exception as e:
                captured.append({"capture_error": f"{type(e).__name__}: {e}"})
        return await original(self, input_messages)

    _bu_svc.Agent.get_next_action = wrapped


async def main() -> int:
    from browser_use import Agent, Browser, BrowserConfig
    from src.models import model_manager
    from src.tools._browser_json_extractor import install_tolerant_extractor
    from src.tools.auto_browser import _unwrap_for_browser_use, _resolve_wire_id

    model_manager.init_models(use_local_proxy=True)
    install_tolerant_extractor()

    requested = "or-qwen3-next-80b-a3b-instruct"
    lookup_id = f"langchain-{requested}"
    if lookup_id not in model_manager.registed_models:
        raise SystemExit(f"missing langchain registration: {lookup_id}")
    raw_model = model_manager.registed_models[lookup_id]
    print(f"[probe] raw_model class={type(raw_model).__name__}")
    print(f"[probe] wire_id={_resolve_wire_id(raw_model)!r}")
    llm = _unwrap_for_browser_use(raw_model)
    print(f"[probe] llm class={type(llm).__name__}")

    _install_capture()

    task = (
        "Go to https://www.iana.org/ and find the headline of the latest news "
        "item. Click into the news section and report the title of the most "
        "recent news article. Be precise."
    )

    browser = Browser(config=BrowserConfig(headless=True, disable_security=True))

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        tool_calling_method="raw",
        max_actions_per_step=1,
        use_vision=False,  # qwen3-next-80b text-only — no image input support
    )
    print(f"[probe] agent built, tool_calling_method={agent.tool_calling_method}")
    print("[probe] running task (max 6 steps, 240s wall clock)...")

    t0 = time.monotonic()
    try:
        await asyncio.wait_for(agent.run(max_steps=6), timeout=240)
    except asyncio.TimeoutError:
        print("[probe] (wall-clock-timeout 240s)")
    except Exception as e:
        print(f"[probe] exception {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        try:
            await browser.close()
        except Exception:
            pass

    elapsed = time.monotonic() - t0
    print(f"[probe] elapsed={elapsed:.1f}s, captured={len(captured)} responses")

    empty = nonempty = 0
    for c in captured:
        if "capture_error" in c:
            continue
        if c.get("content_empty"):
            empty += 1
        else:
            nonempty += 1

    n_steps = getattr(getattr(agent, "state", None), "n_steps", None)
    print()
    print("=" * 60)
    print(f"  Empty-content responses : {empty}")
    print(f"  Non-empty responses     : {nonempty}")
    print(f"  Final agent.state.n_steps : {n_steps}")
    print("=" * 60)

    print()
    if empty == 0 and nonempty > 0:
        print("[probe] VERDICT: PASS — qwen3-next-80b emits content reliably. Apply override.")
        return 0
    if empty > 0:
        print(f"[probe] VERDICT: FAIL — {empty} empty-content responses (qwen3-next-80b also broken).")
        return 1
    print("[probe] VERDICT: INCONCLUSIVE — no responses captured.")
    return 2


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
