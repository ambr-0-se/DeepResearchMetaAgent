"""Probe: when qwen3.6-plus on OR raw-mode emits content='', is the
structured AgentOutput JSON present in `output.tool_calls`?

This is the verification gate for Option 2 (patch get_next_action to
recover from content='' by reading raw_msg.tool_calls). If tool_calls
is populated with the AgentOutput schema, Option 2 is viable. If
tool_calls is empty/garbage, Option 2 cannot help.

Method:
  1. Build the same LangChain wrapper auto_browser uses for `or-qwen3.6-plus`.
  2. Wrap `llm.ainvoke` to capture every AIMessage response.
  3. Run a real browser_use Agent in raw mode (the production path) on a
     task that browser_use histories show tends to produce content='' on
     qwen — multi-step navigation with cookie banner.
  4. After each invoke, log: content present?, tool_calls present?, when
     content is empty, dump full tool_calls structure.
  5. Run for ≤6 steps / 240s wall clock.

Output: classification of every empty-content response — does it carry
recoverable JSON in tool_calls?
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


# Capture for every llm.ainvoke call
captured: list[dict] = []


def _install_capture():
    """Patch browser_use Agent.get_next_action to capture the AIMessage
    *before* the parser is called. Any pydantic-protected wrapper layer
    is bypassed because we go straight to the agent service module.
    """
    from browser_use.agent import service as _bu_svc

    original = _bu_svc.Agent.get_next_action

    async def wrapped(self, input_messages):
        # The raw branch in browser_use 0.1.48 calls self.llm.invoke(),
        # but we want the underlying response. We can replicate the raw
        # branch's behaviour up to the parse step ourselves to capture
        # the AIMessage, then defer to the original method.
        if self.tool_calling_method == "raw":
            try:
                output = self.llm.invoke(input_messages)
                content = getattr(output, "content", None)
                tool_calls = getattr(output, "tool_calls", None)
                additional_kwargs = getattr(output, "additional_kwargs", None) or {}
                response_metadata = getattr(output, "response_metadata", None) or {}
                captured.append({
                    "content_preview": (str(content) or "")[:300],
                    "content_empty": not str(content or "").strip(),
                    "tool_calls": tool_calls,
                    "additional_kwargs_keys": list(additional_kwargs.keys()),
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

    requested = "or-qwen3.6-plus"
    lookup_id = f"langchain-{requested}"
    if lookup_id not in model_manager.registed_models:
        raise SystemExit(f"missing langchain registration: {lookup_id}")
    raw_model = model_manager.registed_models[lookup_id]
    print(f"[probe] raw_model class={type(raw_model).__name__}")
    print(f"[probe] wire_id={_resolve_wire_id(raw_model)!r}")
    llm = _unwrap_for_browser_use(raw_model)
    print(f"[probe] llm class={type(llm).__name__}")

    _install_capture()
    print("[probe] browser_use Agent.get_next_action wrapped for capture")

    # Multi-step task historically associated with empty content from qwen.
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
        # Production setting (raw mode) — same as auto_browser.py runtime.
        tool_calling_method="raw",
        max_actions_per_step=1,
    )
    print(f"[probe] agent built, tool_calling_method={agent.tool_calling_method}")
    print("[probe] running task (max 6 steps, 240s wall clock)...")

    t0 = time.monotonic()
    try:
        await asyncio.wait_for(agent.run(max_steps=6), timeout=240)
    except asyncio.TimeoutError:
        print("[probe] (wall-clock-timeout 240s — expected for stuck loops)")
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

    # Classify
    empty_with_tc = empty_no_tc = nonempty = 0
    for i, c in enumerate(captured):
        if "capture_error" in c:
            print(f"[probe] capture_err[{i}]: {c['capture_error']}")
            continue
        is_empty = c.get("content_empty")
        tc = c.get("tool_calls")
        has_tc = isinstance(tc, list) and len(tc) > 0
        if is_empty and has_tc:
            empty_with_tc += 1
        elif is_empty:
            empty_no_tc += 1
        else:
            nonempty += 1

    print()
    print("=" * 60)
    print(f"  empty content + tool_calls populated  : {empty_with_tc}")
    print(f"  empty content + NO tool_calls         : {empty_no_tc}")
    print(f"  non-empty content                     : {nonempty}")
    print("=" * 60)

    # Dump the full tool_calls structure for the first empty-with-tc case
    print()
    for i, c in enumerate(captured):
        if "capture_error" in c:
            continue
        if c.get("content_empty") and isinstance(c.get("tool_calls"), list) and len(c["tool_calls"]) > 0:
            print(f"--- empty-content response #{i} (tool_calls populated) ---")
            print(f"  content_preview: {c['content_preview']!r}")
            print(f"  finish_reason: {c['finish_reason']!r}")
            print(f"  completion_tokens: {c['completion_tokens']}")
            print(f"  tool_calls (truncated to 3 calls):")
            for j, tc in enumerate(c["tool_calls"][:3]):
                tc_dump = json.dumps(tc, default=str, indent=2)[:1500]
                print(f"    [{j}] {tc_dump}")
            break
    else:
        # No empty-with-tc found
        for i, c in enumerate(captured):
            if "capture_error" in c:
                continue
            if c.get("content_empty"):
                print(f"--- empty-content response #{i} (NO tool_calls) ---")
                print(f"  content_preview: {c['content_preview']!r}")
                print(f"  finish_reason: {c['finish_reason']!r}")
                print(f"  additional_kwargs_keys: {c['additional_kwargs_keys']}")
                print(f"  tool_calls: {c.get('tool_calls')!r}")
                break

    # VERDICT
    print()
    if empty_with_tc > 0:
        print(f"[probe] VERDICT: tool_calls IS populated when content='' "
              f"({empty_with_tc} cases) — Option 2 is viable.")
        return 0
    if empty_no_tc > 0:
        print(f"[probe] VERDICT: empty content has NO recoverable tool_calls "
              f"({empty_no_tc} cases) — Option 2 CANNOT help.")
        return 1
    print("[probe] VERDICT: did not reproduce empty-content state in this probe. "
          "Inconclusive.")
    return 2


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
