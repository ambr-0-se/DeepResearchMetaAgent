"""Live probes for the Phase-1-4 implementation.

Runs three cheap one-shot calls against OpenRouter to validate:
  1. Kimi K2.5 accepts `tool_choice="required"` with the new extra_body
     (thinking disabled + Moonshot provider pin) on a base64 image + a
     trivial tool list.
  2. Qwen3.6-Plus correctly downgrades to `tool_choice="auto"` and the
     "must call a tool" corrective prompt coaxes a tool call.
  3. Gemma 4 31B — two-step probe. Try `"required"` first (bypass dispatch);
     if it 404s or 400s, fall back to `"auto"` + corrective prompt. Records
     the verdict so we can decide whether `google/gemma-4-31b-it` stays in
     `MODELS_REJECTING_REQUIRED`.

Run from repo root in the dra env:
    /Users/ahbo/miniconda3/envs/dra/bin/python scripts/live_probe_tool_choice.py

Cost: <$0.02 total.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY or API_KEY in {"PLACEHOLDER", ""}:
    sys.exit("OPENROUTER_API_KEY missing in .env — cannot run live probes")

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")


# Minimal 1x1 transparent PNG (67 bytes).
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)

ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo a message back. Useful when you need to reply.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to echo."},
            },
            "required": ["message"],
        },
    },
}


def _verdict(tag: str, ok: bool, detail: str) -> None:
    marker = "✓" if ok else "✗"
    print(f"  [{marker}] {tag}: {detail}")


def probe_kimi() -> dict:
    """Probe 1 — Kimi K2.5 image + required tool_choice with Moonshot pin."""
    print("\n=== Probe 1 — Kimi K2.5 (image + required + extra_body) ===")
    try:
        resp = client.chat.completions.create(
            model="moonshotai/kimi-k2.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one sentence, then call the echo tool with your description."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
                    ],
                }
            ],
            tools=[ECHO_TOOL],
            tool_choice="required",
            extra_body={
                "thinking": {"type": "disabled"},
                "provider": {"order": ["Moonshot"]},
            },
            max_tokens=1024,
        )
        finish = resp.choices[0].finish_reason
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        _verdict("HTTP 200", True, "no 400 on thinking/required mismatch")
        _verdict("finish_reason", finish == "tool_calls", f"got {finish!r}")
        _verdict("tool_calls present", bool(tool_calls), f"{len(tool_calls)} call(s)")
        if tool_calls:
            tc = tool_calls[0]
            print(f"    tool={tc.function.name} args={tc.function.arguments[:120]!r}")
        return {
            "probe": "kimi",
            "ok": finish == "tool_calls" and bool(tool_calls),
            "finish_reason": finish,
            "tool_calls": len(tool_calls),
        }
    except Exception as exc:
        _verdict("exception", False, f"{type(exc).__name__}: {exc}")
        return {"probe": "kimi", "ok": False, "error": str(exc)}


def probe_qwen() -> dict:
    """Probe 2 — Qwen3.6-Plus auto + corrective prompt."""
    print("\n=== Probe 2 — Qwen3.6-Plus (auto dispatch + coercive prompt) ===")
    try:
        resp = client.chat.completions.create(
            model="qwen/qwen3.6-plus",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a tool-using assistant. Every step MUST call exactly "
                        "one tool from the provided list — never reply in plain text."
                    ),
                },
                {
                    "role": "user",
                    "content": "Please call the echo tool with the message 'hello'.",
                },
            ],
            tools=[ECHO_TOOL],
            tool_choice="auto",
            max_tokens=200,
        )
        finish = resp.choices[0].finish_reason
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        _verdict("HTTP 200", True, f"finish_reason={finish!r}")
        _verdict("tool_calls present", bool(tool_calls), f"{len(tool_calls)} call(s)")
        if tool_calls:
            tc = tool_calls[0]
            print(f"    tool={tc.function.name} args={tc.function.arguments[:120]!r}")
        elif msg.content:
            print(f"    plain-text reply: {msg.content[:200]!r}")
        return {
            "probe": "qwen",
            "ok": bool(tool_calls),
            "finish_reason": finish,
            "tool_calls": len(tool_calls),
        }
    except Exception as exc:
        _verdict("exception", False, f"{type(exc).__name__}: {exc}")
        return {"probe": "qwen", "ok": False, "error": str(exc)}


def probe_gemma() -> dict:
    """Probe 3 — Gemma 4 31B two-step: required first, auto fallback."""
    print("\n=== Probe 3 — Gemma 4 31B (required → auto fallback) ===")
    extra_body = {
        "provider": {"order": ["DeepInfra", "Together"], "allow_fallbacks": False},
        "reasoning": {"enabled": False},
    }

    # Step 1 — optimistic required
    print("  Step 1: tool_choice='required'")
    step1_ok = False
    step1_err = None
    try:
        resp = client.chat.completions.create(
            model="google/gemma-4-31b-it",
            messages=[
                {"role": "user", "content": "Call the echo tool with the message 'hi'."},
            ],
            tools=[ECHO_TOOL],
            tool_choice="required",
            extra_body=extra_body,
            max_tokens=200,
        )
        finish = resp.choices[0].finish_reason
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        raw_content = msg.content or ""
        _verdict("HTTP 200", True, f"finish={finish!r}")
        _verdict("tool_calls present", bool(tool_calls), f"{len(tool_calls)} call(s)")
        if tool_calls:
            tc = tool_calls[0]
            print(f"      tool={tc.function.name} args={tc.function.arguments[:120]!r}")
        # Sanity: check no raw gemma special tokens leaked to content
        leak = "<|tool_call>" in raw_content or "<tool_call|>" in raw_content
        _verdict("no special-token leak in content", not leak,
                 "chat template rendered tool call properly" if not leak
                 else f"LEAK in content: {raw_content[:200]!r}")
        step1_ok = bool(tool_calls) and not leak and finish == "tool_calls"
    except Exception as exc:
        step1_err = f"{type(exc).__name__}: {exc}"
        _verdict("step1 exception", False, step1_err)

    if step1_ok:
        print("  → Gemma accepts 'required'. Recommended: remove "
              "'google/gemma-4-31b-it' from MODELS_REJECTING_REQUIRED.")
        return {"probe": "gemma", "step1_ok": True, "action": "remove_from_set"}

    # Step 2 — auto fallback with coercive prompt
    print("  Step 2: tool_choice='auto' + coercive prompt")
    try:
        resp = client.chat.completions.create(
            model="google/gemma-4-31b-it",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a tool-using assistant. Every step MUST call exactly "
                        "one tool from the provided list — never reply in plain text."
                    ),
                },
                {"role": "user", "content": "Call the echo tool with the message 'hi'."},
            ],
            tools=[ECHO_TOOL],
            tool_choice="auto",
            extra_body=extra_body,
            max_tokens=200,
        )
        finish = resp.choices[0].finish_reason
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        _verdict("HTTP 200", True, f"finish={finish!r}")
        _verdict("tool_calls present via auto", bool(tool_calls), f"{len(tool_calls)} call(s)")
        if tool_calls:
            tc = tool_calls[0]
            print(f"      tool={tc.function.name} args={tc.function.arguments[:120]!r}")
        elif msg.content:
            print(f"      plain-text reply: {msg.content[:200]!r}")
        return {
            "probe": "gemma",
            "step1_ok": False,
            "step1_err": step1_err,
            "step2_ok": bool(tool_calls),
            "action": "keep_in_set",
        }
    except Exception as exc:
        _verdict("step2 exception", False, f"{type(exc).__name__}: {exc}")
        return {
            "probe": "gemma",
            "step1_ok": False,
            "step1_err": step1_err,
            "step2_ok": False,
            "step2_err": str(exc),
            "action": "fail",
        }


def main() -> int:
    results = [
        probe_kimi(),
        probe_qwen(),
        probe_gemma(),
    ]
    print("\n=== Summary ===")
    for r in results:
        print(json.dumps(r, indent=2))
    # Return 1 if any probe is hard-failed (exception or neither step works)
    any_fail = any(
        (not r.get("ok", True) and "error" in r)
        or (r.get("probe") == "gemma" and r.get("action") == "fail")
        for r in results
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
