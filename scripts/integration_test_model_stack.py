"""Integration test: exercise the actual OpenAIServerModel → dispatch → retry path.

Validates that the registered ``or-kimi-k2.5``, ``or-qwen3.6-plus``, and
``or-gemma-4-31b-it`` models work end-to-end through the production stack
(``model_manager.init_models()`` → ``OpenAIServerModel._prepare_completion_kwargs``
→ hybrid ``tool_choice`` dispatch → OR request). This is complementary to
``scripts/live_probe_tool_choice.py`` — the probe uses the raw OpenAI SDK and
would pass even if our wrapper was broken; this script exercises the wrapper.

Also sanity-checks:
- Kimi extra_body is applied (request succeeds with ``tool_choice="required"``
  + thinking-disabled; would 400 if the extra_body plumbing regressed).
- Qwen gets downgraded to ``"auto"`` via the prefix rule (look for the
  one-shot ``[tool_choice]`` INFO log).
- Gemma provider pin keeps the request on DeepInfra/Together (request
  succeeds with ``"required"``; provider-pin failure would 404).
- Retry guard fires as a non-streaming fallback for Qwen if the first turn
  comes back empty-handed (exercised indirectly — ``SimpleNamespace`` tool
  is easy for every model to call, so usually first-turn success).

Run from repo root in the dra env:
    /Users/ahbo/miniconda3/envs/dra/bin/python scripts/integration_test_model_stack.py

Cost: <$0.02 total (same order as the probe).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.models.base import ChatMessage  # noqa: E402
from src.models import model_manager  # noqa: E402
from src.models import tool_choice as tc  # noqa: E402

# Capture the dispatch module's INFO log so we can assert the dedup line fires
# exactly once per (run, model) when the dispatcher downgrades.
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


def _make_tool() -> SimpleNamespace:
    """Minimal duck-typed Tool for message_manager.get_tool_json_schema."""
    return SimpleNamespace(
        name="echo",
        description="Echo a message back.",
        parameters={
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo.",
                },
            },
        },
    )


async def _run_turn(model_name: str, expect_downgrade: bool) -> dict:
    print(f"\n=== {model_name} (expect_downgrade={expect_downgrade}) ===")
    model = model_manager.registed_models.get(model_name)
    if model is None:
        return {"model_name": model_name, "ok": False, "error": "not registered"}

    # Snapshot log state so we can assert the downgrade log fired for this
    # specific model (or NOT for models that should pass "required" through).
    tc._reset_logged_downgrades_for_tests()

    messages = [
        ChatMessage(
            role="user",
            content="Please call the echo tool with the message 'hello from integration test'.",
        ),
    ]
    tool = _make_tool()

    try:
        # max_tokens=1024 covers Kimi's response pattern — even with
        # thinking disabled, it emits a paragraph of prose before the tool
        # call. Production agents inherit a generous default from the model
        # registration; hard-coding it here keeps the integration test
        # self-contained and reproducible.
        resp: ChatMessage = await model(
            messages,
            stop_sequences=["Observation:", "Calling tools:"],
            tools_to_call_from=[tool],
            max_tokens=1024,
        )
    except Exception as exc:
        print(f"  ✗ {type(exc).__name__}: {exc}")
        return {"model_name": model_name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    tool_calls = resp.tool_calls or []
    print(f"  model.model_id = {model.model_id!r}")
    print(f"  tool_calls = {len(tool_calls)}")
    if tool_calls:
        tc0 = tool_calls[0]
        print(f"  -> name={tc0.function.name} args={tc0.function.arguments!r}")
    elif resp.content:
        print(f"  plain-text content: {resp.content[:400]!r}")
    # Echo raw finish_reason when available (wrapped response stashes the
    # SDK object at resp.raw; choices[0] is the completion choice).
    try:
        finish = resp.raw.choices[0].finish_reason
        print(f"  finish_reason = {finish!r}")
    except Exception:
        pass

    downgrade_fired = model.model_id in tc._LOGGED_DOWNGRADES

    result = {
        "model_name": model_name,
        "wire_id": model.model_id,
        "tool_calls": len(tool_calls),
        "downgrade_fired": downgrade_fired,
        "ok": len(tool_calls) >= 1
              and (downgrade_fired == expect_downgrade),
    }
    if expect_downgrade and not downgrade_fired:
        print("  ✗ expected dispatch downgrade, did not fire")
    if not expect_downgrade and downgrade_fired:
        print("  ✗ unexpected dispatch downgrade fired")
    if not tool_calls:
        print("  ✗ no tool_calls in response")
    if result["ok"]:
        print("  ✓ all checks passed")
    return result


async def main() -> int:
    model_manager.init_models(use_local_proxy=False)

    # Sanity: verify all 4 matrix slugs + their langchain wrappers are registered.
    expected = [
        "mistral-small", "langchain-mistral-small",
        "or-kimi-k2.5", "langchain-or-kimi-k2.5",
        "or-qwen3.6-plus", "langchain-or-qwen3.6-plus",
        "or-gemma-4-31b-it", "langchain-or-gemma-4-31b-it",
    ]
    missing = [k for k in expected if k not in model_manager.registed_models]
    print(f"Registration smoke: Missing: {missing}")
    if missing:
        print(f"  ✗ cannot proceed — some models are unregistered")
        return 1

    results = []
    # Kimi: should NOT downgrade (not in defensive set, no matching prefix);
    # its extra_body must allow "required" to work despite Moonshot's default
    # thinking=on constraint.
    results.append(await _run_turn("or-kimi-k2.5", expect_downgrade=False))

    # Qwen: MUST downgrade (qwen/ prefix rule); retry guard carries reliability
    # if the model replies in plain text.
    results.append(await _run_turn("or-qwen3.6-plus", expect_downgrade=True))

    # Gemma: should NOT downgrade after the D5 live probe removed it from
    # the defensive set; the provider pin keeps it on DeepInfra/Together.
    results.append(await _run_turn("or-gemma-4-31b-it", expect_downgrade=False))

    print("\n=== Summary ===")
    for r in results:
        status = "✓" if r.get("ok") else "✗"
        print(f"  {status} {r}")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
