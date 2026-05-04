# Handoff: PENDING — Kimi K2.5 vision + per-model `tool_choice` hybrid dispatch

**Status:** IMPLEMENTED + LIVE-VERIFIED (2026-04-18) — **140/140** unit sweep on 2026-04-19 (`scripts/run_handoff_pytest_sweep.sh`); farm repeat + matrix smoke (**I2**) still to do.
**Session where decisions were made:** 2026-04-18 (continues from [`HANDOFF_TEST_EVAL.md`](HANDOFF_TEST_EVAL.md))
**Blocks:** GAIA test-split submission run (cannot start until live probes pass).
**Pointers back:** [`HANDOFF_INDEX.md`](../../HANDOFF_INDEX.md), [`HANDOFF_PROVIDER_MATRIX.md`](HANDOFF_PROVIDER_MATRIX.md)

---

## Post-implementation addendum (2026-04-18)

### Corrections relative to the design spec below

1. **Lookup-table key: wire id, not alias.** The spec below (§Change 2, §Execution checklist) said to key `MODELS_REJECTING_REQUIRED` + `_QWEN_AUTO_PREFIXES` on the registered aliases (`or-qwen*`, `or-gemma-4-31b-it`, `or-kimi-k2.5`). Exploration proved `_register_openrouter_models` does *not* reassign `OpenAIServerModel.model_id` after construction, so at the agent call site `self.model.model_id` holds the **wire slug** (`qwen/qwen3.6-plus`, `google/gemma-4-31b-it`, `moonshotai/kimi-k2.5`). The shipped implementation correctly keys on wire ids — named entry `google/gemma-4-31b-it`, prefix tuple `("qwen/",)`. Unit tests assert both paths.
2. **Dispatch lives in the model layer, not every agent call site.** The `tool_choice="required"` default was set only in `OpenAIServerModel._prepare_completion_kwargs` — no agent currently passes the value explicitly. D2 ("apply to all sub-agents") was satisfied by a single override inside `_prepare_completion_kwargs` that calls `pick_tool_choice(self.model_id, tool_choice)`. Every agent inherits the behaviour without code changes. The retry guard lives in `GeneralAgent._step_stream` + `ToolCallingAgent._step_stream` — the two points that inspect `response.tool_calls`.
3. **New module.** Dispatch helpers live in `src/models/tool_choice.py` (separate from `src/models/models.py`) to avoid the circular import that would occur between `openaillm.py` and `models.py`.

### Implemented changes (commit-group index)

| Commit group | Files | Status |
|-------------|-------|--------|
| 1. Kimi K2.5 — thinking off + Moonshot pin | `src/models/models.py` (OR registration loop + Kimi entry) | ✅ implemented |
| 2. Hybrid `tool_choice` dispatch + retry guard | `src/models/tool_choice.py` (new), `src/models/openaillm.py`, `src/agent/general_agent/general_agent.py`, `src/base/tool_calling_agent.py`, `tests/test_tool_choice_dispatch.py` (new, 21 tests, all green) | ✅ implemented |
| 3. Qwen swap to `or-qwen3.6-plus` (D1) | `scripts/gen_eval_configs.py`, regen'd `configs/config_gaia_c{0,2,3,4}_qwen.py` | ✅ implemented |
| 4. Gemma 4 31B 4th matrix slot (D4 + D5) | `src/models/models.py` (registration + provider pin + reasoning off), `scripts/gen_eval_configs.py`, `scripts/run_eval_matrix.sh` (ALL_MODELS + per-model concurrency cap), new `configs/config_gaia_c{0,2,3,4}_gemma.py` | ✅ implemented |

### Test-sweep status — **140/140 pass** (dra env, 2026-04-19)

Run one-file-at-a-time (pytest-multi-file collection interacts badly with the
`src.logger` stub in `tests/test_skill_seed.py` — see
[CLAUDE.md §Local dev environment](../../CLAUDE.md) for the documented
workaround). **Shortcut:** `bash scripts/run_handoff_pytest_sweep.sh` (re-execs
into `conda` env `dra` when `mmengine` is missing on default `python`).

| File | Pass | Notes |
|------|------|-------|
| `test_failover_model` | 12/12 | |
| `test_reasoning_preservation` | 12/12 | |
| `test_tier_b_tool_messages` | 4/4 | |
| `test_process_tool_calls_guard` | 8/8 | |
| `test_max_steps_yield_order` | 6/6 | |
| `test_rc2_diagnostic_hook` | 4/4 | |
| `test_tool_generator` | 12/12 | |
| `test_review_schema` | 26/26 | |
| `test_skill_registry` | 28/28 | |
| `test_skill_seed` | 6/6 | |
| `test_tool_choice_dispatch` | 22/22 | (+1 vs 2026-04-18 table — parametrized coverage) |
| **Total** | **140/140** | |

### Live probe results — all green (2026-04-18)

Ran `python scripts/live_probe_tool_choice.py` in the `dra` env against OR.

| Probe | Result | Detail |
|-------|--------|--------|
| **1. Kimi K2.5** — image + `tool_choice="required"` + `extra_body={thinking: disabled, provider.order: [Moonshot]}` | ✓ pass | `finish_reason="tool_calls"`; tool `echo` called with a 1-sentence image description. No 400 on thinking/required mismatch. (First attempt at `max_tokens=200` was token-starved — the fix itself works fine; `max_tokens=1024` cleared it.) |
| **2. Qwen3.6-Plus** — text + `tool_choice="auto"` + coercive system prompt | ✓ pass | `finish_reason="tool_calls"`; tool `echo` called with `{"message": "hello"}`. |
| **3. Gemma 4 31B** — two-step: `"required"` first, `"auto"` fallback | ✓ pass on Step 1 | HTTP 200, `finish_reason="tool_calls"`, `echo({"message": "hi"})`, **no special-token leak** in `content`. Provider pin `DeepInfra+Together` + `reasoning.enabled=false` is sufficient for clean forced tool use. |

### Follow-up edits applied after probes

- Removed `"google/gemma-4-31b-it"` from `MODELS_REJECTING_REQUIRED` in `src/models/tool_choice.py` per Probe 3's positive result. Kept the set declared (now empty) as an extensibility hook for future non-Qwen entries.
- Updated unit test `test_models_rejecting_required_is_empty_after_gemma_probe` to assert the set is empty and that Gemma resolves to `"required"`. One retry-guard test that relied on Gemma's named-entry path switched to `qwen/qwen3-coder-next` (prefix path) so the retry guard still has a deterministic auto-dispatch model to exercise.
- Updated the Gemma registration comment in `src/models/models.py` to record the probe outcome.
- `scripts/live_probe_tool_choice.py` committed alongside so future regressions can be reproduced in minutes.

### Review-response edits (2026-04-19, after integration test + code review)

- **Kimi OR routing — silent pin bug fixed** (`fix(providers): Kimi OR routing — correct Moonshot slug + reasoning-off param name`). The earlier pin `provider.order=["Moonshot"]` was silently ignored — OR's verbatim slug is `"Moonshot AI"` (with space). With `allow_fallbacks=True` (default), OR was free-routing across all 16 Kimi providers; the original live probes "succeeded" by accident. Enforced the pin (`allow_fallbacks=False`) and switched the reasoning-control key from the direct-API `thinking={"type":"disabled"}` to the OR-canonical `reasoning={"enabled": False}`. Verified via a 5-option probe 2026-04-19: only `reasoning.enabled=false` produces `finish_reason="tool_calls"` against the actual Moonshot AI provider.
- **Kimi `stop` + `tool_choice="required"` production bug** (`fix(providers): Kimi stop-strip ...`). Surfaced by `scripts/integration_test_model_stack.py` (which exercises the real `OpenAIServerModel` stack, not raw OpenAI SDK). Moonshot AI halts with `finish_reason="stop"` and empty `tool_calls` when `stop` is non-empty — even for stop strings that cannot appear in the output (verified with literal `XYZZYZZY:`). `GeneralAgent._step_stream` always passes `stop=["Observation:", "Calling tools:"]`, so without the fix **every Kimi turn in real GAIA runs would have returned no tool calls.** Added `kimi-k2.5` + `kimi-k2.5-no-thinking` to `UNSUPPORTED_STOP_MODELS` in `src/models/message_manager.py`.
- **Streaming-path retry-guard advisory.** Warn at `GeneralAgent.__init__` / `ToolCallingAgent.__init__` if `stream_outputs=True` AND the model is in the auto-dispatch set. Not a hard raise — no current matrix cell hits the combination — but a misconfiguration tripwire.
- **DashScope wire-id invariant assertion.** Fail loudly at registration if any future DashScope `model_id` adopts the `qwen/` prefix that would collide with the hybrid-dispatch rule.
- **Constants-drift test.** New `test_retry_constants_match_across_production_files` regex-parses `MAX_TOOL_RETRIES` + `_TOOL_CHOICE_RETRY_PROMPT` from both the async and sync guard files (they are duplicated to avoid a circular dep between `src/agent/` and `src/base/`) and asserts they agree. Empirically verified to catch drift by patching one file and observing the test fail.

### Methodology caveats to cite in the paper

- **`reasoning_content` is dropped on retry turns.** The retry guard echoes `ChatMessage(role="assistant", content=chat_message.content or "")` — the `reasoning_content` field (DeepSeek-reasoner, Qwen3-thinking, etc.) is not carried. For any model matched by `needs_reasoning_echo()`, the provider would 400 on the retry turn. **Current matrix is unaffected** — `or-qwen3.6-plus` runs with `enable_thinking=False` and is the only auto-dispatched model; no `*-thinking` variant is in the auto-dispatch set. This is a latent risk if the matrix is ever extended to include a thinking-mode Qwen slug.
- **LangChain wrappers (`langchain-or-*`) inherit no `extra_body`.** The browser-use tool's LangChain wrapper is registered without Kimi's reasoning-off pin or Gemma's provider pin. All current 16 configs only reference the LangChain wrapper for `auto_browser_use_tool`, which does not use `tool_choice`, so the missing `extra_body` is inert. Explicit caveat for any future config that routes a non-browser step through the LangChain wrapper.

---

## TL;DR — two approved changes, neither yet implemented

> **[SUPERSEDED for implementation detail]** The design pseudocode below
> (especially the alias-keyed ``MODELS_REJECTING_REQUIRED`` set and
> ``or-qwen*`` / ``or-gemma-4-31b-it`` literals) reflects the pre-exploration
> spec. The shipped code keys on **wire ids** (``qwen/`` prefix,
> ``google/gemma-4-31b-it`` → removed after the D5 probe passed). See the
> [post-implementation addendum](#post-implementation-addendum-2026-04-18)
> at the top for the canonical current state.

### Change 1 — Kimi swap: enable vision via `extra_body`

The matrix currently routes Kimi through `or-kimi-k2.5` → OpenRouter slug `moonshotai/kimi-k2.5` but without extra routing hints. Research dispatch on 2026-04-18 (see §Research findings below) found:

- `moonshotai/kimi-k2.5` **IS** native multimodal (text+image), 262K context, currently free tier.
- Moonshot's own docs require `tool_choice="required"` calls to have thinking **disabled** via `extra_body={"thinking":{"type":"disabled"}}`. Otherwise the API returns 400.
- Free-tier OR can silently route to non-Moonshot backends that diverge on `thinking` / sampling-lock semantics. Pinning provider order to Moonshot makes behavior reproducible.

**Approved change:** add to the OR Kimi registration so **every** caller of `or-kimi-k2.5` gets the correct routing:

```python
extra_body = {
    "thinking": {"type": "disabled"},
    "provider": {"order": ["Moonshot"]},
}
```

Applies in `src/models/models.py` where `or-kimi-k2.5` is registered (Moonshot OpenRouter block).

After this change, the matrix's 4 Kimi cells gain vision capability (critical for GAIA screenshots + image-containing questions).

### Change 2 — Per-model `tool_choice` hybrid dispatch (Option C)

The agent harness currently sends `tool_choice="required"` unconditionally. Multiple Qwen slugs on OpenRouter reject this value (404 "No endpoints found that support the provided 'tool_choice' value" for 404-class, 400 provider-rejection for others). Research also confirmed **Alibaba's own DashScope OpenAI-compat endpoint doesn't document `"required"` support** — this is an ecosystem-wide Qwen limitation, not OR-specific.

Agent (operator) picked **Option C: hybrid** over universal-auto or named-dispatcher — the smallest change that future-proofs without retraining every agent.

**Approved design:**

1. Maintain a lookup table `MODELS_REJECTING_REQUIRED` in `src/models/models.py` listing model aliases (not slugs) known to reject `tool_choice="required"`.
2. Add a helper `pick_tool_choice(model_id, default="required")` that returns `"auto"` when the model is in that set, else `default`.
3. In every call site that currently passes `tool_choice="required"` (primarily `src/base/tool_calling_agent.py`; secondarily any sub-agent that overrides), use `pick_tool_choice(self.model.model_id)` instead of the hard-coded value.
4. Wrap the model call with a **retry guard**: if `tool_choice` resolves to `"auto"` and the response has zero `tool_calls`, inject a corrective user message ("You replied in plain text. Every step REQUIRES a tool call. Choose one now; if you intended the final answer, call `final_answer_tool`.") and retry up to `MAX_TOOL_RETRIES=2`. On final failure, fall through the existing error-handling path.

Initial lookup table population (from research + live probes):

```python
MODELS_REJECTING_REQUIRED = {
    # 404-class — confirmed zero OR providers accept tool_choice="required"
    "or-qwen3.6-plus",
    "or-qwen3.5-flash-02-23",
    "or-qwen3.5-plus-02-15",
    "or-qwen3-vl-32b-instruct",
    "or-qwen2.5-vl-72b-instruct",
    # 400-class — provider bug on AtlasCloud; note: can alternatively be
    # rescued by extra_body provider pinning (["deepinfra","parasail","baseten"])
    "or-qwen3.5-27b",
    "or-qwen3.5-122b-a10b",
}
```

The table is **model-alias-keyed**, matching how `model.model_id` is set after registration (see `_register_openrouter_models` in `models.py` — the alias survives past the constructor where `OpenAIServerModel.model_id` is reassigned to the real wire id). **Implementation note:** double-check that the alias (`or-qwen...`) rather than the wire id (`qwen/qwen...`) is the right key; if the wire id is what `model.model_id` holds at the call site, the table must key on `qwen/...` instead. Confirm with a one-line print before wiring.

---

## Context: research that led to these decisions

### Research dispatch 2026-04-18 (sub-agents)

Three parallel sub-agents investigated claims that conflicted with OpenRouter's `supported_parameters` metadata:

**Kimi K2.5** ([Moonshot docs](https://platform.kimi.ai/docs/guide/kimi-k2-5-quickstart), OR model page, GitHub issues `zed-industries/zed#37032`, `opencode#11541`, Together AI quickstart):
- Native multimodal via 15T-token continued-pretraining over K2-Base with vision+text.
- Images must be base64-encoded data URLs ≤100 MB (hard constraint); URLs rejected.
- `tool_choice="required"` is explicitly rejected when `thinking.type=="enabled"` — docs state: *"When thinking is enabled, tool_choice can only be 'auto' or 'none'."*
- Sampling locks: temp=0.6 / top_p=0.95 in non-thinking mode; temp=1.0 / top_p=0.95 in thinking mode. Overrides 400. Our `MessageManager.get_clean_completion_kwargs` (HANDOFF_PROVIDER_MATRIX.md `7632470`) already strips these — apply still holds.
- Free-tier OR can silently fall back between sub-providers; pin via `provider.order=["Moonshot"]` for deterministic behaviour.

**MiniMax M2.x** ([MiniMax docs](https://platform.minimax.io/docs/release-notes/models), OR pages, HuggingFace `MiniMaxAI/MiniMax-M2`):
- **All M2 / M2.1 / M2.5 / M2.7 variants are text-only.** No hidden VL variant exists on GitHub, HuggingFace, MiniMax platform, or OR. OR metadata is accurate.
- Only `minimax-01` has vision (via the older VL-01 component) but OR metadata omits `tools` from supported_parameters — function calling is not reliably exposed.
- Verdict: MiniMax **cannot** occupy a VL+tool matrix slot. Out of scope for this project.

**Qwen `tool_choice="required"` failures** ([OR provider routing docs](https://openrouter.ai/docs/guides/routing/provider-selection), [DashScope tool_choice docs](https://www.alibabacloud.com/help/en/model-studio/qwen-function-calling), `vllm-project/vllm#19051`, `zed-industries/zed#36094`, `musistudio/claude-code-router#409`):
- **Root cause: backend-provider-level, not model-level.** OR's routing layer filters out providers that don't advertise support for the specific `tool_choice` VALUE being requested. When all candidate providers for a model reject `"required"`, the request 404s with "No endpoints found".
- 400-class failures (AtlasCloud on `qwen3.5-27b`, `qwen3.5-122b-a10b`; Cerebras on `qwen3-coder`): provider accepted routing but upstream inference stack rejected at runtime. Fixable via `provider.only=["deepinfra","parasail","baseten"] + require_parameters=true`.
- 404-class failures: NO provider accepts `"required"`. Provider pin cannot rescue these. Only fixable by changing the call itself (named tool_choice, or `"auto"` with prompt coercion).
- Even DashScope's native OpenAI-compat endpoint documents only `"auto"`, `"none"`, named-function — not `"required"`. So this is a **broad Qwen-ecosystem limitation**.

Full reports exist in the 2026-04-18 research dispatch conversation — preserved here via reference. Agent IDs: `ad8460f08763c6400` (Kimi), `a432b1e9f1664df39` (MiniMax), `a1e23337daed04f73` (Qwen).

### Options weighed for Option C selection

Four options for handling the Qwen `"required"` ecosystem issue were considered:

| # | Approach | Code change | Compatibility | Reliability | Token overhead |
|---|---|---|---|---|---|
| 1 | Keep `"required"`, pick only working Qwens | 0 | narrow | 100% | 0 |
| 2 | Provider pinning in `extra_body` | 0 (config) | unblocks 400-class only | 100% | 0 |
| **3** | **Hybrid dispatch + retry guard (selected)** | **small (one file)** | **works everywhere** | **95–99% first try; retry catches rest** | **3–8% only on the hybrid-fallback models** |
| 4 | Universal `"auto"` + prompt + retry | small | works everywhere | 95–99% + retry | 3–8% everywhere (even well-behaved models) |
| 5 | Named-dispatcher-tool refactor | large (all agents) | works everywhere | 100% | 0 (but bigger prompts) |

Option 3 wins on risk/reward for a paper-submission run. Option 5 is reserved for if we ever need to ship dozens of new models without individual testing.

---

## [SUPERSEDED — historical spec only] Code change specification

> **The shipped code does NOT match the spec in this section.** It was written
> before the codebase exploration on 2026-04-18 that proved
> ``OpenAIServerModel.model_id`` holds the **wire id** at the agent call site,
> not the registration alias. The shipped implementation therefore keys the
> lookup on wire ids (``qwen/`` prefix, named entries like
> ``google/gemma-4-31b-it``), not aliases (``or-qwen*``, ``or-gemma-4-31b-it``,
> ``or-kimi-k2.5``). The spec below is kept as historical context for the
> research trail — do NOT use it as an implementation reference.
>
> Canonical post-implementation summary lives in the
> [post-implementation addendum](#post-implementation-addendum-2026-04-18)
> at the top of this document.

### Files to touch

| File | Change |
|------|--------|
| `src/models/models.py` | (a) Add `extra_body` to the OR Kimi (`or-kimi-k2.5`) registration entry. (b) Define `MODELS_REJECTING_REQUIRED` constant near the top. (c) Define `pick_tool_choice(model_id, default)` helper. |
| `src/base/tool_calling_agent.py` | At the `tool_choice="required"` call site, replace with `pick_tool_choice(self.model.model_id)`. Wrap the call in a retry loop when the resolved value is `"auto"`. |
| `src/agent/adaptive_planning_agent/adaptive_planning_agent.py` | Same treatment if it independently passes `tool_choice`. |
| Other sub-agent classes (`general_agent`, `deep_analyzer_agent`, `deep_researcher_agent`, `browser_use_agent`) | Grep for `tool_choice` overrides and apply the same pattern. |
| `src/base/async_multistep_agent.py` | Same (likely the primary call site). |
| `tests/test_tool_choice_dispatch.py` (NEW) | Unit tests for `pick_tool_choice` table behaviour + retry guard (mock model that alternates plain-text / tool_call responses). |

### Implementation sketch — `pick_tool_choice`

```python
# In src/models/models.py

MODELS_REJECTING_REQUIRED: set[str] = {
    # 404-class — no OR provider accepts tool_choice="required"
    "or-qwen3.6-plus",
    "or-qwen3.5-flash-02-23",
    "or-qwen3.5-plus-02-15",
    "or-qwen3-vl-32b-instruct",
    "or-qwen2.5-vl-72b-instruct",
    # 400-class — provider bug (could alt-rescue via provider pin)
    "or-qwen3.5-27b",
    "or-qwen3.5-122b-a10b",
    # Extend as new models surface the same issue.
}


def pick_tool_choice(model_id: str, default: str = "required") -> str:
    """Resolve the tool_choice value for a given model alias.

    Agents send ``tool_choice="required"`` by default so every turn produces a
    tool call. Some OpenRouter-hosted models (notably the Qwen3/3.5 family
    routed through AtlasCloud / Cerebras / older 2025 providers) reject that
    value with 404 / 400 — in which case we downgrade to ``"auto"`` and let
    the caller's retry-guard coerce plain-text responses back into tool calls.
    """
    if model_id in MODELS_REJECTING_REQUIRED:
        return "auto"
    return default
```

### Implementation sketch — retry guard

```python
# In src/base/tool_calling_agent.py (and mirrored in other call sites)

MAX_TOOL_RETRIES = 2

tool_choice = pick_tool_choice(self.model.model_id)
response = await self.model(messages=history, tools=tools, tool_choice=tool_choice)

if tool_choice == "auto":
    retries = 0
    while response.tool_calls is None or len(response.tool_calls) == 0:
        if retries >= MAX_TOOL_RETRIES:
            break  # fall through to existing error-handling path
        history = history + [
            {"role": "assistant", "content": response.content or ""},
            {"role": "user", "content": (
                "You replied in plain text, but every step REQUIRES a tool call. "
                "Choose exactly one tool from the list above and call it now. "
                "If you intended to give a final answer, call `final_answer_tool`."
            )},
        ]
        response = await self.model(messages=history, tools=tools, tool_choice="auto")
        retries += 1
```

### Logging expectations

Emit an INFO-level log line the first time a model goes through the `"auto"` path in a run, so the SLURM log clearly shows which cells were affected:

```
[tool_choice] or-qwen3.5-27b -> auto (model rejects "required")
```

And a WARNING every time the retry guard fires (after the first attempt):

```
[tool_choice retry 1/2] or-qwen3.5-27b replied in plain text; injecting corrective.
```

### Kimi `extra_body` change (specific)

Locate the OR Kimi row in `_register_openrouter_models` — model `moonshotai/kimi-k2` / `or-kimi-k2.5` etc. Add:

```python
extra_body = {
    "thinking": {"type": "disabled"},
    "provider": {"order": ["Moonshot"]},
}
```

and pass it through `OpenAIServerModel(extra_body=extra_body, ...)`. The plumbing is already there (HANDOFF_PROVIDER_MATRIX `7632470`); we just need to populate the value for this slug.

---

## Validation plan (after implementation)

1. **Unit tests:**
   - `tests/test_tool_choice_dispatch.py` covers `pick_tool_choice` map semantics + retry guard (mock model returns text first, tool_call second).
   - Existing `tests/test_failover_model.py`, `tests/test_reasoning_preservation.py`, etc. must all stay green (run the 10-file sweep documented in `HANDOFF_TEST_EVAL.md` §I0 / pre-flight).

2. **Live verification — Kimi vision:**
   ```python
   # In a throwaway probe script, send a base64 image to or-kimi-k2.5 with
   # tool_choice="required" AFTER the extra_body change. Expect:
   #  - no 400 about thinking
   #  - response.choices[0].finish_reason == "tool_calls"
   ```

3. **Live verification — Qwen `"auto"` dispatch:**
   ```python
   # Probe qwen/qwen3.5-27b with provider pin + tool_choice="auto" + a
   # strong "must call a tool" prompt. Expect a tool_calls response.
   ```

4. **Smoke matrix re-run:**
   - One cell per model: `sbatch run_matrix_slurm.sh smoke <model> c0` (or the `scripts/run_eval_matrix.sh smoke ...` path if running locally).
   - Confirm at least one `[tool_choice]` INFO log line for Qwen cells if the Qwen slug is swapped back to `qwen3.5-27b`.
   - Confirm `dra.jsonl` has at least one row with `prediction != null` for each model.

5. **Regen configs if needed:**
   - If the decision is ALSO to swap Qwen back to `qwen/qwen3.5-27b` (to exercise the new dispatch path), update `scripts/gen_eval_configs.py` MODELS row and run `python scripts/gen_eval_configs.py`.
   - Otherwise, the current config (`qwen3-next-80b-a3b-instruct`, which does support `"required"`) stays.

---

## Decisions confirmed by operator (2026-04-18, post-checkpoint)

### (D1) Qwen slug → **`or-qwen3.6-plus`**

Operator chose Qwen3.6-Plus on OpenRouter (one of the 404-class Qwens per earlier live probe). This makes Change 2 (hybrid `tool_choice` dispatch) **load-bearing** — without it, every Qwen cell will 404 on the first call. `or-qwen3.6-plus` has vision (text+image+video per OR metadata), 1M context, $0.325/$1.95 per M in/out.

The `scripts/gen_eval_configs.py` `MODELS` table row for qwen needs reverting from `or-qwen3-next-80b-a3b-instruct` back to `or-qwen3.6-plus`, and the 4 Qwen configs regenerated.

### (D2) Retry guard scope → **apply to all sub-agents**

`pick_tool_choice(model.model_id)` + retry guard wraps every model call that previously sent `tool_choice="required"`. Grep inventory of call sites to patch:

```
src/base/tool_calling_agent.py      # primary
src/base/async_multistep_agent.py   # primary async variant
src/agent/general_agent/*.py
src/agent/deep_analyzer_agent/*.py
src/agent/deep_researcher_agent/*.py
src/agent/browser_use_agent/*.py    # (relies on LangChain wrapper — confirm which call site passes tool_choice)
src/agent/planning_agent/*.py
src/agent/adaptive_planning_agent/*.py
```

### (D3) Qwen family blanket `"auto"` → **all `or-qwen*` / `qwen/*` slugs**

Operator overrode my narrower recommendation (only the known-failing 400/404 slugs). Reasoning: OR's backend routing for Qwen is inconsistent across the whole family; a slug that works today can silently start failing tomorrow when OR shifts providers. Safest policy is treating the whole Qwen family uniformly as `"auto"`.

**Updated helper:**

```python
# src/models/models.py

MODELS_REJECTING_REQUIRED: set[str] = {
    # Named entries for models outside the Qwen prefix rule.
    # D5 (2026-04-18): Gemma 4 31B on OR — defensive until live-probe confirms
    # tool_choice="required" works. If Probe 1 succeeds, delete this entry.
    "or-gemma-4-31b-it",
}

_QWEN_AUTO_PREFIXES: tuple[str, ...] = (
    "or-qwen",      # all OpenRouter Qwen aliases we register
    "qwen/",        # raw OR wire ids, in case a caller ever uses one
    "langchain-or-qwen",  # LangChain wrappers for OR Qwens
)

def pick_tool_choice(model_id: str, default: str = "required") -> str:
    """Resolve tool_choice for a given model alias.

    Qwen ecosystem (whole family) → "auto" + prompt coercion (D3).
    Other models in MODELS_REJECTING_REQUIRED → "auto".
    Everyone else → default ("required").
    """
    if model_id in MODELS_REJECTING_REQUIRED:
        return "auto"
    if model_id.startswith(_QWEN_AUTO_PREFIXES):
        return "auto"
    return default
```

One consequence: the retry-guard overhead (potentially +3–8% tokens per Qwen turn when it text-replies) applies to every Qwen cell uniformly. Matches the methodological principle of applying the same harness rules across all cells of the same model.

---

## Addendum — 2026-04-18 late-session research dispatch (post-D1/D2/D3)

### Kimi K2.5 "self-directed agent swarm" paradigm — does it confound the ablation?

Research (sub-agent `ac9cfb359a9cdd9db`, consulting Moonshot platform docs, Kimi K2.5 tech report, Trilogy AI / DataCamp / Heroku Inference writeups, OpenRouter model page):

- **No server-side orchestration.** Both `api.moonshot.ai/v1/chat/completions` and OpenRouter's `moonshotai/kimi-k2.5` are plain single-step OpenAI-compat endpoints — one forward pass per request, tool_calls returned to the caller, caller executes. No hidden sub-agent loop.
- **"Agent Swarm" is a client-side harness.** The "100 sub-agents / 1,500 tool calls / 4.5× speedup" marketing number is produced by Moonshot's *Kimi Code CLI* runner on top of the base API, not by the API itself. Same category as our DRA meta-agent layer — a separate orchestrator that makes many model calls.
- **The "agentic" capability is weights-level**, not runtime. K2.5 was post-trained with Parallel-Agent RL (PARL) on agent trajectories. This is the same kind of post-training that Claude Sonnet 4, GPT-4.1, and Gemini 2.5 Pro all receive — capability lives in the weights.

**Verdict: suitable for the ablation.** At the chat-completions API layer K2.5 is exactly the "plain LLM" our C0/C1/C2/C3 experiment assumes. What we inherit is a model whose weights bias toward agent-flavored tool emission — but every modern frontier model carries that same bias, and it affects the *absolute* score level uniformly across all 4 conditions, not the *delta* that the ablation actually measures.

**Methodology caveat to document in the paper:**

> Kimi K2.5 was post-trained with Parallel-Agent RL (PARL; [Moonshot tech report, arXiv 2602.02276](https://arxiv.org/abs/2602.02276)) on parallel-decomposition agent trajectories. This may produce native parallelism in tool-call patterns even in the baseline C0 condition. Because the `modify_subagent` action space in C1–C3 explicitly supports parallel sub-agent invocation, this bias is absorbed uniformly across conditions rather than favouring any single condition. Readers interpreting absolute-score numbers across models should treat PARL-trained models (Kimi K2.5, possibly Qwen3.x if they are similarly RL-trained) differently from non-PARL controls.

**Do NOT use:**
- Kimi.com Agent Swarm beta UI (consumer product)
- Kimi Code CLI harness (would confound by *adding* a second meta-agent layer beneath ours)

### Gemma 4 (2026-04-02 release) — suitable for our matrix?

Research (sub-agent `ab6bb23a4c5e769a1`, consulting DeepMind model card, Google developers blog, HF Welcome-Gemma-4 post, OpenRouter model pages, llama.cpp tool-calling fix thread, τ2-bench community runs):

**Architecture & pricing:**

| Variant | Total | Active | Ctx | Modalities | OR price (in/out /M) |
|---------|-------|--------|-----|------------|---------------------|
| `google/gemma-4-31b-it` | 30.7B dense | 30.7B | 256K | text, image, video | $0.13 / $0.38 |
| `google/gemma-4-26b-a4b-it` | 25.2B MoE | ~3.8B | 256K | text, image, video | $0.08 / $0.35 |
| `:free` variants | — | — | 262K | — | $0 (free tier) |

License: Apache 2.0 (fully open, commercial use — upgrade from Gemma license on G1/G2/G3).

**Tool-calling reliability:** Gemma 3 27B scored 6.6 on τ2-bench (essentially broken). **Gemma 4 31B scores 86.4 on τ2-bench** — >10× improvement. Native tool-call special tokens in the tokenizer (not prompt-template hacks). OR metadata claims `tools` + verified by community smoke tests in llama.cpp / transformers. `tool_choice="required"` not explicitly confirmed on OR — budget one live smoke probe before committing.

**Published benchmarks (Apr 2026):**

| Benchmark | Gemma-4-31B | Gemma-4-26B-A4B | Gemma-3-27B | Notes |
|---|---|---|---|---|
| MMLU Pro | 85.2 | 82.6 | 67.6 | |
| GPQA Diamond | 84.3 | 82.3 | 42.4 | 26B-A4B beats gpt-oss-120B here with 4B active |
| AIME 2026 | 89.2 | 88.3 | 20.8 | |
| LiveCodeBench v6 | 80.0 | 77.1 | — | |
| MMMU Pro (vision) | 76.9 | 73.8 | 49.7 | |
| **τ2-bench (agentic)** | **86.4** | — | 6.6 | Closest published proxy for GAIA-style multi-step tool use |
| LMArena Elo | 1452 | 1441 | — | |

No published GAIA number — we'd be among the first to publish one.

**Server-side agent behavior?** None. Gemma 4 is a plain chat-completions LLM. "Configurable thinking mode" is inline thinking tokens within a single forward pass (same as Gemini 2.5 / Qwen3), not a sub-agent loop. Clean for ablation.

**Verdict: suitable, recommended as a 4th matrix slot** rather than replacing an existing model. Rationale:

- Adds Google provenance (currently matrix has Mistral / Moonshot / Alibaba — no US frontier lab without API lock-in)
- **Only dense frontier model in the matrix** if Gemma-4-31B is picked — methodological counterpoint to Kimi K2.5 MoE and Qwen3.x MoE variants
- Apache 2.0 + open weights give a clean reproducibility story for the paper
- Confirmed strong tool-use (τ2-bench 86.4 vs Gemma 3's 6.6 — massive generation leap; native tool tokens, not hacks)
- Budget-compatible: 31B at $0.13/$0.38 is ~13% cheaper input, ~35% cheaper output than Mistral-Small
- No hidden server-side agent

**Matrix expansion cost (if added):** 4 models × 4 conditions = **16 cells** instead of 12. Smoke (default **3 Q** each via `run_eval_matrix.sh`) = **48 Q** (set `LIMIT=5` for 80 Q). Full test split (~300 Q each) = ~4,800 Q. Current budget estimate for full submission run: **+33% on top of the current 12-cell estimate** (+$10–$35).

**Caveats:**
- `tool_choice="required"` support unconfirmed — treat Gemma 4 with the hybrid dispatch defensively (add to `MODELS_REJECTING_REQUIRED` set until a live probe confirms it works).
- Modalities are text+image+video but not audio — GAIA has some audio questions (small fraction); would fall through to `deep_analyzer_tool` OCR/audio handlers.

---

## Updated matrix proposal (pending operator confirmation)

| Slot | `model_id` | VL | tool_choice handling | Cost (in/out /M) |
|------|-----------|-----|-----------------------|-------------------|
| Mistral | `mistral-small` (native) | ✅ | `"required"` works | $0.15 / $0.60 |
| Kimi | `or-kimi-k2.5` (OR) | ✅ after Change 1 | `"required"` works after Change 1 (thinking disabled) | free tier currently |
| Qwen | **`or-qwen3.6-plus`** (OR) — D1 | ✅ (per OR metadata) | **hybrid dispatch → "auto"** after Change 2 (D3) | $0.325 / $1.95 |
| **Gemma 4** (D4 confirmed) | **`or-gemma-4-31b-it`** (OR, paid; `:free` excluded) | ✅ (text+image+video, no audio) | **hybrid dispatch → "auto" defensively (D5)** — remove from set after smoke probe succeeds; provider pin DeepInfra+Together, `reasoning.enabled=false`, concurrency ≤ 4 | $0.13 / $0.38 |

---

## Decisions confirmed (D4 / D5 — 2026-04-18 late-session)

### (D4) Add Gemma-4-31B-it as a 4th matrix slot? → **YES**

Operator confirmed **yes**. Matrix becomes 4 models × 4 conditions = **16 cells**. Rationale: dense-vs-MoE counterpoint vs Kimi/Qwen/Mistral MoE variants, Google provenance (no other US frontier lab in matrix without API lock-in), Apache 2.0 open weights for reproducibility, τ2-bench 86.4 for strong tool-use signal, and paper novelty (first published GAIA number for Gemma 4). Cost delta: +$10–35 on full test-split run vs current 12-cell budget.

**Picked slug: `google/gemma-4-31b-it`** (paid, NOT `:free` — see D5 research below for why `:free` is excluded).

### (D5) Blanket-auto treatment for Gemma 4? → **YES (defensive), paired with smoke-probe + provider pin**

Research dispatch 2026-04-18 (sub-agent `a0f950f0e496c8030`, consulting OpenRouter docs, Artificial Analysis provider pages, HuggingFace Welcome-Gemma-4 blog, Google AI function-calling docs, vLLM PRs #38847 / #39043 / #39392 / #39468, llama.cpp PR #21326, Zed issue #36094) returned **INCONCLUSIVE leaning YES for paid slugs, NO for `:free`**. No primary source explicitly confirms or denies `tool_choice="required"` works end-to-end on OR Gemma 4 at this time. Circumstantial evidence is favorable (7/8 paid providers advertise `tools`; all vLLM-backed providers inherit generic `"required"` support via guided decoding) but the Gemma 4 tool-calling parsers were actively being patched through 2026-04-11 — close enough to our 2026-04-18 run to warrant defensive treatment rather than optimism.

**Approved implementation for Gemma 4:**

1. **Add `or-gemma-4-31b-it` to `MODELS_REJECTING_REQUIRED`** at registration time (not to the Qwen prefix rule — keep the two mechanisms separate so we can audit them independently). Retry guard carries the reliability. Removal is cheap once a live probe succeeds.
2. **Provider pin via `extra_body`** at registration:
   ```python
   extra_body = {
       "provider": {"order": ["DeepInfra", "Together"], "allow_fallbacks": False},
       "reasoning": {"enabled": False},  # avoid thinking-mode contamination of tool output (vLLM #39043)
   }
   ```
   Both DeepInfra and Together are vLLM-backed and are most likely to run the latest gemma4 parser. Novita is excluded (doesn't advertise `tools`). Google AI Studio is excluded (historical non-support of `"required"`).
3. **Exclude `:free` variants from the eval.** Google AI Studio routing is the worst case for forced tool use.
4. **Smoke-probe before full run** (gate to the full test-split submission):
   - Probe 1: `tool_choice="required"` via OR `google/gemma-4-31b-it` + the pinned `extra_body` above. Expected failure mode if un-supported: 404 "No endpoints found that support the provided 'tool_choice' value" — identical to the Qwen 404 class.
   - Probe 2: `tool_choice="auto"` with explicit "must call a tool" prompt. Expected: `tool_calls` in response.
   - If Probe 1 succeeds → remove `or-gemma-4-31b-it` from `MODELS_REJECTING_REQUIRED`, keep the provider pin.
   - If Probe 1 fails → keep in the set; retry-guard handles the re-prompt.
5. **Concurrency cap.** vLLM #39392 documents a Gemma 4 parser bug under parallel load that emits all-pad tokens non-deterministically. Set `max_concurrency ≤ 4` for Gemma 4 cells in `scripts/run_eval_matrix.sh` (trivial: pass through to `examples/run_gaia.py --concurrency 4` for the Gemma stream only, leave Mistral/Kimi/Qwen at current default).
6. **Chat-template sanity check.** Gemma 4 tool-call tokens (`<|tool_call>` / `<tool_call|>`) only fire with Google's updated Jinja template (llama.cpp merge 2026-04-11). A provider on a stale template returns tool calls as plain text inside `content` — our parser sees zero `tool_calls` even when `tool_choice="required"` "succeeded". The provider pin to DeepInfra+Together mitigates this; verify once during Probe 1 by inspecting raw `response.choices[0].message` content.

**Methodology caveat to add to the paper** (cross-ref the Kimi PARL caveat from §Addendum above):

> Gemma 4 cells were run with the OpenRouter provider pool restricted to DeepInfra and Together (both vLLM-backed) for deterministic behaviour. Reasoning mode was disabled via `extra_body={"reasoning":{"enabled":false}}` to prevent thinking-channel contamination of tool-call output (vLLM issue #39043, open as of 2026-04-18). Concurrency was capped at 4 to avoid the `<pad>` token parser bug documented in vLLM issue #39392.

### D5 research bottom line (for the paper appendix)

| Aspect | Finding | Source |
|--------|---------|--------|
| Providers serving `google/gemma-4-31b-it` on OR | 8 total: Google AI Studio, DeepInfra, Together, Parasail, Clarifai, GMI (FP8), W&B, Novita | [Artificial Analysis — Gemma 4 31B providers](https://artificialanalysis.ai/models/gemma-4-31b/providers) |
| Providers advertising `tools` | 7/8 (all except Novita) | same |
| Native tool-call special tokens | `<|tool_call>` / `<tool_call|>` added at Gemma 4 | [HF Welcome-Gemma-4](https://huggingface.co/blog/gemma4) |
| `tool_choice="required"` documented? | No explicit yes/no in Google's function-calling doc | [Google AI — Gemma 4 function calling](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4) |
| vLLM support | Generic `"required"` via guided decoding; Gemma 4 parser added in #38847 | [vLLM tool-calling docs](https://docs.vllm.ai/en/latest/features/tool_calling/) + [PR #38847](https://github.com/vllm-project/vllm/pull/38847) |
| Known bugs ≤ 2026-04-18 | Parser crashes (vLLM #39043), all-pad under concurrency (#39392), format mis-parses (#39468); llama.cpp template fix #21326 merged 2026-04-11 | per-link |
| OR provider filtering by `tool_choice` VALUE | Confirmed: 404s when no candidate provider advertises the value (same mechanism as Qwen failures) | [OR tool-calling guide](https://openrouter.ai/docs/guides/features/tool-calling), [Zed #36094](https://github.com/zed-industries/zed/issues/36094) |

---

## Execution checklist for the next session

Operator decisions confirmed: **D1 = `or-qwen3.6-plus`**, **D2 = apply to all sub-agents**, **D3 = blanket `"auto"` for every `or-qwen*` / `qwen/*` / `langchain-or-qwen*` slug**, **D4 = YES (add `or-gemma-4-31b-it` as 4th matrix slot)**, **D5 = YES defensively (add `or-gemma-4-31b-it` to `MODELS_REJECTING_REQUIRED` with provider pin + concurrency cap + smoke probe gate)**.

- [ ] `git pull origin main` → HEAD includes `c233881` or later
- [ ] Re-read this handoff + `HANDOFF_PROVIDER_MATRIX.md` + `HANDOFF_TEST_EVAL.md`
- [ ] **Change 1 — Kimi extra_body** (`src/models/models.py`, OR Kimi registration): add `extra_body={"thinking":{"type":"disabled"}, "provider":{"order":["Moonshot"]}}`.
- [ ] **Change 2 — hybrid `tool_choice` dispatch** (`src/models/models.py` + all call sites from D2 grep list): add `MODELS_REJECTING_REQUIRED` set + `_QWEN_AUTO_PREFIXES` tuple + `pick_tool_choice` helper; replace every hard-coded `tool_choice="required"` with `pick_tool_choice(self.model.model_id)`.
- [ ] **Change 2b — retry guard** in each tool_calling call site: re-prompt up to 2× when `"auto"`-resolved response has no `tool_calls`. INFO-log first `"auto"` resolution per (run, model); WARNING-log each retry.
- [ ] **Config regen (D1):** edit `scripts/gen_eval_configs.py` `MODELS` qwen row from `or-qwen3-next-80b-a3b-instruct` → `or-qwen3.6-plus` (+ LangChain wrapper alias `langchain-or-qwen3.6-plus`); run `python scripts/gen_eval_configs.py`.
- [ ] **(D4) Gemma 4 slot — registration + config:**
  - Register `or-gemma-4-31b-it` in the `_register_openrouter_models` block of `src/models/models.py` with:
    ```python
    extra_body = {
        "provider": {"order": ["DeepInfra", "Together"], "allow_fallbacks": False},
        "reasoning": {"enabled": False},
    }
    ```
  - Register LangChain wrapper alias `langchain-or-gemma-4-31b-it` mirroring existing LangChain wrapper pattern.
  - Extend `MODELS` in `scripts/gen_eval_configs.py` with `("gemma", "or-gemma-4-31b-it", "langchain-or-gemma-4-31b-it", ...)`; regen the 4 new `config_gaia_c{0,2,3,4}_gemma.py` files (`python scripts/gen_eval_configs.py`).
  - Extend `ALL_MODELS` in `scripts/run_eval_matrix.sh` to include `gemma`.
  - Cap Gemma-stream concurrency at 4 (vLLM #39392 parser-under-concurrency workaround): pass `--concurrency 4` for the Gemma stream only; leave Mistral/Kimi/Qwen at the current default.
- [ ] **(D5) Gemma 4 tool_choice defensive entry:** add `"or-gemma-4-31b-it"` to `MODELS_REJECTING_REQUIRED` (alongside any other named entries, distinct from the Qwen prefix rule). Retry guard carries reliability until the smoke probe succeeds.
- [ ] Add `tests/test_tool_choice_dispatch.py` — unit tests for `pick_tool_choice` table + Qwen prefix rule + Gemma named entry + retry guard (mock model alternating text/tool_call).
- [ ] Run the 10-file pytest sweep — must stay 118/118 green + new tests.
- [ ] **Live probes (small budget, one-shot each):**
  - Kimi: base64 image + `tool_choice="required"` via OR `moonshotai/kimi-k2.5` with new extra_body. Expect no 400, `finish_reason="tool_calls"`.
  - Qwen3.6-Plus: text + `tool_choice="auto"` + strong "must call a tool" prompt. Expect `tool_calls` in response (not plain text).
  - Gemma 4 — **two-step probe**:
    1. `tool_choice="required"` against `or-gemma-4-31b-it` with the pinned `extra_body`. If 200 + `finish_reason="tool_calls"` → remove `or-gemma-4-31b-it` from `MODELS_REJECTING_REQUIRED`; keep provider pin.
    2. If step 1 returns 404 "No endpoints found…" or 400 → keep `or-gemma-4-31b-it` in `MODELS_REJECTING_REQUIRED`; run step 2 with `tool_choice="auto"` + strong tool-required prompt; confirm `tool_calls` present.
    3. In either case, inspect raw `response.choices[0].message` content to confirm chat-template fires the tool-call special tokens rather than emitting them as plain text.
- [ ] Commit in logical groups:
  1. `fix(providers): Kimi K2.5 — thinking-disabled + Moonshot provider-pin via extra_body`
  2. `feat(tool_choice): hybrid dispatch with retry guard for Qwen family + named set`
  3. `fix(providers): switch Qwen matrix back to or-qwen3.6-plus (D1 — now unblocked by Change 2)`
  4. `feat(matrix): add Gemma-4-31B-it as 4th matrix slot (D4 + D5 defensive auto + provider pin)`
- [ ] Push to `origin/main`
- [ ] Annotate this handoff's "To Do" checklist as done, update `HANDOFF_INDEX.md` row #9 status to "Validated → Completed" once the live probes pass.
- [ ] Resume the main execution protocol from `HANDOFF_TEST_EVAL.md` §I0 pre-flight (integration track)

---

## Session state at checkpoint (2026-04-18)

### Commits on `origin/main` relevant to this handoff

```
2398b0b  docs(handoff): HANDOFF_TEST_EVAL.md + index row #8
bdb1ed0  fix(providers): swap Qwen matrix to or-qwen3-next-80b-a3b-instruct
f5a8a64  docs(handoffs): move per-topic handoff docs to docs/handoffs/
6f5ddd1  feat(slurm): run_matrix_slurm.sh SBATCH wrapper
c2fd507  merge(main): integrate origin/main 446051c
7f5de4f  docs(handoff): annotate #1–#7 with 2026-04-18 code-validation outcomes
4162bcc  fix(mcp): reject unclosed-fence scripts
0e7903c  feat(eval): scripts/validate_handoffs.sh
905a1fa  feat(browser): configurable inner max_steps
7ee9ae1  fix(runtime): MCP + Tool.from_code + LogLevel + GAIA filters
a98da9a  fix(providers): Qwen thinking-mode + Kimi→OpenRouter + pretty_text
```

### Current matrix configs (to be updated per D1)

| Slot | Current `model_id` | Vision | tool_choice="required" |
|------|---------------------|--------|------------------------|
| Mistral | `mistral-small` (native) | ✅ | ✅ works |
| Kimi | `or-kimi-k2.5` (OR) | ❌ (until Change 1 lands) | ⚠️ needs thinking off (Change 1) |
| Qwen | `or-qwen3-next-80b-a3b-instruct` (OR direct) | ❌ text-only | ✅ works |

### Test suite baseline

118 passed / 0 failed across:
`test_failover_model`, `test_reasoning_preservation`, `test_tier_b_tool_messages`, `test_process_tool_calls_guard`, `test_max_steps_yield_order`, `test_rc2_diagnostic_hook`, `test_tool_generator`, `test_review_schema`, `test_skill_registry`, `test_skill_seed`.

`Missing: []` on registration smoke for `mistral-small`, `or-kimi-k2.5`, `or-qwen3-next-80b-a3b-instruct`, `langchain-or-qwen3-next-80b-a3b-instruct`.

All 12 configs load + `pretty_text` clean.

### Still-unpushed `run_matrix_slurm.sh` assumption

SBATCH wrapper requests CPU-only (no GPU) at 24h wall-clock. Matches the API-only matrix. Verified in earlier commit (`6f5ddd1`).

### Environment assumptions

- HKU CS Phase-3 gateway: `gpu3gate1.cs.hku.hk` via HKUVPN
- Project path on farm: `/userhome/cs2/ambr0se/DeepResearchMetaAgent`
- Conda env name: `dra` (must exist; `make install-requirements` path if not)
- `.env` keys populated: `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `FIRECRAWL_API_KEY`, `HF_TOKEN`. `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` / `MOONSHOT_API_KEY` may be placeholders.
