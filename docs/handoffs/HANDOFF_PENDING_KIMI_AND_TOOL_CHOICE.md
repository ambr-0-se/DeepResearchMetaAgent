# Handoff: PENDING — Kimi K2.5 vision + per-model `tool_choice` hybrid dispatch

**Status:** Drafting — code changes approved by operator, not yet implemented.
**Session where decisions were made:** 2026-04-18 (continues from [`HANDOFF_TEST_EVAL.md`](HANDOFF_TEST_EVAL.md))
**Blocks:** GAIA test-split submission run (cannot start until both changes land + re-verify).
**Pointers back:** [`HANDOFF_INDEX.md`](../../HANDOFF_INDEX.md), [`HANDOFF_PROVIDER_MATRIX.md`](HANDOFF_PROVIDER_MATRIX.md)

---

## TL;DR — two approved changes, neither yet implemented

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

## Code change specification (for the implementation session)

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
   - Existing `tests/test_failover_model.py`, `tests/test_reasoning_preservation.py`, etc. must all stay green (run the 10-file sweep documented in `HANDOFF_TEST_EVAL.md` §Pre-flight).

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

## Decisions still open at checkpoint time

### (D1) Does the Qwen slug change back to `qwen3.5-27b` now that the dispatch can handle it?

Three sub-options:
- **(a) Keep `or-qwen3-next-80b-a3b-instruct`** (current; native `"required"` works; text-only; weakest benchmarks in the set).
- **(b) Swap to `or-qwen3.5-27b`** (user's original preference; VL; requires the new `"auto"` path + probably `provider.only=["deepinfra","parasail","baseten"]` extra_body).
- **(c) Swap to `or-qwen3-vl-235b-a22b-instruct`** (flagship VL; native `"required"`; mid-tier cost $0.20/$0.88; live-verified).

If **(b)**, the first live probe after implementation must confirm the dispatch fallback works for this specific slug; otherwise fall back to **(c)**.

### (D2) Does the `"auto"` retry guard apply to sub-agents too, or only the planner?

Recommended: apply everywhere — `tool_choice="required"` appears in several places (`tool_calling_agent`, per-agent overrides). One helper, applied consistently. Keeps the harness uniform.

### (D3) Does the hybrid dispatch need per-agent overrides?

E.g., should `browser_use_agent` always use `"auto"` (because its sub-model is the LangChain wrapper, which might route differently) regardless of the main-agent model_id?

Recommended: initially NO — use the model_id lookup only. If observed failures on a specific sub-agent/model combo, add a second-level override keyed on (model_id, agent_type).

---

## Execution checklist for the next session

- [ ] `git pull origin main` → HEAD includes `2398b0b` or later
- [ ] Re-read this handoff + `HANDOFF_PROVIDER_MATRIX.md` + `HANDOFF_TEST_EVAL.md`
- [ ] Confirm D1/D2/D3 decisions with operator before writing code
- [ ] Implement **Change 1 (Kimi extra_body)** — ≤ 30 lines in `src/models/models.py`
- [ ] Implement **Change 2 (hybrid tool_choice)** — `pick_tool_choice` helper + retry guard
- [ ] Add `tests/test_tool_choice_dispatch.py`
- [ ] Run the 10-file pytest sweep — must stay 118/118 green (+ new tests)
- [ ] Live probe Kimi with image + tool_choice=required; Qwen3.5-27b with auto
- [ ] If D1 picked (b) or (c), regen configs via `scripts/gen_eval_configs.py`
- [ ] Commit in logical groups:
  1. `fix(providers): Kimi K2.5 — thinking-disabled + Moonshot provider-pin via extra_body`
  2. `feat(tool_choice): hybrid dispatch with retry guard for models rejecting "required"`
  3. If D1 triggers config regen: `fix(providers): switch Qwen matrix to <new slug>`
- [ ] Push to `origin/main`
- [ ] Annotate this handoff's "To Do" checklist as done, update `HANDOFF_INDEX.md` row to "Ready to execute → Executing"
- [ ] Resume the main execution protocol from `HANDOFF_TEST_EVAL.md` §S0 pre-flight

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
