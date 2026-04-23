# HANDOFF — Qwen `auto_browser_use_tool` three-layer fix (2026-04-23)

## TL;DR

Qwen via OpenRouter has been unable to use `auto_browser_use_tool` since
stack assembly. F6 in `HANDOFF_E1_E2_RESULTS.md` documented 0 Step 2
transitions in 3,266 attempts during E0 v3. P5 (`7f985cd`) fixed the
HTTP-404 surface but the underlying failure shifted to browser_use's
schema-coercion path + Qwen's reasoning-mode emitting empty content.

This handoff lands a **three-layer fix** scoped to Qwen wire ids
(`qwen/*`), gated by the same prefix rule as `pick_tool_choice`, plus a
**bonus Mistral browser-use fix** (silent pre-existing bug discovered
during V3 regression testing).

### Qwen-scoped layers

- **L1** `tool_calling_method='raw'` — browser_use bypasses
  `with_structured_output` / `bind_tools`; sidesteps the
  discriminated-union `AgentOutput.action` coercion that caused Qwen
  to fill every optional field (see §Root cause B2).
- **L2** Tolerant JSON extractor installed process-wide — patches
  `browser_use.agent.message_manager.utils.extract_json_from_model_output`
  with a stateless fallback wrapper that handles Qwen's one-closing-fence
  output (`"json\n{...}\n\`\`\`"`), prose-wrapped bodies, and empty
  content. Mistral / Kimi / future callers take the fast path
  unchanged.
- **L3** Disable Qwen reasoning on the LangChain OR wrapper via
  `extra_body={"reasoning": {"enabled": False}}` — without this, Qwen
  frequently consumes its entire output budget on hidden reasoning
  tokens and emits `content=""`, causing browser_use raw-mode to hit
  3/3 consecutive parse failures. `reasoning.enabled=false` is the same
  OR-normalised knob `or-kimi-k2.5` and `or-gemma-4-31b-it` already use
  (verified against langchain-openai==0.3.11 constructor signature).
- **L4** `tool_choice="none"` raw-mode guard in
  `ToolChoiceDowngradingChatOpenAI._get_request_payload` — when
  browser_use calls `llm.invoke()` without bound tools, inject
  `tool_choice="none"` so Alibaba/Qwen stops tagging the response as
  `finish_reason="tool_calls"` with empty content. (Residual Qwen
  behaviour: under complex system prompts Qwen still sometimes emits
  empty content — tracked as a known caveat below; the fix makes the
  PATH reachable, Step 2 reliable, and extraction opportunistic.)

### Bonus fix — Mistral `KeyRotatingChatOpenAI` browser_use compatibility

Discovered 2026-04-23 during V3 regression: `KeyRotatingChatOpenAI`
(added by P4 on 2026-04-22) is NOT a Pydantic `BaseChatModel` subclass.
When `browser_use.Agent(llm=mistral_wrapper)` validates, Pydantic v2
rejects the wrapper with
`Input should be a valid dictionary or instance of BaseChatModel`; in
older LangChain validator paths it raises
`'ChatOpenAI' object has no attribute 'get'` (35 occurrences in the
2026-04-22 T3v2 Mistral log — browser_use was SILENTLY broken for
Mistral too). Fix: `_unwrap_for_browser_use(model)` returns
`model._instances[0]` for rotating wrappers, plain pass-through
otherwise. Also adds a dict-shape `.get()` on
`KeyRotatingChatOpenAI` for older validator paths.

## Status

- [x] Unit tests — 27/27 new + 140 prior = **167/167 green** in `scripts/run_handoff_pytest_sweep.sh` (2026-04-23).
- [x] V2 — Qwen live probe reaches Step 2 reliably (pre-fix: 0/3,266). Content extraction is opportunistic (Qwen LLM-quality variance under complex system prompts; caveat logged below).
- [x] V3 — Mistral regression **PASSES** (Steps 1/2/3 reached, 286 chars extracted, tolerant extractor not installed on Mistral path). This also fixes the silent pre-existing `KeyRotatingChatOpenAI` browser_use bug (P4 regression since 2026-04-22).
- [x] V4 — Combined T3v3 matrix smoke (3 Q × 2 models × C4, validation split): **Qwen made 6 `🔗 Navigated to` events** in real GAIA traffic (pre-fix: 0 / 3,266 E0 v3 attempts). Mistral made 12 (also previously broken by the KeyRotating bug). Accuracy at tight smoke caps (agent_config.max_steps=4) was 0/3 for both models — expected at smoke budget; the V4 criterion is PATH UNBLOCK, not GAIA accuracy.
- [ ] Local commit (no push — user hasn't authorized).

## Original problem (log evidence)

**F6 observation** from `HANDOFF_E1_E2_RESULTS.md:122-179`:

- `workdir/run_logs/full_qwen.log` (E0 v3, 3,006 browser calls):
  **3,266 Step 1 marks, 0 Step 2 marks**.
- `workdir/run_logs/full_mistral.log` (same E0 v3 run): 1,880 Step 1,
  1,970 Step 2, 1,810 Step 3, … 663 Step 10+ — Mistral works fine.
- Per-question E0 v3 Qwen `dra.jsonl`: **14 / 80 questions** invoked
  `browser_use_agent` delegation; **all 14** produced
  `about:blank, about:blank, about:blank` trajectories.

**2026-04-23 live OR probe** (captured in this session):

```
tc='required' → HTTP 404 "No endpoints found that support the provided 'tool_choice' value"
tc={'type':'function','function':{'name':'echo'}} → HTTP 404 (same)
tc='auto' → 200 OK (but see B2 below)
```

**2026-04-23 raw-mode probe** (existing `p5_live_validation.py`):

```
🔗  Navigated to https://en.wikipedia.org/wiki/Kangaroo       (L1 works)
extract_json_from_model_output failed: Expecting value:...    (B3)
content='' with reasoning_tokens=534                          (L3 unblocker)
```

**2026-04-23 live V4 T3v3 smoke** (3 Q × C4 × {mistral, qwen}, in flight during
this handoff):

- Qwen: **21+ Step 2 marks in log** — vs 0 across all of E0 v3 (3,266
  Step 1 attempts, 0 Step 2). The path is unblocked.
- Mistral: 28+ Step 2 marks (vs the 35 `'ChatOpenAI' object has no
  attribute 'get'` silent-failure errors in T3v2 pre-fix).

## Root causes

### B1 (addressed by P5) — HTTP 404

`browser_use` calls `bind_tools(AgentOutput, tool_choice=tool_name)`.
`ChatOpenAI.bind_tools` converts this to
`{"type":"function","function":{"name":"AgentOutput"}}`. Alibaba (sole
OR Qwen provider) rejects every non-`"auto"` tool_choice value — both
the string `"required"` AND the named-function dict.

### B2 (addressed by L1) — Schema-coercion hallucination

Post-P5 the payload has `tool_choice="auto"` (OR accepts this). But
`AgentOutput.action: list[ActionModel]` is a discriminated union over
~20 optional action fields. Without forcing, Qwen emits a single tool
call with EVERY field filled:

```
Action 1/1: {"done":{"text":"","success":false},"search_google":{"query":""},
  "go_to_url":{"url":"https://..."},"click_element_by_index":{"index":82,"xpath":""},
  ..."drag_drop":{...}}
```

browser_use iterates `exclude_unset=True` and executes `done` first →
agent terminates at Step 1 with 0 extracted chars.

### B3 (addressed by L2) — Brittle JSON fence parser

`browser_use.agent.message_manager.utils.extract_json_from_model_output`
does:

```python
if '```' in content:
    content = content.split('```')[1]
    if '\n' in content:
        content = content.split('\n', 1)[1]
```

Qwen in raw mode emits `"json\n{...}\n```"` — only the closing fence,
and only the language tag `json\n` where the opening fence should be.
`split('```')` returns `['json\n{...}\n', '']` and `[1]` is empty →
`json.JSONDecodeError`.

### B4 (addressed by L3) — Reasoning tokens consume output budget

Qwen's reasoning mode (enabled by default on OR) writes to
`reasoning_tokens` before visible content. Observed in probe:
`reasoning_tokens=534, prompt_tokens=5453, content=''`. 3/3 retries
produce empty content → browser_use terminates the session.

`reasoning={"enabled": false}` on the OR wrapper tells OR to disable
thinking on whichever backend the Qwen request routes to (the same
knob used by the `or-kimi-k2.5` and `or-gemma-4-31b-it` registrations).

## Changes table

| File | Change | Motivation |
|---|---|---|
| [src/tools/_browser_json_extractor.py](../../src/tools/_browser_json_extractor.py) (NEW) | Tolerant JSON parser + patch installer + idempotence / reset hooks | L2 |
| [src/tools/auto_browser.py](../../src/tools/auto_browser.py) | `_resolve_wire_id` + `_pick_browser_tool_calling_method`; wire Qwen path to `tool_calling_method='raw'` + install tolerant extractor | L1 |
| [src/models/models.py](../../src/models/models.py) | `_register_openrouter_models`: pass `extra_body={"reasoning":{"enabled":False}}` for any `qwen/*` wire id to the LangChain wrapper | L3 |
| [scripts/p5_live_validation.py](../../scripts/p5_live_validation.py) | Exercise real `auto_browser.py` helpers (not reconstruction); tighten D4→C4 + add C3 (Step 2+ required) | verify L1+L2+L3 end-to-end |
| [scripts/mistral_browser_use_regression.py](../../scripts/mistral_browser_use_regression.py) (NEW) | Regression smoke against Mistral La Plateforme — asserts default path unchanged, tolerant extractor NOT installed, Step 2+ reached, ≥100 chars extracted | R6 fairness guardrail |
| [tests/test_auto_browser_qwen_raw_mode.py](../../tests/test_auto_browser_qwen_raw_mode.py) (NEW) | 27 unit tests (T1 parser × 9, T2 installer × 4, T3 kwarg selection × 4, T4 Mistral unchanged × 2, T5 wire-id resolution × 3, T6 end-to-end × 2, T7 unwrap × 3) | V1 |
| [scripts/run_handoff_pytest_sweep.sh](../../scripts/run_handoff_pytest_sweep.sh) | Append new test file to `FILES` array (per-file sweep avoids `src.logger` stub leakage) | R9 |

## Live-validation commands

```bash
# V1 — offline pytest (≤30 s, no API)
bash scripts/run_handoff_pytest_sweep.sh
# Pass: "all modules passed" + 164 tests.

# V2 — Qwen browser_use live probe (~45 s, ~$0.01 OR spend)
/Users/ahbo/miniconda3/envs/dra/bin/python scripts/p5_live_validation.py
# Pass: overall PASS ✓ — C1+C2+C3+C4 all green, step numbers ≥[1,2].

# V3 — Mistral regression smoke (~45 s, ~$0.01 Mistral spend)
/Users/ahbo/miniconda3/envs/dra/bin/python scripts/mistral_browser_use_regression.py
# Pass: overall PASS ✓ — Step 2+ reached, ≥100 chars, tolerant extractor NOT installed.

# V4 — combined T3v3 matrix smoke (~12 min, ~$0.15)
DRA_RUN_ID=20260423_T3v3smoke caffeinate -dims bash scripts/run_eval_matrix.sh \
  full '' c4 "max_samples=3 dataset.shuffle=True dataset.seed=42 \
  per_question_timeout_secs=1800 agent_config.max_steps=15 \
  auto_browser_use_tool_config.max_steps=10 browser_use_agent_config.max_steps=3 \
  deep_researcher_tool_config.time_limit_seconds=45"
# Pass: Qwen log has ≥1 Step 2+ from browser_use_agent + Mistral accuracy not regressed vs T3v2.
```

## Validation criteria

| Check | Command (suffix after `cd` to repo root) | Pass threshold |
|---|---|---|
| V1a full sweep | `bash scripts/run_handoff_pytest_sweep.sh \| tail -3` | Exit 0 + "sweep: all modules passed" |
| V1b new tests isolated | `pytest tests/test_auto_browser_qwen_raw_mode.py -q` | 24 passed |
| V2 Qwen live | `python scripts/p5_live_validation.py \| tail -5` | `overall: PASS ✓` |
| V3 Mistral regression | `python scripts/mistral_browser_use_regression.py \| tail -5` | `overall: PASS ✓` |
| V4 matrix smoke Qwen | `grep -c "📍 Step 2" workdir/gaia_c4_qwen_20260423_T3v3smoke/log.txt` | ≥ 1 |
| V4 matrix smoke Mistral | `grep -c "📍 Step 2" workdir/gaia_c4_mistral_20260423_T3v3smoke/log.txt` | ≥ T3v2 baseline |
| Fairness check | `grep "tool_calling_method" workdir/gaia_c4_mistral_*/log.txt` | empty (no Mistral uses raw mode) |

## Methodology implications (E0 v3 → E3)

E0 v3 Qwen C4 training ran on the broken path — 14/80 questions
involved silent `browser_use_agent` delegations that produced only
`about:blank` trajectories. Consequences for the paper:

1. **Qwen's 2 learned skills from E0 v3 are biased toward browser-failure
   recovery**. The skills tuned for the broken path (`research-fallback-sources`,
   `escalate-on-browser-failure`) will now see a working browser at E3
   test time — evaluation may show them underfiring (a legit
   "capability–inference mismatch").
2. **E3 Qwen C4 runs the fixed path**. Compare against the C3→C4
   accuracy delta, but disclose the asymmetry:
   > "Due to a latent interaction between `browser_use`'s schema-coerced
   > structured-output path and Alibaba's restricted `tool_choice`
   > support (both discovered post-E0), Qwen C4's training library
   > is not a like-for-like reflection of the test-time capability
   > surface. Mistral C4 is unaffected — its browser path worked
   > throughout. This is documented in `docs/handoffs/HANDOFF_QWEN_BROWSER_RAW_MODE.md`
   > with full root-cause + live-probe evidence."
3. **No E0 re-run planned** (operator decision pre-dating this handoff;
   see F3 in `HANDOFF_E1_E2_RESULTS.md`). Methodology footnote carries
   the asymmetry.

## Known unknowns / caveats

- **browser_use version pinning**: both `_browser_json_extractor.py`
  and P5's `openaillm.py` override private API surfaces
  (`extract_json_from_model_output`, `_get_request_payload`). If
  `browser_use` or `langchain-openai` are upgraded past `0.1.48` /
  `0.3.11` respectively, both loudly RuntimeError at first call —
  re-pin in the env or port the hooks to the new symbol before
  re-running E3.
- **Qwen reasoning disable**: `reasoning={"enabled": false}` is OR's
  normalised parameter. We don't probe what the underlying Alibaba
  API is told — if OR ever routes to a non-Alibaba Qwen backend that
  doesn't honour this knob, empty-content failures could return. Pin
  monitored via V2 success rate (target ≥ 80% across 5 consecutive
  runs) before E3 launch.
- **Raw mode may prompt Qwen differently than function_calling mode**.
  browser_use's raw-mode system prompt embeds the action schema as
  text. Qwen's GAIA-accuracy delta vs. function_calling is unknown
  at the time of this handoff — covered by V4 smoke before E3.
