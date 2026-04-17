# Handoff: Multi-Provider Integration + GAIA Eval Matrix (Mistral / Kimi / Qwen × C0–C4)

**Session date:** 2026-04-17 → 2026-04-18
**Branch:** `main` (pushed — both commits live on `origin/main`)
**Commits:** `7632470` (multi-provider + 3 correctness fixes) → `9883a3a` (Qwen failover + Kimi no-thinking + 12-cell eval matrix + parallel runner)
**Scope:** Adds 5 model providers (DeepSeek V3.2, Mistral Small 4, Qwen3 family, Kimi K2.5, MiniMax M2.7) to the model registry, fixes 3 multi-turn correctness risks discovered during integration, and ships a 3-models × 4-conditions GAIA eval pipeline with a Qwen DashScope→OpenRouter auto-failover.

---

## TL;DR Checklist

### Completed (this session)

- [x] Registered 5 new providers (native + OpenRouter paths) — `src/models/models.py`
- [x] Added `extra_body` support to `OpenAIServerModel` for vendor-specific request fields (DashScope `enable_thinking`, Moonshot `thinking={"type":"disabled"}`)
- [x] **Fix A:** `reasoning_content` round-trip for DeepSeek-reasoner / Qwen3-thinking — `ChatMessage.reasoning_content` field, capture in `openaillm.py`, echo in `MessageManager`, role-merge skip when reasoning present
- [x] **Fix B:** Moonshot Kimi sampling lock — strip `temperature/top_p/n/penalty/logprobs` after caller-kwargs merge in `MessageManager.get_clean_completion_kwargs`
- [x] **Fix C:** DashScope `enable_thinking` injection — `extra_body` kwarg flows through `self.kwargs` to `chat.completions.create`
- [x] Registered Kimi `kimi-k2.5-no-thinking` variant (extra_body disables thinking) — required for C3/C4 JSON output
- [x] Implemented `FailoverModel` (`src/models/failover.py`) — one-way DashScope→OpenRouter switch on quota-exhaustion errors, registered as `qwen3.6-plus-failover`
- [x] Generated 12 single-model eval configs via `scripts/gen_eval_configs.py`
- [x] Built parallel batch runner `scripts/run_eval_matrix.sh` (3 model streams parallel; conditions sequential per stream; smoke vs full modes)
- [x] 28 unit tests passing — 12 reasoning/sampling/extra_body, 16 failover detection + sticky-switch + streaming
- [x] Code-reviewer subagent sign-off (GREEN, no BLOCK/HIGH issues)
- [x] Both commits pushed to `origin/main`

### To Do Next Session (GPU Farm)

- [ ] `git pull origin main` on GPU farm; confirm HEAD = `9883a3a`
- [ ] Populate `.env` with `MISTRAL_API_KEY`, `MOONSHOT_API_KEY`, `DASHSCOPE_API_KEY`, `OPENROUTER_API_KEY`, `FIRECRAWL_API_KEY` (see "How to set up `.env`" below)
- [ ] Tier 0–2 pre-flight (config sanity, model registration smoke, single-question per model)
- [ ] Tier 3 — C3/C4 JSON-output validation per model (catches `response_format` conflicts early)
- [ ] Tier 5 — `bash scripts/run_eval_matrix.sh smoke` (5 questions × 12 cells = 60 questions on validation split)
- [ ] Inspect Qwen failover trigger (look for `[FailoverModel:qwen3.6-plus-failover] primary ... quota exhausted` line)
- [ ] If smoke passes, kick off `bash scripts/run_eval_matrix.sh full` for the test-split submission run
- [ ] Score every cell with `scripts/analyze_results.py` and stash the per-cell numbers in this doc before promoting to Completed

---

## Original Problems

### Problem A — Need diverse SOTA models for GAIA leaderboard submission

The codebase had OpenAI / Anthropic / Google / local-vLLM-Qwen registered. For a fair multi-model GAIA evaluation we needed first-class native + OpenRouter support for DeepSeek V3.2, Mistral Small 4, Qwen3 family (Max / Plus / Coder + thinking variants), Moonshot Kimi K2.5, and MiniMax M2.7. All five are OpenAI-SDK compatible, so the integration pattern is uniform — but each ships with its own quirks that break the agent's multi-turn tool loop in subtle, silent ways.

### Problem B — Three latent correctness risks surfaced by integration review

A code-reviewer pass on the 5-provider draft flagged three risks that would silently corrupt evals:

**B1 — Reasoning content stripped on receive (DeepSeek-reasoner, Qwen3-thinking)**

`openaillm.py:261` filtered the SDK message dump to `{"role","content","tool_calls"}`, dropping `reasoning_content`. DeepSeek-reasoner returns HTTP 400 on turn 2+ of a tool loop if `reasoning_content` is not echoed back; Qwen3 thinking-mode behaves the same when `enable_thinking=True` is persisted across turns. Without a fix, **multi-turn tool-calling on either model fails silently after turn 1.**

**B2 — Moonshot Kimi sampling-param lock**

Moonshot fixes `temperature` to 1.0 (thinking) / 0.6 (non-thinking), `top_p` to 0.95, `n` to 1. Any caller-injected override 400s. The agent's `_prepare_completion_kwargs` does `completion_kwargs.update(kwargs)` last, so a future caller passing `temperature=0.5` would crash every Kimi call. Need a strip applied **after** the caller merge so provider constraints always win.

**B3 — DashScope `enable_thinking` has no injection path**

DashScope accepts `enable_thinking=true` as a top-level request field on the OpenAI-compat endpoint, but it's not part of the OpenAI schema. The OpenAI SDK accepts `extra_body={...}` for vendor extras, but `OpenAIServerModel.__init__` had no parameter to thread it through.

### Problem C — Single-model eval constraint requires per-(model, condition) configs

The user wants single-model GAIA runs (no mixed-model setups). With 3 models × 4 conditions (C0/C2/C3/C4), that's 12 distinct configs, each pinning one `model_id` across the planner + 3 sub-agents + 3 tools + LangChain wrapper. mmengine config inheritance replaces dicts wholesale rather than deep-merging, so each file must redeclare every agent/tool config — too repetitive to hand-maintain, especially as model IDs change.

### Problem D — Qwen on free DashScope tier needs auto-failover to OpenRouter

DashScope offers a generous free tier; OpenRouter offers paid Qwen access. The eval should consume free quota first, then transparently switch to OpenRouter on quota exhaustion — without restarting the run or hand-editing configs mid-eval.

### Problem E — Kimi K2.5 thinking-on default breaks JSON output

C3 ReviewAgent and C4 SkillExtractor both rely on `response_format={"type":"json_object"}`. Moonshot disallows this while thinking is on (default). Without an explicit thinking-disabled variant, C3/C4 silently degrade to C2-equivalent on Kimi.

---

## Changes & Commits

### Commit `7632470` — multi-provider + 3 correctness fixes

| File | Change |
|------|--------|
| [src/models/models.py](src/models/models.py) | New `_register_dashscope_models`, `_register_mistral_models`, `_register_moonshot_models`, `_register_minimax_models`. Native DeepSeek `else` branch added. OpenRouter slug list extended with `or-deepseek-v3.2`, `or-mistral-small`, `or-qwen3-max`, `or-qwen3.6-plus`, `or-qwen3-coder-next`, `or-kimi-k2.5`, `or-minimax-m2.7`. Qwen3 thinking variants registered with `extra_body={"enable_thinking": True}`. |
| [src/models/openaillm.py](src/models/openaillm.py) | `OpenAIServerModel.__init__` accepts `extra_body: dict \| None`; merged into `self.kwargs` so it forwards through every completion call. `generate()` reads `reasoning_content` off the SDK message object via `getattr` **before** `model_dump` so the include-set filter no longer strips it. `generate_stream` reads `choice.delta.reasoning_content` per chunk. |
| [src/models/base.py](src/models/base.py) | `ChatMessage` gains `reasoning_content: str \| None`. `ChatMessageStreamDelta` same. `agglomerate_stream_deltas` accumulates the field. `from_dict` reads it. JSON round-trip preserved. |
| [src/models/message_manager.py](src/models/message_manager.py) | New predicates `is_moonshot_kimi`, `is_minimax`, `needs_reasoning_echo`. `_get_chat_completions_message_list` echoes `reasoning_content` on assistant messages only when the predicate matches; role-merge path skipped when reasoning present (prevents fabricated attribution). `get_clean_completion_kwargs` strips Kimi-banned params and clamps MiniMax temperature **after** caller-kwargs merge. |
| [.env.template](.env.template) | Added `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `MISTRAL_API_KEY`, `MOONSHOT_API_KEY`, `MINIMAX_API_KEY` (+ `_API_BASE` overrides). |
| [tests/test_reasoning_preservation.py](tests/test_reasoning_preservation.py) | NEW — 12 unit tests for fixes A/B/C. Loads modules via `importlib` to bypass `src/__init__.py` heavy imports, runs in any minimal env. |

Diff stats: 6 files, +673 / −16.

### Commit `9883a3a` — Qwen failover + Kimi no-thinking + 12-cell eval matrix + parallel runner

| File | Change |
|------|--------|
| [src/models/failover.py](src/models/failover.py) | NEW — `FailoverModel` wrapper. Proxies `generate` / `generate_stream` / `model_id` / `_last_input_token_count` / arbitrary attribute access. One-way switch on quota-exhaustion errors only (DashScope free-tier patterns + HTTP 402/403). Conservative detection: bare 429 rate-limit messages do NOT trigger switch. Type-only imports of `OpenAIServerModel` / `ChatMessage` under `TYPE_CHECKING` so it's testable in isolation. |
| [src/models/models.py](src/models/models.py) | Imports `FailoverModel`. `_register_qwen_failover_models` runs LAST in `init_models` — wraps `qwen3.6-plus` (DashScope) + `or-qwen3.6-plus` (OpenRouter) into `qwen3.6-plus-failover` when both keys are present. New `kimi-k2.5-no-thinking` variant via `extra_body={"thinking":{"type":"disabled"}}`. |
| [scripts/gen_eval_configs.py](scripts/gen_eval_configs.py) | NEW — generator for the 12-config matrix. Single template; one row per (model_label, model_id, langchain_alias) × (condition, base_config). Run with `python scripts/gen_eval_configs.py` (or `--dry-run`). |
| [configs/config_gaia_<c0\|c2\|c3\|c4>_<mistral\|kimi\|qwen>.py](configs/) | 12 NEW generated files. Each inherits from the matching C-condition base and overrides `tag` + every agent/tool `model_id` to the chosen model. Mistral uses `mistral-small`, Kimi uses `kimi-k2.5-no-thinking`, Qwen uses `qwen3.6-plus-failover`. Browser-use LangChain wrapper alias matches. `max_samples=None` and `concurrency=4` defaults — both overridable via `--cfg-options`. |
| [scripts/run_eval_matrix.sh](scripts/run_eval_matrix.sh) | NEW — parallel batch runner. Three model streams (`mistral`, `kimi`, `qwen`) launched in parallel; the four conditions run sequentially within each stream so each model's API key is not contended. Modes: `smoke` (cap 5 questions on `validation` split) / `full` (no cap, default `test` split). Optional model / condition filters. Per-stream logs in `workdir/run_logs/`. |
| [tests/test_failover_model.py](tests/test_failover_model.py) | NEW — 16 unit tests: quota-pattern detection (DashScope free-tier + 402 + response-body match), sticky one-way switch, transient 429 NOT switching, streaming failover before first chunk, attribute proxying, alias precedence over child `model_id`. Loads `failover.py` under unique module name to avoid sys.modules pollution. |
| [CLAUDE.md](CLAUDE.md) | Documented new model registry, failover semantics, provider quirks, and matrix overview. |

Diff stats: 18 files, +2,424 / −0 (the rest of the +673 from `7632470` plus all-new files here).

**Combined commit range stats:** 30 files changed, +3,097 / −27.

### Why two commits and not one

Commit `7632470` is mechanical multi-provider integration with safety fixes — landed first so the model registry is testable in isolation. Commit `9883a3a` builds the eval pipeline on top — separate so a revert of the matrix work doesn't lose the provider integration.

### Not in either commit (intentional scope exclusions)

- The Qwen3-VL / DeepSeek-VL vision variants — not needed for the 3-model run since Mistral, Kimi, and Qwen3.6-plus are all unified multimodal.
- A `kimi-k2.5-thinking` ablation variant — easy follow-up if you want a head-to-head later.
- Reasoning-token billing accounting — `TokenUsage` only tracks input/output. DeepSeek bills reasoning separately. Cost-tracking scripts undercount by the reasoning amount.
- `<think>` preservation through `general_agent.py::_prune_messages_if_needed` at ≥85% context. MiniMax `<think>` blocks could be truncated mid-string by the pruner. Deferred — current evals rarely cross 85% threshold.

---

## How to Set Up `.env`

```bash
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
cp .env.template .env
# then edit .env with the keys below
```

### Required keys for the 3-model matrix

| Variable | Where to get it | Required for |
|---|---|---|
| `MISTRAL_API_KEY` | https://console.mistral.ai → API Keys | All 4 Mistral configs |
| `MOONSHOT_API_KEY` | https://platform.moonshot.ai → API Keys | All 4 Kimi configs |
| `DASHSCOPE_API_KEY` | https://account.alibabacloud.com → Model Studio → API Keys | Qwen primary (free tier) |
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys | Qwen failover backup |
| `FIRECRAWL_API_KEY` | https://www.firecrawl.dev → API Keys | `web_searcher_tool` (used by all sub-agents) |

Also keep a valid `OPENAI_API_KEY` populated — `model_manager.init_models()` may probe it during registration of OpenAI models even if no config selects them.

### Endpoints — leave defaults unless region-specific

```
DASHSCOPE_API_BASE=https://dashscope-intl.aliyuncs.com/compatible-mode/v1   # ← change to dashscope.aliyuncs.com/compatible-mode/v1 for CN region
MISTRAL_API_BASE=https://api.mistral.ai/v1
MOONSHOT_API_BASE=https://api.moonshot.ai/v1
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
```

If on HK/CN networks, the intl DashScope endpoint may resolve to Singapore. Verify with a manual `curl https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models -H "Authorization: Bearer $DASHSCOPE_API_KEY"` first; switch to the CN endpoint if `qwen3.6-plus` returns 404.

### `.gitignore` check

```bash
grep -E "^\.env$" .gitignore || echo ".env" >> .gitignore
```

---

## How to Test on GPU Farm

### Prerequisites

```bash
ssh <gpu-farm-host>
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
git pull origin main
git log -1 --oneline   # Should show 9883a3a

conda activate dra
bash scripts/ensure_playwright_browsers.sh   # if browser-use is in scope
```

### Tier 0 — Config sanity (no API calls, ~5 sec)

```bash
for cfg in configs/config_gaia_c{0,2,3,4}_{mistral,kimi,qwen}.py; do
  python -c "from mmengine.config import Config; Config.fromfile('$cfg'); print('OK $cfg')"
done
```

### Tier 1 — Model registration smoke (~5 sec, requires keys)

```bash
python -c "
from src.models.models import ModelManager
m = ModelManager(); m.init_models(use_local_proxy=False)
needed = ['mistral-small', 'kimi-k2.5-no-thinking', 'qwen3.6-plus', 'qwen3.6-plus-failover',
          'or-qwen3.6-plus', 'or-mistral-small', 'or-kimi-k2.5']
missing = [n for n in needed if n not in m.registed_models]
print('Missing:', missing)
print('Registered total:', len(m.registed_models))
"
```

`Missing: []` means all keys are in `.env` and `qwen3.6-plus-failover` registered correctly.

### Tier 2 — Single-question smoke per model (~3-5 min each)

```bash
python examples/run_gaia.py --config configs/config_gaia_c0_mistral.py --cfg-options max_samples=1 dataset.split=validation
python examples/run_gaia.py --config configs/config_gaia_c0_kimi.py    --cfg-options max_samples=1 dataset.split=validation
python examples/run_gaia.py --config configs/config_gaia_c0_qwen.py    --cfg-options max_samples=1 dataset.split=validation
```

Watch for the failure patterns in the table below.

### Tier 3 — C3/C4 JSON-output validation (~5-10 min)

C3/C4 are most likely to surface provider-specific JSON issues — run each once before kicking off the matrix:

```bash
python examples/run_gaia.py --config configs/config_gaia_c3_kimi.py    --cfg-options max_samples=2 dataset.split=validation
python examples/run_gaia.py --config configs/config_gaia_c3_qwen.py    --cfg-options max_samples=2 dataset.split=validation
python examples/run_gaia.py --config configs/config_gaia_c4_mistral.py --cfg-options max_samples=2 dataset.split=validation
```

### Tier 5 — Full smoke matrix (~30-90 min, 60 questions total)

```bash
bash scripts/run_eval_matrix.sh smoke
# Tail logs in parallel:
tail -f workdir/run_logs/smoke_*.log
```

### Tier 6 — Full submission run (test split, no cap)

Only after Tiers 0-5 are clean:

```bash
bash scripts/run_eval_matrix.sh full
```

Cost estimate: ~$30-100 depending on test-set size, model speeds, and step counts. Run inside `tmux` or as a SLURM batch — full sweep is hours.

---

## How to Validate the Fixes

### Validation 1 — Reasoning content round-trip (Fix A)

If running any **DeepSeek-reasoner** or **Qwen3-thinking** config (not in the default 12, but if added later):

```bash
NEW_RUN=workdir/<run_dir>/log.txt

# Should NOT see HTTP 400 on multi-turn tool loops
grep -c "400.*reasoning_content" "$NEW_RUN"
# Expected: 0

# Confirm reasoning_content is being captured (raw=full response object, reasoning preserved)
grep -c "reasoning_content" workdir/<run_dir>/dra.jsonl
# Expected: > 0 if reasoner is in the loop
```

### Validation 2 — Kimi sampling lock (Fix B)

```bash
NEW_RUN=workdir/gaia_c0_kimi/log.txt

grep -c "400.*temperature\|400.*top_p\|400.*Invalid sampling" "$NEW_RUN"
# Expected: 0 — Kimi must accept every request

grep -c "ChatMessage with role 'tool' requires tool_call_id" "$NEW_RUN"
# Expected: 0 — tool-loop should round-trip cleanly
```

### Validation 3 — Qwen `extra_body` forwarding (Fix C)

If you add a `qwen3.6-plus-thinking` config later:

```bash
# Server-side log on DashScope console should show enable_thinking=true on requests
# Locally, confirm reasoning_content present in dra.jsonl after a thinking-mode run
grep -c "reasoning_content" workdir/<run_dir>/dra.jsonl
```

### Validation 4 — Qwen failover trigger

Hard to deterministically force without burning real quota. Two approaches:

**Approach A (natural):** Let the Qwen smoke run consume DashScope free tier. When quota exhausts, watch for:

```bash
grep "FailoverModel:qwen3.6-plus-failover.*primary.*quota exhausted" \
  workdir/run_logs/smoke_qwen.log
```

After this line, all subsequent Qwen calls in that process route to OpenRouter. Cross-check by inspecting OpenRouter usage dashboard for the matching timeframe — should show requests to `qwen/qwen3.6-plus`.

**Approach B (forced):** Set `DASHSCOPE_API_KEY=invalid_test_key` and run one question. Note: an invalid key returns 401, NOT a quota error — `_looks_like_quota_exhaustion` returns False and failover correctly does NOT trigger. So Approach B is only useful if you mock the response. Stick with Approach A.

**Pass criterion:** if the smoke run completes without DashScope quota exhaustion, failover is untested but not broken — defer to the full run. If quota does exhaust, the log line above MUST appear and downstream calls MUST succeed via OpenRouter.

### Validation 5 — Per-cell answer correctness

```bash
for d in workdir/gaia_*_{mistral,kimi,qwen}/; do
  echo "=== $d ==="
  python scripts/analyze_results.py "$d/dra.jsonl" 2>/dev/null | head -10
done
```

**Pass criterion (smoke):** every cell shows non-zero answered count. Any cell with 0 answers indicates a model-config mismatch — investigate that workdir's `log.txt` first.

**Pass criterion (full submission):** Compare condition deltas within each model:
- C2 should equal or beat C0 (reactive diagnose/modify helps or is neutral)
- C3 should equal or beat C2 (review step adds signal)
- C4 should equal or beat C3 (skill library compounds)
- Cross-model: not directly comparable (different reasoning strengths) — report side-by-side, not as a single ranking

### Validation 6 — No regressions

```bash
# Existing model-layer tests still pass
python -m pytest tests/test_failover_model.py tests/test_reasoning_preservation.py tests/test_tier_b_tool_messages.py -q
# Expected: 28 passed
```

---

## Failure-Mode Cheatsheet

| Model | Failure pattern | Likely cause | Fix |
|---|---|---|---|
| **Mistral** | `400 — invalid model_id` | Mistral renamed model | Update `mistral-small-2603` in `_register_mistral_models` |
| **Kimi** | `400 — temperature locked` | A caller still injecting temp | Verify `MessageManager.get_clean_completion_kwargs` is on the call path |
| **Kimi** | `400 — response_format with thinking` | Wrong variant chosen | Confirm config uses `kimi-k2.5-no-thinking`, not `kimi-k2.5` |
| **Qwen** | `404 model not found` | Wrong endpoint region | Switch `DASHSCOPE_API_BASE` to CN endpoint |
| **Qwen** | `[FailoverModel] primary quota exhausted` | Free tier already consumed | Expected — confirms failover working |
| **All** | `OPENAI_API_KEY` errors at init | OpenAI registration probes the key | Set OpenAI key to any valid value |
| **C3 silent fallback** | `ReviewStep falling back` warnings | Model can't comply with `response_format` | Review degrades to C2 — log it, but eval still completes |
| **All** | `crawl4ai` import error | Conda env missing browser-use deps | `pip install crawl4ai` in the `dra` env |

---

## Context the Next Session Will Need

### Key file paths

- **This handoff:** `HANDOFF_PROVIDER_MATRIX.md`
- **Older handoff (silent failures):** `HANDOFF_SILENT_FAILURES.md` — its fixes (`ba28f21`) are also live on `origin/main` and apply to all configs in this matrix
- **Index:** `HANDOFF_INDEX.md`
- **Config generator:** `scripts/gen_eval_configs.py` — regenerate, never hand-edit, the 12 generated configs
- **Parallel runner:** `scripts/run_eval_matrix.sh smoke|full [model] [condition]`
- **Per-stream logs:** `workdir/run_logs/<mode>_<model>.log`
- **Per-cell results:** `workdir/gaia_<condition>_<model>/dra.jsonl`

### Useful commands

```bash
# Show the two commits in this handoff
git show 7632470 --stat
git show 9883a3a --stat

# Re-generate configs (do this if model IDs change)
python scripts/gen_eval_configs.py

# Single-cell run
python examples/run_gaia.py --config configs/config_gaia_c3_mistral.py

# Score a single cell
python scripts/analyze_results.py workdir/gaia_c3_mistral/dra.jsonl

# Compare two cells (e.g. C0 vs C3 on Mistral)
python scripts/compare_results.py \
  workdir/gaia_c0_mistral/dra.jsonl \
  workdir/gaia_c3_mistral/dra.jsonl

# Token-usage roll-up for a cell
grep -E "Input tokens: [0-9,]+ \| Output tokens: [0-9,]+" workdir/<run_dir>/log.txt | \
  awk -F'[|:,]' '{i+=$3; o+=$5} END {print "Input:", i, "Output:", o}'

# Resume an interrupted cell — run_gaia.py reads dra.jsonl and skips done questions
python examples/run_gaia.py --config configs/config_gaia_c4_qwen.py
```

### Known unknowns / caveats

- **DashScope free-tier sizing:** the daily quota for `qwen3.6-plus` on the intl endpoint is not documented in stable form. First Qwen run will reveal it empirically — log the exact token count at which failover fires.
- **C4 skill drift:** `enable_skill_extraction=True` in C4 configs mutates `src/skills/` in-place at task end. For frozen-library evaluation, override with `--cfg-options enable_skill_extraction=False`. **Important:** if you don't, two consecutive C4 runs are not directly comparable (the second runs against a larger library).
- **MiniMax `<think>` pruning:** unrelated to the 3-model matrix (MiniMax not in scope here), but if MiniMax is added later, the `general_agent.py::_prune_messages_if_needed` at ≥85% context can truncate `<think>` blocks mid-string. Deferred — see "Not in either commit" above.
- **Cost ceiling:** full matrix on test split is potentially expensive. Set a kill-switch — `pkill -f "config_gaia_.*_<model>"` will kill one stream without affecting the others.
- **Kimi vision per-question:** `kimi-k2.5-no-thinking` claims to retain vision (only thinking is disabled, not multimodal). If a GAIA multimodal question silently produces text-only reasoning, suspect that the `extra_body={"thinking":{"type":"disabled"}}` payload also disabled vision — verify with one image-attachment task in Tier 2.

### Baseline to beat (from `HANDOFF_SILENT_FAILURES.md`)

The 2026-03-27 baseline (`gaia_verify10_127871_20260327_214432`) was 6/10 answered on Qwen-on-vLLM. With the silent-failure fixes from `ba28f21` AND the model-layer correctness fixes from this handoff, expect strict improvement on the same questions when run via `config_gaia_c0_qwen.py` with the same 10-question subset. If it's not strictly better, that's a regression worth digging into.
