# Handoff Index

Entry point for change handoffs that still need validation on the GPU farm.
Each linked document is **self-contained** — read the specific handoff doc for
original problem, changes, test commands, and validation criteria.

> Per-topic handoff docs live in [`docs/handoffs/`](docs/handoffs/) (see the
> [directory README](docs/handoffs/README.md) for authoring conventions). This
> index stays at the repo root because the operator is trained to open it
> first.

---

## Pending Handoffs

| # | Title | Doc | Status | Commit(s) | Pushed? |
|---|-------|-----|--------|-----------|---------|
| 1 | Silent-failure fixes for browser + analyzer tools | [HANDOFF_SILENT_FAILURES.md](docs/handoffs/HANDOFF_SILENT_FAILURES.md) | Code-validated (unit + partial runtime, 2026-04-18) — awaiting GPU-farm test-split run | `ba28f21` | Yes |
| 2 | Multi-provider integration + GAIA eval matrix (Mistral/Kimi/Qwen × C0–C4) with Qwen DashScope→OpenRouter failover | [HANDOFF_PROVIDER_MATRIX.md](docs/handoffs/HANDOFF_PROVIDER_MATRIX.md) | Code-validated (24 unit tests; failover live-fired 2026-04-18; see local-validation fixes below) — awaiting test-split accuracy | `7632470` → `9883a3a` → `a98da9a` (local fixes: Qwen thinking, Kimi→OpenRouter) | Yes |
| 3 | `modify_subagent` prompt + tool-description guidance expansion (all 7 actions covered, failure-mode→action table, condition-scoped anti-patterns for C2/C3/C4) | [HANDOFF_MODIFY_SUBAGENT_GUIDANCE.md](docs/handoffs/HANDOFF_MODIFY_SUBAGENT_GUIDANCE.md) | Code-validated (93 adaptive-tool mentions in C2+ runtime logs, 2026-04-18) — awaiting test-split | `764c6bf` → `b73eb39` | Yes |
| 4 | Core C3 / C4 implementation — structural REVIEW step + cross-task skill library (the four experimental conditions C0/C2/C3/C4 themselves) | [HANDOFF_C3_C4_IMPLEMENTATION.md](docs/handoffs/HANDOFF_C3_C4_IMPLEMENTATION.md) | Code-validated (60 unit tests: 26 review_schema + 28 skill_registry + 6 skill_seed; smoke matrix targets 4/4 C4 `SkillRegistry` banners — awaiting `[REVIEW]` marker under traffic + test-split) | `60065a8` → `433c30e` → `0643089` → `d247605` | Yes |
| 5 | RC1 premature `final_answer_tool` guard + duplicate-yield bug fix + RC2 exception-chain diagnostic hook + prompt contradictions fix | [HANDOFF_RC1_FINAL_ANSWER_GUARD.md](docs/handoffs/HANDOFF_RC1_FINAL_ANSWER_GUARD.md) | Code-validated (18 unit tests, 2026-04-18) — awaiting guard-fire in real traffic | `54e7707` → `a9a6985` → `c52cf91` → `912685f` → `d36f4d4` | Yes |
| 6 | Pass 2 Qwen-4B (vLLM) tuning — sub-agent `max_steps` 3/3/5→7/7/7 and `context_prune_threshold_ratio`=0.75 on `config_gaia_adaptive_qwen.py` | [HANDOFF_PASS2_QWEN_TUNING.md](docs/handoffs/HANDOFF_PASS2_QWEN_TUNING.md) | **N/A for 3-model API matrix** (Qwen-4B local-vLLM only; reinstate when returning to on-prem Qwen-4B runs) | `63486ca` | **No — push pending** |
| 7 | ToolGenerator hardening — allowlist + AST imports, repair retry, prompt examples, `Tool.from_code(expected_tool_name)`, collision-safe dynamic tool names, unit tests | [HANDOFF_TOOLGENERATOR.md](docs/handoffs/HANDOFF_TOOLGENERATOR.md) | Code-validated (12/12 unit tests after `7ee9ae1` schema fallback fix, 2026-04-18) — awaiting test-split | `0161321` → `7ee9ae1` (Tool.from_code parameters synthesis fix) | Yes |
| 8 | GAIA test-split execution protocol for the (now-)16-cell matrix on the HKU CS GPU farm (staged S0→S1→S2→[C4 train/freeze]→S4 + grep sweep + resume protocol) | [HANDOFF_TEST_EVAL.md](docs/handoffs/HANDOFF_TEST_EVAL.md) | Ready to execute — **unblocked 2026-04-18** (#9 implemented + live-verified); **2026-04-19:** `smoke_validate_handoffs_234` + `validate_handoffs.sh` aligned to 16 cells (`463d791`) | `463d791` + prior matrix commits on `main` | Yes |
| 9 | Kimi K2.5 vision via `extra_body.thinking=disabled + provider.order=[Moonshot]`, per-model `tool_choice` hybrid dispatch with retry guard (D1–D3), Gemma-4-31B as 4th matrix slot (D4/D5) with DeepInfra/Together provider pin | [HANDOFF_PENDING_KIMI_AND_TOOL_CHOICE.md](docs/handoffs/HANDOFF_PENDING_KIMI_AND_TOOL_CHOICE.md) | **IMPLEMENTED + LIVE-VERIFIED (2026-04-18)** — 139/139 unit tests green, 3/3 live probes passed (Kimi/Qwen/Gemma); **farm:** full pytest sweep + matrix smoke still to do | `fe3de8d` → `829d4d8` → `c17f24e` → `27d48e4` | Yes |

### Local-validation follow-ups (2026-04-18 session) — not original handoffs, but required to unblock local test-split prep

| Commit | Scope |
|--------|-------|
| `a98da9a` | Qwen `enable_thinking=False` on base variants; Kimi matrix switched to `or-kimi-k2.5` (OpenRouter) per operator direction; `del _os, _datetime` in all 16 generated/base configs so `mmengine.Config.pretty_text` doesn't crash yapf |
| `7ee9ae1` | MCP fence extraction rewritten to take only the first fenced block; `Tool.from_code` now synthesizes `parameters` from `inputs` when absent (fixes 3 ToolGenerator tests); `LogLevel.WARNING` added (fixes 3 skill_registry tests); `GAIADataset` gets `skip_file_attachments` + `task_ids` kwargs; `workdir/` added to `.gitignore` |
| `905a1fa` | `AutoBrowserUseTool.max_steps` now configurable (default 50 preserved); pass `auto_browser_use_tool_config.max_steps=8` for smoke validation |
| `4162bcc` | MCP fence extraction: unclosed opening fence now logs an explicit "missing closing ``` marker" rejection instead of exec'ing trailing junk |
| `0e7903c` | `scripts/validate_handoffs.sh` — per-run grep sweep producing a pass/info matrix across handoffs #1/#2/#3/#4/#5/#7 |
| `463d791` | **2026-04-19:** `smoke_validate_handoffs_234.sh` Tier-0 loads **16** matrix configs (adds Gemma); registration smoke matches `gen_eval_configs` defaults (`or-kimi-k2.5`, `or-qwen3.6-plus`, `or-gemma-4-31b-it` + langchain wrappers); `validate_handoffs.sh` greps **all four** models; `run_matrix_slurm.sh` / `run_eval_matrix.sh` comments 16-cell |

Farm operators: `git pull origin main` — follow-ups through `463d791` are on **`origin/main`** (as of 2026-04-19).

---

## Completed / Archived

_(none yet — move rows here once their validation pass is signed off on the GAIA test split.)_

---

## Conventions

- **One doc per logical change set.** If two fixes are independent (can be validated / reverted separately), they get separate docs.
- **Filename:** `HANDOFF_<SHORT_TOPIC>.md` in the repo root. Short topic is a noun phrase (e.g. `SILENT_FAILURES`, `PROVIDER_MATRIX`, `REVIEW_STEP_BUG`).
- **Each doc must include:**
  1. TL;DR checklist (done / to do)
  2. Original problem with log evidence or reproducer
  3. Changes table mapped to commit hash(es)
  4. GPU-farm test commands (prereqs, how to kick off, where results land)
  5. Validation criteria (concrete grep / diff / pass thresholds)
  6. Known unknowns / caveats
- **Status values in the table above:** `Drafting` → `Ready to validate` → `Validating` → `Validated` (move to Completed) / `Blocked` / `Rolled back`.
- **Pushed?** column reminds the next session whether `git push origin main` is still owed before anyone can `git pull` on the farm.

---

## Operator setup: API keys and model backends (matrix defaults)

The **checked-in 16-cell GAIA configs** (`configs/config_gaia_c{0,2,3,4}_{mistral,kimi,qwen,gemma}.py`) are generated by `scripts/gen_eval_configs.py` with these defaults (updated 2026-04-18 per handoff #9):

| Role | Default `model_id` | Required env vars | Notes |
|------|-------------------|-------------------|-------|
| Mistral | `mistral-small` (native La Plateforme) | `MISTRAL_API_KEY` | |
| Kimi | `or-kimi-k2.5` (OpenRouter) | `OPENROUTER_API_KEY` | `extra_body={thinking: disabled, provider.order: [Moonshot]}` baked into OR registration — gives Kimi vision on GAIA image questions while satisfying Moonshot's `tool_choice="required"` constraint. |
| Qwen | `or-qwen3.6-plus` (OpenRouter) | `OPENROUTER_API_KEY` | Whole Qwen family is blanket-downgraded to `tool_choice="auto"` at dispatch time (hybrid dispatch D3) + retry guard; vision + 1M context. |
| Gemma | `or-gemma-4-31b-it` (OpenRouter, paid — **`:free` excluded**) | `OPENROUTER_API_KEY` | Provider pin `DeepInfra+Together`, `reasoning.enabled=false`, concurrency ≤ 4 (vLLM #39392). Live-probed 2026-04-18 — accepts `tool_choice="required"` directly. |

**`OPENAI_API_KEY` is optional** for those runs: if it is unset or empty, `ModelManager` **skips** remote OpenAI + LangChain `langchain-gpt-*` wrappers (no crash). You only need it when a config selects an OpenAI model.

**`MOONSHOT_API_KEY`** is NOT required for the matrix defaults since Kimi routes via OpenRouter. If you want to switch Kimi to the native Moonshot API, edit `MODELS` in `scripts/gen_eval_configs.py` (`or-kimi-k2.5` → `kimi-k2.5-no-thinking`) and regen.

**`DASHSCOPE_API_KEY`** is NOT required for the matrix defaults since Qwen routes via OpenRouter directly (no failover wrapper). The `qwen3.6-plus-failover` alias remains registered but no current config uses it.

**Always:** `FIRECRAWL_API_KEY` for `web_searcher_tool`; `conda activate dra` (project's `dra` conda env is the only one with `crawl4ai` installed — see [CLAUDE.md](../../CLAUDE.md) for details); Playwright if the browser tool runs.

**Env replication:** `make install` (Poetry via `pyproject.toml`) or `make install-requirements` (via `requirements.txt`) — both manifests include `crawl4ai>=0.6.3`, `python-dotenv>=1.0.1`, `openai>=1.90.0`, and every runtime dep needed by `src.*` + the eval scripts.

**Quick smoke for handoffs #2–#4 (16 matrix configs + prompts + C3/C4 unit tests + 4-model registration):**  
`bash scripts/smoke_validate_handoffs_234.sh`

**Firecrawl balance:**  
`python scripts/check_firecrawl_credits.py` (uses `GET …/v1/team/credit-usage`). Credit use per GAIA run depends on how often the agent calls scrape/crawl, not on the number of LLM tokens — treat any estimate as approximate.

---

## What counts as “full” handoff validation vs smoke

| Level | What you run | What it proves |
|-------|----------------|------------------|
| **Smoke (minutes)** | `bash scripts/smoke_validate_handoffs_234.sh`, targeted `pytest`, `max_samples=1` GAIA, `python scripts/check_firecrawl_credits.py` | Wiring, registration, prompt render, no obvious regressions in isolated tests |
| **Full (per handoff doc)** | Full GAIA splits, matrix `smoke`/`full`, score/grep thresholds in each `HANDOFF_*.md` | Original research / release criteria |

To make **problem-solved** checks easier without full GAIA: prefer **one grep per fix** on a small run (e.g. `premature-final-answer guard` count, zero `No such file or directory: 'code.txt'`), keep **`scripts/smoke_validate_handoffs_234.sh`** green, and add **`STRICT_QWEN_FAILOVER=1`** to the smoke script when you require OpenRouter for Qwen failover registration.

---

## Time-limited validation scope (each bash job under about five minutes)

**Scope change:** Full GAIA or full **16-cell** matrix runs are **out of scope** when every bash invocation must finish in **under about five minutes**.

**Is that enough to “test”?**  
**Yes, for a limited definition:** it is **enough** to catch wiring regressions (config load, model registration, imports, a **single** GAIA question or `max_samples=1`, Tier-0 config loops, targeted pytest files). It is **not** enough to satisfy the original pass/fail thresholds in each handoff (accuracy vs baselines, full-matrix per-cell scores, DashScope quota → failover behavior, distribution of `AgentMaxStepsError` over hundreds of tasks, and so on). Treat any green short run as **smoke / sanity only**, not a completed handoff validation.

**Practical caps (keep wall-clock under ~5 minutes):** use `python examples/run_gaia.py ... --cfg-options max_samples=1 dataset.split=validation` (or `max_samples=2` if still within budget), run **one** matrix cell or **one** pytest module per bash job, and split work across **multiple** short jobs instead of `run_eval_matrix.sh full` or full test-split scripts.

**Sign-off:** When operating under this constraint, do **not** move handoff rows to **Completed / Archived** on score alone; note in the row or PR that validation was **smoke-only (≤5 min/job)** per this section.

---

## Progress snapshot (2026-04-19)

**Done (repo / local):** `origin/main` includes prior follow-ups (`a98da9a`, `7ee9ae1`, …) and **`463d791`** (16-cell handoff smoke + `validate_handoffs.sh` + SLURM/matrix script comments). `bash scripts/smoke_validate_handoffs_234.sh` is green with keys (Tier 0 = 16 configs, registration = 4 models + wrappers). Spot-check: `pytest tests/test_tool_choice_dispatch.py tests/test_failover_model.py` → **34 passed** (`dra` env, 2026-04-19).

**Not done (GPU farm / eval):** pytest sweep on farm (#9), **S0→S1→S2** smoke matrix (`sbatch run_matrix_slurm.sh smoke`), optional **C4 train + freeze-smoke**, then **S4** test-split + scoring + `validate_handoffs.sh <DRA_RUN_ID>` report — still required before moving rows to **Completed / Archived**. Handoff **#6** remains N/A for the API matrix.

## For the Next Session

1. Read this index first.
2. On the farm: `git pull origin main`, then run **farm pytest** (#9) and **S0→S1→S2** per [HANDOFF_TEST_EVAL.md](docs/handoffs/HANDOFF_TEST_EVAL.md); attach `validate_handoffs.sh` output for the matrix `DRA_RUN_ID`.
3. When the **test-split** run is signed off, update each row's Status, move to Completed, and note the validating run's tag.
