# Handoff Index

Entry point for change handoffs that still need validation on the GPU farm.
Each linked document is **self-contained** — read the specific handoff doc for
original problem, changes, test commands, and validation criteria.

> Per-topic handoff docs live in `[docs/handoffs/](docs/handoffs/)` (see the
> [directory README](docs/handoffs/README.md) for authoring conventions). This
> index stays at the repo root because the operator is trained to open it
> first.

---

## Pending Handoffs


| #   | Title                                                                                                                                                                                                                          | Doc                                                                                              | Status                                                                                                                                                                                                                  | Commit(s)                                                                       | Pushed?               |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------- |
| 1   | Silent-failure fixes for browser + analyzer tools                                                                                                                                                                              | [HANDOFF_SILENT_FAILURES.md](docs/handoffs/HANDOFF_SILENT_FAILURES.md)                           | Code-validated (unit + partial runtime, 2026-04-18) — awaiting GPU-farm test-split run                                                                                                                                  | `ba28f21`                                                                       | Yes                   |
| 2   | Multi-provider integration + GAIA eval matrix (Mistral/Kimi/Qwen × C0–C4) with Qwen DashScope→OpenRouter failover                                                                                                              | [HANDOFF_PROVIDER_MATRIX.md](docs/handoffs/HANDOFF_PROVIDER_MATRIX.md)                           | Code-validated (24 unit tests; failover live-fired 2026-04-18; see local-validation fixes below) — awaiting test-split accuracy                                                                                         | `7632470` → `9883a3a` → `a98da9a` (local fixes: Qwen thinking, Kimi→OpenRouter) | Yes                   |
| 3   | `modify_subagent` prompt + tool-description guidance expansion (all 7 actions covered, failure-mode→action table, condition-scoped anti-patterns for C2/C3/C4)                                                                 | [HANDOFF_MODIFY_SUBAGENT_GUIDANCE.md](docs/handoffs/HANDOFF_MODIFY_SUBAGENT_GUIDANCE.md)         | Code-validated (93 adaptive-tool mentions in C2+ runtime logs, 2026-04-18) — awaiting test-split                                                                                                                        | `764c6bf` → `b73eb39`                                                           | Yes                   |
| 4   | Core C3 / C4 implementation — structural REVIEW step + cross-task skill library (the four experimental conditions C0/C2/C3/C4 themselves)                                                                                      | [HANDOFF_C3_C4_IMPLEMENTATION.md](docs/handoffs/HANDOFF_C3_C4_IMPLEMENTATION.md)                 | Code-validated (60 unit tests: 26 review_schema + 28 skill_registry + 6 skill_seed; smoke matrix targets 4/4 C4 `SkillRegistry` banners — awaiting `[REVIEW]` marker under traffic + test-split)                        | `60065a8` → `433c30e` → `0643089` → `d247605`                                   | Yes                   |
| 5   | RC1 premature `final_answer_tool` guard + duplicate-yield bug fix + RC2 exception-chain diagnostic hook + prompt contradictions fix                                                                                            | [HANDOFF_RC1_FINAL_ANSWER_GUARD.md](docs/handoffs/HANDOFF_RC1_FINAL_ANSWER_GUARD.md)             | Code-validated (18 unit tests, 2026-04-18) — awaiting guard-fire in real traffic                                                                                                                                        | `54e7707` → `a9a6985` → `c52cf91` → `912685f` → `d36f4d4`                       | Yes                   |
| 6   | Pass 2 Qwen-4B (vLLM) tuning — sub-agent `max_steps` 3/3/5→7/7/7 and `context_prune_threshold_ratio`=0.75 on `config_gaia_adaptive_qwen.py`                                                                                    | [HANDOFF_PASS2_QWEN_TUNING.md](docs/handoffs/HANDOFF_PASS2_QWEN_TUNING.md)                       | **N/A for 3-model API matrix** (Qwen-4B local-vLLM only; reinstate when returning to on-prem Qwen-4B runs)                                                                                                              | `63486ca`                                                                       | **No — push pending** |
| 7   | ToolGenerator hardening — allowlist + AST imports, repair retry, prompt examples, `Tool.from_code(expected_tool_name)`, collision-safe dynamic tool names, unit tests                                                          | [HANDOFF_TOOLGENERATOR.md](docs/handoffs/HANDOFF_TOOLGENERATOR.md)                               | Code-validated (12/12 unit tests after `7ee9ae1` schema fallback fix, 2026-04-18) — awaiting test-split                                                                                                                 | `0161321` → `7ee9ae1` (Tool.from_code parameters synthesis fix)                 | Yes                   |
| 8   | GAIA test-split execution protocol for the (now-)16-cell matrix on the HKU CS GPU farm (staged **I0–I3** integration + **E0–E3** evaluation + grep sweep + resume protocol; optional `scripts/integration_i3_c4_pipeline.sh`)                                                                   | [HANDOFF_TEST_EVAL.md](docs/handoffs/HANDOFF_TEST_EVAL.md)                                       | Ready to execute — **unblocked 2026-04-18**; **2026-04-19:** 16-cell smoke + `validate_handoffs` (`463d791`); **I2** default **3 Q/cell** + smoke step caps + parallel model streams (`run_eval_matrix.sh`) | `463d791` + prior matrix commits on `main`                                      | Yes                   |
| 9   | Kimi K2.5 vision via `extra_body.thinking=disabled + provider.order=[Moonshot]`, per-model `tool_choice` hybrid dispatch with retry guard (D1–D3), Gemma-4-31B as 4th matrix slot (D4/D5) with DeepInfra/Together provider pin | [HANDOFF_PENDING_KIMI_AND_TOOL_CHOICE.md](docs/handoffs/HANDOFF_PENDING_KIMI_AND_TOOL_CHOICE.md) | **IMPLEMENTED + LIVE-VERIFIED** — **140/140** unit tests (one-file sweep, 2026-04-19; `bash scripts/run_handoff_pytest_sweep.sh`), 3/3 live probes (2026-04-18); **farm:** repeat sweep + matrix smoke (**I2**) still to do | `fe3de8d` → `829d4d8` → `c17f24e` → `27d48e4`                                   | Yes                   |


### Local-validation follow-ups (2026-04-18 session) — not original handoffs, but required to unblock local test-split prep


| Commit    | Scope                                                                                                                                                                                                                                                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `a98da9a` | Qwen `enable_thinking=False` on base variants; Kimi matrix switched to `or-kimi-k2.5` (OpenRouter) per operator direction; `del _os, _datetime` in all 16 generated/base configs so `mmengine.Config.pretty_text` doesn't crash yapf                                                                                                                    |
| `7ee9ae1` | MCP fence extraction rewritten to take only the first fenced block; `Tool.from_code` now synthesizes `parameters` from `inputs` when absent (fixes 3 ToolGenerator tests); `LogLevel.WARNING` added (fixes 3 skill_registry tests); `GAIADataset` gets `skip_file_attachments` + `task_ids` kwargs; `workdir/` added to `.gitignore`                    |
| `905a1fa` | `AutoBrowserUseTool.max_steps` now configurable (default 50 preserved); pass `auto_browser_use_tool_config.max_steps=8` for smoke validation                                                                                                                                                                                                            |
| `4162bcc` | MCP fence extraction: unclosed opening fence now logs an explicit "missing closing ``` marker" rejection instead of exec'ing trailing junk                                                                                                                                                                                                              |
| `0e7903c` | `scripts/validate_handoffs.sh` — per-run grep sweep producing a pass/info matrix across handoffs #1/#2/#3/#4/#5/#7                                                                                                                                                                                                                                      |
| `463d791` | **2026-04-19:** `smoke_validate_handoffs_234.sh` Tier-0 loads **16** matrix configs (adds Gemma); registration smoke matches `gen_eval_configs` defaults (`or-kimi-k2.5`, `or-qwen3.6-plus`, `or-gemma-4-31b-it` + langchain wrappers); `validate_handoffs.sh` greps **all four** models; `run_matrix_slurm.sh` / `run_eval_matrix.sh` comments 16-cell |
| `5299b41` | [`scripts/run_handoff_pytest_sweep.sh`](scripts/run_handoff_pytest_sweep.sh) — one-file-at-a-time **140**-test sweep for handoff #9 / farm CI (same modules as [HANDOFF_PENDING_KIMI_AND_TOOL_CHOICE.md](docs/handoffs/HANDOFF_PENDING_KIMI_AND_TOOL_CHOICE.md)); auto `conda run -n dra` when `mmengine` missing |
| `9d66808` | **2026-04-19:** **I2** smoke default **3 Q/cell** + `SMOKE_CFG_OPTIONS` step caps in [`scripts/run_eval_matrix.sh`](scripts/run_eval_matrix.sh); [HANDOFF_TEST_EVAL.md](docs/handoffs/HANDOFF_TEST_EVAL.md) §Full chain = **I2 → E0 → E1 → E2 → E3** (legacy S2→…→S4); output retention + parallel streams documented |
| `44b0227` | **C4 train / E3 split correctness:** `DATASET_SPLIT=validation` + `sbatch run_matrix_slurm.sh full '' c4` trains skills on **full validation** (config default is `test`); **unset** before **E3** so all 16 cells score **`test`**; frozen C4 cells still use `agent_config.*` overrides per [HANDOFF_TEST_EVAL.md](docs/handoffs/HANDOFF_TEST_EVAL.md) |
| `4d6c2e6` | **2026-04-19:** **`DATASET_SPLIT` is now mandatory in full mode** — [`scripts/run_eval_matrix.sh`](scripts/run_eval_matrix.sh) refuses unset, refuses `c4 + test` (train-on-test leakage), refuses `c0/c2/c3 + validation` (eval-on-val). Closes the two silent footguns that could have invalidated E3. E3 16-cell shape (`ONLY_CONDITION=""`) is no longer valid on test — submit per-condition (`full '' c0` / `c2` / `c3`) plus frozen C4 via `examples/run_gaia.py`. [`scripts/integration_i3_c4_pipeline.sh`](scripts/integration_i3_c4_pipeline.sh) now echoes `I3a skill count: N total (S seeded, L learned)` after I3a so seed-only runs are visible before I3c. [HANDOFF_TEST_EVAL.md](docs/handoffs/HANDOFF_TEST_EVAL.md) glossary + §E0/§E3 snippets updated to `DATASET_SPLIT=... sbatch --export=ALL ...`. |
| `3b6b7bd` | **2026-04-19:** **Tighter I-track smoke caps** in both [`scripts/run_eval_matrix.sh`](scripts/run_eval_matrix.sh) and [`scripts/integration_i3_c4_pipeline.sh`](scripts/integration_i3_c4_pipeline.sh) for faster local integration runs: `agent_config.max_steps` 10→6, `auto_browser_use_tool_config.max_steps` 8→4 (largest wall-time knob — CAPTCHA loops), `browser_use_agent_config.max_steps` 3→2, `deep_researcher_tool_config.time_limit_seconds` 30→20. Analyzer/researcher agent `max_steps=2` unchanged. **Full-mode runs unaffected** (`cell_cmd` full branch does not read `SMOKE_CFG_OPTIONS`, and `auto_browser_use_tool_config.max_steps=15` stays pinned in generated configs per [f73a666](https://github.com/ambr-0-se/DeepResearchMetaAgent/commit/f73a666)). Per-cell smoke wall drops from ~5-10 min to ~1-3 min on stuck-browser questions. |
| `208a199` | **2026-04-19:** **Conda re-exec probe now checks `crawl4ai`, not just `mmengine`** in [`scripts/run_handoff_pytest_sweep.sh`](scripts/run_handoff_pytest_sweep.sh) and [`scripts/integration_i3_c4_pipeline.sh`](scripts/integration_i3_c4_pipeline.sh). Local-mac repro: base `/Users/ahbo/miniconda3/bin/python` has `mmengine` but not `crawl4ai`, so the old probe falsely concluded "no re-exec needed" and the sweep later crashed on `src/__init__.py` → `src/utils/url_utils.py` → `from crawl4ai import AsyncWebCrawler`. Per CLAUDE.md, `crawl4ai` is the dra-only canary; probe `import mmengine, crawl4ai` together. Farm side unaffected (bare base env lacks both). |


Farm operators: `git pull origin main` — follow-ups through `5299b41`+ are on `origin/main` (as of 2026-04-19).

---

## Completed / Archived

*(none yet — move rows here once their validation pass is signed off on the GAIA test split.)*

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


| Role    | Default `model_id`                                            | Required env vars    | Notes                                                                                                                                                                                                 |
| ------- | ------------------------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mistral | `mistral-small` (native La Plateforme)                        | `MISTRAL_API_KEY`    |                                                                                                                                                                                                       |
| Kimi    | `or-kimi-k2.5` (OpenRouter)                                   | `OPENROUTER_API_KEY` | `extra_body={thinking: disabled, provider.order: [Moonshot]}` baked into OR registration — gives Kimi vision on GAIA image questions while satisfying Moonshot's `tool_choice="required"` constraint. |
| Qwen    | `or-qwen3.6-plus` (OpenRouter)                                | `OPENROUTER_API_KEY` | Whole Qwen family is blanket-downgraded to `tool_choice="auto"` at dispatch time (hybrid dispatch D3) + retry guard; vision + 1M context.                                                             |
| Gemma   | `or-gemma-4-31b-it` (OpenRouter, paid — `:free` excluded) | `OPENROUTER_API_KEY` | Provider pin `DeepInfra+Together`, `reasoning.enabled=false`, concurrency ≤ 4 (vLLM #39392). Live-probed 2026-04-18 — accepts `tool_choice="required"` directly.                                      |


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


| Level                      | What you run                                                                                                                        | What it proves                                                                |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **I-track / smoke (minutes)** | `bash scripts/smoke_validate_handoffs_234.sh` (**I0**), matrix `smoke` (**I2**), optional `integration_i3_c4_pipeline.sh` (**I3**), targeted `pytest`, `max_samples=1` GAIA, `python scripts/check_firecrawl_credits.py` | Wiring, registration, prompt render, no obvious regressions in isolated tests |
| **E-track / full (per handoff doc)** | Full GAIA **test** split, matrix `full` (**E3**), **E0–E2** C4 prep when needed, score/grep thresholds in each `HANDOFF_*.md`                                               | Original research / release criteria                                          |


To make **problem-solved** checks easier without full GAIA: prefer **one grep per fix** on a small run (e.g. `premature-final-answer guard` count, zero `No such file or directory: 'code.txt'`), keep `scripts/smoke_validate_handoffs_234.sh` green, and add `STRICT_QWEN_FAILOVER=1` when you require OpenRouter for Qwen failover registration.

---

## Time-limited validation scope (each bash job under about five minutes)

**Scope change:** Full GAIA or full **16-cell** matrix runs are **out of scope** when every bash invocation must finish in **under about five minutes**.

**Is that enough to “test”?**  
**Yes, for a limited definition:** it is **enough** to catch wiring regressions (config load, model registration, imports, a **single** GAIA question or `max_samples=1`, Tier-0 config loops, targeted pytest files). It is **not** enough to satisfy the original pass/fail thresholds in each handoff (accuracy vs baselines, full-matrix per-cell scores, DashScope quota → failover behavior, distribution of `AgentMaxStepsError` over hundreds of tasks, and so on). Treat any green short run as **smoke / sanity only**, not a completed handoff validation.

**Practical caps (keep wall-clock under ~5 minutes):** use `python examples/run_gaia.py ... --cfg-options max_samples=1 dataset.split=validation` (or `max_samples=2` if still within budget), run **one** matrix cell or **one** pytest module per bash job, and split work across **multiple** short jobs instead of `run_eval_matrix.sh full` or full test-split scripts.

**Sign-off:** When operating under this constraint, do **not** move handoff rows to **Completed / Archived** on score alone; note in the row or PR that validation was **smoke-only (≤5 min/job)** per this section.

---

## Progress snapshot (2026-04-19)

**Done (repo / local):** `origin/main` includes prior follow-ups (`a98da9a`, `7ee9ae1`, …), `463d791` (16-cell handoff smoke + `validate_handoffs.sh` + SLURM/matrix script comments), and **`9d66808`** (**I2** = default **3 Q/cell** + smoke step caps; docs: **I2→E0→E1→E2→E3**, retention, parallelism). `bash scripts/smoke_validate_handoffs_234.sh` is green with keys (Tier 0 = 16 configs, registration = 4 models + wrappers). **Handoff #9 unit sweep:** `bash scripts/run_handoff_pytest_sweep.sh` → **140/140** passed (`dra`, 2026-04-19).

**Not done (GPU farm / eval):** repeat `bash scripts/run_handoff_pytest_sweep.sh` (CI parity), then **I0→I1→I2** (`sbatch run_matrix_slurm.sh smoke` — default **3 Q/cell** + smoke step caps; **four models in parallel**). Optionally **I3** when C4 code/config changes (`bash scripts/integration_i3_c4_pipeline.sh`). Before **E3**, complete **E0 → E1 → E2** when reporting **C4** on `test` (see [HANDOFF_TEST_EVAL.md](docs/handoffs/HANDOFF_TEST_EVAL.md) §Full chain). Then **E3** + scoring + `validate_handoffs.sh <DRA_RUN_ID>`. **Retain** all `workdir/` run dirs and `logs/` until review. Handoff **#6** remains N/A for the API matrix.

## For the Next Session

1. Read this index first.
2. On the farm: `git pull origin main`, then `bash scripts/run_handoff_pytest_sweep.sh` (#9 CI parity) and **I0→I1→I2** per [HANDOFF_TEST_EVAL.md](docs/handoffs/HANDOFF_TEST_EVAL.md) (I2 defaults: 3 Q/cell + caps; parallel streams). Optionally **I3** for a one-model C4 pipeline smoke. Run **E0 → E1 → E2** before **E3** if C4 is in scope; keep all outputs for review.
3. When the **test-split** run is signed off, update each row's Status, move to Completed, and note the validating run's tag.