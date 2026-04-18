# Handoff: GAIA test-split evaluation — 16-cell matrix on the HKU CS GPU farm

**Session date:** 2026-04-18
**Branch / HEAD push:** `main` at `27d48e4` (post-handoff-#9 implementation)
**Scope:** Executes the GAIA submission run for the APAI4799 meta-agent paper.
Produces **4 models × 4 conditions = 16 `dra.jsonl` files** on the GAIA test
split (~300 questions each; ~4,800 Q total), plus the C4 training pass
needed to harden the skill library before the scored evaluation.

> **Matrix-size update (2026-04-18):** expanded from 12 cells to 16 after
> handoff #9 landed (D4 added Gemma-4-31B as the 4th model slot). Cost +~$10-35
> on the full test-split run vs the old 12-cell budget.

**Dependencies:**
- [`HANDOFF_SILENT_FAILURES.md`](HANDOFF_SILENT_FAILURES.md) (`ba28f21`) — browser/analyzer silent-fail fixes
- [`HANDOFF_PROVIDER_MATRIX.md`](HANDOFF_PROVIDER_MATRIX.md) (`7632470` → `9883a3a`) — Mistral / Kimi / Qwen registration + failover
- [`HANDOFF_MODIFY_SUBAGENT_GUIDANCE.md`](HANDOFF_MODIFY_SUBAGENT_GUIDANCE.md) (`764c6bf` → `b73eb39`) — C2+ prompt expansions
- [`HANDOFF_C3_C4_IMPLEMENTATION.md`](HANDOFF_C3_C4_IMPLEMENTATION.md) (`60065a8` → `d247605`) — ReviewStep + Skill library
- [`HANDOFF_RC1_FINAL_ANSWER_GUARD.md`](HANDOFF_RC1_FINAL_ANSWER_GUARD.md) (`54e7707` → `d36f4d4`) — RC1/RC2 guards
- [`HANDOFF_TOOLGENERATOR.md`](HANDOFF_TOOLGENERATOR.md) (`0161321` → `7ee9ae1`) — ToolGenerator hardening

Plus the 2026-04-18 local-validation follow-ups listed in
[`HANDOFF_INDEX.md`](../../HANDOFF_INDEX.md): `a98da9a`, `7ee9ae1`, `905a1fa`,
`4162bcc`, `0e7903c`, `c2fd507`, `6f5ddd1`, and this session's Qwen-swap commit.

---

## TL;DR Checklist

### Before the farm

- [ ] `git pull origin main` → HEAD includes this session's Qwen-swap commit
- [ ] `.env` populated on farm with: `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`,
      `FIRECRAWL_API_KEY`, **`HF_TOKEN`** (required — GAIA is a gated dataset)
- [ ] `DASHSCOPE_API_KEY` / `MOONSHOT_API_KEY` can be placeholders —
      **Kimi uses `or-kimi-k2.5`** (OpenRouter, with
      `extra_body.thinking=disabled + provider.order=[Moonshot]`),
      **Qwen uses `or-qwen3.6-plus`** (OpenRouter; hybrid `tool_choice`
      dispatch resolves to `"auto"` per the Qwen-family prefix rule),
      **Gemma uses `or-gemma-4-31b-it`** (OpenRouter paid; provider pin
      `DeepInfra+Together`, `reasoning.enabled=false`, per-stream
      concurrency capped at 4)
- [ ] `OPENAI_API_KEY` optional — `ModelManager` short-circuits on empty
- [ ] Pull the GAIA dataset once into `data/GAIA/` (see "Prerequisites" below)

### On the farm

- [ ] **S0** pre-flight: `bash scripts/smoke_validate_handoffs_234.sh`
- [ ] **S1** 1-Q canary for one (model, condition) cell to prove the pipeline
- [ ] **S2** smoke matrix: `sbatch run_matrix_slurm.sh smoke`
- [ ] **C4 training pass** (see §C4 Train/Freeze below) — OPTIONAL but
      recommended for publishable C4 numbers
- [ ] **Post-S2 snapshot** — `cp -a` C4 `skills/` into
      `workdir/c4_trained_libraries/{model}_skills` (see §C4 Train/Freeze,
      lines 171–192)
- [ ] **Farm-side freeze smoke** (recommended before S4; ~30 min, ~$0.80):
      3-Q × 4 models with `agent_config.*` override — verifies the
      mechanism on the farm against real learned libraries
- [ ] **S4** test-split submission: `sbatch run_matrix_slurm.sh full`
- [ ] Collect `dra.jsonl` → run `scripts/analyze_results.py` per cell
- [ ] `bash scripts/validate_handoffs.sh <DRA_RUN_ID>` → attach pass/info
      summary to this handoff when promoting to Completed

---

## Matrix definition

16 cells = 4 models × 4 conditions:

| Condition | Meta-agent capability added | Configs / model slot |
|-----------|------------------------------|----------------------|
| **C0** | — (vanilla `PlanningAgent` baseline) | `configs/config_gaia_c0_<model>.py` |
| **C2** | Reactive `diagnose_subagent` + `modify_subagent` | `configs/config_gaia_c2_<model>.py` |
| **C3** | C2 + structural REVIEW step | `configs/config_gaia_c3_<model>.py` |
| **C4** | C3 + cross-task skill library (pre-seeded + learned) | `configs/config_gaia_c4_<model>.py` |

| Model slot | Real slug (model_id) | Cost (in/out /M) | tool_choice handling | Rationale / caveats |
|------------|----------------------|-------------------|-----------------------|---------------------|
| **Mistral** | `mistral-small` (native La Plateforme) | $0.15 / $0.60 | `"required"` works | Dense ~24B; uses `MISTRAL_API_KEY`. |
| **Kimi** | `or-kimi-k2.5` (OpenRouter) | free tier | `"required"` works after extra_body fix (thinking off) | `extra_body={thinking: disabled, provider.order: [Moonshot]}` pins routing so free-tier OR can't silently fall back to a sub-provider with diverging thinking semantics. Enables vision on GAIA image questions. |
| **Qwen** | `or-qwen3.6-plus` (OpenRouter, D1) | $0.325 / $1.95 | **hybrid dispatch → "auto"** (Qwen-family prefix rule, D3) | Vision + 1M context. OR providers for the whole Qwen family reject `"required"`; hybrid dispatch + retry guard coax plain-text replies back into tool calls. |
| **Gemma** (D4) | `or-gemma-4-31b-it` (OpenRouter paid) | $0.13 / $0.38 | `"required"` works directly (verified 2026-04-18) | Dense 31B, Apache 2.0, only non-MoE in the matrix. Provider pin `DeepInfra+Together` + `reasoning.enabled=false`; `:free` variant excluded (Google AI Studio lacks reliable `tools` + `required`). Per-stream concurrency capped at 4 (vLLM #39392 pad-parser bug). |

---

## Prerequisites

```bash
ssh gpu3gate1.cs.hku.hk                        # HKU CS Phase-3 gateway
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
git pull origin main                           # ensure HEAD has this handoff's commits

conda activate dra                             # must already exist; if not, run `make install-requirements`
bash scripts/ensure_playwright_browsers.sh     # browser_use_agent needs Chromium

# GAIA dataset (one-time; ~200 MB)
python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='gaia-benchmark/GAIA',
                  repo_type='dataset',
                  local_dir='data/GAIA',
                  token=os.environ['HF_TOKEN'])"
ls data/GAIA  # should contain 2023/ with validation + test subdirs
```

---

## Execution protocol (staged; each gate must pass before the next)

### S0 — Pre-flight (free, ~30 sec)

```bash
bash scripts/smoke_validate_handoffs_234.sh
```

Expected: all 12 configs load, model registration reports `Missing: []`, C3
schema + C4 skill parser unit tests green.

### S1 — Single-cell canary (~10 min, <$0.50)

```bash
sbatch run_matrix_slurm.sh smoke mistral c0        # cheapest model, baseline condition, 5 Q
squeue -u $USER                                    # wait for it to finish
tail -f logs/matrix_<JOBID>.out                    # watch live
```

Pass criterion: a non-empty `workdir/gaia_c0_mistral_<run_id>/dra.jsonl` with
at least one row containing a non-null `prediction`.

### S2 — Smoke matrix (~1-2 h, $2-5)

```bash
sbatch run_matrix_slurm.sh smoke                   # 5 Q × 16 cells = 80 Q, validation split
```

Pass criteria (check `logs/matrix_<JOBID>.out` via the auto-run
`validate_handoffs.sh` summary at the bottom):

- 16 `dra.jsonl` files written (one per cell)
- `Handoff #2`: 0 Kimi sampling-lock 400s, 0 Qwen thinking-mode 400s
- `Handoff #3`: >0 `modify_subagent` / `diagnose_subagent` mentions in
  C2 / C3 / C4 cell logs
- `Handoff #4 C3`: ≥1 `enable_review=True; building ReviewStep` banner per
  C3 cell; ≥1 `[REVIEW]` marker once the planner delegates
- `Handoff #4 C4`: 3 `SkillRegistry built at workdir/.../skills (C4)` banners;
  `[seed_skills_dir] seeded ...` fires ≥ 7× per C4 cell
- 0 Python tracebacks in any per-cell `log.txt` (the stream-log MCP-stdio
  parse errors are known cosmetic noise — ignore)

If S2 fails for any cell, **stop** and diagnose before moving to S4.
Typical triggers:
- Mistral → `DeepResearchTool RetryError[ValueError]` usually means
  Firecrawl credits exhausted — run `python scripts/check_firecrawl_credits.py`.
- Kimi → 401 from OpenRouter means `OPENROUTER_API_KEY` is wrong for that
  model scope.
- Qwen → 404 "no endpoints support tool_choice" means either (a) the matrix
  config is on an older slug — should be `or-qwen3.6-plus` (see
  `scripts/gen_eval_configs.py` `MODELS` table), or (b) hybrid dispatch
  didn't fire — confirm the once-per-run `[tool_choice] qwen/... -> auto`
  INFO log is present and that `src/models/tool_choice.py` exists.
- Gemma → 404 on `tool_choice` means the provider pin expired or a new OR
  backend entered the pool; restrict to `DeepInfra+Together` via the
  registration `extra_body` in `src/models/models.py`.
- Gemma → garbled content / `<|tool_call>` leak in text means a stale
  chat template on a specific provider; narrow the provider pin further.

### C4 Train/Freeze pass (OPTIONAL, ~3-6 h, $5-15) — recommended for paper

**Why this matters.** C4's `enable_skill_extraction=True` mutates the
`skills_dir` at the end of every task. If you run C4 on the test split with
extraction still enabled, then:

1. The skill library grows **during scoring** — question N's result depends
   on the skills extracted by questions 1..N-1, so results are
   order-dependent and not cleanly reproducible.
2. You're conflating two things the paper is supposed to separate:
   *skill-library utility* (how much the seeded + learned skills help) vs.
   *online-learning dynamics* (how well the extractor adds good skills).

Standard ML methodology → **train then freeze**:

```bash
# 1. Training: let C4 evolve skills on the labelled validation split.
#    All 4 models in parallel; extraction stays on.
sbatch run_matrix_slurm.sh full '' c4
# => workdir/gaia_c4_{mistral,kimi,qwen,gemma}_<TRAIN_RUN_ID>/
#    each ends with a `skills/` dir containing seeded + learned SKILL.md.

# 2. Snapshot the trained libraries and stage them as the starting point
#    for the scored run. Run ONCE before S4.
TRAIN_RUN_ID=<copy from step 1 logs>
mkdir -p workdir/c4_trained_libraries
for m in mistral kimi qwen gemma; do
  cp -r workdir/gaia_c4_${m}_${TRAIN_RUN_ID}/skills \
        workdir/c4_trained_libraries/${m}_skills
done
```

For the S4 scored run, pass an override so C4 cells load the trained
library and **do not** extract further (see S4 below).

### Freeze-smoke validation (2026-04-18, Mac — Mistral × 1 Q)

The train/freeze override mechanism was validated end-to-end locally before
committing to any farm training pass, using a **synthetic "trained library"**
(the 7 canonical seeds from `src/skills/` + a uniquely-named canary
`freeze-canary-mistral`) pinned via `--cfg-options`. This caught a latent
bug in the override target.

**Bug found.** The config file contains `agent_config = planning_agent_config`
at its tail ([`configs/config_gaia_c4_mistral.py:84`](../../configs/config_gaia_c4_mistral.py)),
but `mmengine.Config.fromfile` materialises the two names as **independent
`ConfigDict` instances** (different `id()`). `merge_from_dict` with a
dotted key mutates only the keyed dict, so
`planning_agent_config.skills_dir=<snapshot>` updates `planning_agent_config`
while `agent_config` retains the file-load default
`workdir/gaia_c4_<model>_<run_id>/skills`. `create_agent()` at
[`src/agent/agent.py:108`](../../src/agent/agent.py) then reads
`config.agent_config`, so the override is silently ignored. Result: the
"frozen" run actually runs in **C4 training mode** (extraction ON, fresh
per-run `skills_dir` that re-seeds from `src/skills/`). Without this local
validation, the farm C4 S4 run would have produced order-dependent,
extraction-contaminated numbers — exactly the failure mode the train/freeze
protocol exists to prevent.

**Fix.** Override `agent_config.*` instead of `planning_agent_config.*`.
The corrected commands above (lines 207–229) reflect this.

**Verification results** (run id `freeze_smoke_mac_20260418_221849`, full log at
`logs/c4_freeze_smoke_mistral.log`):

| # | Check | Command / site | Result |
|---|-------|---------------|--------|
| 1 | `SkillExtractor active (C4 training mode)` banner suppressed | `grep -c "SkillExtractor active" logs/c4_freeze_smoke_mistral.log` | **0** ✓ |
| 2 | Snapshot skill dir count unchanged | `ls workdir/c4_trained_libraries/mistral_skills/ \| wc -l` | **8 / 8** ✓ (7 seeds + canary; identical pre/post) |
| 3 | No writes to snapshot | implicit — `SkillExtractor` not constructed (line 120 gate) | ✓ |
| 4 | Library was actually consumed | `grep -c "Calling tool: 'activate_skill'"` | **1** ✓ (`handling-file-attachments`) |
| 5 | Canary visible in registry injection (proves snapshot pinning) | `grep -n "freeze-canary-mistral" logs/c4_freeze_smoke_mistral.log` | **line 541** ✓ |

Additional positive signals in the log head:

- `[AdaptivePlanningAgent] enable_skills=True; building SkillRegistry at workdir/c4_trained_libraries/mistral_skills (C4)` — override path honoured
- `[seed_skills_dir] workdir/c4_trained_libraries/mistral_skills already seeded (marker present); skipping.` — `.seeded` correctly prevents re-seed over a pre-built snapshot
- `[AdaptivePlanningAgent] Initialized with 5 tools, 3 managed agents, review_step=on, skill_registry=on`

**Local-only gotcha (not needed on the farm).** On the Mac dev box, the MCP
server subprocess is spawned with `command='python'` and inherits PATH, not
the parent's interpreter. The default shell `python` resolves to base
miniconda (no `fastmcp`), causing `ModuleNotFoundError: No module named
'fastmcp'` → `Connection closed`. Fix locally with an extra cfg-option:
`mcp_tools_config.mcpServers.LocalMCP.command=/Users/.../miniconda3/envs/dra/bin/python`.
On the farm this is a non-issue because `conda activate dra` in the SBATCH
wrapper puts the right `python` in PATH before the subprocess spawns.

**What's NOT yet validated.** The synthetic library contained only seed
skills + canary, so this run proves the override *mechanism* but not the
full train-then-freeze *loop* with a genuinely trained library. The first
farm C4 Train pass is still the first end-to-end test of real skill
extraction + later freeze. Reserve the 1 h budget slot mentioned in
"Known unknowns" for that.

### Farm-side freeze smoke (~30 min, ~$0.80) — recommended before S4

Integration test: the Mac validation above proved the *mechanism*; this
step proves the same override also holds on the HKU CS farm environment
(`conda activate dra` instead of the Mac `python` workaround) and against
a genuinely-trained library (seeds + newly-extracted skills), **before**
committing the 8–24 h, $30–100 S4 scored run.

**Inputs:** requires the Post-S2 snapshot step (lines 171–192) to have
completed, so `workdir/c4_trained_libraries/{mistral,kimi,qwen,gemma}_skills/`
exist and each contains the `.seeded` marker.

**Cost caps** (applied inline via `--cfg-options` — important; the Mac
dry run spent ~15 min on a single CAPTCHA retry loop because defaults
allow 50-step browser sessions). Smoke budget per cell ≈ 3 Q × ~$0.05 =
~$0.15; × 4 models = ~$0.60 + orchestration overhead.

- `agent_config.max_steps=10` — plan budget (default 25)
- `auto_browser_use_tool_config.max_steps=8` — internal browser loop cap
  (default 50; recommended smoke value per commit `905a1fa`)
- `deep_analyzer_agent_config.max_steps=2` (default 3)
- `deep_researcher_agent_config.max_steps=2` (default 3)
- `browser_use_agent_config.max_steps=3` (default 5)
- `deep_researcher_tool_config.time_limit_seconds=30` (default 60)

These caps are smoke-appropriate only — do **not** reuse them in S4.

**Run** (one 3-Q cell per model, in parallel is fine):

```bash
for m in mistral kimi qwen gemma; do
  sbatch --job-name=gaia-c4-freeze-${m} --time=1:00:00 \
         --output=logs/c4_freeze_smoke_${m}_%j.out \
         --error=logs/c4_freeze_smoke_${m}_%j.err \
         --wrap "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dra \
                 && cd /userhome/cs2/ambr0se/DeepResearchMetaAgent \
                 && python examples/run_gaia.py \
                      --config configs/config_gaia_c4_${m}.py \
                      --cfg-options \
                        max_samples=3 \
                        dataset.split=validation \
                        agent_config.skills_dir=workdir/c4_trained_libraries/${m}_skills \
                        agent_config.enable_skill_extraction=False \
                        agent_config.max_steps=10 \
                        auto_browser_use_tool_config.max_steps=8 \
                        deep_analyzer_agent_config.max_steps=2 \
                        deep_researcher_agent_config.max_steps=2 \
                        browser_use_agent_config.max_steps=3 \
                        deep_researcher_tool_config.time_limit_seconds=30"
done
```

**Canary (optional but recommended).** Inject a uniquely-named planner-scope
skill into each snapshot before this run, so you can assert it appears in
the planner's registry-injection block in the smoke log. Pattern:

```bash
for m in mistral kimi qwen gemma; do
  mkdir -p "workdir/c4_trained_libraries/${m}_skills/freeze-canary-farm-${m}"
  cat > "workdir/c4_trained_libraries/${m}_skills/freeze-canary-farm-${m}/SKILL.md" <<EOF
---
name: freeze-canary-farm-${m}
description: Canary — proves snapshot pinning on the farm. Safe to ignore.
metadata:
  consumer: planner
  skill_type: verification_pattern
  source: seeded
  verified_uses: 0
  confidence: 0.5
---
# freeze-canary-farm-${m}
Canary body. Never triggers naturally.
EOF
done
```

**Pass criteria** (per model, mirroring the Mac validation table):

| # | Check | Command | Expect |
|---|-------|---------|--------|
| 1 | Extractor not constructed | `grep -c "SkillExtractor active (C4 training mode)" logs/c4_freeze_smoke_${m}_*.out` | **0** |
| 2 | No writes to snapshot | `find workdir/c4_trained_libraries/${m}_skills -name SKILL.md \| wc -l` before vs after | equal counts, same names |
| 3 | Skill bodies unchanged | body-only `diff` of each snapshot `SKILL.md` (exclude frontmatter — `increment_verified_uses` mutates it legitimately) | all empty |
| 4 | Library actually read | `jq -r '.intermediate_steps[]?.tool_calls[]?.name // empty' workdir/gaia_c4_${m}_<FREEZE_RUN_ID>/dra.jsonl \| grep -c activate_skill` | **> 0** |
| 5 | Canary visible | `grep -c "freeze-canary-farm-${m}" logs/c4_freeze_smoke_${m}_*.out` | **> 0** |
| 6 | Override banner landed | `grep "building SkillRegistry at" logs/c4_freeze_smoke_${m}_*.out` | shows the snapshot path, not `workdir/gaia_c4_${m}_<run>/skills` |

**Fail → stop-gate.** If any model fails any check, do NOT submit S4 C4
cells. Most likely culprits: shell quoting swallowing the `--cfg-options`
args in `--wrap` (switch to a standalone `script.sh` + `sbatch script.sh`),
or a regression in the generated configs (regenerate from
`scripts/gen_eval_configs.py` and retry).

**Pass → ready for S4.** Attach the 6-row pass table per model to this
handoff before launching S4.

### S4 — Test-split submission (~8-24 h, $30-100)

Full matrix, test split, all 16 cells. Long job; use SLURM for
disconnect-survival.

**Plain (no C4 training pass):**
```bash
sbatch run_matrix_slurm.sh full
```

**With frozen trained libraries (recommended for C4 paper numbers):**
Override the 3 C4 cells' skill config via `--cfg-options`. Easiest path is a
thin wrapper; submit each model's C4 cell individually:

```bash
# C0/C2/C3 for all three models → normal
sbatch run_matrix_slurm.sh full '' c0
sbatch run_matrix_slurm.sh full '' c2
sbatch run_matrix_slurm.sh full '' c3

# C4 × {mistral, kimi, qwen} → one-off each, with the trained library pinned
#
# IMPORTANT — override namespace is `agent_config.*`, NOT `planning_agent_config.*`.
# The config file aliases `agent_config = planning_agent_config` but mmengine's
# Config.fromfile materialises the two names as independent dicts, and
# create_agent() in src/agent/agent.py reads `config.agent_config`. A
# `planning_agent_config.*` override merges successfully at the top level but
# is silently ignored by the agent, leaving C4 running in training mode
# (extraction ON, fresh skills_dir). Validated locally on Mac 2026-04-18 —
# see "Freeze-smoke validation" subsection below.
for m in mistral kimi qwen; do
  sbatch --job-name=gaia-c4-$m --time=24:00:00 \
         --output=logs/c4_${m}_%j.out --error=logs/c4_${m}_%j.err \
         --wrap "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dra \
                 && cd /userhome/cs2/ambr0se/DeepResearchMetaAgent \
                 && python examples/run_gaia.py \
                      --config configs/config_gaia_c4_${m}.py \
                      --cfg-options \
                        agent_config.skills_dir=workdir/c4_trained_libraries/${m}_skills \
                        agent_config.enable_skill_extraction=False"
done
```

### Post-run

```bash
# Per-cell score summary
for d in workdir/gaia_c*_<SUBMIT_RUN_ID>/; do
  echo "=== $d ==="
  python scripts/analyze_results.py "$d/dra.jsonl" | head
done

# Condition-vs-condition deltas within a model
python scripts/compare_results.py \
  workdir/gaia_c0_mistral_<id>/dra.jsonl \
  workdir/gaia_c3_mistral_<id>/dra.jsonl

# Handoff grep sweep — attach to this doc when moving to Completed
bash scripts/validate_handoffs.sh <SUBMIT_RUN_ID> > validation_report_<SUBMIT_RUN_ID>.txt
```

---

## What to watch for (per-handoff warning signs)

| Handoff | Red flag in logs | Likely meaning |
|---------|-------------------|-----------------|
| #1 | `auto_browser_use_tool returned no extracted content` | Browser page never rendered — known degradation, not a regression. Should be 0 or low if Playwright Chromium is installed. |
| #1 | `No such file or directory: 'code.txt'` | Analyzer silent-fail fix regressed. **Should be 0.** |
| #2 | `400 ... temperature|top_p|Invalid sampling` in Kimi logs | Sampling-lock fix regressed. **Should be 0.** |
| #2 | `tool_choice ... thinking mode` in Qwen logs | `enable_thinking=False` didn't propagate. **Should be 0.** |
| #2 | `AllocationQuota.FreeTierOnly` on Qwen | DashScope hit the exhausted tier. With the 2026-04-18 Qwen swap, configs no longer touch DashScope — if this appears, the operator is running a pre-swap config. |
| #2 | `No endpoints found that support the provided 'tool_choice' value` | The Qwen matrix config is still on `or-qwen3.6-plus`. Regenerate configs from the latest `scripts/gen_eval_configs.py`. |
| #3 | 0 `modify_subagent` / `diagnose_subagent` mentions in C2/C3/C4 | Prompt didn't reach the adaptive agent — check template_path override in the config. |
| #4 C3 | 0 `enable_review=True; building ReviewStep` banners across C3 cells | C3 config flag broke; check `enable_review=True` in `planning_agent_config`. |
| #4 C3 | 0 `[REVIEW]` markers despite the banner | Planner never delegated (agent timed out on step 1 only). Diagnose by reading the cell's log.txt — expected pattern is ≥3 delegations per question. |
| #4 C4 | 0 `SkillRegistry built` / 0 `seed_skills_dir seeded` | C4 init broke. Check `enable_skills=True`, `skills_dir=workdir/{tag}/skills`, `DRA_RUN_ID` env var present. |
| #4 C4 | Library has questions' content **inlined** (e.g. specific years / URLs / numeric constants) | Extractor entity-blocklist regressed. See `_extractor.py` stage 4. |
| #5 | RC1 `premature-final-answer guard` fires | Normal — means the guard caught a mis-issued `final_answer_tool`. Should not BLOCK the run. |
| #5 | `RC2 diagnostic scope error in sub-agent` | Also normal; means RC2's diagnostic hook caught a scoped error. Not a regression. |
| #7 | `Error executing script for tool` in stream log | Dynamic tool failed to register. With MCP fence-extraction fix, the log should also say `skipped — fenced script missing closing \`\`\` marker` for the known-bad seed scripts. Non-fatal. |

---

## Common obstacles

- **Firecrawl quota** — `scripts/check_firecrawl_credits.py` reports remaining
  credits. Credit use is per-scrape, not per-token, so a full matrix with
  heavy browser use can burn through fast. A low balance manifests as
  `DeepResearchTool RetryError[ValueError]` in cell logs.
- **OpenRouter rate limiting** — Kimi, Qwen, and Gemma all route through
  OpenRouter. Concurrent requests (4 parallel streams × 4 conditions) on
  one key can throttle. If you see clustered 429s, run the streams
  sequentially instead of parallel by editing
  `scripts/run_eval_matrix.sh` `run_model_stream` to not background.
- **HKU CS Phase-3 gateway disconnect** — SBATCH handles this; interactive
  `tmux` sessions do not if the gateway node itself reboots. Prefer SBATCH.
- **Playwright Chromium missing** — `scripts/ensure_playwright_browsers.sh`
  runs at SBATCH startup; if it fails the browser_use_agent tool errors out
  cleanly (the silent-failure fix from Handoff #1 surfaces the real cause).
- **Python traceback in cell log** — any non-MCP traceback means something
  regressed; preserve the run dir and attach the log to the promoted handoff.

---

## Resume protocol (Ctrl-C / SIGTERM / preemption)

Every cell's `examples/run_gaia.py` reads its existing `dra.jsonl` at
startup and skips already-answered questions. To resume a killed cell:

```bash
# Get the DRA_RUN_ID of the interrupted run (from workdir/run_logs/matrix_runid.txt)
DRA_RUN_ID=<prior_id> bash scripts/run_eval_matrix.sh full '' <cond>
# or re-submit the whole SLURM job pinning the run id:
DRA_RUN_ID=<prior_id> sbatch run_matrix_slurm.sh full
```

For C4 cells specifically: resuming preserves the evolved `skills_dir` (the
`.seeded` marker blocks re-seeding — see `_seed.py`). If you intentionally
want to start the C4 library fresh, clear `workdir/gaia_c4_<model>_<id>/skills/`
before resubmitting.

---

## Exit / sign-off criteria

- `validation_report_<SUBMIT_RUN_ID>.txt` shows:
  - 0 Handoff #1 red flags
  - 0 Handoff #2 sampling / thinking-mode 400s
  - >0 Handoff #3 adaptive-tool mentions
  - 4 / 4 `SkillRegistry built` (Handoff #4 C4 — one per model's C4 cell)
  - ≥1 `[REVIEW]` marker per C3 / C4 cell (Handoff #4 C3)
  - 1+ `[tool_choice] qwen/qwen3.6-plus -> auto` INFO line per Qwen cell (Handoff #9 hybrid dispatch verification)
  - All 16 cells produce a non-empty `dra.jsonl`
- Per-cell accuracy numbers computed via `scripts/analyze_results.py`
- Move each `HANDOFF_INDEX.md` row to **Completed / Archived** with the
  submission run id and the `validation_report_*.txt` path.
- Delete **nothing**; this handoff's evidence is the paper's audit trail.

---

## Known unknowns at hand-off time

- **OpenRouter Qwen3.6-Plus stability** — provider-health issues on the
  OpenRouter side during the ~8-24 h submission run would silently degrade
  Qwen cells. Mitigate by running Qwen cells **first** and spot-checking
  after ~30 min; if the latency or error rate spikes, `scancel` that job
  and swap to `or-qwen3-max` (also verified live) as a fallback before
  relaunching.
- **Gemma 4 31B provider drift** — only DeepInfra + Together are pinned.
  If both have outages or degrade simultaneously, Gemma cells will start
  401/404/429-ing. Either widen the provider list (add Parasail or GMI)
  in `src/models/models.py` or accept partial matrix coverage.
- **Local `browser-use` asyncio cancellation** — confirmed broken on macOS,
  probably fine on Linux. If Mistral cells show 20+ min runtimes past the
  configured `per_question_timeout_secs`, it's the same bug — raise the
  timeout or accept the limitation.
- **C4 train/freeze protocol is the intended methodology**, but it was not
  exercised end-to-end locally because the local path has other issues.
  First farm run is the first real test of the train-then-freeze loop.
  Reserve a 1 h budget slot to diagnose if it misbehaves.
- **GPU farm wall-clock limits** — `run_matrix_slurm.sh` requests 24 h. Full
  matrix on test split with 4 parallel streams typically fits, but a
  browser-heavy condition on slow providers (plus Gemma's concurrency cap
  of 4) can blow the wall. If SLURM kills the job mid-run, resubmit with
  the same `DRA_RUN_ID` (see "Resume protocol").

---

## Files touched in this handoff (none — docs + configs only)

This handoff does not itself change source code — it documents the
execution protocol for the work in handoffs #1-#9. The runtime changes
that enabled this 16-cell matrix (Kimi extra_body, hybrid `tool_choice`
dispatch with retry guard, Qwen swap to `or-qwen3.6-plus`, and Gemma-4-31B
addition) are tracked in handoff #9 (commits `fe3de8d` → `829d4d8` →
`c17f24e` → `27d48e4`).

Current matrix-defining files (post-handoff-#9):

| File | Role |
|------|------|
| `scripts/gen_eval_configs.py` | MODELS rows: mistral → `mistral-small`; kimi → `or-kimi-k2.5`; qwen → `or-qwen3.6-plus`; gemma → `or-gemma-4-31b-it` |
| `configs/config_gaia_{c0,c2,c3,c4}_{mistral,kimi,qwen,gemma}.py` | 16 regenerated cell configs |
| `scripts/run_eval_matrix.sh` | `ALL_MODELS=(mistral kimi qwen gemma)` with `GEMMA_CONCURRENCY` cap (default 4) |
| `src/models/models.py` | OR registrations: Kimi (thinking off + Moonshot pin), Gemma (DeepInfra/Together pin + reasoning off) |
| `src/models/tool_choice.py` | Hybrid dispatch for the Qwen family |
| `src/agent/general_agent/general_agent.py` + `src/base/tool_calling_agent.py` | Retry guard for the "auto" path |
| `docs/handoffs/HANDOFF_TEST_EVAL.md` | This document |
