# Handoff: GAIA test-split evaluation — 12-cell matrix on the HKU CS GPU farm

**Session date:** 2026-04-18
**Branch / HEAD push:** `main` at `6f5ddd1` (+ whatever this session adds on top)
**Scope:** Executes the GAIA submission run for the APAI4799 meta-agent paper.
Produces **3 models × 4 conditions = 12 `dra.jsonl` files** on the GAIA test
split (~300 questions each; ~3,600 Q total), plus the C4 training pass
needed to harden the skill library before the scored evaluation.

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
      **Kimi uses `or-kimi-k2.5`** (OpenRouter) and **Qwen uses
      `or-qwen3-next-80b-a3b-instruct`** (OpenRouter direct, no failover)
- [ ] `OPENAI_API_KEY` optional — `ModelManager` short-circuits on empty
- [ ] Pull the GAIA dataset once into `data/GAIA/` (see "Prerequisites" below)

### On the farm

- [ ] **S0** pre-flight: `bash scripts/smoke_validate_handoffs_234.sh`
- [ ] **S1** 1-Q canary for one (model, condition) cell to prove the pipeline
- [ ] **S2** smoke matrix: `sbatch run_matrix_slurm.sh smoke`
- [ ] **C4 training pass** (see §C4 Train/Freeze below) — OPTIONAL but
      recommended for publishable C4 numbers
- [ ] **S4** test-split submission: `sbatch run_matrix_slurm.sh full`
- [ ] Collect `dra.jsonl` → run `scripts/analyze_results.py` per cell
- [ ] `bash scripts/validate_handoffs.sh <DRA_RUN_ID>` → attach pass/info
      summary to this handoff when promoting to Completed

---

## Matrix definition

12 cells = 3 models × 4 conditions:

| Condition | Meta-agent capability added | Configs / model slot |
|-----------|------------------------------|----------------------|
| **C0** | — (vanilla `PlanningAgent` baseline) | `configs/config_gaia_c0_<model>.py` |
| **C2** | Reactive `diagnose_subagent` + `modify_subagent` | `configs/config_gaia_c2_<model>.py` |
| **C3** | C2 + structural REVIEW step | `configs/config_gaia_c3_<model>.py` |
| **C4** | C3 + cross-task skill library (pre-seeded + learned) | `configs/config_gaia_c4_<model>.py` |

| Model slot | Real slug (model_id) | Rationale |
|------------|----------------------|-----------|
| **Mistral** | `mistral-small` (native La Plateforme) | `MISTRAL_API_KEY` |
| **Kimi** | `or-kimi-k2.5` (OpenRouter) | Native Moonshot path kept as placeholder per operator direction |
| **Qwen** | `or-qwen3-next-80b-a3b-instruct` (OpenRouter direct) | DashScope free tier exhausted; `or-qwen3.6-plus` rejects `tool_choice="required"`; `qwen3-next-80b-a3b-instruct` is the cheapest + fastest tool-call-compatible Qwen variant (~$0.09 / $1.10 per M input/output; 0.7s latency; 262K context). **NB:** failover wrapper `qwen3.6-plus-failover` is still registered but no config uses it now. |

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
sbatch run_matrix_slurm.sh smoke                   # 5 Q × 12 cells = 60 Q, validation split
```

Pass criteria (check `logs/matrix_<JOBID>.out` via the auto-run
`validate_handoffs.sh` summary at the bottom):

- 12 `dra.jsonl` files written (one per cell)
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
- Qwen → 404 "no endpoints support tool_choice" means the matrix config is
  on an older slug (should be `or-qwen3-next-80b-a3b-instruct`, see
  `scripts/gen_eval_configs.py` `MODELS` table).

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
#    All 3 models in parallel; extraction stays on.
sbatch run_matrix_slurm.sh full '' c4
# => workdir/gaia_c4_{mistral,kimi,qwen}_<TRAIN_RUN_ID>/
#    each ends with a `skills/` dir containing seeded + learned SKILL.md.

# 2. Snapshot the trained libraries and stage them as the starting point
#    for the scored run. Run ONCE before S4.
TRAIN_RUN_ID=<copy from step 1 logs>
mkdir -p workdir/c4_trained_libraries
for m in mistral kimi qwen; do
  cp -r workdir/gaia_c4_${m}_${TRAIN_RUN_ID}/skills \
        workdir/c4_trained_libraries/${m}_skills
done
```

For the S4 scored run, pass an override so C4 cells load the trained
library and **do not** extract further (see S4 below).

### S4 — Test-split submission (~8-24 h, $30-100)

Full matrix, test split, all 12 cells. Long job; use SLURM for
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
for m in mistral kimi qwen; do
  sbatch --job-name=gaia-c4-$m --time=24:00:00 \
         --output=logs/c4_${m}_%j.out --error=logs/c4_${m}_%j.err \
         --wrap "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dra \
                 && cd /userhome/cs2/ambr0se/DeepResearchMetaAgent \
                 && python examples/run_gaia.py \
                      --config configs/config_gaia_c4_${m}.py \
                      --cfg-options \
                        planning_agent_config.skills_dir=workdir/c4_trained_libraries/${m}_skills \
                        planning_agent_config.enable_skill_extraction=False"
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
- **OpenRouter rate limiting** — Kimi and Qwen both route through
  OpenRouter. Concurrent requests (3 parallel streams × 4 conditions) on
  one key can throttle. If you see clustered 429s, run the 3 streams
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
  - 3 / 3 `SkillRegistry built` (Handoff #4 C4)
  - ≥1 `[REVIEW]` marker per C3 / C4 cell (Handoff #4 C3)
  - All 12 cells produce a non-empty `dra.jsonl`
- Per-cell accuracy numbers computed via `scripts/analyze_results.py`
- Move each `HANDOFF_INDEX.md` row to **Completed / Archived** with the
  submission run id and the `validation_report_*.txt` path.
- Delete **nothing**; this handoff's evidence is the paper's audit trail.

---

## Known unknowns at hand-off time

- **OpenRouter Qwen3-Next stability** — this is a newer model slug; a
  provider-health issue on the OpenRouter side during the ~8-24 h submission
  run would silently degrade Qwen cells. Mitigate by running Qwen cells
  **first** and spot-checking after ~30 min; if the latency or error rate
  spikes, `scancel` that job and swap to `or-qwen3-max` (also verified live,
  same day) as a fallback before relaunching.
- **Local `browser-use` asyncio cancellation** — confirmed broken on macOS,
  probably fine on Linux. If Mistral cells show 20+ min runtimes past the
  configured `per_question_timeout_secs`, it's the same bug — raise the
  timeout or accept the limitation.
- **C4 train/freeze protocol is the intended methodology**, but it was not
  exercised end-to-end locally because the local path has other issues.
  First farm run is the first real test of the train-then-freeze loop.
  Reserve a 1 h budget slot to diagnose if it misbehaves.
- **GPU farm wall-clock limits** — `run_matrix_slurm.sh` requests 24 h. Full
  matrix on test split with 3 parallel streams typically fits, but a
  browser-heavy condition on slow providers can blow the wall. If SLURM
  kills the job mid-run, resubmit with the same `DRA_RUN_ID` (see
  "Resume protocol").

---

## Files touched in this handoff (none — docs + configs only)

This handoff does not itself change source code — it documents the
execution protocol for the work in handoffs #1-#7. The Qwen matrix swap
(`or-qwen3.6-plus-failover` → `or-qwen3-next-80b-a3b-instruct`) is
tracked in this session's dedicated commit.

| File | Change |
|------|--------|
| `scripts/gen_eval_configs.py` | Kimi → `or-kimi-k2.5`, Qwen → `or-qwen3-next-80b-a3b-instruct` in `MODELS` table |
| `configs/config_gaia_{c0,c2,c3,c4}_qwen.py` | Regenerated — all agent / tool slots now pin `or-qwen3-next-80b-a3b-instruct` |
| `src/models/models.py` | New OR alias registration for `or-qwen3-next-80b-a3b-instruct` |
| `docs/handoffs/HANDOFF_TEST_EVAL.md` | This document |
