# E1 + E2 results — snapshot + freeze smoke (2026-04-20)

**Status:** EXECUTED with mixed outcomes — mechanism verified, two real findings need follow-up before E3.
**Scope:** `DRA_RUN_ID=20260420_E2freeze` (E2) and the `_skills_v3` snapshots under `workdir/c4_trained_libraries/` (E1).
**Upstream:** E0 v3 training run (`DRA_RUN_ID=20260420_E0v3`, handoff `HANDOFF_E0_V3_PHASE2_RERUN.md`).
**Downstream gate:** E3 — see "Blockers before E3" below.

---

## E1 — Snapshot after E0 (post-Phase-2)

**Ran:** 2026-04-20 local (Mac), after E0 v3 Phase-2 completed. Isolated snapshot per model; source `dra.jsonl` and `skills/` preserved in the E0 run directories.

| Model  | Source E0 run                               | Snapshot destination                                   | Skills (total / seeded / learned) |
|--------|---------------------------------------------|--------------------------------------------------------|-----------------------------------|
| mistral | `workdir/gaia_c4_mistral_20260420_E0v3/skills` | `workdir/c4_trained_libraries/mistral_skills_v3`      | 14 / 7 / **7** |
| qwen    | `workdir/gaia_c4_qwen_20260420_E0v3/skills`    | `workdir/c4_trained_libraries/qwen_skills_v3`         |  9 / 7 / **2** |

Command used (for audit — re-runnable if snapshots ever need to be recreated from the same source):

```bash
for m in mistral qwen; do
  src="workdir/gaia_c4_${m}_20260420_E0v3/skills"
  dst="workdir/c4_trained_libraries/${m}_skills_v3"
  cp -a "$src" "$dst"          # -a preserves the .seeded marker → no re-seed on mount
  [ -f "$dst/.seeded" ] || echo "WARN: $dst missing .seeded marker"
done
```

Kimi and Gemma are excluded from the active matrix (see `HANDOFF_TEST_EVAL.md` §Matrix definition); no `kimi_skills_v3` / `gemma_skills_v3` exist and E3 will not score C4 cells for those models.

**Seeded-count parity** (7 on each snapshot) matches the committed `src/skills/` seed set — confirms the `.seeded` marker correctly suppressed re-seeding. **Learned-count asymmetry** (7 Mistral vs. 2 Qwen) is the honest artefact of 80-row E0 training: Qwen's SkillExtractor dedup stage rejected more candidates, consistent with its higher E0 correct-count (16 vs. 11) leaving fewer failure trajectories with extractable structure.

---

## E2 — Freeze smoke (local)

### Execution

- **Launcher:** `scripts/launch_e2_freeze_smoke.sh` (committed 2026-04-21). Pins `DRA_RUN_ID=20260420_E2freeze`, `DATASET_SPLIT=validation`, `max_samples=3`, `dataset.shuffle=True dataset.seed=42`, and the override namespace `agent_config.skills_dir=<snapshot>` + `agent_config.enable_skill_extraction=False` per the CLAUDE.md-documented gotcha.
- **Env:** Mac-local (`dra` conda env); Mistral + Qwen streams launched concurrently via `nohup` + `disown`, with `caffeinate` to keep the machine awake. No SLURM — the handoff `HANDOFF_TEST_EVAL.md` §E2 describes the farm variant but this run was local-only.
- **Questions:** the first 3 rows of a seed=42 shuffle of the validation split — identical across models by construction, so any per-model score difference is model/library-attributable.

### Per-task outcomes (official GAIA scorer)

| Task ID prefix | Ground truth | Mistral C4 (frozen) | Qwen C4 (frozen) |
|----------------|--------------|---------------------|------------------|
| `6f37996b` | `b, e` | **✅ correct** (`b,e`, 109 s, 13 steps) | **✅ correct** (`b, e`, 311 s, 6 steps) |
| `023e9d44` | `8`    | ❌ timeout (no prediction, **3298 s**) | ❌ timeout (no prediction, **3273 s**) |
| `e961a717` | `12`   | ❌ timeout (no prediction, **3308 s**) | **✅ correct** (`12`, 700 s, 12 steps) |

**Totals: Mistral 1/3, Qwen 2/3.** Both agreed on the easy categorical. Mistral timed out on one question Qwen solved cleanly in 700 s — not a library shortfall; Mistral spent 865 s on a single planner step (visible in `log.txt`) before ever reaching the rest of its budget.

### Pass criteria (mirrors `HANDOFF_TEST_EVAL.md` §E2 table)

| # | Check | Mistral | Qwen | Status |
|---|-------|---------|------|--------|
| 1 | Extractor NOT constructed (`grep -c "SkillExtractor active (C4 training mode)"`) | 0 | 0 | ✅ pass |
| 2 | No writes to snapshot (file count stable across run) | 14 | 9 | ✅ pass |
| 3 | Skill bodies unchanged (body-only diff excluding frontmatter) | — | — | *not checked — no pre-run body snapshot was taken; add to launcher for E3* |
| 4 | Library actually read (`activate_skill` **invoked** as a tool call) | **0 invocations** | **0 invocations** | ❌ **fail** (see finding F2) |
| 5 | Canary visible (planner-scope canary skill surfaces in registry block) | — | — | *not executed — canary step skipped in this local run* |
| 6 | Override banner names the snapshot path | `[SkillRegistry] loaded 14 skills from workdir/c4_trained_libraries/mistral_skills_v3` | `loaded 9 skills from workdir/c4_trained_libraries/qwen_skills_v3` | ✅ pass |

**Summary:** 3/4 executed checks pass (1, 2, 6); pass #4 fails on both models; passes #3 and #5 were not set up locally.

---

## Findings

### F1 — Per-question timeout does not enforce its 1800 s budget (CRITICAL for E3)

**Observation.** All three timeout rows reported wall durations of **3273 s – 3308 s** against a nominal `per_question_timeout_secs=1800`.

**Trace.**
- `examples/run_gaia.py:175` wraps `agent.run(task=...)` in `asyncio.wait_for(..., timeout=per_question_timeout)`.
- In `workdir/gaia_c4_mistral_20260420_E2freeze/log.txt`, task `023e9d44` started at `21:36:55` and the `"Question timed out after 1800s"` warning emitted at `22:31:47` — **54 m 52 s** wall, not 30 m.
- The log shows normal step progression past the 30-minute mark (e.g. "Step 1: Duration 865.24 seconds" at `22:10:28`). The outer `wait_for` fires its cancellation scheduler at +30 min, but the inner task does not unwind promptly because at least one sub-agent / tool path still blocks on synchronous-looking cleanup beyond the `auto_browser_use_tool` deadlock that commit `aa78edc` fixed.

**Why it matters.**
- Budget projection is off by almost 2× for any timeout row. At the E3 scale (2 models × 160 test questions × worst-case timeout fraction), a 1500 s overshoot per timeout renders the current cost/time estimate non-conservative.
- The timeout **also causes F3 below** (no skill extraction), because the post-timeout branch in `run_gaia.py` never returns control to `adaptive_planning_agent.run` for the post-run extraction block.

**Fix direction.** Two layers, both required:
1. **Hard wall-clock guard in `run_gaia.py`.** After the outer `asyncio.wait_for` raises `TimeoutError`, give the inner task a bounded cleanup window (e.g. `asyncio.shield(..., timeout=30)`), then force-terminate if still alive — don't await it indefinitely.
2. **Audit sub-agent / tool paths that block in `CancelledError`.** `aa78edc` patched `auto_browser.py`. Likely offenders still unpatched: `deep_researcher` inner call (the current per-call timeout in `453ae24` may be suppressed when the outer event loop is cancelling), and any MCP-backed tool that awaits subprocess stdout.

### F2 — `activate_skill` not invoked on the 3 smoke questions

**Observation.** `SkillRegistry` loads correctly (banner + 14/9 skill counts in the log), but the `activate_skill` tool was **called 0 times** on either model in the `intermediate_steps` JSON for all 3 rows. The 20 / 16 `activate_skill` mentions per-row are from the injected registry-block *prompt text*, not tool invocations.

**Interpretation.** Three non-exclusive explanations:
1. **Sample is too small** — 3 questions may not intersect the 14 (Mistral) / 9 (Qwen) skill topics well. The first question (`6f37996b`) is a ~100 s categorical that doesn't need workflow scaffolding. The two timeout rows never reached a planning step where a skill would have been activated.
2. **Prompt phrasing is permissive, not directive** — the system prompt says *"Call `activate_skill` with a skill name to load its full workflow"* but does not mandate consulting the registry before delegation. Planners observed to go straight to `deep_analyzer_agent` / `browser_use_agent` without a skill probe.
3. **Library coverage gap** — with 9 Qwen skills, topical coverage of a random 3-question sample is ≤30% at best, even with perfect prompt adherence.

**Why it matters.** F2 does not invalidate the E2 freeze-mechanism verification (1, 2, 6 all pass). It does mean we have **zero positive evidence that the frozen library influences behaviour** on this smoke, which is the whole point of running E2 before E3. The E3 score gap between C3 and C4 is the *ground truth* answer to the activation question, but a silent smoke is a missed early-warning opportunity.

**Fix direction.**
1. **Extend the smoke to 10–15 questions** on the next E2 run. Topic intersection with a 14-skill library should become non-trivial.
2. **Inject the canary skill** (pass #5) before any future E2 — a planner-scope canary that matches every task is the cheapest way to assert the registry-consumer path works end-to-end.
3. **(Optional) add `activate_skill` to the planner's prompted checklist** with a "did you consult the registry?" nudge before the first delegation. This is a prompt change, not a pipeline change; measure via C3→C4 delta, not local smoke.

### F3 — Timeout rows produce no skill-extraction signal during E0 training (CRITICAL for E0 interpretation)

**Observation.** In `src/agent/adaptive_planning_agent/adaptive_planning_agent.py:323`, `super().run(...)` is awaited inside a `try:` block, and `_maybe_extract_skill(...)` is called after it returns. When the outer `asyncio.wait_for` in `run_gaia.py` cancels the inner agent task on timeout, `super().run()` raises `CancelledError` and the extraction branch is never reached. The `finally:` block only calls `_reset_to_original_state()`.

**Why it matters.** Any E0 timeout row contributes **zero** to the skill library — no matter how much partial progress the planner made, no matter how rich the REVIEW verdicts were on the intermediate delegations. Given E0 v3 timeout counts (Mistral 8 of 16 Phase-2 re-attempts, plus uncounted timeouts in the main 80-row draws), this is a non-trivial chunk of the training signal quietly discarded.

**Paper-methodology implication.** The current learned-skill counts (7 Mistral, 2 Qwen from 80 rows each) are **after** this discard, not the full pipeline's yield. Worth stating explicitly in the methodology section.

**Fix direction.** Move `_maybe_extract_skill` into an outer `try/except CancelledError` that runs the extractor with its own bounded timeout (~60 s) against whatever partial `self.memory.steps` exists. Extraction's six-stage worthiness filter is conservative — if partial trajectories have nothing extractable, the filter rejects them; if they do, the library gets new signal. Caveat to pre-register in the paper: **timeout-sourced skills need a methodological note** so they're not mistaken for "the agent completed and reflected" rows.

---

## Artefacts

- **E1 snapshots** (read-only after this handoff unless explicitly rolled over):
  - `workdir/c4_trained_libraries/mistral_skills_v3/` (14 SKILL.md files + `.seeded`)
  - `workdir/c4_trained_libraries/qwen_skills_v3/` (9 SKILL.md files + `.seeded`)
- **E2 runs:**
  - `workdir/gaia_c4_mistral_20260420_E2freeze/{dra.jsonl,log.txt}`
  - `workdir/gaia_c4_qwen_20260420_E2freeze/{dra.jsonl,log.txt}`
- **Launcher:** `scripts/launch_e2_freeze_smoke.sh`
- **Upstream handoff:** `docs/handoffs/HANDOFF_E0_V3_PHASE2_RERUN.md`

---

## Blockers before E3

1. **F1 (timeout enforcement)** — must land before E3. Proposed fix: §A.1 in session plan (hard wall-clock guard in `run_gaia.py`). Without it, the 8–24 h E3 estimate is off by 1.5–2× in the worst case.
2. **F3 (extraction on timeout)** — not strictly required for E3 (E3 freezes extraction), but worth landing before any future E0 re-run. Proposed fix: §A.2 in session plan.
3. **Re-run E2 with F1 fixed and a 10–15 question sample + canary.** Repeat pass #4 specifically — if `activate_skill` is still not invoked at a meaningful rate, revisit prompt phrasing before E3.

## Items explicitly not blockers

- **F2 alone** (small-sample non-activation) is not a blocker; E3's C3→C4 delta is the authoritative test.
- **Kimi / Gemma exclusion** is stable and documented (`HANDOFF_TEST_EVAL.md` §Matrix definition). No action.
- **Seeded-vs-learned asymmetry** (Mistral 7 / Qwen 2) is an honest artefact, not a bug.
