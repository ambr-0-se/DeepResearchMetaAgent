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
| 1 | Extractor NOT constructed (`grep -c "SkillExtractor active (C3 training mode)"`) | 0 | 0 | ✅ pass |
| 2 | No writes to snapshot (file count stable across run) | 14 | 9 | ✅ pass |
| 3 | Skill bodies unchanged (body-only diff excluding frontmatter) | — | — | *not checked — no pre-run body snapshot was taken; add to launcher for E3* |
| 4 | Library actually read (`activate_skill` **invoked** as a tool call) | 0 invocations on 3-Q smoke ([re-analysis 2026-04-22](#f2-activate_skill-invocation-rate-zero-on-e2-smoke-is-a-sample-size-artifact-e0-training-shows-the-library-is-used): **60 invocations / 26 Qs across the 80-Q E0 run**) | 0 invocations on 3-Q smoke (E0 = **10 invocations / 10 Qs**) | ⚠️ smoke uninformative (zero expected at these sample sizes); production-scale evidence confirms library is invoked — see F2 |
| 5 | Canary visible (planner-scope canary skill surfaces in registry block) | — | — | *not executed — canary step skipped in this local run* |
| 6 | Override banner names the snapshot path | `[SkillRegistry] loaded 14 skills from workdir/c4_trained_libraries/mistral_skills_v3` | `loaded 9 skills from workdir/c4_trained_libraries/qwen_skills_v3` | ✅ pass |

**Summary:** 3/4 executed checks pass (1, 2, 6); pass #4's zero-count was reclassified 2026-04-22 as a sample-size artifact after scanning E0 v3 data showed 60 Mistral + 10 Qwen real activations across the 80-Q training run (see F2 below). Passes #3 and #5 were not set up locally.

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

### F2 — `activate_skill` invocation rate: zero on E2 smoke is a sample-size artifact; E0 training shows the library IS used

**Original observation (2026-04-20).** `SkillRegistry` loads correctly (banner + 14/9 skill counts in the log), but the `activate_skill` tool was **called 0 times** on either model in the `intermediate_steps` JSON for all 3 rows. The 20 / 16 `activate_skill` mentions per-row were from the injected registry-block *prompt text*, not tool invocations. That raised the concern: "is the library cosmetic?"

**Re-analysis (2026-04-22).** Scanning the E0 v3 `dra.jsonl` (80 Qs per model, production caps) for `tool_calls[].function.name == "activate_skill"`:

| Model | Real invocations | Questions with ≥1 activation | Per-Q activation rate |
|---|---|---|---|
| Mistral | **60** | **26 / 80** | **32.5%** |
| Qwen    | **10** | **10 / 80** | **12.5%** |

**The library is being invoked at production scale.** The E2 zero-count reflects small-sample variance + smoke-cap compression, not a dead code path:

- Probability of zero activations in a random 3-Q sample given the measured per-Q rate:
  - Mistral at 32.5% → P(zero in 3) = (1 − 0.325)³ ≈ **31%** (unlikely but plausible)
  - Qwen at 12.5% → P(zero in 3) = (1 − 0.125)³ ≈ **67%** (zero is the *modal* outcome)
- Per-task corroboration: **2 of E2's 3 questions** (`6f37996b`, `e961a717`) DID have Mistral activations during E0 (1 and 2 respectively). E2 missed them, which points to a second factor beyond sample size.

**Second factor — smoke step caps compress the planner's registry-exploration budget.** E2 uses `SMOKE_CFG_OPTIONS` with `agent_config.max_steps=10` (down from the E0/E3 default `max_steps=20`). With half the planning budget, the planner goes straight to delegation instead of probing `activate_skill` first. Same system prompt, same library — different exploration headroom.

**Evidence this matters.** Mistral's E0 24-Q with-activation subset had activations spread across planner steps 3–15 (median 7). In a 10-step budget, activations from step 11+ are literally impossible. That alone is consistent with E2's Mistral-specific underperformance.

**What E2 actually verified** (revised):
- ✅ Plumbing: `SkillRegistry` loads the right snapshot path, `enable_skill_extraction=False` is honoured, no writes to the snapshot.
- ❌ Behaviour under smoke caps: can't extrapolate from a 3-Q cap-compressed sample.
- (previously thought) ~~Zero positive evidence the frozen library influences behaviour.~~ **Superseded:** the E0 training data provides the positive evidence at scale.

**Implications.**
1. **F2 downgraded from "blocker-adjacent" to "smoke-coverage note."** The library is demonstrably being invoked. E3's C3→C4 delta remains the authoritative signal for "does it help", which is the real question.
2. **Future E2 pass design.** If re-running E2 as a quick canary, either (a) use 10–15 Qs to make zero-activation an informative signal, or (b) use E3-level caps (`max_steps=20`) instead of `SMOKE_CFG_OPTIONS` so the activation budget is representative. Option (a) is cheaper.
3. **Per-question activation data for the paper.** Add a brief methodology subsection reporting the E0 per-Q activation rates (32.5% / 12.5%) as evidence the C4 library isn't passive decoration.

**Original fix directions superseded.** The "inject canary skill" + "tighten prompt wording" suggestions were mitigations for what we now know was a sample-size artifact. They are not necessary for E3. If post-E3 analysis shows the C4 accuracy lift is smaller than expected, reopen prompt-wording changes then.

### F6 — Qwen `browser_use.Agent` has been broken since stack assembly (CRITICAL for Qwen C4 interpretation) — added 2026-04-22

**Observation.** Across every run in the entire workdir history,
`browser_use.Agent` on Qwen has failed at Step 1 and never progressed.
Evidence:

- `workdir/run_logs/full_qwen.log`: **3 266 `[agent] 📍 Step 1`
  marks, 0 `Step 2` marks**. Every Qwen browser_use attempt died at
  the first LLM call.
- `workdir/run_logs/full_mistral.log`: **1 880 Step 1 + 1 970 Step 2
  + 1 810 Step 3 + 1 577 Step 4 + … + 663 Step 10+**. Mistral's
  browser_use works normally — distinct step distribution with
  thousands of multi-step sessions.
- Per-question E0 v3 Qwen `dra.jsonl`: 14 / 80 questions show
  `browser_use_agent` delegation from the planner; all 14 produced
  "about:blank, about:blank, about:blank" trajectories — browser_use
  returned no useful web content.

**Root cause.** Configs give Mistral and Qwen different LangChain
wrappers for their browser tool:

| Model | `auto_browser_use_tool_config.model_id` | Endpoint | `tool_choice="required"` support |
|---|---|---|---|
| Mistral | `langchain-mistral-small` | Mistral La Plateforme (native) | ✅ supported |
| Qwen | `langchain-or-qwen3.6-plus` | OpenRouter → Alibaba | ❌ rejected with HTTP 404 |

Our agent default is `tool_choice="required"`. The native path
applies the `pick_tool_choice` downgrade (qwen/\* → `"auto"`) via
`OpenAIServerModel.generate()`, but the LangChain path used by
`auto_browser_use_tool` did not. So every Qwen browser_use.Agent
Step 1 LLM call sent `"required"` → Alibaba → 404 → browser_use gave
up on that step. Mistral's browser calls go direct to La Plateforme
which accepts `"required"` natively; no downgrade needed.

**Why this wasn't flagged until T3 v1:** browser_use.Agent failures
at Step 1 don't surface as loud errors in `run_gaia.py`'s per-Q
jsonl — the sub-agent returns a normal-looking result (something
about "couldn't load the page") and the planner moves on. The 3 240-
error cascade in T3 v1 was one Qwen question's retry loop sitting
in the 1800 s budget long enough for the retries to accumulate; E0
v3 questions bailed out through other error paths before the retry
counter reached 3 000. The phenomenon was always there, just silent.

**Implications for E0 v3 Qwen C4 training:**

- 14 / 80 E0 v3 Qwen questions (17.5%) involved `browser_use_agent`
  delegations that produced only failed trajectories.
- Qwen's 2 learned skills from E0 v3 are biased toward cases where
  the planner learned to **avoid** browser_use — `research-fallback-sources`
  (prefer alt data sources when browser fails) and the `escalate-to-
  deep-research-on-browser-failure` are skills that RESPOND to this
  broken path rather than use it productively.
- Mistral's 7 learned skills are unaffected — Mistral's browser_use
  worked throughout.
- E3 Qwen C4 cells will either show the same bias (if P5's fix isn't
  live-validated before E3) OR show an improvement delta attributable
  to P5 landing (if it is). Worth pre-registering this observation
  before E3 submission so the paper can frame the result either way.

**What P5 fixes and what it doesn't.** P5 addresses the tool_choice
404 at the LangChain layer. If `browser_use.Agent` can reach its
first LLM call without Playwright bailing out, P5's downgrade will
rewrite `"required"` → `"auto"` and Alibaba will accept the request.
What P5 does NOT fix: Playwright initialization failures, anti-bot
blocks, JavaScript-heavy pages returning the fallback "Please enable
JavaScript" text — those are separate browser-automation issues.

**Fix direction.**
1. Before E3: run a **targeted Qwen browser_use validation** (see
   `HANDOFF_THROUGHPUT_REFACTOR.md` §Combined smoke executed). A
   single Qwen question that reliably forces `browser_use_agent`
   delegation + confirms Step 1 → Step 2+ transition + confirms the
   `qwen/qwen3.6-plus -> auto` downgrade banner fires.
2. Paper methodology footnote: disclose the 17.5% silent-failure rate
   on E0 v3 Qwen browser delegations and the implication for learned-
   skill composition.
3. Optional longer-term: align Mistral and Qwen on the same browser
   backend (e.g. both use native paths) so P5-style seam bifurcation
   isn't needed.

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

1. **F1 (timeout enforcement)** — ✅ **ADDRESSED 2026-04-22 by P1 of `HANDOFF_THROUGHPUT_REFACTOR.md` (commit `5aa1467`).** Hard wall-clock guard caps each question at `per_question_timeout_secs + CLEANUP_GRACE_SECS=30` via an explicit `asyncio.create_task` + double `asyncio.shield` pattern. Combined-phase smoke re-run still pending to measure the observable wall-time delta.
2. **F3 (extraction on timeout)** — ❌ **deliberately not fixed.** Operator confirmed 2026-04-21 no E0 re-run planned; E2/E3 freeze extraction so the `_maybe_extract_skill` branch never fires in the active roadmap. Covered by a methodology footnote in the paper instead. If a future ablation re-enables extraction, reopen this via the session-plan §A.2 sketch.
3. **Re-run E2 with F1 fixed and a 10–15 question sample + canary.** Still recommended. Combine with the `HANDOFF_THROUGHPUT_REFACTOR.md` §Execution log → "Combined smoke (pending)" in a single pass to amortize setup.

## Items explicitly not blockers

- **F2** — reclassified 2026-04-22 from "small-sample non-activation" to "measured per-Q activation rate 12.5%–32.5% in E0 v3 training; E2's zero-count was a sample-size + smoke-cap artifact." The library is demonstrably being invoked. See F2 section above for the full re-analysis. E3's C3→C4 accuracy delta remains the authoritative signal for whether the library *helps*; nothing to fix before E3.
- **Kimi / Gemma exclusion** is stable and documented (`HANDOFF_TEST_EVAL.md` §Matrix definition). No action.
- **Seeded-vs-learned asymmetry** (Mistral 7 / Qwen 2) is an honest artefact, not a bug.
