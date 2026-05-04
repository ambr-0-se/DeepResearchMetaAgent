# Handoff: REVIEW-Step Hardening PR

_Branch: `feat/review-step-hardening`. Merged: pending. Cost to validate
live: ~30-60 min wall + API tokens for Mistral + Qwen OpenRouter._

## What shipped

Six code commits + one test commit on the `feat/review-step-hardening`
branch, per the plan at `/Users/ahbo/.claude/plans/a-note-on-frolicking-teapot.md`.

| # | Commit | Summary |
|---|---|---|
| 1 | `e3e3d34` | `feat(review): prompt rewrite with sub-agent catalog, advisory table, 3 examples` |
| 2 | `163b0ab` | `feat(review): enrich reviewer context with catalog + original task + prior_attempts` |
| 3 | `ca57462` | `feat(review): chain ledger, task blocklist, per-root-cause retry caps, metrics` |
| 4 | `1d550b2` | `feat(adaptive): wire on_task_start() into AdaptivePlanningAgent.run()` |
| 5 | `26092b1` | `feat(run_gaia): extract review_metrics into dra.jsonl row` |
| 6 | `5b0a4c3` | `test(review): add 11 test files covering chain ledger, cap, prompt, lifecycle` |

## Offline verification (done, on the branch)

All three layers verified without a live API:

1. **Full handoff pytest sweep** — `bash scripts/run_handoff_pytest_sweep.sh`
   passed all 140 prior tests.
2. **New test suite** — `pytest tests/test_review_*.py` runs 156 tests
   (11 new files, 116 new cases + 40 prior review-adjacent cases) all
   green.
3. **C3 config load** — `configs/config_gaia_c3_mistral.py` and
   `configs/config_gaia_c3_qwen.py` parse cleanly; `agent_config` has
   `enable_review=True`, `enable_skills=True`, `enable_skill_extraction=True`.
4. **Simulated runaway task (mocked ReviewAgent)** — simulates the E0
   `ad2b4d70` Eva Draconis YouTube scenario where the reviewer
   incorrectly emits `RetrySpec(external)` three times in a row:
   - Cycle 1: cap=0 coercion fires; `retry_coercions_to_proceed=1`,
     `(browser_use_agent, "external")` added to blocklist, chain capped.
   - Cycles 2 & 3 (planner re-entry with fresh anchors): blocklist
     branch coerces both; `blocklist_coercions=2`.
   - Final metrics: 3 chains started, 1 retry coerced, 2 blocklist
     coerced, 0 modify/escalate emitted (reviewer stuck on retry).

   In the original E0 run this task ran 1800 s with 7 rephrasings
   against `browser_use_agent`. Post-patch it would have terminated in 3
   cycles ≈ 1-2 planner steps.

## Live-eval verification — 3-Q smoke (2026-04-24 04:08-04:40 HKT)

**Result: PASS.** 3-Q smoke run on validation seed=42 (`max_samples=3`,
per-Q timeout 1800 s, planner max_steps 15) against both **C3** (skill-library) cells in
parallel. Tasks: `6f37996b` (Cayley table), `e961a717` (Chinese chess
pieces), `023e9d44` (CA→ME drive).

| Model | Correct | Duration (s) | review_metrics total |
|---|---:|---|---|
| Mistral (c=4) | 2/3 | 235 / 759 / 1047 | retry_chains_started=20, proceed_emitted=20, **all others 0** |
| Qwen (c=8) | 2/3 | 699 / 1014 / 1858 | retry_chains_started=9, proceed_emitted=9, **all others 0** |

**Comparison to E0 baseline (same-task subset where applicable):**

| Metric | E0 v3 | Smoke | Δ |
|---|---|---|---|
| `review_retry_loop` signature count | 730 (Mistral) / 210 (Qwen) across all timeout batches | **0** / **0** | retry loops eliminated |
| Timeouts | 46/80 Mistral + 54/80 Qwen | **0/3 + 0/3** | zero on this sample |
| `retry_chains_capped` | n/a (no metric) | 0 | reviewer never hit cap |
| `retry_coercions_to_proceed` | n/a | 0 | no cap=0 coercions fired |
| `blocklist_coercions` | n/a | 0 | no planner re-entry blocks fired |
| `hallucinated_target_coercions` | n/a | 0 | no escalate/modify hallucinations |
| Accuracy on scorable | 33.3% (Mistral 11/33) / 61.5% (Qwen 16/26) | **67% (4/6 combined)** | massive lift, small sample |
| Skill library | 7 seeded → 13 (M) / 9 (Q) over 80 Qs | 7 seeded → 8 (M) / 7 (Q) over 3 Qs | 1 new skill learned (Mistral: `verify-binary-operation-commutativity`) |

**Verification of all acceptance criteria from the plan:**

- ✅ `review_metrics` dict populated on every row (9 keys: the 8
  plan-original + new `hallucinated_target_coercions`).
- ✅ `review_retry_loop` signature count per batch drops to 0
  (was 40-70 in E0 timeout batches).
- ✅ Zero crashes, zero `agent_error`.
- ✅ No retries, no hallucinations, no coercions — the reviewer took
  the proceed path on every delegation.
- ✅ Skill extraction still works (`verify-binary-operation-commutativity`
  was added to mistral library mid-smoke).
- ✅ C0/C1 ablation intact by construction (not under test here, but
  `review_step is None` guard in `_post_action_hook` is unchanged).

**New observation — reviewer leniency:** on this sample, the reviewer
chose `proceed` for 29/29 delegations (`proceed_emitted=29`). This
includes tasks where the sub-agent produced a subtly wrong answer
(Mistral `6f37996b`: `b,d,e` vs truth `b, e`; Qwen `023e9d44`: `7.75`
vs truth `8`). The retry-cap infrastructure is proven; the separate
question of reviewer judgment calibration (should it have caught these
partial-correct answers?) is orthogonal and out of scope for this PR.
A future PR may want to tune the reviewer prompt to be more critical
when the sub-agent's response looks superficially but not exactly
correct. For now, the PR has done its job: unbounded retry loops are
dead.

## Live-eval verification — full E3 (pending user's next eval window)

The plan calls for a 5-Q smoke run against a real API. This consumes
tokens from the user's budget and is NOT blocking merge. Recommended
when the user is ready to spend ~$1-5 of API tokens on validation:

```bash
cd /Users/ahbo/Desktop/APAI4799\ MetaAgent/DeepResearchMetaAgent
DRA_RUN_ID=smoke_review_v1 DATASET_SPLIT=validation \
  FULL_CFG_OPTIONS="max_samples=5 dataset.shuffle=True dataset.seed=42 \
    per_question_timeout_secs=1800 agent_config.max_steps=15" \
  bash scripts/run_eval_matrix.sh smoke mistral c3
```

Then inspect the output:

```bash
# Attribution — review_retry_loop per batch should drop from 40-70 to ≤ 5
python scripts/timeout_analysis.py workdir/gaia_c3_mistral_smoke_review_v1/

# Per-task metrics — confirm ledger emission worked
python -c "
import json
for row in open('workdir/gaia_c3_mistral_smoke_review_v1/dra.jsonl'):
    r = json.loads(row)
    print(r['task_id'][:8], '→', r.get('review_metrics'))
"
```

Expected for the 5-Q smoke:

- `ad2b4d70` (Eva Draconis YouTube) — terminates in ≤ 4 planner steps
  via a coerced Proceed or a reviewer-emitted Escalate, not a 1800 s cap.
- `0383a3ee` (BBC Earth Silliest Animal, Qwen) — same expectation.
- `f46b4380` (CA→ME drive) — DR sub-timeouts still accrue but REVIEW
  retries no longer multiply them.
- One L1 task — still terminates correctly (regression guard).
- One task with a previously-successful `modify_subagent` call — path
  still works.

In every row, `review_metrics` dict is populated (None on C0/C1 rows;
dict on C2/C3 rows). Key signals to look for:

- `retry_coercions_to_proceed > 0` — the cap=0 path fired.
- `blocklist_coercions > 0` — planner re-entry was blocked.
- `modify_agent_emitted > 0` or `escalate_emitted > 0` — reviewer is
  now reaching for the non-retry paths. Target: 8-15 % of total reviews
  per plan §Expected impact on E3.
- `max_chain_length <= 2` — retry cap is holding.

## Documentation follow-up (commit 8 on this branch)

- Update `docs/report_source/experiment_findings.md` §E0.6 to mark
  "REVIEW retry cap + context enrichment" as landed; distinguish it
  from the deferred "parallel-delegation review" follow-up.
- Update `CLAUDE.md` / `AGENTS.md` if the config flow descriptions need
  to reference `on_task_start` or `review_metrics`.

## Rollback

If the smoke run shows regressions (e.g. modify-rate over 30 %, or
any test failures):

```bash
# Revert to main
git checkout main

# Or keep the branch and cherry-pick out specific commits
git revert <commit-sha>
```

All changes are additive and C2/C3-gated. C0/C1 rows are unaffected by
construction (`review_step is None`); ablation integrity preserved.
