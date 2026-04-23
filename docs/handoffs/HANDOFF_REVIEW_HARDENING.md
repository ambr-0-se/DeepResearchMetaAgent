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
3. **C4 config load** — `configs/config_gaia_c4_mistral.py` and
   `configs/config_gaia_c4_qwen.py` parse cleanly; `agent_config` has
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

## Live-eval verification (deferred to user's next eval window)

The plan calls for a 5-Q smoke run against a real API. This consumes
tokens from the user's budget and is NOT blocking merge. Recommended
when the user is ready to spend ~$1-5 of API tokens on validation:

```bash
cd /Users/ahbo/Desktop/APAI4799\ MetaAgent/DeepResearchMetaAgent
DRA_RUN_ID=smoke_review_v1 DATASET_SPLIT=validation \
  FULL_CFG_OPTIONS="max_samples=5 dataset.shuffle=True dataset.seed=42 \
    per_question_timeout_secs=1800 agent_config.max_steps=15" \
  bash scripts/run_eval_matrix.sh smoke mistral c4
```

Then inspect the output:

```bash
# Attribution — review_retry_loop per batch should drop from 40-70 to ≤ 5
python scripts/timeout_analysis.py workdir/gaia_c4_mistral_smoke_review_v1/

# Per-task metrics — confirm ledger emission worked
python -c "
import json
for row in open('workdir/gaia_c4_mistral_smoke_review_v1/dra.jsonl'):
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

In every row, `review_metrics` dict is populated (None on C0/C2 rows;
dict on C3/C4 rows). Key signals to look for:

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

All changes are additive and C3/C4-gated. C0/C2 rows are unaffected by
construction (`review_step is None`); ablation integrity preserved.
