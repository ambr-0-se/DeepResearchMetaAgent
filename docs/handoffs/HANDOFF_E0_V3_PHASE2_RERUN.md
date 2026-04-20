# E0 v3 — Phase 2 infra-error re-attempt rule (pre-registered)

**Registered:** 2026-04-20 (before Phase 1 resume or Phase 2 launch)
**Scope:** `DRA_RUN_ID=20260420_E0v3` only.
**Author context:** the E0 v3 run was gracefully paused at 10:13:35 HKT by operator request with both streams short of their 80-row target (Mistral 64/80, Qwen 44/80). 16 of Mistral's 64 rows failed due to a verified OpenRouter→Mistral edge outage (08:28–09:25 HKT). This document locks in the re-attempt rule before Phase 2 runs, so no classification choice is made after seeing Phase-2 outcomes.

## The two-phase resume protocol

### Phase 1 — Complete the sample (Option A behaviour)

```bash
bash scripts/resume_e0.sh
```

- Appends seed=42-ordered draws until each model reaches 80 rows.
- All prior rows (correct, wrong, timeout, provider-error) are preserved via `DRA_RESUME_PRESERVE_ALL=1`.
- Expected deltas:
  - Mistral: 64 → 80 (+16 new draws, rows 65–80 of the shuffle order)
  - Qwen: 44 → 80 (+36 new draws, rows 45–80 of the shuffle order)
- The 16 Mistral infra-error rows **remain in `dra.jsonl`** after Phase 1. Phase 2 will handle them.

### Phase 2 — One-shot infra-error re-attempt

Launched by `scripts/rerun_e0_infra_errors.sh` (written after Phase 1 completes, so its assertion logic can verify the 80-row precondition). For each model:

1. Count rows whose `agent_error` matches the regex below. If 0, no-op (Qwen path).
2. Back up `dra.jsonl` → `dra.jsonl.pre_rerun_<TS>.bak`.
3. Back up `skills/` → `skills.pre_rerun_<TS>/`.
4. Atomic rewrite: drop infra-error rows from `dra.jsonl`.
5. Assert new row count = 80 − N_infra (rollback from backup on mismatch).
6. Launch one `run_eval_matrix.sh full <model> c4` with `max_samples=N_infra` and identical `FULL_CFG_OPTIONS` as Phase 1.
7. Post-run verify: `dra.jsonl` back to 80 rows.

## Eligibility regex

Applied to `agent_error` (case-sensitive, Python `re.search`):

```
upstream connect error|reset reason: overflow|HTTP 5\d{2}|Service Unavailable|Bad Gateway|provider.*unavailable|model_overloaded
```

**Not eligible (count as final, no re-attempt):**

- Per-question timeout (deliberate 1800s budget cap — retrying just hits it again at similar cost)
- Wrong answer (no error — real training/eval signal)
- Parsing errors, tool errors, any error not matching the regex

## Expected match counts at registration

Verified via frozen regex on `dra.jsonl` 2026-04-20 before Phase 1 launch:

| Model | Rows at pause | Infra matches | Phase-2 re-attempts |
|-------|---------------|---------------|---------------------|
| Mistral | 64 | 16 | 16 |
| Qwen | 44 | 0 | 0 (no-op; rule applied symmetrically) |

Sample snippets from the 16 Mistral matches:

```
task=3cef3a44  err='Error while generating output:\nupstream connect error or disconnect/reset before headers. '
task=544b7f0c  err='Error while generating output:\nupstream connect error or disconnect/reset before headers. '
```

All 16 clustered 08:28–09:25 HKT (15 within a 20-minute window 08:28–08:49); median 0 intermediate steps per row — the planner's first chat completion was rejected before any reasoning happened.

## Rule invariants

- **Applied symmetrically to both models.** Qwen's natural 0-match outcome is an honest reflection of provider reliability during E0, not a rule carve-out.
- **One re-attempt per eligible row.** If the re-attempt also fails (infra or any other reason), the new outcome is final — no second re-attempt.
- **Targeting guarantee via seed=42 shuffle invariant.** Because `GAIADataset(shuffle=True, seed=42)` runs the Fisher-Yates shuffle BEFORE `done_questions` filtering in `run_gaia.py`, and all flagged rows' shuffle indices are ≤ 80 (they were among the first 80 drawn), setting `max_samples = N_infra` after dropping exactly those N rows causes the first-N-of-remaining slice to be exactly the dropped task_ids. The re-run script re-confirms this invariant before launch by reconstructing the expected target set.
- **Same `DRA_RUN_ID=20260420_E0v3`, same `skills_dir`.** The skill library inherits Phase 1 state; re-attempts and any new skill extractions land in the same library.

## Methodology caveat for the paper

Re-attempts have access to a fuller skill library than the original attempts did (Phase 1 extractions accumulated first). This is acceptable because original infra-failures had 0 intermediate steps and therefore could not have benefited from any library state at the original attempt time — any non-zero trace from the re-attempt is strictly more training signal. Will be documented in the paper's methodology section as a pre-registered robustness procedure.

## Audit trail

For each model with N_infra > 0, Phase 2 produces:

- `workdir/gaia_c4_<model>_20260420_E0v3/dra.jsonl.pre_rerun_<TS>.bak` — original 80-row file before Phase 2 edit
- `workdir/gaia_c4_<model>_20260420_E0v3/skills.pre_rerun_<TS>/` — skill library snapshot before Phase 2 launch
- `workdir/run_logs/rerun_infra_<model>_<TS>.log` — Phase 2 stream log

After Phase 2 completes, this handoff is updated with final match counts and outcomes (OK / still-infra / timeout / other) per model, preserving the git history of the pre-registration.

## Final outcomes (post-Phase 2, 2026-04-20 21:25 HKT)

### Execution

- **Phase 1 complete** at 21:09 HKT (Mistral) / 21:09 HKT (Qwen) — both models at 80/80 rows.
- **Phase 2 launched** at 19:12 HKT for Mistral (parallel with Qwen Phase 1 tail — safe per per-model workdir isolation). Script: `scripts/rerun_e0_infra_errors.sh mistral`.
- **Phase 2 complete** at 21:25 HKT for Mistral. Qwen Phase 2 was a no-op (0 matches, rule applied symmetrically).
- **Wall clock Phase 2**: ~2h13m for 16 re-attempts at concurrency=4.

### Artifacts

- `workdir/gaia_c4_mistral_20260420_E0v3/dra.jsonl.pre_rerun_20260420_191202.bak` — pre-Phase-2 80-row snapshot
- `workdir/gaia_c4_mistral_20260420_E0v3/skills.pre_rerun_20260420_191202/` — skill library at Phase 2 launch (6 learned + 7 seed)
- `workdir/run_logs/rerun_infra_mistral_20260420_191202.log` — Phase 2 launcher log
- `workdir/run_logs/full_mistral.log` — continued stream log (appended)

### Per-task outcomes — Mistral's 16 re-attempts

Classified via official GAIA scorer + pre-registered infra regex:

| Outcome | Count | % | Task IDs |
|---------|-------|---|----------|
| **CORRECT** (newly recovered) | **3** | 18.75% | `a3fbeb63`, `4b650a35`, `4d0aa727` |
| Wrong (model attempted, missed) | 3 | 18.75% | `3cef3a44`, `0383a3ee`, `e9a2c537` |
| Gave-up (`Unable to determine`) | 2 | 12.50% | `65afbc8a`, `5cfb274c` |
| Timeout 1800s (full budget consumed) | 8 | 50.00% | `544b7f0c`, `ecbc4f94`, `72e110e7`, `8131e2c0`, `5b2a14e8`, `ebbc1f13`, `5d0080cb`, `87c610df` |
| **Still-infra (re-attempt re-failed)** | **0** | **0%** | — |
| Other errors | 0 | 0% | — |

All 16 rows produced non-zero intermediate steps this time (vs 0 steps on original Phase 1 infra-failure), confirming the re-attempts were substantive. The provider outage (08:28–09:25 HKT cluster) is no longer observable in the stack.

### Aggregate impact on E0 v3 scores

| Model | Condition | Correct (before / after Phase 2) | Errors (before / after Phase 2) |
|-------|-----------|----------------------------------|--------------------------------|
| Mistral | C4 | 8 / **11** (+3) | 55 (incl. 16 infra) / 47 (0 infra) |
| Qwen | C4 | 16 / 16 (no-op) | 54 (0 infra) / 54 |

### Methodology conformance

- Regex applied unchanged from pre-registration (no edits after seeing outcomes).
- One re-attempt per row; no second re-attempts on the 13 non-correct outcomes.
- No retargeting: the 16 dropped task_ids exactly matched the max_samples=16 slice after shuffle+done_questions filter (shuffle invariant verified in launch log: `Found 64 previous results! | Limited to 16 tasks (max_samples=16)`).
- Rule applied symmetrically: Qwen's 0-match result is an honest reflection of provider reliability during E0 (no carve-out).

## Files touched by Phase 2

- `scripts/rerun_e0_infra_errors.sh` (new) — one-shot launcher; written 2026-04-20 after Phase 1 completion verified.
- `workdir/gaia_c4_mistral_20260420_E0v3/dra.jsonl` (in-place rewrite + backup)
- `workdir/gaia_c4_qwen_20260420_E0v3/dra.jsonl` (unchanged; no matches)

No code changes to `run_gaia.py`, `run_eval_matrix.sh`, `resume_e0.sh`, or any src/ module.
