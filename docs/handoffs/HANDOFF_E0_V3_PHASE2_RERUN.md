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

## Files touched by Phase 2

- `scripts/rerun_e0_infra_errors.sh` (new) — one-shot launcher; written after Phase 1 completes.
- `workdir/gaia_c4_mistral_20260420_E0v3/dra.jsonl` (in-place rewrite + backup)
- `workdir/gaia_c4_qwen_20260420_E0v3/dra.jsonl` (unchanged; no matches)

No code changes to `run_gaia.py`, `run_eval_matrix.sh`, `resume_e0.sh`, or any src/ module.
