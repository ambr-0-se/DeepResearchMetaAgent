# Handoff Index

Entry point for change handoffs that still need validation on the GPU farm. Each linked document is **self-contained** — read the specific handoff doc for original problem, changes, test commands, and validation criteria.

---

## Pending Handoffs

| # | Title | Doc | Status | Commit(s) | Pushed? |
|---|-------|-----|--------|-----------|---------|
| 1 | Silent-failure fixes for browser + analyzer tools | [HANDOFF_SILENT_FAILURES.md](HANDOFF_SILENT_FAILURES.md) | Ready to validate | `ba28f21` | Yes |
| 2 | Multi-provider integration + GAIA eval matrix (Mistral/Kimi/Qwen × C0–C4) with Qwen DashScope→OpenRouter failover | [HANDOFF_PROVIDER_MATRIX.md](HANDOFF_PROVIDER_MATRIX.md) | Ready to validate | `7632470` → `9883a3a` | Yes |
| 3 | `modify_subagent` prompt + tool-description guidance expansion (all 7 actions covered, failure-mode→action table, condition-scoped anti-patterns for C2/C3/C4) | [HANDOFF_MODIFY_SUBAGENT_GUIDANCE.md](HANDOFF_MODIFY_SUBAGENT_GUIDANCE.md) | Ready to validate | `764c6bf` | Yes |
| 4 | Core C3 / C4 implementation — structural REVIEW step + cross-task skill library (the four experimental conditions C0/C2/C3/C4 themselves) | [HANDOFF_C3_C4_IMPLEMENTATION.md](HANDOFF_C3_C4_IMPLEMENTATION.md) | Ready to validate | `60065a8` → `433c30e` → `0643089` → `d247605` | Yes |

---

## Completed / Archived

_(none yet — move rows here once their validation pass is signed off)_

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

## For the Next Session

1. Read this index first.
2. Pick the top row that is `Ready to validate` and open its doc.
3. The doc tells you what to push, run, grep, and compare.
4. When done, update the row's Status, move it to Completed, and note the validating run's tag.
