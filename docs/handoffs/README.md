# Handoff Documents

Per-change-set documentation for work that needs validation on the GPU farm.
**Start at [`../../HANDOFF_INDEX.md`](../../HANDOFF_INDEX.md)** (kept at repo root so the
operator's muscle memory — "open HANDOFF_INDEX.md first" — still works); this
directory holds the per-topic companion docs it links to.

## What lives here

One file per logical change set — each a self-contained record of a single
code movement (bug fix, feature, refactor) plus the validation criteria that
prove it shipped correctly. The content is append-only history: you don't
edit these after landing the work, you just add a sibling when a later change
supersedes or builds on the earlier one.

```
docs/handoffs/
├── README.md                                   # this file
├── HANDOFF_SILENT_FAILURES.md                  # browser/analyzer silent-fail fixes
├── HANDOFF_PROVIDER_MATRIX.md                  # Multi-provider + GAIA matrix (4 models × C0–C4; see index for 16-cell)
├── HANDOFF_MODIFY_SUBAGENT_GUIDANCE.md         # prompt + tool-description expansions
├── HANDOFF_C3_C4_IMPLEMENTATION.md             # ReviewStep + Skill library
├── HANDOFF_RC1_FINAL_ANSWER_GUARD.md           # RC1/RC2 final-answer guards
├── HANDOFF_PASS2_QWEN_TUNING.md                # Qwen-4B vLLM tuning (local-vLLM only)
├── HANDOFF_TOOLGENERATOR.md                    # ToolGenerator hardening
└── HANDOFF_<TOPIC>.md                          # future change sets — see template below
```

## Why per-change files, not one big CHANGELOG

1. **Narrative stays with the change.** Git commit messages answer "what",
   but `HANDOFF_<TOPIC>.md` answers "why it was needed", "what to grep for
   to prove it's working", and "known unknowns". That context rots fast in
   a one-line PR description.
2. **Independent validation trails.** Each doc has a test-command section
   that can be run in isolation on the GPU farm. If two changes are
   independent (revert-able separately), they get separate docs.
3. **Research reproducibility.** This repo is an APAI4799 research project
   (see root `CLAUDE.md`); these docs are the paper-trail for how the
   experimental conditions C0/C2/C3/C4 were built and validated.

## Authoring conventions

Every handoff doc must contain:

1. **TL;DR checklist** — done / to do
2. **Original problem** — log evidence or a minimal reproducer
3. **Changes table** — each row maps one file to its commit hash(es)
4. **GPU-farm test commands** — prereqs, how to kick off, where results land
5. **Validation criteria** — concrete grep / diff / pass thresholds
6. **Known unknowns / caveats** — what a future reader might trip on

Filename: `HANDOFF_<SHORT_TOPIC>.md`. Short topic is a noun phrase (e.g.
`SILENT_FAILURES`, `PROVIDER_MATRIX`, `REVIEW_STEP_BUG`).

Status values (tracked in `HANDOFF_INDEX.md`, not in the per-topic file):
`Drafting` → `Ready to validate` → `Validating` → `Validated` (Completed) /
`Blocked` / `Rolled back` / `N/A`.

## Lifecycle

```
 drafting  ─→  pending push  ─→  on the farm  ─→  validated
    │              │                  │              │
    │              │                  │              └─ move row in INDEX to "Completed / Archived"
    │              │                  └─ grep/diff against validation criteria
    │              └─ git push origin main before git pull on the farm
    └─ write this file alongside the commit(s)
```

Docs are **never deleted** even after validation passes — they're the
historical record of why the code looks the way it does. When a later
handoff supersedes an earlier one, note the supersession in both docs
(and update `HANDOFF_INDEX.md` to mark the earlier as "Superseded by #N").
