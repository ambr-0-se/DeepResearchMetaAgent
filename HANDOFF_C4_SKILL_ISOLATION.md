# HANDOFF: Per-run C4 skill-library isolation

## TL;DR

**Problem.** Every C4 config pointed at the shared `skills_dir="src/skills"`. This caused three failure modes for the 3-models × 4-conditions GAIA evaluation matrix:
1. **Cross-model contamination.** Whichever model's C4 stream finished first seeded the next model with its extracted skills, confounding per-model attribution.
2. **Parallel filesystem race.** The matrix runner launches mistral/kimi/qwen streams in parallel. They would have written into the same directory simultaneously.
3. **Same-model overwrite.** Every re-run of a model mutated `src/skills/` in place, erasing prior runs and dirtying the git tree.

**Fix.** Each C4 invocation writes into its own `workdir/gaia_c4_<model>_<DRA_RUN_ID>/skills/`, seeded from the canonical `src/skills/` on first construction. `.seeded` marker makes seeding idempotent; `_latest` symlink provides stable inspection path. `workdir/` is already gitignored so the tracked tree stays clean.

**Plan reference:** `/Users/ahbo/.claude/plans/synthetic-gliding-bonbon.md` (approved by user).

### Done
- [x] `src/skills/_seed.py` — `seed_skills_dir(dst, src)` helper
- [x] `src/agent/adaptive_planning_agent/adaptive_planning_agent.py` — calls seed helper before `SkillRegistry(...)`
- [x] `scripts/gen_eval_configs.py` — emits `_RUN_ID` prelude + per-run `tag` + `skills_dir` for C4 configs only
- [x] `configs/config_gaia_c4.py` (hand-written base) — same per-run pattern applied
- [x] `configs/config_gaia_c4_{mistral,kimi,qwen}.py` — regenerated from the updated generator
- [x] `scripts/run_eval_matrix.sh` — exports shared `DRA_RUN_ID` and updates `_latest` symlinks per C4 cell
- [x] `tests/test_skill_seed.py` — 6 unit tests (fresh-seed, noop-on-marker, reseed-after-marker-deleted, self-copy-noop, missing-src, marker-content) all passing
- [x] Docs: `CLAUDE.md`, `README.md`, `src/skills/README.md`, `HANDOFF_PROVIDER_MATRIX.md` updated

### To do (GPU farm)
- [ ] `git push origin main`
- [ ] Smoke: `DRA_RUN_ID=smoke_$(date +%s) bash scripts/run_eval_matrix.sh smoke '' c4` — expect 3 disjoint `workdir/gaia_c4_{mistral,kimi,qwen}_smoke_*/skills/` directories each with 7 seed skills + `.seeded` marker
- [ ] Isolation check: run the same smoke twice with different `DRA_RUN_ID`, `diff -r` the two runs' skill dirs → no overlap of learned skills
- [ ] Resume check: Ctrl-C mid-run, restart with same `DRA_RUN_ID` → seed-copy skipped, prior learned skills survive
- [ ] Full: `bash scripts/run_eval_matrix.sh full` — confirm `git status` after shows zero tracked changes under `src/skills/`

## Design

**Run ID.** `DRA_RUN_ID` env var. Fallback: fresh `YYYYMMDD_HHMMSS` timestamp at config load. Matrix runner exports one value for the whole invocation so the three parallel streams co-stamp into one `workdir/gaia_c4_<model>_<run_id>/` per model.

**Layout.**
```
workdir/gaia_c4_<model>_<run_id>/
  dra.jsonl               # eval results
  skills/
    <seed skills...>      # copied from src/skills on first task
    <newly-extracted>/    # appended by SkillExtractor at task end
    .seeded               # marker; prevents re-copy on resume
workdir/gaia_c4_<model>_latest   # symlink → most-recent run dir
```

**Seeding.** `seed_skills_dir(dst, src)`:
1. Early-return if `dst.resolve() == src.resolve()` (legacy standalone `skills_dir="src/skills"` case — avoids dirtying the canonical source).
2. `mkdir(parents=True, exist_ok=True)` on `dst`.
3. If `dst / .seeded` exists, return 0.
4. For each `entry in src.iterdir()`: skip unless it's a directory containing `SKILL.md`. Otherwise `shutil.copytree(entry, dst / entry.name)`, catching `FileExistsError` to preserve pre-existing user content.
5. Write `.seeded` marker **last** (with ISO timestamp + source path + seeded names) so mid-seed crashes re-seed cleanly next start.

## Changes

| File | Purpose |
|------|---------|
| `src/skills/_seed.py` (new) | The `seed_skills_dir` helper + `SEED_MARKER_FILENAME` constant |
| `src/agent/adaptive_planning_agent/adaptive_planning_agent.py` | Import and invoke `seed_skills_dir` before `SkillRegistry(skills_path)` in `_build_skill_registry_from_config` |
| `scripts/gen_eval_configs.py` | `_C4_RUN_ID_PRELUDE` constant; C4 branch emits `tag = f"gaia_c4_<model>_{_RUN_ID}"` and `skills_dir = f"workdir/{tag}/skills"` |
| `configs/config_gaia_c4.py` | Hand-applied RUN_ID prelude + per-run `tag` + derived `skills_dir` |
| `configs/config_gaia_c4_{mistral,kimi,qwen}.py` | Regenerated from updated generator |
| `scripts/run_eval_matrix.sh` | `export DRA_RUN_ID=…` at top; `_latest` symlink retargeted per C4 cell via `ln -sfn`; expanded banner + final Results block |
| `tests/test_skill_seed.py` (new) | 6 unit tests exercising the seed helper |
| `CLAUDE.md`, `README.md`, `src/skills/README.md`, `HANDOFF_PROVIDER_MATRIX.md` | Doc sweep |

## Verification

**Unit (local, already passing):**
```bash
pytest tests/test_skill_seed.py -v
# 6 passed in ~0.03s
```

**Smoke (GPU farm):**
```bash
DRA_RUN_ID=smoke_$(date +%s) bash scripts/run_eval_matrix.sh smoke '' c4
ls workdir/gaia_c4_{mistral,kimi,qwen}_smoke_*/skills/
# Expect: 7 seed skills + .seeded marker in each of 3 disjoint dirs
readlink workdir/gaia_c4_mistral_latest
# Expect: gaia_c4_mistral_smoke_<epoch>
```

**Isolation invariant:**
```bash
A=run_a; B=run_b
DRA_RUN_ID=$A bash scripts/run_eval_matrix.sh smoke mistral c4
DRA_RUN_ID=$B bash scripts/run_eval_matrix.sh smoke mistral c4
diff -r workdir/gaia_c4_mistral_$A/skills/ workdir/gaia_c4_mistral_$B/skills/
# Expect: identical seed set; any learned skills diverge
```

**Git cleanliness (the original complaint):**
```bash
bash scripts/run_eval_matrix.sh full
git status -- src/skills/
# Expect: no output (no tracked changes under src/skills/)
```

## Known unknowns / caveats

- **Symlinks on NFS.** `ln -sfn` is wrapped in `|| true`; if the GPU farm home refuses symlinks the `_latest` convenience is lost but runs still work — find by timestamped name.
- **Resume atomicity.** `.seeded` marker is written last. If the process is killed mid-`copytree` before the marker is written, next start re-copies (may raise `FileExistsError` on partial child) — the helper swallows `FileExistsError` and preserves pre-existing dirs, so a partial dir becomes the "canonical" copy for that skill. Acceptable tradeoff; the alternative (remove-and-retry) would silently mask unrelated failures.
- **Analysis scripts.** `scripts/analyze_results.py` and `scripts/compare_results.py` take the JSONL path as a positional argument — no changes needed, just longer paths.
