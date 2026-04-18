# Handoff: Silent-Failure Fixes for Browser + Analyzer Tools

**Session date:** 2026-04-17
**Baseline eval:** `workdir/gaia_verify10_127871_20260327_214432/` (6/10 answered, 3/10 "Unable to determine")
**Commit:** `ba28f21` — `fix(tools): surface silent failures in browser + analyzer tools`
**Branch:** `main` (local only — **not yet pushed** to `origin/main`)

---

## TL;DR Checklist

### Completed

- [x] Diagnosed two silent-failure patterns in the 2026-03-27 GAIA eval
- [x] Implemented defensive fixes in browser + analyzer tools
- [x] Added prompt guardrail against hallucinated file paths
- [x] Verified YAML parses and Python compiles
- [x] Reviewed fixes with Haiku code-reviewer: **FIXES BOTH PROBLEMS** (defensive layer carries load; prompt is a safety net)
- [x] Committed as `ba28f21` (3 files, +46/−7)

### To Do Next Session

- [ ] `git push origin main` (commit is local only)
- [ ] Pull on GPU farm: `ssh <gpu-farm> && cd DeepResearchMetaAgent && git pull`
- [ ] Reproduce the 10-question `gaia_verify10` baseline with fixes applied
- [ ] Validate fix 1 (browser) by checking Task 2 no longer loops on empty observations
- [ ] Validate fix 2 (analyzer) by checking no `No such file or directory: 'code.txt'` errors
- [ ] Compare answer correctness vs. the 2026-03-27 baseline (target: strictly better than 6/10)
- [ ] If successful, consider scaling up to the new 12-config matrix (C0–C4 × Qwen/Kimi/Mistral)

---

## Original Problems

### Problem 1 — Silent browser tool failure

**Where:** `src/tools/auto_browser.py` (`auto_browser_use_tool`)

**Symptom in log:** Around line 1900 of `workdir/gaia_verify10_127871_20260327_214432/log.txt`, the tool was called 5+ times back-to-back with empty `Observation:\n` after each call. The parent agent looped until it hit max_steps and emitted `Unable to determine`.

**Root cause:** When `browser_use.Agent.run()` exhausted its own internal `max_steps=50` without extracting anything:
1. `history.extracted_content()` returned `[]`
2. `"\n".join([])` returned `""`
3. `ToolResult(output="", error=None)` was returned → tool reported **success with empty content**
4. The calling agent couldn't distinguish this from a legitimate empty page and retried the same call with the same args

**Impact on eval:** Task 2 (`17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc`) "Find fish species from Finding Nemo" failed despite being trivially answerable.

### Problem 2 — Hallucinated file paths

**Where:** `src/tools/deep_analyzer.py` → `MarkitdownConverter.convert()` at `src/tools/markdown/mdconvert.py:186`

**Symptom in log:** Five occurrences of `[Errno 2] No such file or directory: 'code.txt'` / `'unlambda_code.txt'`. Model retried the same hallucinated filename repeatedly.

**Root cause:** The Qwen model received a GAIA task containing inline Unlambda code (no file attachment), but was trained by prompt examples like `{"document": "document.pdf"}` to put file paths in tool arguments. It invented `"source": "code.txt"` and passed it to `deep_analyzer_tool`. There was no existence check before forwarding to `MarkitdownConverter`, which tried to open the non-existent file and raised `FileNotFoundError`. The error was caught silently at `mdconvert.py:186`, returning `None`. The downstream `.text_content` access would then crash with `AttributeError` (latent bug).

**Impact on eval:** Unlambda task (`14569e28-c88c-43e4-8c32-097d35b9a67d`) contributed to the 3/10 failure count.

### Problem 3 — Nature p-value task (NOT fixable, for reference)

Task `04a04a9b-226c-43fd-b319-d5e89743676f` asked for the "average p-value of Nature articles in 2020." Data is genuinely not public. Model correctly refused. **No fix needed** — this is working as intended.

---

## Changes & Commits

### Commit `ba28f21` — three files

| File | Change |
|------|--------|
| [src/tools/auto_browser.py](src/tools/auto_browser.py) | `_browser_task` now raises `RuntimeError` with diagnostic (internal steps + visited URLs + recommendation to switch agent) when no content is extracted. `forward` wraps the exception as `ToolResult(error=str(e))`. |
| [src/tools/deep_analyzer.py](src/tools/deep_analyzer.py) | `_analyze` now checks `os.path.isfile(source)` for non-URI sources before converting. Returns a LLM-digestible error message instructing the caller to pass inline content via `task`. Also guards against `converter.convert()` returning `None` (fixes latent `AttributeError`). |
| [src/agent/deep_analyzer_agent/prompts/deep_analyzer_agent.yaml](src/agent/deep_analyzer_agent/prompts/deep_analyzer_agent.yaml) | New "IMPORTANT — `source` parameter rules" block added after the image-generator example, explicitly forbidding invented file names and directing inline content to the `task` parameter. |

Diff stats: `+46 / −7` across three files.

### Not in the commit (intentionally excluded)

- `src/models/models.py` — unrelated multi-provider work (DeepSeek/Kimi/Mistral/MiniMax integration)
- 12 new `configs/config_gaia_*_*.py` files — unrelated evaluation matrix
- `src/models/failover.py`, `tests/test_failover_model.py`, `scripts/gen_eval_configs.py`, `scripts/run_eval_matrix.sh` — unrelated failover work

---

## How to Test on GPU Farm

### Prerequisites

```bash
# 1. Push the commit to remote (run locally first)
cd "/Users/ahbo/Desktop/APAI4799 MetaAgent/DeepResearchMetaAgent"
git push origin main

# 2. On GPU farm, pull latest
ssh <gpu-farm-host>
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
git pull origin main
git log -1 --oneline  # Should show ba28f21

# 3. Confirm conda env + Playwright browsers are ready
conda activate dra
bash scripts/ensure_playwright_browsers.sh
```

### Reproduce the baseline (10-question verify)

The original failing run used `run_combined_test.sh` with the tag `gaia_verify10_127871`. Re-run the same script on the **same 10 questions** so we can compare apples-to-apples.

```bash
cd /userhome/cs2/ambr0se/DeepResearchMetaAgent
sbatch run_combined_test.sh
```

- SLURM output: `logs/combined_test_<JOBID>.out`
- Results: `workdir/gaia_verify10_<JOBID>_<timestamp>/dra.jsonl`
- Submission: `workdir/gaia_verify10_<JOBID>_<timestamp>/submission.jsonl`

Wall-clock: ~30 min to 6 h depending on task mix and model speed. Watch the log:

```bash
squeue -u $USER
tail -f logs/combined_test_<JOBID>.out
```

### (Optional) Targeted single-task re-run

`run_gaia.py` does **not** have a `--task-ids` flag. To test just the two previously-failing tasks, either:

**Option A:** Let the full 10-question run execute; then inspect only the two task IDs in the output. Cheaper than writing custom filter code, and you get the full picture.

**Option B:** Temporarily add a filter in `examples/run_gaia.py` after line 263 (`tasks_to_run = tasks_to_run[:int(max_samples)]`) such as:

```python
target_ids = {"17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc",   # fish species
              "14569e28-c88c-43e4-8c32-097d35b9a67d"}   # unlambda
tasks_to_run = [t for t in tasks_to_run if t["task_id"] in target_ids]
```

**Do not commit this filter** — it's only for quick iteration.

---

## How to Validate the Fixes

Run these checks against the new `workdir/gaia_verify10_<NEW_JOBID>_*/` directory and compare to the `gaia_verify10_127871_20260327_214432` baseline.

### Validation 1 — Fix 1 (browser empty observations)

```bash
NEW_RUN=workdir/gaia_verify10_<NEW_JOBID>_*/log.txt
OLD_RUN=workdir/gaia_verify10_127871_20260327_214432/log.txt

# Old run: should show many blank "Observation:\n" lines from browser tool
grep -cE "^Observation:$" "$OLD_RUN"

# New run: empty observations from browser tool should drop substantially
grep -cE "^Observation:$" "$NEW_RUN"

# Expected new behavior: the new RuntimeError diagnostic should appear
grep -c "auto_browser_use_tool returned no extracted content" "$NEW_RUN"
# Should be > 0 if the browser failed at all; if 0, even better (browser worked)
```

**Pass criterion:** Either (a) no browser failures at all, or (b) browser failures now surface as a visible error with the diagnostic string — **not** as blank observations.

### Validation 2 — Fix 2 (hallucinated paths)

```bash
# Old run: 5 file-not-found errors
grep -c "No such file or directory: 'code.txt'" "$OLD_RUN"
grep -c "No such file or directory: 'unlambda_code.txt'" "$OLD_RUN"

# New run: these errors must NOT appear — the existence check blocks them
grep -c "No such file or directory: 'code.txt'" "$NEW_RUN"
grep -c "No such file or directory: 'unlambda_code.txt'" "$NEW_RUN"

# Expected new behavior: if the agent still hallucinates, the tool now emits this
grep -c "ERROR: source=.* is neither an existing file nor a URI" "$NEW_RUN"
# > 0 means the defensive layer caught a hallucination (good — visible signal)
# = 0 means the prompt update prevented it (even better)
```

**Pass criterion:** Zero `No such file or directory: 'code.txt'` entries. Any `ERROR: source=...` entries are a feature, not a bug — they mean the agent tried to hallucinate but got caught and can now self-correct.

### Validation 3 — Answer correctness

```bash
python scripts/analyze_results.py workdir/gaia_verify10_<NEW_JOBID>_*/dra.jsonl
```

Or eyeball `submission.jsonl`:

```bash
cat workdir/gaia_verify10_<NEW_JOBID>_*/submission.jsonl
```

**Pass criterion:**
- Baseline: 6 answered / 3 "Unable to determine" / 1 empty
- Target: **at least 7 answered**, with Task 2 (fish species) flipping to a real answer. Unlambda task may still be hard for Qwen but should not fail with hallucinated-filename noise.

### Validation 4 — No regressions

```bash
# Quick sanity: confirm no unhandled exceptions from the new defensive code
grep -c "AttributeError: 'NoneType' object has no attribute 'text_content'" "$NEW_RUN"
# Must be 0 — the None guard in deep_analyzer.py fixes this latent bug
```

---

## Context the Next Session Will Need

### Key file paths

- **Approved plan:** `~/.claude/plans/enchanted-toasting-cosmos.md` (on local machine only)
- **Baseline eval dir:** `workdir/gaia_verify10_127871_20260327_214432/`
- **Baseline log:** `workdir/gaia_verify10_127871_20260327_214432/log.txt` (2 MB)
- **Baseline submissions:** `workdir/gaia_verify10_127871_20260327_214432/submission.jsonl`

### Baseline answers (for comparison)

```
Task  1: c61d22de-...  complex society       ✓ answered
Task  2: 17b5a6a3-...  Unable to determine   ✗ browser tool silent failure
Task  3: 04a04a9b-...  Unable to determine   ✗ data genuinely unavailable (NOT fixable)
Task  4: 14569e28-...  .                     ✗ Unlambda — hallucinated code.txt
Task  5: e1fc63a2-...  17500                 ✓ answered
Task  6: 32102e3e-...  Time-Parking 2...     ✓ answered
Task  7: 8e867cd7-...  2                     ✓ answered
Task  8: 3627a8be-...  8000                  ✓ answered
Task  9: 7619a514-...  Unable to determine   ✗ (cause not fully diagnosed)
Task 10: ec09fa32-...  50                    ✓ answered
```

### Known unknowns

- Task 9 (`7619a514-5fa8-43ef-9143-83b66a43d7a4`) also returned "Unable to determine" but its failure mode was not investigated. If the new run still fails it, investigate separately — it may be a third pattern.
- The 2026-04 multi-provider matrix (12 configs) is new untracked work. The fixes in `ba28f21` apply to all of them (tool-level, not config-level) but have only been reasoned-about for the Qwen-on-vLLM code path. If testing against Kimi/Mistral/DeepSeek, watch for provider-specific wrinkles.

### Useful commands

```bash
# Show the commit diff
git show ba28f21

# Show token usage from any eval
grep -E "Input tokens: [0-9,]+ \| Output tokens: [0-9,]+" workdir/<run_dir>/log.txt | \
  awk -F'[|:,]' '{i+=$3; o+=$5} END {print "Input:", i, "Output:", o}'

# Compare two runs side-by-side
python scripts/compare_results.py \
  workdir/gaia_verify10_127871_20260327_214432/dra.jsonl \
  workdir/gaia_verify10_<NEW_JOBID>_*/dra.jsonl
```
