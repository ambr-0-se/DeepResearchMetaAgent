# Handoff: Pass 2 — Qwen-4B (vLLM) Budget + Pruning Tuning

**Session date:** 2026-04-18
**Baseline eval:** `workdir/gaia_test_127877_20260327_230408/` (GAIA test set 15/301 = 4.98%)
**Commit:** `63486ca` — `perf(qwen-4b): Pass 2 tuning — raise sub-agent max_steps, tighten pruning`
**Branch:** `main` — **needs push** (see the index — all pending handoffs get pushed together)
**Scope:** `configs/config_gaia_adaptive_qwen.py` only. NOT the 12 matrix configs. Stacks on top of `HANDOFF_RC1_FINAL_ANSWER_GUARD.md`; validate that one first, then this one.

---

## TL;DR Checklist

### Completed

- [x] Raised sub-agent `max_steps` 3 → 7 (deep_researcher, deep_analyzer, browser_use). 56+ `AgentMaxStepsError: Reached max steps.` log lines in the baseline run.
- [x] Raised `browser_use_agent.max_steps` 5 → 7 (web navigation is step-hungry).
- [x] Added `context_prune_threshold_ratio=0.75` override on all 4 agent config dicts (default is 0.85 in `src/utils/token_utils.py`; Qwen's 32 k wall needs earlier pruning).
- [x] Added `tests/test_config_qwen_tuning.py` — 2 AST tests guarding the 4 max_steps + 4 pruning ratio values.
- [x] Verified locally: config parses, tests pass.
- [x] Committed as `63486ca` (2 files, +104/−4).

### To Do Next Session (on GPU farm)

- [ ] Pull: `git pull` — confirm HEAD = `63486ca`.
- [ ] Run the Pass 2 regression test: `pytest tests/test_config_qwen_tuning.py -v` — target: 2/2 pass.
- [ ] **Validate RC1 + Pass 2 together**: run the 10-question smoke (see HANDOFF_RC1 step 3) and then the full test-set run (HANDOFF_RC1 step 4). Validation criteria below compose with RC1's.
- [ ] **Collect V5 measurement** for the NEXT tuning pass:
  - `grep -c "AgentMaxStepsError" workdir/gaia_adaptive_qwen/log.txt` — target: strictly less than the baseline's 56.
  - `grep "Step [0-9]*: Duration" workdir/gaia_adaptive_qwen/log.txt | awk -F'[:| ]' '{print $2,$4}' | sort -n` — distribution of max step reached per sub-agent run. Compute P95 of step counts on successful runs; that's the input for the next tuning pass.
  - `grep -c "Context Pruning" workdir/gaia_adaptive_qwen/log.txt` — count of prune events. Compare to baseline.
  - `grep -c "This model's maximum context length" workdir/gaia_adaptive_qwen/log.txt` — count of vLLM 400s. **Target: 0.** If > 0, Pass 2.2 is not aggressive enough; drop ratio to 0.70 or reduce `context_prune_tail_segments` 4 → 2.

---

## Original Problem

Two clusters in the 4.98% baseline (same run as RC1):

| Cluster | Evidence |
|---|---|
| Sub-agent `max_steps` hits | 56 occurrences of `AgentMaxStepsError: Reached max steps.` in `workdir/gaia_test_127877_20260327_230408/log.txt`. Config had `deep_researcher_agent.max_steps=3`, `deep_analyzer_agent.max_steps=3`, `browser_use_agent.max_steps=5`. Qwen-4B is a weak instruction-follower and needs more turns to recover from mid-delegation errors. |
| Context overflow | 9 tasks hit vLLM's 32 768-token ceiling. Observed input sizes in `log.txt`: 45 128, 55 762, 111 893 tokens. The 85% prune threshold fires at ~27 853 tokens; by the time one 8–12 k tool result arrives on top, the prompt is already over 32 k. |

## Changes (commit `63486ca`)

### 2.1 — sub-agent max_steps 3/3/5 → 7/7/7

`configs/config_gaia_adaptive_qwen.py`:
- `deep_researcher_agent_config.max_steps`: 3 → 7
- `deep_analyzer_agent_config.max_steps`: 3 → 7
- `browser_use_agent_config.max_steps`: 5 → 7
- `planning_agent_config.max_steps`: 25 (unchanged)

7 is a starting point pending GPU-farm P95 measurement of successful-run step counts per sub-agent. Wall-clock bounded by `per_question_timeout_secs=1200`; worst-case budget is still under the wall.

### 2.2 — context_prune_threshold_ratio per-config override

Same file; added `context_prune_threshold_ratio=0.75` to each of the 4 agent config dicts. Read by `src/agent/general_agent/general_agent.py:168` via `getattr(self.config, "context_prune_threshold_ratio", DEFAULT_CONTEXT_PRUNE_THRESHOLD_RATIO)` — the override on the agent dict wins, the default in `src/utils/token_utils.py:22` stays 0.85 for everything else (ARC, matrix configs, non-Qwen GAIA).

With `max_model_len=32 768` and `context_prune_reserve_tokens=4096`:
- 0.85 default → pruning fires at 27 853 tokens, effective budget 23 757.
- 0.75 Qwen override → pruning fires at 24 576 tokens, effective budget 20 480. 8 k of headroom before the 32 k wall.

### Regression test

`tests/test_config_qwen_tuning.py`:
- `test_sub_agent_max_steps_raised_to_seven` — AST-parses the 4 agent dicts and asserts the expected `max_steps` values.
- `test_context_prune_threshold_ratio_override_on_every_agent_config` — asserts all 4 dicts carry the 0.75 override.

## Why NOT the 12 matrix configs

`config_gaia_c{0,2,3,4}_{qwen,kimi,mistral}.py` all use `qwen3.6-plus-failover` / `kimi-k2.5-no-thinking` / `mistral-small-failover` — API-served models with **128 k context windows** and materially better instruction-following than a local vLLM 4B. They haven't been GPU-validated yet, and their `max_steps=3/3/5` budgets may be correct for them. Touching all 12 without measurement adds blast radius without signal. **Wait for the matrix baseline run; then tune per-config based on that run's step-distribution and context-pressure data.** Note: matrix configs are generated by `scripts/gen_eval_configs.py` — any budget changes should go into the generator template, not hand-edited into the 12 files.

## GPU-Farm Test Commands

```bash
cd DeepResearchMetaAgent
conda activate dra
git pull && git log -1 --oneline                        # confirm HEAD = 63486ca

# 1. Regression test
pytest tests/test_config_qwen_tuning.py -v              # 2/2 pass

# 2. Smoke (stacks on RC1 smoke — run that first)
python examples/run_gaia.py --config configs/config_gaia_adaptive_qwen.py --cfg-options max_samples=10 dataset.split=validation

# 3. Full GAIA test-set run (same script as the RC1 handoff)
sbatch run_gaia_test_eval.sh
```

## Validation Criteria (concrete)

After step 2 above:

```bash
cd workdir/gaia_adaptive_qwen
grep -c "AgentMaxStepsError" log.txt      # strictly < baseline 56 across the full set; for 10-q smoke, target 0
grep -c "Context Pruning" log.txt         # should be > 0 — tighter threshold means we prune more aggressively
grep -c "This model's maximum context length" log.txt   # TARGET: 0 (baseline had 9 on the full set)

# Composite with RC1
grep -c "premature-final-answer guard" log.txt   # > 0 from RC1

# Did the previously passing task still pass?
python scripts/analyze_results.py dra.jsonl | grep "Time-Parking"
```

After step 3 (full test-set):

- **Score**: strictly > 4.98%. Forecast with RC1 + Pass 2 stacked: 10–15% (not a blowout; model capability is still a ceiling on this config).
- **`AgentMaxStepsError` count**: target < 30 (baseline 56). If it went UP, 7 is still too low; raise to 10 next pass.
- **Context-overflow 400 count**: target 0 (baseline 9). If any occur, 0.75 is not aggressive enough.
- **`per_question_timeout_secs` hits**: should NOT increase (baseline 5). If many more tasks are timing out, the 7-step budget × 25-planner-step math is overshooting — the timeout becomes a new bottleneck and Pass 2 either needs to be rolled back or the planner `max_steps` lowered.

## Known Unknowns / Caveats

- **7 is a starting point, not the final value**: this is a guess informed by the 56-count evidence but not by a measured step distribution. If the forthcoming GPU-farm run shows successful-run P95 step counts are 4, then 7 is overkill and 5 would be enough. If P95 is 9, then 7 is still too low.
- **0.75 may not be aggressive enough for 45 k / 55 k / 111 k overshoots**: the 111 k case in the baseline is 3.4× the context wall — no threshold-only fix can recover that single task; it needs per-observation truncation or `tail_segments 4 → 2`. Logged as follow-up, not in this pass.
- **`per_question_timeout_secs=1200` is the wall**: if 7-step sub-agents chain deeply (planner 25 × sub-agent 7 × 30 s = ~87 min worst case), the timeout trips before max_steps does. This is bounded and acceptable for the initial run, but if timeout-trips dominate failures, the response is to EITHER lower sub-agent max_steps OR raise `per_question_timeout_secs` (to e.g. 1800), not both.
- **No measurement protocol change**: the same `log.txt` grep patterns described above are the measurement. No need to instrument the code further for Pass 2.
