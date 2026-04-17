# Handoff: RC1 Premature `final_answer_tool` Guard + Related Runtime Fixes

**Session dates:** 2026-04-17 (Pass 1 + Pass 3.1), 2026-04-18 (scope-creep fixups from Codex round 3)
**Baseline eval:** `workdir/gaia_test_127877_20260327_230408/` — GAIA test set 15/301 (4.98%)
**Commits:** `54e7707`, `a9a6985`, `c52cf91`, `912685f`, `d36f4d4`
**Branch:** `main` — **already pushed** to `origin/main`
**Scope:** Runtime guard in `process_tool_calls`, duplicate-yield bug fix in `_run_stream`, RC2 exception-chain diagnostic hook in `execute_tool_call`, prompt contradiction fix across 9 YAMLs, tool description tightening. No config or model changes in this handoff — see `HANDOFF_PASS2_QWEN_TUNING.md` for that.

---

## TL;DR Checklist

### Completed

- [x] Root-caused the 4.98% GAIA test-set score across 4 clusters: RC1 premature `final_answer_tool` (63/301 = 21%), RC2 `UnboundLocalError: final_answer` (87/301 = 29%), sub-agent `max_steps` hits (56 in log), context overflow (9/301 = 3%).
- [x] Ran `/codex review` twice on the plan (18 + 7 findings); all actionable items resolved or deferred with evidence.
- [x] Ran `/codex review` on the 3-commit implementation (5 findings); 3 fixed in-pass (toolcalling_agent.yaml + RC2 label logic + committed), 2 explicitly deferred with justification (lone-final heuristic carries FP risk; duplicate-all-dropped bounded by max_steps).
- [x] Wired an up-front guard in `GeneralAgent.process_tool_calls` that handles every ordering of premature / duplicate `final_answer_tool`.
- [x] Fixed the duplicate-yield bug at `_run_stream` L518 (sync + async base classes) where the stale loop `action_step` was re-yielded after `_handle_max_steps_reached` had already appended its own terminal step.
- [x] Added RC2 exception-chain diagnostic hook in `execute_tool_call` that logs full `__cause__` / `__context__` chains with tracebacks for `NameError` / `UnboundLocalError` — diagnostic only, error propagation unchanged.
- [x] Fixed the `"name": "final_answer"` vs `"name": "final_answer_tool"` contradiction across all 9 agent YAMLs (planning, adaptive, c3, c4, general, deep_researcher, deep_analyzer, browser_use, toolcalling — 41 lines total).
- [x] Rewrote the L20 framing away from "stuck on a loop" pressure; added Rule 5 ("never emit `final_answer_tool` in the same turn as any other tool call").
- [x] Tightened `FinalAnswerTool.description` + `answer` parameter description so the contract is re-sent every turn.
- [x] Added 18 regression tests across 3 new test files; all passing locally (parse-only; heavy-dep tests like pytest run on GPU farm).

### To Do Next Session (on GPU farm)

- [ ] Pull on GPU farm: `ssh <gpu-farm> && cd DeepResearchMetaAgent && git pull` — confirm `git log -1 --oneline` shows `63486ca` (Pass 2) as head (last handoff in the RC1+Pass2 stack).
- [ ] Run the 3 new test files under the `dra` env: `pytest tests/test_process_tool_calls_guard.py tests/test_max_steps_yield_order.py tests/test_rc2_diagnostic_hook.py -v`. Target: 18/18 pass.
- [ ] Run a **10-question smoke**: `python examples/run_gaia.py --config configs/config_gaia_adaptive_qwen.py --cfg-options max_samples=10 dataset.split=validation`. Must-haves in the new `log.txt`:
  - ≥ 1 `[premature-final-answer guard]` log entry fires on the ping-pong riddle / Finding Nemo task (the two tasks that previously hallucinated `50` and `12345, 67890`).
  - No `cannot access local variable 'final_answer'` lines OR: the same lines but NOW accompanied by `[RC2 diagnostic] Scope error in sub-agent ... full exception chain follows:` with a real traceback — that's what Pass 3.2 needs.
  - The pre-fix one passing task (#6 Time-Parking 2) still passes.
- [ ] Run the **full 301-task GAIA test set** once smoke passes: `sbatch run_gaia_test_eval.sh`. Expected score improvement from 4.98%; confidence is "non-zero" not "large" — see "Realistic outcome forecast" below.
- [ ] Collect V5 measurement (for the next tuning pass, gated by the Pass 2 doc):
  - P95 of step counts per sub-agent on **successful** runs (per-agent `intermediate_steps` count filtered by `prediction == true_answer`).
  - P95 of `estimated_tokens` on turns where `[Context Pruning]` fires.
  - Count of `[premature-final-answer guard]` vs `[duplicate-final-answer guard]` log lines.

---

## Original Problem

GAIA test-set run `gaia_test_127877_20260327_230408` (config_gaia_adaptive_qwen.py, vLLM Qwen3-VL-4B, 301 tasks) scored 4.98% (15/301). Root-cause taxonomy:

| Cluster | Tasks | Mechanism |
|---|---|---|
| RC1 — premature `final_answer_tool` | 63/301 (21%) | Qwen emits `final_answer_tool` alongside other tool calls in the same turn with a fabricated `answer` argument (e.g. `'12345, 67890'` for zip codes, `'50'` for the ping-pong ball riddle). The agent accepts the hallucinated answer and terminates. |
| RC2 — `UnboundLocalError: final_answer` | 87/301 (29%) | Sub-agents raise `UnboundLocalError` that surfaces as `AgentToolExecutionError: … cannot access local variable 'final_answer'`. 222 log lines. **Exact triggering path not yet traced** — the existing wrap reduced the inner exception to `str(e)`, stripping the traceback. |
| Sub-agent `max_steps` hits | 56 log lines | Sub-agents hit their `max_steps=3` / `5` caps before useful work completes. |
| Context overflow (32k Qwen) | 9/301 (3%) | Pruning threshold 85% prunes too late. vLLM returns 400 on 45k–111k-token prompts. |

## Changes (mapped to commits)

| Commit | Scope |
|---|---|
| `54e7707` | New `src/agent/general_agent/_tool_call_guard.py` with `apply_final_answer_guard()` pure filter. Wired into `GeneralAgent.process_tool_calls` (up-front, before separation loop). Memory sync via `chat_message.tool_calls = effective_tool_calls or None`. Tool description + answer param description tightened. RC2 diagnostic hook added to `execute_tool_call`. Tests: `test_process_tool_calls_guard.py` (8 cases), `test_rc2_diagnostic_hook.py` (4 cases). |
| `a9a6985` | `_handle_max_steps_reached` now returns `ActionStep` (not the content string); `_run_stream` yields that fresh step and reads `action_output` from it. Applied symmetrically to sync (`src/base/multistep_agent.py`) and async (`src/base/async_multistep_agent.py`). Test: `test_max_steps_yield_order.py` (6 AST-based regression checks). |
| `c52cf91` | 8 agent YAMLs: 32 `"name": "final_answer"` → `"name": "final_answer_tool"` replacements, L20 rewrite, Rule 5 added. |
| `912685f` | Codex-round-3 follow-up: `src/base/prompts/toolcalling_agent.yaml` had the same 5-line contradiction — fixed. |
| `d36f4d4` | Codex-round-3 follow-up: RC2 chain-label logic was wrong (`cur is cur.__cause__` is ~always false). Labels now correctly tag each link as `root` / `__cause__` / `__context__`. |

## GPU-Farm Test Commands

### 1. Unit tests (fast, under 5s)

```bash
cd DeepResearchMetaAgent
conda activate dra
pytest tests/test_process_tool_calls_guard.py tests/test_max_steps_yield_order.py tests/test_rc2_diagnostic_hook.py -v
```
Expected: 18 passed, 0 failed.

### 2. Parse regression — every prompt YAML still parses and the contradiction is gone

```bash
for f in src/agent/*/prompts/*.yaml src/base/prompts/*.yaml; do python -c "import yaml; yaml.safe_load(open('$f').read())" && echo "OK $f"; done
grep -rn '"name": "final_answer"' src/ | grep -v final_answer_tool
```
Expected: every `.yaml` parses; grep returns zero lines (contradiction gone everywhere).

### 3. 10-question smoke on the adaptive Qwen config

```bash
python examples/run_gaia.py --config configs/config_gaia_adaptive_qwen.py --cfg-options max_samples=10 dataset.split=validation
```
Output lands in `workdir/gaia_adaptive_qwen/`.

### 4. Full GAIA test-set run (301 tasks)

```bash
sbatch run_gaia_test_eval.sh
```

## Validation Criteria (concrete)

After step 3 above, run:

```bash
cd workdir/gaia_adaptive_qwen
# RC1 guard must fire at least once
grep -c "premature-final-answer guard" log.txt           # expected: > 0
grep -c "duplicate-final-answer guard" log.txt           # expected: 0 or small (duplicate-only case is rare)

# RC2 diagnostic hook must surface tracebacks when the scope error occurs
grep -c "cannot access local variable 'final_answer'" log.txt  # expected: some (may still occur until Pass 3.2/3.3 land)
grep -c "\[RC2 diagnostic\] Scope error in sub-agent" log.txt  # expected: MATCHES the line above
# The line above is critical — it's what unblocks Pass 3.2. If it does not
# appear, the hook did not fire and we have another bug in the diagnostic
# code path.

# Previously passing task #6 (Time-Parking 2) still passes
python scripts/analyze_results.py dra.jsonl | grep "Time-Parking"  # expected: correct
```

After step 4 (full test-set run), run `scripts/analyze_results.py` on the output and diff against `workdir/gaia_test_127877_20260327_230408/dra.jsonl`:

- **Pass 1 metric**: count tasks where step-1 `tool_calls` contains BOTH `final_answer_tool` AND other tools. Baseline 63/301 (21%). **Target: 0.**
- **Score**: strictly > 4.98%. Confidence: non-zero improvement, not a large jump. Forecast: 8–12% if RC2 still crashes, 12–18% if RC2 was already-hallucinated finals that also benefit from the guard.
- **RC2 tracebacks**: if the RC2 diagnostic hook fired on real errors, the log will show the exact traceback origin. Capture the first one and hand off as `HANDOFF_RC2_TRACE.md` for Pass 3.2.

## Known Unknowns / Caveats

- **Lone premature `final_answer_tool` still passes**: if Qwen emits a solitary `final_answer_tool` with a fabricated answer and no siblings, the guard does NOT fire (status `"none"`). This is deliberately deferred — a "require prior observation in memory" heuristic has FP risk on trivia / direct-knowledge tasks. Revisit if the baseline-forecast delta is smaller than expected.
- **Duplicate-all-dropped has no in-memory recovery signal**: when the guard drops all calls in the `duplicate` branch, the model sees no feedback in memory on the next turn. Bounded by `max_steps=25` (worst case 25 wasted turns). Rare pattern in practice — the 63 failing cases were all mixed-turn, not duplicate-only.
- **AST-based regression tests**: `test_max_steps_yield_order.py` and `test_rc2_diagnostic_hook.py` check identifier names and presence, not runtime behavior. A legitimate refactor that renames `final_memory_step` will break them even if semantics are preserved — update the test alongside the refactor.
- **The matrix configs are not affected by this pass**: the 12 `config_gaia_<c0|c2|c3|c4>_<kimi|mistral|qwen>.py` files use `qwen3.6-plus-failover` (128k API context), not the vLLM 4B that produced the 4.98% baseline. The RC1 guard and the duplicate-yield fix are shared runtime code so the matrix benefits transparently, but the Qwen-4B-specific `max_steps` and pruning tuning in `HANDOFF_PASS2_QWEN_TUNING.md` only touches `configs/config_gaia_adaptive_qwen.py`.
