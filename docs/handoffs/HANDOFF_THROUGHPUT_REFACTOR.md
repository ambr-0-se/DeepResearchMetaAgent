# Throughput + timeout-enforcement refactor — implementation plan

**Status:** ALL FOUR PHASES LANDED (2026-04-22). Unit tests green across the
full 140-test sweep + 43 new tests; smoke re-run with all phases active
pending (see "Execution log" below). Original pre-registration kept intact
below for audit.
**Goal:** make E3 cost/time projections trustworthy and cut wall time on Mistral's rate-limit-bound cells.
**Scope:** four phases, landed in strict order. Each is independently useful; each has its own smoke gate before promoting.
**Out of scope:** A.2 (extraction on timeout) — dropped per 2026-04-21 decision; no E0 re-run planned, so E2/E3's frozen-library path never triggers extraction.

---

## Summary of phases and landing order

| # | Phase | Files touched (primary) | Risk | Expected effect |
|---|-------|-------------------------|------|-----------------|
| **P1** | **A.1 hard wall-clock guard** | `examples/run_gaia.py` | Low | Per-question wall time capped at 1830 s (was 3300 s observed) |
| **P2** | **Streaming worker pool** | `examples/run_gaia.py` | Very low | Eliminates straggler blocking; ≈ –20 % to –40 % wall on heterogeneous cells |
| **P3** | **Qwen concurrency 4 → 8** | `scripts/gen_eval_configs.py` + regen 4 Qwen configs | Very low | ≈ +50 % Qwen throughput; evidence says zero 429s at 4 |
| **P4** | **Mistral multi-key round-robin** | `src/models/openaillm.py`, `src/models/models.py`, `.env.template`, tests | Medium | Halves Mistral rate-limit queueing (2 keys → 10 msg/min effective) |

Each phase is a separate commit (two if a docs row follows). Smoke-gate between phases per §Validation.

---

## P1 — A.1 hard wall-clock guard

### Motivation

`asyncio.wait_for(agent.run(...), timeout=1800)` under Python 3.11 does not abandon the inner task — it calls `task.cancel()` and then awaits task completion. When a sub-tool blocks in unshielded sync I/O (evidence: E2 task `023e9d44` ran 3298 s against 1800 s cap), the outer timeout is effectively uncapped. This makes E3 cost projection unreliable by up to 2×.

### Files affected

- **`examples/run_gaia.py`** — `answer_single_question()` timeout block (lines ~167–206)

No other file changes.

### Design

Replace the implicit-cancel `asyncio.wait_for` with explicit task creation + bounded cleanup:

```python
# Module-level constant
CLEANUP_GRACE_SECS = 30   # max time to wait for cancelled task to unwind

# Inside answer_single_question, replace lines ~174-206:
agent_task = asyncio.create_task(
    agent.run(task=augmented_question),
    name=f"run_{example['task_id']}",
)
try:
    final_result = await asyncio.wait_for(
        asyncio.shield(agent_task),
        timeout=per_question_timeout,
    )
except asyncio.TimeoutError:
    agent_task.cancel()
    try:
        await asyncio.wait_for(agent_task, timeout=CLEANUP_GRACE_SECS)
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
        # Abandon. Background task keeps running; relying on already-landed
        # per-call 120s HTTP timeout (fbd0dd1) + browser 15s cleanup
        # (aa78edc) to reap its state eventually.
        logger.error(
            f"Task {example['task_id']}: cancel() did not complete in "
            f"{CLEANUP_GRACE_SECS}s — abandoning. Proceeding with next task."
        )
    # Preserve the exception semantics the outer code expects:
    exception = asyncio.TimeoutError(
        f"Per-question timeout ({per_question_timeout}s) exceeded"
    )
    raised_exception = True
    output = None
    intermediate_steps = []
    iteration_limit_exceeded = True
    break
```

The `asyncio.shield` wrap is important: it makes the outer `wait_for`'s auto-cancel a no-op on the inner task, so we retain control of cancellation timing.

### Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Leaked background tasks** hold httpx connections, Playwright instances, MCP subprocesses | Medium | Already bounded by `fbd0dd1` (120 s LLM HTTP timeout) + `aa78edc` (15 s browser cleanup). Worst-case leak per question window = ~135 s, self-reaping. At concurrency=4 this is ≤4 leaked tasks at any time. Memory/FD impact negligible for the run length. |
| **Semaphore interaction** (with P2 later): if a leaked task never dies, does it hold a semaphore slot? | Low | No — semaphore released on `async with sem:` block exit, which happens when the cancellation path returns, not when the leaked task finally dies. Verified by semaphore semantics. |
| **CancelledError masking** — current code catches `Exception`, not `BaseException`; Cancel is a BaseException | Low | The `except (asyncio.TimeoutError, asyncio.CancelledError, Exception)` form explicitly catches CancelledError in the cleanup window, so no masking. Outside the timeout block, no change. |
| **`iteration_limit_exceeded=True`** on timeout might now differ from pre-change behaviour | Low (no behavioural change) | Field was already set to True on timeout in the old code (line 203). Preserved. |
| **Worst-case leak compounding** — if every question leaks, background tasks pile up | Low | Upper bound: `total_questions × 135 s`. For a 160-Q test cell that's 6h of tail background activity spread across the run, not a hard wall at the end. Monitor `ps`/`lsof` during smoke. |
| **Python version dependency** — `asyncio.TaskGroup` not used in P1, but `shield` needs a well-behaved cancel | None | `shield` is 3.8+; project is 3.11+. Safe. |

### Tests (NEW)

File: `tests/test_run_gaia_timeout.py`

```python
@pytest.mark.asyncio
async def test_hard_cap_on_cancel_ignoring_task():
    """Task that swallows CancelledError must not hold run_gaia past cleanup grace."""
    # Construct a fake agent.run that ignores cancel.
    async def ignore_cancel():
        while True:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
    # Patch create_agent to return an object whose .run = ignore_cancel
    # Assert: total wall time ≤ per_question_timeout + CLEANUP_GRACE_SECS + 1s slack
    ...

@pytest.mark.asyncio
async def test_happy_path_unchanged():
    """Fast task returns normally without timeout path activation."""
    ...

@pytest.mark.asyncio
async def test_cooperative_cancel_completes_promptly():
    """Task that honours cancel finishes well before CLEANUP_GRACE_SECS."""
    ...
```

### Validation gate

Run `bash scripts/launch_e2_freeze_smoke.sh` (3 Qs per model, 2 models, already-known timeouts in the sample). Expected:

- Mistral `023e9d44` + `e961a717`: agent_error = `"Per-question timeout (1800s) exceeded"`, end_time − start_time ≤ **1830 s** each (was ~3300 s).
- Qwen `023e9d44`: same pattern.
- `6f37996b` (known fast): unchanged, ≤ 400 s both models.

If any timeout row shows > 1830 s wall, P1 has a bug — rollback and investigate.

### Rollback

One commit. `git revert <hash>` restores the previous `asyncio.wait_for` block. No data migrations.

---

## P2 — Streaming worker pool

### Motivation

`examples/run_gaia.py:294-298` runs questions in batches of `concurrency` via `asyncio.gather`. `gather` waits for the slowest in each batch before starting the next, so a single 1800 s stuck question blocks three peers for ≤1800 s of scheduling latency — pure throughput loss with no behavioural signal.

### Files affected

- **`examples/run_gaia.py`** — lines 293–298 and the async `main()` body

### Design

Replace batch-gather with semaphore-bounded TaskGroup:

```python
# replaces lines 293-298
concurrency = max(1, int(getattr(config, "concurrency", 4)))
sem = asyncio.Semaphore(concurrency)

async def _bounded(task):
    async with sem:
        try:
            return await answer_single_question(config, task)
        except Exception as e:
            # CRITICAL: catch Exception (NOT BaseException) so CancelledError
            # still propagates for Ctrl-C; but wrap here so a worker crash
            # does NOT cancel siblings via TaskGroup.
            logger.exception(
                f"worker crashed on task {task.get('task_id','?')}: {e}"
            )
            return None

async with asyncio.TaskGroup() as tg:
    for task in tasks_to_run:
        tg.create_task(_bounded(task))

logger.info("| All tasks complete (streaming worker pool).")
```

### Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **TaskGroup cancels siblings on first unhandled exception** | Medium (catastrophic if hit) | `_bounded` catches `Exception` explicitly. `answer_single_question` already handles its own exceptions internally and writes a jsonl row. The only way to escape `_bounded` is `BaseException` (KeyboardInterrupt, SystemExit, CancelledError) — those SHOULD cancel siblings (user intent). Added guard rail: assertion in tests that a known-throwing mock does not cancel the other workers. |
| **Concurrent `append_answer` writes** | None | `append_answer` already uses `threading.Lock()` for file append (run_gaia.py:26,48). Safe under both threads and asyncio tasks. |
| **Loss of batch-completion log lines** (current code logs `"Batch N done"`) | Very low | Replace with a per-task completion log inside `_bounded`, or a periodic progress line. Not needed for correctness. |
| **Progress monitoring relies on batch boundaries** (e.g. `scripts/monitor_tick.py`) | Low | Check — `monitor_tick.py` reads `dra.jsonl` row count, not batch counts. Grep confirms. |
| **All tasks fail silently** (every question hits worker crash) | Low | Fail-safe: if `dra.jsonl` has no rows after all tasks complete, log ERROR and exit non-zero. Add to the post-TaskGroup block. |
| **Python version** (`TaskGroup` needs 3.11+) | None | `pyproject.toml` pins `>=3.11,<4.0`. Safe. |
| **Semaphore counter vs batch sizing** | None | `async with sem:` slot is held during `answer_single_question`; released on exit (normal or exception). Identical semantics to gather-batch except no barrier between batches. |
| **Semaphore with 0 or negative `concurrency`** | Low | `max(1, int(...))` guard. If config sets 0, treat as 1. |

### Tests (NEW)

File: `tests/test_run_gaia_worker_pool.py`

```python
@pytest.mark.asyncio
async def test_concurrency_cap_respected():
    """At no instant do more than `concurrency` workers run simultaneously."""
    # Instrument _bounded with a shared counter; assert peak ≤ concurrency.
    ...

@pytest.mark.asyncio
async def test_all_tasks_complete_no_drops():
    """All N tasks run once; none skipped."""
    ...

@pytest.mark.asyncio
async def test_worker_exception_isolated():
    """A worker that raises does NOT cancel siblings."""
    # Mix a raising task with normal tasks; assert normal ones still complete.
    ...

@pytest.mark.asyncio
async def test_straggler_does_not_block_peers():
    """A slow task does not delay the start of queued tasks."""
    # 4 tasks, concurrency=2: 1 slow (5s), 3 fast (0.1s each).
    # Assert total wall ≤ 5s + small ε (would be ~5s with gather-batch of 2:
    # batch 1 = max(5, 0.1) = 5s; batch 2 = 0.1s; total = 5.1s.
    # With streaming pool: fast 3 finish in ~0.3s while slow still running,
    # total = 5s exactly).
    ...
```

### Validation gate

Same E2 smoke launcher. Expected:

- Total wall clock to finish 6 Qs (3 per model, parallel) ≤ the old wall clock. With the current 1/3 timeout rate, gather-batch-of-3 wall = max-duration = timeout wall. With streaming pool, first correct answer lands as soon as the fast one finishes; timeout row runs in parallel.
- No regression: all 6 rows appear in jsonl.

### Rollback

One commit. Revertable. `monitor_tick.py` and other analysis scripts rely on jsonl only, not on log lines, so analysis continuity is preserved.

---

## P3 — Qwen concurrency 4 → 8

### Motivation

E0 evidence: Qwen made 3,006 LLM calls at concurrency=4 with **zero** 429s, **zero** rate-limit warnings, **zero** `Retry-After` headers, and all traffic routing through OpenRouter → Alibaba. Headroom is real. Doubling per-cell concurrency is the cheapest throughput gain in the plan.

### Files affected

- **`scripts/gen_eval_configs.py`** — model list, adds a `concurrency` override per model slot (only Qwen gets 8; others stay at 4)
- **`configs/config_gaia_c{0,2,3,4}_qwen.py`** — regenerated, each `concurrency = 4` → `concurrency = 8`
- (Optional) **`scripts/run_eval_matrix.sh`** — add a `QWEN_CONCURRENCY` envvar override path mirroring the existing `GEMMA_CONCURRENCY`, for rollback without regenerating configs

No code changes outside config generation.

### Design

Per-model concurrency override in the generator template:

```python
# scripts/gen_eval_configs.py — in MODELS tuples, add an optional `concurrency`:
MODELS = [
    ("mistral", "mistral-small", "langchain-mistral-small", <...>, 4),
    ("kimi",    "or-kimi-k2.5",   <...>,                       4),
    ("qwen",    "or-qwen3.6-plus", "langchain-or-qwen3.6-plus", <...>, 8),
    ("gemma",   "or-gemma-4-31b-it", <...>,                     4),
]
```

Template body emits `concurrency = <N>` derived from the tuple. Generator signature change is backward-compatible if 4 is the default when absent.

Then: `python scripts/gen_eval_configs.py` to regenerate the 4 Qwen configs.

### Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Hidden OR rate limits surface at 8** | Low (evidence says no) | Smoke-gate with a 5-Q Qwen run, grep log for `429` / `Retry-After` / `RateLimitError`. If non-zero, revert to 4. |
| **Sub-provider (Alibaba) capacity** | Low | OR routed 3,006 calls to Alibaba during E0 without any rate-limit signal. Doubling the in-flight count to ≤8 is still within their per-account concurrency. |
| **Memory pressure** (2× browser instances if every Q hits browser_use_agent) | Low | Current memory peak during E0 Qwen cell is comfortable. 2× is still within farm/Mac limits. Add memory check in smoke-gate assertion. |
| **Cost spike** | Very low (linear) | Throughput improvement translates to faster wall, same token cost. No cost-per-token change. |
| **Kimi / Gemma configs unintentionally bumped** | None | Explicit per-model tuple; only Qwen sees 8. |

### Tests

Config-only change; covered by existing `tests/smoke_validate_handoffs_234.sh` (loads all configs, verifies defaults). Add one assertion:

```python
# tests/test_gen_eval_configs.py (NEW, short)
def test_qwen_concurrency_is_8():
    for cond in ["c0","c2","c3","c4"]:
        cfg = Config.fromfile(f"configs/config_gaia_{cond}_qwen.py")
        assert cfg.concurrency == 8, f"{cond}/qwen concurrency should be 8"
def test_non_qwen_concurrency_unchanged():
    for m in ["mistral","kimi","gemma"]:
        for cond in ["c0","c2","c3","c4"]:
            cfg = Config.fromfile(f"configs/config_gaia_{cond}_{m}.py")
            assert cfg.concurrency == 4, f"{cond}/{m} concurrency drifted"
```

### Validation gate

Run `bash scripts/run_eval_matrix.sh smoke qwen c0` (3 Qs at concurrency=8). Expected:

- Zero 429 / `Retry-After` / `RateLimitError` in the log.
- Wall clock ≤ 70% of a concurrency=4 baseline on the same 3 Qs.
- `dra.jsonl` has 3 rows (no drops from semaphore underflow).

### Rollback

Regenerate with the old tuple OR set `concurrency=4` via `--cfg-options concurrency=4` at launch (no regen needed thanks to mmengine overrides). Trivial.

---

## P4 — Mistral multi-key round-robin

### Motivation

Mistral E0 evidence: 9,286 `Chat completion timed out after 120s` + 7 explicit 429s across the run, against a 5 req/min/key free-tier limit. A second key (already available per operator) doubles the effective ceiling. Both the native `OpenAIServerModel` path AND the LangChain `ChatOpenAI` wrapper (used by `auto_browser_use_tool`) must rotate — otherwise browser calls bypass the rotation and consume the shared window.

### Files affected

- **`src/models/openaillm.py`** — new class `KeyRotatingOpenAIServerModel` subclassing `OpenAIServerModel`, wrapping N `openai.AsyncOpenAI` clients with a round-robin picker + per-key cooldown on 429.
- **`src/models/models.py`** — `_register_mistral_models()` detects list vs string for `api_key`, constructs rotating variants for both native and LangChain wrappers. For LangChain: a thin `KeyRotatingChatOpenAI` shim that dispatches `invoke` / `ainvoke` / `stream` across N `ChatOpenAI` instances.
- **`.env.template`** — document `MISTRAL_API_KEY_2` (placeholder).
- **`tests/test_key_rotation.py`** (NEW) — round-robin distribution, cooldown on 429, fall-through when all cooling, degrades cleanly to single-key behavior when list length = 1.

### Design — native path

New class skeleton:

```python
# src/models/openaillm.py

import itertools
import time
from dataclasses import dataclass, field

@dataclass
class _KeyState:
    client: "AsyncOpenAI"
    cool_until: float = 0.0  # epoch seconds; 0 = ready

class KeyRotatingOpenAIServerModel(OpenAIServerModel):
    """
    Drop-in replacement for OpenAIServerModel that rotates across N API keys.
    Round-robin; per-key cooldown on 429 (uses Retry-After header when present,
    else DEFAULT_COOLDOWN_SECS). Falls through to any ready key if the next
    rotation pick is cooling. If all keys cooling, picks the earliest-ready one.
    """
    DEFAULT_COOLDOWN_SECS: float = 13.0   # Mistral free tier: 5/min → 12s window + 1s slack

    def __init__(self, model_id: str, api_keys: list[str], api_base: str, **kwargs):
        super().__init__(model_id=model_id, api_key=api_keys[0], api_base=api_base, **kwargs)
        self._keys = [_KeyState(AsyncOpenAI(api_key=k, base_url=api_base)) for k in api_keys]
        self._cycle = itertools.cycle(range(len(self._keys)))

    def _pick(self) -> _KeyState:
        now = time.monotonic()
        n = len(self._keys)
        # Try up to n rotations for a ready key.
        for _ in range(n):
            idx = next(self._cycle)
            if self._keys[idx].cool_until <= now:
                return self._keys[idx]
        # All cooling — return earliest-ready.
        return min(self._keys, key=lambda k: k.cool_until)

    def _mark_429(self, state: _KeyState, retry_after_secs: float | None) -> None:
        cooldown = retry_after_secs or self.DEFAULT_COOLDOWN_SECS
        state.cool_until = time.monotonic() + cooldown

    # override the HTTP call path — specific hook depends on OpenAIServerModel
    # internals; likely `async def _create_chat_completion` or similar
```

Integration in `_register_mistral_models`:

```python
def _register_mistral_models(self):
    api_keys = [
        k for k in (
            os.getenv("MISTRAL_API_KEY"),
            os.getenv("MISTRAL_API_KEY_2"),
        ) if k and k != PLACEHOLDER
    ]
    if not api_keys:
        logger.warning("No MISTRAL_API_KEY* set, skipping Mistral models")
        return
    api_base = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
    logger.info(f"Registering Mistral models ({len(api_keys)} key{'s' if len(api_keys)>1 else ''})")

    for m in [{"model_name": "mistral-small", "model_id": "mistral-small-2603"}, ...]:
        model_name, model_id = m["model_name"], m["model_id"]
        if len(api_keys) > 1:
            registered = KeyRotatingOpenAIServerModel(
                model_id=model_id, api_keys=api_keys, api_base=api_base,
                custom_role_conversions=custom_role_conversions,
            )
            langchain_model = KeyRotatingChatOpenAI(
                model=model_id, api_keys=api_keys, base_url=api_base,
            )
        else:
            # existing single-key path unchanged
            ...
        self.registed_models[model_name] = registered
        self.registed_models[f"langchain-{model_name}"] = langchain_model
```

### Design — LangChain shim

LangChain's `ChatOpenAI` takes one API key at construction. The shim wraps N instances and dispatches by rotation:

```python
# src/models/openaillm.py

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

class KeyRotatingChatOpenAI:
    """
    Minimal shim presenting the ChatOpenAI interface used by auto_browser_use_tool.
    Round-robins `invoke` / `ainvoke` / `stream` / `astream` across N ChatOpenAI
    instances (one per key).

    Not a subclass of BaseChatModel — we duck-type exactly the methods the
    browser tool calls. If the tool's interface expands, add methods here;
    fail-loudly via __getattr__ if an unexpected attribute is accessed.
    """
    def __init__(self, model, api_keys, base_url, **kwargs):
        self._clients = [
            ChatOpenAI(model=model, api_key=k, base_url=base_url, **kwargs)
            for k in api_keys
        ]
        self._cycle = itertools.cycle(range(len(self._clients)))
        self._cool_until = [0.0] * len(self._clients)

    def _pick(self):
        now = time.monotonic()
        n = len(self._clients)
        for _ in range(n):
            idx = next(self._cycle)
            if self._cool_until[idx] <= now:
                return idx, self._clients[idx]
        idx = min(range(n), key=lambda i: self._cool_until[i])
        return idx, self._clients[idx]

    async def ainvoke(self, *a, **kw):
        idx, c = self._pick()
        try:
            return await c.ainvoke(*a, **kw)
        except Exception as e:
            if _is_rate_limit(e):
                self._cool_until[idx] = time.monotonic() + _cooldown_from_error(e)
                # One retry on the next key; if that also 429s, surface.
                idx2, c2 = self._pick()
                if idx2 != idx:
                    return await c2.ainvoke(*a, **kw)
            raise

    # Same for invoke, stream, astream.
```

The `_is_rate_limit` / `_cooldown_from_error` helpers inspect the exception for a 429 status and a Retry-After header.

### Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **LangChain wrapper rotation is duck-typed** — if `auto_browser_use_tool` uses a method not on the shim | Medium | Grep all call sites first. Add a `__getattr__` that raises with a specific "missing method X; add to KeyRotatingChatOpenAI" error so the failure is loud. Integration test: 5-Q browser-heavy smoke exercises the wrapper. |
| **429 detection wrong** — not catching the right exception shape | Medium | Test with a mock that raises `openai.RateLimitError` and assert the cool-until mutation. Also test an arbitrary `Exception` does NOT trigger cooldown. |
| **Retry-After parsing** | Low | Use header if present (`e.response.headers.get("retry-after")`). Fall back to `DEFAULT_COOLDOWN_SECS = 13.0` (Mistral: 5/min → 12s window + 1s slack). |
| **Thundering herd if all keys cool at once** | Low | `_pick()` returns earliest-ready. Callers still wait for the short cooldown (≤13 s) instead of 60 s+. |
| **Secret leakage into logs** | Medium | Never log `api_keys` directly. Log only `len(api_keys)` and redact-by-default. Audit existing `OpenAIServerModel.__repr__` for key exposure — currently none, confirm. |
| **Backward compat when only 1 key** | Low | Explicit `if len(api_keys) > 1:` branch; single-key path unchanged. Tests: set only `MISTRAL_API_KEY`, assert registration matches current behaviour exactly. |
| **itertools.cycle + asyncio** — two coroutines next() simultaneously | Very low | `itertools.cycle.__next__` is a C-level atomic in CPython. Under asyncio (single-threaded event loop), no race regardless. |
| **Mistral API may reject some requests for spec reasons, not rate limit** | Low | `_is_rate_limit` is strict — only 429 + `rate_limited` / `Rate limit exceeded` messages trigger cooldown. Non-rate-limit errors propagate unchanged. |
| **Mock in tests drifts from real `openai.RateLimitError` shape** | Low | Pin to actual `openai` version in pyproject; test fixture uses the real exception class. |
| **Registration order** — `_register_mistral_models()` runs before other models; shim import must not cycle | Low | Import `KeyRotatingChatOpenAI` at module top of `models.py`. No circular risk. |

### Tests (NEW)

File: `tests/test_key_rotation.py`

```python
def test_round_robin_distribution():
    """10 picks with 2 keys → 5/5 distribution."""
    m = KeyRotatingOpenAIServerModel(model_id="test", api_keys=["a","b"], api_base="http://x")
    picks = [m._pick() for _ in range(10)]
    assert sum(1 for p in picks if p is m._keys[0]) == 5

def test_429_sets_cooldown():
    ...

def test_cooldown_respects_retry_after():
    ...

def test_all_cooling_returns_earliest_ready():
    ...

def test_single_key_degrades_to_no_rotation():
    """With only MISTRAL_API_KEY set, plain OpenAIServerModel is registered, not KeyRotatingOpenAIServerModel."""
    ...

def test_langchain_shim_covers_all_auto_browser_methods():
    """Every method `auto_browser_use_tool` calls on the LangChain wrapper is defined on KeyRotatingChatOpenAI."""
    # Introspect auto_browser.py for attribute access on the langchain model;
    # assert each is on the shim.
    ...
```

### Validation gate

1. **Unit:** 15 new tests pass.
2. **Integration:** `scripts/integration_test_model_stack.py` with `MISTRAL_API_KEY` + `MISTRAL_API_KEY_2` set — 10-request burst should split roughly 5/5.
3. **Smoke:** `bash scripts/launch_e2_freeze_smoke.sh` — Mistral 120s-timeout counts should drop meaningfully (from current baseline). Not expected to go to zero (Mistral's SSE streaming is flaky independent of queueing).

Capture before/after metrics for the paper: `grep -c "Chat completion timed out after 120s"` in log.txt before and after P4.

### Rollback

- Single-key fallback is automatic: unset `MISTRAL_API_KEY_2` and restart — registration falls through to the single-client path without code changes.
- Full revert: one commit per file group (`src/models/*`, `tests/*`, `.env.template`); each revertable independently.

---

## Cross-cutting concerns

### Shared risks across all phases

| Concern | Phase(s) | Mitigation |
|---------|----------|-----------|
| **Unrelated test regressions** from async changes | P1, P2 | Run the full 140-test sweep (`scripts/run_handoff_pytest_sweep.sh`) after each phase; block merge on any regression. |
| **`monitor_tick.py` / `analyze_results.py` / `validate_handoffs.sh` brittleness** — they read jsonl + log files | P1, P2, P3 | No format changes to jsonl or log lines. Verify greps still match after P1 (timeout warning text unchanged). |
| **Handoff doc drift** | All | Update `HANDOFF_TEST_EVAL.md` cost/time estimates after P1+P2 land (the 8–24h range shrinks). One doc-only commit at the end. |
| **Config regeneration drift** (P3) | P3 | `gen_eval_configs.py` is the single source of truth; CI check (`scripts/smoke_validate_handoffs_234.sh`) already verifies all 16 configs import cleanly. |
| **Commit granularity** | All | One feature commit + one docs/index commit per phase. Four phases = 8 commits. Each phase independently revertable. |
| **Secret exposure** in env / logs | P4 | `.env.template` uses `MISTRAL_API_KEY_2=<your-second-key-here>` placeholder. No real keys committed. Log messages say `len(api_keys)`, never `api_keys[i][-4:]` tail-reveals. |
| **Auto mode inadvertent pushes** | All | Do not push intermediate phases. Push only after P4 + smoke validation, as one operation. |

### Documentation updates (one doc commit at the end)

- `CLAUDE.md` — env var list gains `MISTRAL_API_KEY_2` (optional); concurrency note for Qwen.
- `docs/handoffs/HANDOFF_TEST_EVAL.md` — update E3 cost/time estimates; note the hard wall-clock cap.
- `docs/handoffs/HANDOFF_E1_E2_RESULTS.md` — cross-reference: "F1 addressed by P1 (4d6c2e6+...)"; "F3 deliberately not fixed (§Methodology caveat)".
- `HANDOFF_INDEX.md` — one follow-up row per phase commit.
- This file (`HANDOFF_THROUGHPUT_REFACTOR.md`) — after each phase lands, fill in the "executed" section with measured deltas.

### Measurement contract (for the paper)

Capture before/after for each phase on the same smoke (`launch_e2_freeze_smoke.sh` with 6 Qs: 3 fixed seed=42 + 3 pseudo-random from index ≥3 with seed=42). Persist under `workdir/throughput_refactor_benchmarks/<phase>/`. Report in §Methodology "Throughput engineering" subsection:

- wall clock per cell
- `Chat completion timed out after 120s` count
- 429 count
- peak concurrent Python subprocess / memory
- timeout overshoot (`end_time − start_time` for rows with `agent_error` matching timeout)

---

## Execution checklist (run in strict order)

- [ ] **P1** land + unit tests green + E2 smoke validates ≤1830 s per timeout → commit + HANDOFF_INDEX row
- [ ] **P2** land + unit tests green + E2 smoke validates wall-time drop → commit + index row
- [ ] **P3** regenerate + config tests green + Qwen smoke validates zero 429s at c=8 → commit + index row
- [ ] **P4** land + 15 new tests green + integration probe 5/5 split + E2 smoke validates ≥30% drop in Mistral 120s-timeouts → commit + index row
- [ ] Docs sweep: CLAUDE.md, TEST_EVAL, E1_E2 cross-references, this file's "executed" section
- [ ] Single push to `origin/main` covering all phases
- [ ] Re-run full E2 with all four phases active, attach metrics table to this handoff

## Abort conditions (roll back immediately)

- Any phase causes an existing test to regress.
- P1: any smoke timeout row shows > 1830 s wall.
- P2: any task from the smoke is missing from the jsonl (dropped by TaskGroup).
- P3: any 429 appears in the Qwen smoke log at concurrency=8.
- P4: Mistral smoke raises a new class of error not seen at concurrency=4/single-key.

---

## Execution log (2026-04-22)

### P1 — landed `5aa1467` (+ index row `803a7ab`)

Files: `examples/run_gaia.py`, `tests/conftest.py` (new), `tests/test_run_gaia_timeout.py` (new).

Implementation variance from the plan: during test-driven iteration I discovered that the first cleanup wait I wrote (`asyncio.wait_for(agent_task, timeout=CLEANUP_GRACE_SECS)`) re-introduced the exact pathology it was guarding against — Python 3.11's `wait_for` on timeout cancels the inner and awaits its completion, which hangs for a task swallowing `CancelledError`. Fix: add a second `asyncio.shield(agent_task)` around the inner wait too so the cleanup wait is strictly bounded. Without it the `test_hard_cap_on_cancel_ignoring_task` test ran 10 s against a 4 s expected upper bound. With the shield it passes in ~3 s. Documented inline in the `except asyncio.TimeoutError:` block.

All 3 P1 tests pass in ~6.3 s. 140-test sweep green.

### P2 — landed `89f3bf0` (+ index row `b52bbb9`)

Files: `examples/run_gaia.py`, `tests/test_run_gaia_worker_pool.py` (new).

No design variance. 4 new tests pass in ~1 s (slow test gated at ~0.8 s wall for the straggler scenario). 140-test sweep green.

### P3 — landed `1b9f82c` (+ index row `653fbb9`)

Files: `scripts/gen_eval_configs.py`, `configs/config_gaia_c{0,2,3,4}_{mistral,kimi,qwen,gemma}.py` (all 16 regenerated), `tests/test_gen_eval_configs_concurrency.py` (new).

Diff against the pre-P3 tracked state: 4 Qwen configs now emit `concurrency = 8`; non-Qwen configs show only a comment refresh (regenerator's timeout-pin paragraph wording slightly newer) — behavioural values unchanged for Mistral / Kimi / Gemma. 17 new tests (16 parametrised `condition × model` + 1 defensive on the MODELS tuple shape) pass in <0.1 s.

### P4 — landed `d4197fd` (+ index row `5d85155`)

Files: `src/models/openaillm.py`, `src/models/models.py`, `.env.template`, `tests/test_key_rotation.py` (new).

Implementation variance:
- `_load_suffix_keys` scans contiguously (stops at first unset). The pre-reg doc had been ambiguous between contiguous-only and scan-all; I chose contiguous because a sparse `_4` with `_3` unset is more likely an operator mistake than intent.
- LangChain wrapper is a pure delegator (not `ChatOpenAI` subclass) — Pydantic fields on `ChatOpenAI` make attribute overrides brittle. `__getattr__` fallback to `_instances[0]` covers `bind_tools` etc. with a documented caveat: returned bound-tool instances are locked to instance 0 and don't rotate. Acceptable because browser-use binds once per agent construction, not per question.
- Tests use real `httpx.Response` + `openai.RateLimitError` (not `SimpleNamespace`) because the SDK constructor inspects `response.request`.

19 new tests pass in ~7 s; all 43 new tests across P1-P4 pass together in ~12.6 s. 140-test handoff sweep unchanged.

### Combined smoke (pending)

Deferred to the operator's next live pass: run `bash scripts/launch_e2_freeze_smoke.sh` with both `MISTRAL_API_KEY` and `MISTRAL_API_KEY_2` set; capture the measurements contract described in "Cross-cutting concerns → Measurement contract". Expected signals:

- **P1:** every timeout row `end_time − start_time ≤ 1830 s` (was 3273–3308 s).
- **P2:** total wall ≤ `max(per-Q wall)` for the 6-Q sample (vs. sum-of-batch-maxes before).
- **P3:** Qwen at c=8 finishes the 3 fixed Qs in ≤70 % of the c=4 baseline; zero 429 in the Qwen log.
- **P4:** Mistral `Chat completion timed out after 120s` count drops ≥30 % from pre-P4 baseline; no new error class.

The operator should append the measured numbers to this section under a new "Combined smoke (executed)" subheading after the run.
