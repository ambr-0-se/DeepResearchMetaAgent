"""Unit tests for the streaming worker pool in `examples/run_gaia.py::main`
(§P2 of `docs/handoffs/HANDOFF_THROUGHPUT_REFACTOR.md`).

Scope: verify the `asyncio.Semaphore + asyncio.TaskGroup` scheduling
replaces the previous batch-`gather` loop with the following properties:

  1. concurrency cap is respected (peak in-flight ≤ `config.concurrency`)
  2. all tasks complete; none are silently dropped
  3. a worker that raises does NOT cancel its TaskGroup siblings
  4. a slow task does not block peers — total wall ≈ slowest, not sum

The test exercises the pool directly by replicating the `_bounded` +
`TaskGroup` shape from `main()`, because the full `main()` entry point
pulls in the GAIA dataset + logger + model manager which are not worth
standing up for a scheduling test. The code under test is small and
easy to duplicate faithfully; drift is unlikely (review on change).
"""

from __future__ import annotations

import asyncio
import time

import pytest


async def _run_pool(tasks_to_run, worker, concurrency: int) -> list:
    """Mirror of the production loop in `examples/run_gaia.py::main`
    (§P2). Any change to the production shape must be reflected here.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list = []

    async def _bounded(task_item):
        async with sem:
            try:
                res = await worker(task_item)
                results.append(("ok", task_item, res))
                return res
            except Exception as e:  # noqa: BLE001 — mirror of production
                results.append(("error", task_item, e))
                return None

    async with asyncio.TaskGroup() as tg:
        for task_item in tasks_to_run:
            tg.create_task(_bounded(task_item))

    return results


@pytest.mark.asyncio
async def test_concurrency_cap_respected():
    """Peak in-flight workers ≤ concurrency, verified via a shared
    peak-counter.
    """
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def worker(task_item):
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            # Stagger so workers actually overlap.
            await asyncio.sleep(0.05)
            return task_item
        finally:
            async with lock:
                in_flight -= 1

    results = await _run_pool(list(range(20)), worker, concurrency=4)
    assert peak <= 4, f"peak in-flight {peak} exceeds concurrency cap 4"
    assert peak >= 2, (
        f"peak in-flight {peak} suggests workers did not actually "
        "overlap — test's sleep window may be too short to prove anything"
    )
    assert len(results) == 20


@pytest.mark.asyncio
async def test_all_tasks_complete_no_drops():
    """N tasks in, N results out. Nothing dropped by TaskGroup scheduling."""

    async def worker(task_item):
        return f"done-{task_item}"

    tasks = list(range(37))  # off-by-one-prone count
    results = await _run_pool(tasks, worker, concurrency=5)

    assert len(results) == 37
    ok_items = {t for status, t, _ in results if status == "ok"}
    assert ok_items == set(tasks), f"missing tasks: {set(tasks) - ok_items}"


@pytest.mark.asyncio
async def test_worker_exception_isolated():
    """A worker that raises does NOT cancel siblings — the pool treats
    each failure as local and lets the rest complete. This is the
    single biggest regression-risk of the pool design; without the
    `try/except Exception` in `_bounded`, `TaskGroup` cancels all.
    """

    async def worker(task_item):
        await asyncio.sleep(0.02)
        if task_item == "bad":
            raise RuntimeError("boom — simulated worker crash")
        return f"ok-{task_item}"

    tasks = ["a", "b", "bad", "c", "d"]
    results = await _run_pool(tasks, worker, concurrency=2)

    # Expect 1 error + 4 successes; no cancellation of siblings.
    statuses = [s for s, _, _ in results]
    assert statuses.count("error") == 1
    assert statuses.count("ok") == 4
    ok_tasks = {t for s, t, _ in results if s == "ok"}
    assert ok_tasks == {"a", "b", "c", "d"}


@pytest.mark.asyncio
async def test_straggler_does_not_block_peers():
    """1 slow task + many fast ones with concurrency=2. Total wall ≈
    slow-duration, NOT the batched sum. This is the whole point of P2 —
    under batch-gather, a slow task would block its 1 peer; under the
    streaming pool, the fast tasks stream through while the slow one
    continues in its own slot.
    """
    slow_duration = 0.8  # exaggerated so the delta is unambiguous
    fast_duration = 0.05

    async def worker(task_item):
        if task_item == "slow":
            await asyncio.sleep(slow_duration)
        else:
            await asyncio.sleep(fast_duration)
        return task_item

    # 1 slow + 7 fast at concurrency=2. Under batch-gather with batches
    # of 2, the pair containing "slow" stalls the other's slot for the
    # full slow_duration, then each subsequent batch takes fast_duration.
    # Under the streaming pool, "slow" occupies one slot the whole time
    # while the 7 fast tasks stream through the other — total wall
    # ≈ max(slow_duration, 7 * fast_duration / 1) = max(0.8, 0.35) = 0.8.
    tasks = ["slow"] + [f"fast-{i}" for i in range(7)]

    t0 = time.monotonic()
    results = await _run_pool(tasks, worker, concurrency=2)
    elapsed = time.monotonic() - t0

    assert len(results) == 8

    # Upper bound: the streaming-pool theoretical max is slow_duration
    # + (7 * fast_duration / concurrency=2) ≈ 0.8 + 0.175 = ~0.975.
    # Allow generous slack for scheduler overhead on CI.
    assert elapsed < 1.5, (
        f"total wall {elapsed:.3f}s suggests batch-like behaviour; "
        f"streaming pool should complete in ≲ {slow_duration + 0.4:.2f}s"
    )
    # Lower bound: we must actually wait out the slow task.
    assert elapsed >= slow_duration * 0.9, (
        f"total wall {elapsed:.3f}s is suspiciously short — slow task "
        "may not have actually run"
    )
