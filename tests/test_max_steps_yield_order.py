"""Regression test for the stale-action-step duplicate-yield bug.

Before the fix, `AsyncMultiStepAgent._run_stream` did this after the max-steps
path:

    if not returned_final_answer and self.step_number == max_steps + 1:
        final_answer = await self._handle_max_steps_reached(task, images)
        yield action_step            # ← stale loop step
    yield FinalAnswerStep(...)

`_handle_max_steps_reached` already appended its own terminal `ActionStep`
to memory, so re-yielding the loop's stale `action_step` misled streaming
consumers about the final state of the agent.

The fix changes `_handle_max_steps_reached` to return the fresh terminal
`ActionStep` (instead of just the content string), and `_run_stream` yields
that fresh step and pulls `action_output` from it. This test guards against
regressions by checking the source code of both sites.

A full runtime test would require instantiating `AsyncMultiStepAgent` with
a real model, memory, prompt templates, etc. That is covered by the
end-to-end GAIA eval; this file catches the specific regression without
the heavy setup.
"""

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASYNC_AGENT = ROOT / "src/base/async_multistep_agent.py"
SYNC_AGENT = ROOT / "src/base/multistep_agent.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _find_async_method(module: ast.Module, class_name: str, method_name: str) -> ast.AsyncFunctionDef:
    for cls in (n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name):
        for stmt in cls.body:
            if isinstance(stmt, ast.AsyncFunctionDef) and stmt.name == method_name:
                return stmt
    raise AssertionError(f"async method {class_name}.{method_name} not found in {module}")


def _find_sync_method(module: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for cls in (n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name):
        for stmt in cls.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == method_name:
                return stmt
    raise AssertionError(f"method {class_name}.{method_name} not found in {module}")


# ---------------------------------------------------------------------------
# _handle_max_steps_reached now returns the ActionStep, not its .content string.
# ---------------------------------------------------------------------------


def test_async_handle_max_steps_reached_returns_action_step_annotation():
    mod = _parse(ASYNC_AGENT)
    method = _find_async_method(mod, "AsyncMultiStepAgent", "_handle_max_steps_reached")
    assert method.returns is not None, "return annotation missing"
    # Accept `ActionStep` or `"ActionStep"` (forward ref)
    ann = ast.unparse(method.returns)
    assert "ActionStep" in ann, f"expected ActionStep return annotation, got {ann!r}"


def test_async_handle_max_steps_reached_returns_final_memory_step():
    """Body must end with `return final_memory_step`, not `return final_answer.content`."""
    mod = _parse(ASYNC_AGENT)
    method = _find_async_method(mod, "AsyncMultiStepAgent", "_handle_max_steps_reached")
    returns = [n for n in ast.walk(method) if isinstance(n, ast.Return)]
    assert len(returns) >= 1
    final_return = returns[-1]
    assert isinstance(final_return.value, ast.Name), (
        f"expected `return final_memory_step`, got {ast.unparse(final_return)!r}"
    )
    assert final_return.value.id == "final_memory_step", (
        f"expected `return final_memory_step`, got `return {final_return.value.id}`"
    )


def test_sync_handle_max_steps_reached_returns_action_step_annotation():
    """Same invariant for the sync version of the base class."""
    mod = _parse(SYNC_AGENT)
    method = _find_sync_method(mod, "MultiStepAgent", "_handle_max_steps_reached")
    assert method.returns is not None
    ann = ast.unparse(method.returns)
    assert "ActionStep" in ann, f"expected ActionStep return annotation, got {ann!r}"


def test_sync_handle_max_steps_reached_returns_final_memory_step():
    mod = _parse(SYNC_AGENT)
    method = _find_sync_method(mod, "MultiStepAgent", "_handle_max_steps_reached")
    returns = [n for n in ast.walk(method) if isinstance(n, ast.Return)]
    assert len(returns) >= 1
    final_return = returns[-1]
    assert isinstance(final_return.value, ast.Name), ast.unparse(final_return)
    assert final_return.value.id == "final_memory_step"


# ---------------------------------------------------------------------------
# _run_stream yields the FRESH memory step, not the stale loop action_step.
# ---------------------------------------------------------------------------


def _run_stream_yield_targets(method: ast.AST) -> list[str]:
    """Collect the names yielded in the max-steps branch — the `if not
    returned_final_answer and self.step_number == max_steps + 1:` block.

    Returns the list of yielded expression source strings.
    """
    for node in ast.walk(method):
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test)
            if "returned_final_answer" in test_src and "max_steps" in test_src:
                return [
                    ast.unparse(stmt.value.value) if isinstance(stmt.value.value, ast.AST) else "?"
                    for stmt in node.body
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Yield, ast.YieldFrom))
                ]
    raise AssertionError("max-steps branch not found in _run_stream")


def test_async_run_stream_yields_final_memory_step_not_stale_action_step():
    mod = _parse(ASYNC_AGENT)
    method = _find_async_method(mod, "AsyncMultiStepAgent", "_run_stream")
    yields = _run_stream_yield_targets(method)
    assert "action_step" not in yields, (
        "_run_stream still yields the stale `action_step` in the max-steps branch. "
        "Regression of the duplicate-yield bug — see Pass 1.5 in the plan."
    )
    assert any("final_memory_step" in y for y in yields), (
        f"expected `yield final_memory_step` in max-steps branch, got {yields}"
    )


def test_sync_run_stream_yields_final_memory_step_not_stale_action_step():
    mod = _parse(SYNC_AGENT)
    method = _find_sync_method(mod, "MultiStepAgent", "_run_stream")
    yields = _run_stream_yield_targets(method)
    assert "action_step" not in yields
    assert any("final_memory_step" in y for y in yields)
