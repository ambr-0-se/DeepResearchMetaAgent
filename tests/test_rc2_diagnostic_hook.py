"""Regression test for the Pass 3.1 RC2 diagnostic hook.

Without this hook, `execute_tool_call` in `general_agent.py` wraps every
sub-agent exception in `AgentToolExecutionError(str(e))`, which reduces a
Python `UnboundLocalError: cannot access local variable 'final_answer'`
(RC2) to a one-line string with no traceback and no `__cause__` /
`__context__` chain. That's what made RC2 impossible to trace in the
original 301-task GAIA eval.

The hook fires only when the inner exception is `NameError` /
`UnboundLocalError` and logs the full exception chain before the wrap.
This is diagnostics-only: the hook does NOT change error propagation.

This test verifies the hook is present in the source (AST-level) so a
future edit can't silently remove it.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "src/agent/general_agent/general_agent.py"


def _execute_tool_call_except_handler() -> ast.ExceptHandler:
    module = ast.parse(AGENT.read_text())
    for cls in (n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "GeneralAgent"):
        for method in cls.body:
            if isinstance(method, ast.AsyncFunctionDef) and method.name == "execute_tool_call":
                for node in ast.walk(method):
                    if isinstance(node, ast.Try):
                        for handler in node.handlers:
                            # Find the broad `except Exception as e:` handler
                            etype = handler.type
                            if isinstance(etype, ast.Name) and etype.id == "Exception":
                                return handler
    raise AssertionError("execute_tool_call broad-except handler not found")


def test_rc2_hook_filters_on_scope_errors():
    handler = _execute_tool_call_except_handler()
    # The first body node should be an `if isinstance(e, (NameError, UnboundLocalError)):`
    body_src = ast.unparse(ast.Module(body=handler.body, type_ignores=[]))
    assert "NameError" in body_src and "UnboundLocalError" in body_src, (
        "RC2 diagnostic hook missing — expected an `isinstance(e, (NameError, UnboundLocalError))` "
        "branch in execute_tool_call's broad-except handler. See Pass 3.1 in the plan."
    )


def test_rc2_hook_walks_cause_and_context():
    handler = _execute_tool_call_except_handler()
    body_src = ast.unparse(ast.Module(body=handler.body, type_ignores=[]))
    assert "__cause__" in body_src, "hook must walk __cause__ for wrapped exceptions"
    assert "__context__" in body_src, "hook must fall back to __context__ when __cause__ is None"


def test_rc2_hook_uses_traceback_format():
    """The hook must emit tracebacks, not just `str(e)` (that's what the wrap already does)."""
    handler = _execute_tool_call_except_handler()
    body_src = ast.unparse(ast.Module(body=handler.body, type_ignores=[]))
    assert "traceback" in body_src and "format_exception" in body_src, (
        "hook must call traceback.format_exception for each chain element"
    )


def test_rc2_hook_does_not_swallow_exception():
    """The hook must be followed by the existing `raise AgentToolExecutionError(...) from e`."""
    handler = _execute_tool_call_except_handler()
    # The handler body must end with a Raise statement (the existing wrap)
    last = handler.body[-1]
    assert isinstance(last, ast.Raise), (
        f"handler must still end with `raise AgentToolExecutionError(...)`, got {ast.unparse(last)!r}"
    )
    raise_src = ast.unparse(last)
    assert "AgentToolExecutionError" in raise_src
    assert "from e" in raise_src, "must preserve exception chain via `raise ... from e`"
