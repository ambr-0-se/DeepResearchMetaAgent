"""
Tests for ReviewStep's directive renderers:
  - _render_task_blocklist_directive
  - _render_chain_capped_directive

These produce the task-text bits the reviewer reads to know "retry is
unavailable" — their wording is load-bearing because the reviewer's
decision hinges on recognising the signal.
"""
import sys
from pathlib import Path

from unittest.mock import MagicMock

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.meta.review_step import ChainState, ReviewStep  # noqa: E402


def _rs():
    p = MagicMock()
    p.managed_agents = {}
    p.max_steps = 15
    rs = ReviewStep(p)
    rs.on_task_start("task")
    return rs


class TestBlocklistDirective:
    def test_empty_returns_empty_string(self):
        rs = _rs()
        assert rs._render_task_blocklist_directive("browser_use_agent") == ""

    def test_single_cause_rendered(self):
        rs = _rs()
        rs._task_blocklist.add(("browser_use_agent", "external"))
        d = rs._render_task_blocklist_directive("browser_use_agent")
        assert "browser_use_agent" in d
        assert "external" in d
        assert "Retry is unavailable" in d

    def test_multiple_causes_rendered_sorted(self):
        rs = _rs()
        rs._task_blocklist.add(("browser_use_agent", "external"))
        rs._task_blocklist.add(("browser_use_agent", "model_limit"))
        d = rs._render_task_blocklist_directive("browser_use_agent")
        # Both causes present
        assert "external" in d
        assert "model_limit" in d
        # Sorted alphabetical (external < model_limit)
        assert d.index("external") < d.index("model_limit")

    def test_block_for_another_agent_does_not_bleed(self):
        rs = _rs()
        rs._task_blocklist.add(("deep_researcher_agent", "external"))
        d = rs._render_task_blocklist_directive("browser_use_agent")
        assert d == ""


class TestChainCappedDirective:
    def test_includes_last_root_cause(self):
        chain = ChainState(
            anchor="xyz", count=2, capped=True,
            last_root_cause="bad_instruction",
        )
        d = ReviewStep._render_chain_capped_directive(chain)
        assert "CHAIN CAPPED" in d
        assert "bad_instruction" in d
        assert "attempts=2" in d

    def test_defaults_to_unknown_when_root_cause_missing(self):
        chain = ChainState(anchor="xyz", count=0, capped=True)
        d = ReviewStep._render_chain_capped_directive(chain)
        assert "unknown" in d

    def test_suggests_valid_next_actions(self):
        chain = ChainState(anchor="xyz", count=2, capped=True,
                           last_root_cause="external")
        d = ReviewStep._render_chain_capped_directive(chain)
        # Must guide the reviewer to a non-retry action
        assert "modify_agent" in d or "escalate" in d or "proceed" in d
