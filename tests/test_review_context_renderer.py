"""
Tests for ReviewStep._format_context_for_review and helpers.

Covers the reviewer task-text rendering: backward-compat path (no
kwargs) must be byte-identical to pre-change shape; enriched path
injects original task, prior attempts, blocklist/cap directives in the
correct order.
"""
import sys
from pathlib import Path

from unittest.mock import MagicMock

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.meta.review_step import (  # noqa: E402
    ChainState, DelegationContext, PriorAttempt, ReviewStep,
)


def _ctx():
    return DelegationContext(
        agent_name="browser_use_agent",
        task_given="find the year",
        expected_outcome="a 4-digit number",
        actual_response="no result",
        step_number=3,
    )


class TestBackwardCompat:
    def test_no_kwargs_byte_identical_to_pre_change(self):
        """The static renderer with no optional args must produce the same
        string shape as the pre-Layer-2 implementation."""
        expected = (
            "agent_name: browser_use_agent\n"
            "task_given:\nfind the year\n\n"
            "expected_outcome (from planner reasoning):\na 4-digit number\n\n"
            "actual_response:\nno result\n\n"
            "(planner step_number: 3)"
        )
        rendered = ReviewStep._format_context_for_review(_ctx())
        assert rendered == expected


class TestOriginalUserTask:
    def test_short_task_rendered_verbatim(self):
        rendered = ReviewStep._format_context_for_review(
            _ctx(), original_user_task="What year was Interstate 40 built?",
        )
        assert "ORIGINAL USER TASK:" in rendered
        assert "What year was Interstate 40 built?" in rendered

    def test_long_task_truncated(self):
        long_task = "A" * 2000
        rendered = ReviewStep._format_context_for_review(
            _ctx(), original_user_task=long_task,
        )
        # ~1500 cap + ellipsis marker
        assert "…" in rendered
        # The full 2000-A string should NOT appear
        assert "A" * 2000 not in rendered

    def test_empty_task_omits_section(self):
        rendered = ReviewStep._format_context_for_review(
            _ctx(), original_user_task="",
        )
        assert "ORIGINAL USER TASK" not in rendered


class TestPriorAttempts:
    def test_prior_attempts_rendered_with_attempt_indices(self):
        priors = [
            PriorAttempt(1, "unsatisfactory", "external", "first try"),
            PriorAttempt(2, "unsatisfactory", "external", "second try"),
        ]
        rendered = ReviewStep._format_context_for_review(
            _ctx(), prior_attempts=priors,
        )
        assert "PRIOR ATTEMPTS" in rendered
        assert "#1" in rendered
        assert "#2" in rendered
        assert "first try" in rendered
        assert "second try" in rendered
        assert "external" in rendered

    def test_none_prior_attempts_omits_section(self):
        rendered = ReviewStep._format_context_for_review(
            _ctx(), prior_attempts=None,
        )
        assert "PRIOR ATTEMPTS" not in rendered

    def test_empty_prior_attempts_omits_section(self):
        rendered = ReviewStep._format_context_for_review(
            _ctx(), prior_attempts=[],
        )
        assert "PRIOR ATTEMPTS" not in rendered

    def test_prior_with_null_root_cause_renders_dash(self):
        priors = [PriorAttempt(1, "satisfactory", None, "digest")]
        rendered = ReviewStep._format_context_for_review(
            _ctx(), prior_attempts=priors,
        )
        assert "root_cause=—" in rendered or "root_cause=-" in rendered


class TestDirectives:
    def test_task_blocklist_directive_injected(self):
        rendered = ReviewStep._format_context_for_review(
            _ctx(), task_blocklist_directive=(
                "IMPORTANT: agent 'X' previously failed with root_cause=external."
            ),
        )
        assert "IMPORTANT" in rendered
        assert "previously failed" in rendered

    def test_chain_capped_directive_injected(self):
        rendered = ReviewStep._format_context_for_review(
            _ctx(), chain_capped_directive="CHAIN CAPPED: cap reached.",
        )
        assert "CHAIN CAPPED" in rendered

    def test_directives_come_before_original_task(self):
        rendered = ReviewStep._format_context_for_review(
            _ctx(),
            original_user_task="outer q",
            task_blocklist_directive="IMPORTANT: cap reached",
        )
        i_dir = rendered.index("IMPORTANT")
        i_task = rendered.index("ORIGINAL USER TASK")
        assert i_dir < i_task


class TestDigestHelper:
    def test_whitespace_collapsed(self):
        assert ReviewStep._digest_task("a   b\n\n c") == "a b c"

    def test_empty(self):
        assert ReviewStep._digest_task("") == ""

    def test_under_cap_preserved(self):
        assert ReviewStep._digest_task("short") == "short"

    def test_over_cap_truncated_with_ellipsis(self):
        d = ReviewStep._digest_task("x" * 500)
        assert len(d) == ReviewStep.PRIOR_ATTEMPT_DIGEST_MAX
        assert d.endswith("…")

    def test_custom_cap(self):
        d = ReviewStep._digest_task("a" * 50, cap=10)
        assert len(d) == 10
        assert d.endswith("…")


class TestContextLengthBudget:
    """CONTEXT_TEXT_SOFT_BUDGET sanity. The guard is a WARNING-only log,
    not a truncation — the rendered text must always fully convey the
    blocklist / prior_attempts directives. Here we verify the guard doesn't
    break rendering on large inputs."""

    def test_constant_is_sane(self):
        # 10 KB is well below typical model context windows but large
        # enough to accommodate 1500 + 5×200 + ~2 KB overhead.
        assert ReviewStep.CONTEXT_TEXT_SOFT_BUDGET >= 8_000
        assert ReviewStep.CONTEXT_TEXT_SOFT_BUDGET <= 32_000

    def test_large_context_still_renders_completely(self):
        """Even when above the soft budget, all required sections must be
        present. Truncation would silently drop the load-bearing directives."""
        priors = [
            PriorAttempt(i, "unsatisfactory", "external", "x" * 140)
            for i in range(1, 6)
        ]
        long_task = "A" * 1400
        rendered = ReviewStep._format_context_for_review(
            _ctx(),
            original_user_task=long_task,
            prior_attempts=priors,
            task_blocklist_directive="IMPORTANT: blocklist",
            chain_capped_directive="CHAIN CAPPED: foo",
        )
        # All sections present regardless of size
        assert "IMPORTANT: blocklist" in rendered
        assert "CHAIN CAPPED: foo" in rendered
        assert "ORIGINAL USER TASK" in rendered
        assert "PRIOR ATTEMPTS" in rendered
        assert "#5" in rendered  # last prior retained


class TestSubAgentCatalog:
    def test_empty_managed_agents(self):
        p = MagicMock()
        p.managed_agents = {}
        rs = ReviewStep(p)
        assert rs._render_sub_agent_catalog() == ""

    def test_populated_catalog(self):
        tool1 = MagicMock()
        tool2 = MagicMock()
        agent = MagicMock()
        agent.description = "analyses structured data"
        agent.tools = {"python_interpreter_tool": tool1, "final_answer_tool": tool2}

        p = MagicMock()
        p.managed_agents = {"deep_analyzer_agent": agent}
        rs = ReviewStep(p)
        cat = rs._render_sub_agent_catalog()
        assert "deep_analyzer_agent" in cat
        assert "analyses structured data" in cat
        assert "python_interpreter_tool" in cat
        # final_answer_tool filtered
        assert "final_answer_tool" not in cat

    def test_multiple_agents_alphabetical(self):
        p = MagicMock()
        p.managed_agents = {}
        for name in ("zeta_agent", "alpha_agent", "mu_agent"):
            a = MagicMock()
            a.description = "d"
            a.tools = {}
            p.managed_agents[name] = a
        rs = ReviewStep(p)
        cat = rs._render_sub_agent_catalog()
        assert cat.index("alpha_agent") < cat.index("mu_agent") < cat.index("zeta_agent")
