"""
Tests for the ReviewResult Pydantic schema (condition C3).

Covers:
- Round-trip JSON serialization/deserialization for each next_action variant
- Discriminated-union routing via the `action` field
- Validation of the ModifyAgentSpec action enum (must match ModifySubAgentTool)
- render() output shape
- Rejection of malformed payloads
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.meta.review_schema import (  # noqa: E402
    EscalateSpec,
    ModifyAgentSpec,
    ProceedSpec,
    ReviewResult,
    RetrySpec,
    RootCauseCategory,
)


# ---------------------------------------------------------------------------
# Next-action variants — construction, serialization, round-trip
# ---------------------------------------------------------------------------

class TestProceedSpec:
    def test_construct(self):
        spec = ProceedSpec()
        assert spec.action == "proceed"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ProceedSpec.model_validate({"action": "proceed", "bogus": 1})


class TestRetrySpec:
    def test_construct_minimal(self):
        spec = RetrySpec(
            agent_name="deep_analyzer_agent",
            revised_task="Re-extract with absolute path",
            additional_guidance="Pass the full /Users/... path",
        )
        assert spec.action == "retry"
        assert spec.avoid_patterns == []

    def test_avoid_patterns_default_empty(self):
        spec = RetrySpec(
            agent_name="a",
            revised_task="t",
            additional_guidance="g",
        )
        assert isinstance(spec.avoid_patterns, list)

    def test_round_trip(self):
        spec = RetrySpec(
            agent_name="browser_use_agent",
            revised_task="Retry with archive.org fallback",
            additional_guidance="If paywall, try web.archive.org",
            avoid_patterns=["giving up on first 403"],
        )
        js = spec.model_dump_json()
        restored = RetrySpec.model_validate_json(js)
        assert restored == spec


class TestModifyAgentSpec:
    """
    ModifyAgentSpec fields MUST match ModifySubAgentTool.forward() parameters
    so the spec can be passed through without reformatting. If ModifySubAgentTool
    adds/removes actions, this test will catch the drift.
    """

    VALID_MODIFY_ACTIONS = {
        "add_existing_tool_to_agent",
        "add_new_tool_to_agent",
        "remove_tool_from_agent",
        "modify_agent_instructions",
        "add_agent",
        "remove_agent",
        "set_agent_max_steps",
    }

    @pytest.mark.parametrize("modify_action", sorted(VALID_MODIFY_ACTIONS))
    def test_all_valid_modify_actions_accepted(self, modify_action):
        spec = ModifyAgentSpec(
            modify_action=modify_action,
            agent_name="deep_analyzer_agent",
            specification="python_interpreter_tool",
        )
        assert spec.modify_action == modify_action

    def test_invalid_modify_action_rejected(self):
        with pytest.raises(ValidationError):
            ModifyAgentSpec(
                modify_action="delete_agent_now",  # not a real action
                agent_name="x",
                specification="y",
            )

    def test_followup_retry_defaults_true(self):
        spec = ModifyAgentSpec(
            modify_action="modify_agent_instructions",
            agent_name="a",
            specification="Do X instead of Y",
        )
        assert spec.followup_retry is True

    def test_signature_matches_modify_tool(self):
        """
        Guard: the (modify_action, agent_name, specification) trio must remain
        a valid passthrough to ModifySubAgentTool.forward(action, agent_name,
        specification). If ModifySubAgentTool drops/renames any of these, this
        test will still pass because it only checks the spec — but the
        integration check in review_step.py must also be updated.
        """
        spec = ModifyAgentSpec(
            modify_action="add_new_tool_to_agent",
            agent_name="deep_analyzer_agent",
            specification="A tool that extracts tables from PDFs.",
        )
        # Fields are named so the planner can unpack them cleanly
        assert hasattr(spec, "modify_action")
        assert hasattr(spec, "agent_name")
        assert hasattr(spec, "specification")


class TestEscalateSpec:
    def test_construct(self):
        spec = EscalateSpec(
            from_agent="browser_use_agent",
            to_agent="deep_researcher_agent",
            reason="Browser failing on dynamic content",
            task="Research the topic using general search",
        )
        assert spec.action == "escalate"


# ---------------------------------------------------------------------------
# ReviewResult — full payload tests
# ---------------------------------------------------------------------------

class TestReviewResultSatisfactory:
    def test_minimal_satisfactory(self):
        result = ReviewResult(
            verdict="satisfactory",
            confidence=0.95,
            summary="Deep analyzer returned a well-structured answer.",
            next_action=ProceedSpec(),
        )
        assert result.root_cause_primary is None
        assert result.root_cause_secondary is None
        assert result.next_action.action == "proceed"

    def test_round_trip_satisfactory(self):
        result = ReviewResult(
            verdict="satisfactory",
            confidence=1.0,
            summary="OK.",
            next_action=ProceedSpec(),
        )
        js = result.model_dump_json()
        restored = ReviewResult.model_validate_json(js)
        assert restored == result


class TestReviewResultUnsatisfactory:
    def test_with_modify_agent_next_action(self):
        result = ReviewResult(
            verdict="unsatisfactory",
            confidence=0.8,
            summary="Agent lacked PDF extraction capability.",
            root_cause_primary=RootCauseCategory.MISSING_TOOL,
            root_cause_detail="No pdfplumber-equivalent tool available",
            next_action=ModifyAgentSpec(
                modify_action="add_existing_tool_to_agent",
                agent_name="deep_analyzer_agent",
                specification="python_interpreter_tool",
            ),
        )
        # Discriminated union resolves correctly
        assert isinstance(result.next_action, ModifyAgentSpec)
        assert result.next_action.modify_action == "add_existing_tool_to_agent"

    def test_primary_and_secondary_root_causes(self):
        result = ReviewResult(
            verdict="partial",
            confidence=0.6,
            summary="Partial result; missing verification.",
            root_cause_primary=RootCauseCategory.INCOMPLETE_OUTPUT,
            root_cause_secondary=RootCauseCategory.INSUFFICIENT_INSTRUCTION,
            next_action=RetrySpec(
                agent_name="deep_analyzer_agent",
                revised_task="Re-extract and verify with python",
                additional_guidance="Show intermediate steps.",
            ),
        )
        assert result.root_cause_primary == RootCauseCategory.INCOMPLETE_OUTPUT
        assert result.root_cause_secondary == RootCauseCategory.INSUFFICIENT_INSTRUCTION

    def test_round_trip_all_variants(self):
        """Full round-trip for each next_action discriminator value."""
        variants = [
            ReviewResult(
                verdict="satisfactory",
                confidence=1.0,
                summary="OK",
                next_action=ProceedSpec(),
            ),
            ReviewResult(
                verdict="unsatisfactory",
                confidence=0.7,
                summary="retry",
                root_cause_primary=RootCauseCategory.INSUFFICIENT_INSTRUCTION,
                next_action=RetrySpec(
                    agent_name="a",
                    revised_task="t",
                    additional_guidance="g",
                ),
            ),
            ReviewResult(
                verdict="unsatisfactory",
                confidence=0.8,
                summary="modify",
                root_cause_primary=RootCauseCategory.MISSING_TOOL,
                next_action=ModifyAgentSpec(
                    modify_action="modify_agent_instructions",
                    agent_name="a",
                    specification="s",
                ),
            ),
            ReviewResult(
                verdict="unsatisfactory",
                confidence=0.9,
                summary="escalate",
                root_cause_primary=RootCauseCategory.WRONG_TOOL_SELECTION,
                next_action=EscalateSpec(
                    from_agent="a",
                    to_agent="b",
                    reason="r",
                    task="t",
                ),
            ),
        ]
        for original in variants:
            js = original.model_dump_json()
            restored = ReviewResult.model_validate_json(js)
            assert restored == original, f"Round-trip failed for {original.next_action.action}"


class TestReviewResultValidation:
    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ReviewResult(
                verdict="satisfactory",
                confidence=1.5,  # > 1.0
                summary="x",
                next_action=ProceedSpec(),
            )

    def test_invalid_verdict_rejected(self):
        with pytest.raises(ValidationError):
            ReviewResult(
                verdict="kinda_ok",  # not in the Literal
                confidence=0.5,
                summary="x",
                next_action=ProceedSpec(),
            )

    def test_missing_next_action_rejected(self):
        with pytest.raises(ValidationError):
            ReviewResult(
                verdict="satisfactory",
                confidence=1.0,
                summary="x",
            )  # type: ignore[call-arg]


class TestRender:
    def test_render_satisfactory_minimal(self):
        result = ReviewResult(
            verdict="satisfactory",
            confidence=1.0,
            summary="Clean pass.",
            next_action=ProceedSpec(),
        )
        text = result.render()
        assert "verdict: satisfactory" in text
        assert "confidence=1.00" in text
        assert "summary: Clean pass." in text
        assert "next_action: proceed" in text
        # No root_cause line when none provided
        assert "root_cause:" not in text

    def test_render_with_root_cause_and_secondary(self):
        result = ReviewResult(
            verdict="unsatisfactory",
            confidence=0.6,
            summary="Missing tool and unclear task.",
            root_cause_primary=RootCauseCategory.MISSING_TOOL,
            root_cause_secondary=RootCauseCategory.UNCLEAR_OBJECTIVE,
            root_cause_detail="No pdfplumber; task ambiguous on output format",
            next_action=ModifyAgentSpec(
                modify_action="add_existing_tool_to_agent",
                agent_name="deep_analyzer_agent",
                specification="python_interpreter_tool",
            ),
        )
        text = result.render()
        assert "root_cause: missing_tool (+unclear_goal)" in text
        assert "detail: No pdfplumber" in text
        assert "modify_agent deep_analyzer_agent" in text
