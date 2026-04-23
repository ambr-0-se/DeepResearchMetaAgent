"""Tests for GAIA scorer parity with the official leaderboard scorer."""

import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.metric.gaia_scorer import question_scorer


class TestQuestionScorerNoneHandling:
    """
    Parity with huggingface.co/spaces/gaia-benchmark/leaderboard scorer.py:
    a None prediction is scored as the literal string "None", never raised.
    """

    def test_none_prediction_on_string_truth_is_false_not_raised(self):
        assert question_scorer(None, "Berlin") is False

    def test_none_prediction_on_number_truth_is_false_not_raised(self):
        assert question_scorer(None, "42") is False

    def test_none_prediction_on_list_truth_is_false_not_raised(self):
        assert question_scorer(None, "a, b, c") is False

    def test_literal_none_prediction_matches_literal_none_truth(self):
        # edge case: if the truth happens to be the string "None", a None
        # prediction should score True because leaderboard-parity coerces
        # None → "None" before normalisation.
        assert question_scorer(None, "None") is True

    def test_string_prediction_still_works_after_guard(self):
        assert question_scorer("Berlin", "Berlin") is True
        assert question_scorer("berlin", "Berlin") is True  # lowercased

    def test_number_prediction_still_works_after_guard(self):
        assert question_scorer("42", "42") is True
        assert question_scorer("$42", "42") is True  # strips $ per official

    def test_list_prediction_still_works_after_guard(self):
        assert question_scorer("a, b, c", "a, b, c") is True
        assert question_scorer("a, b", "a, b, c") is False  # length guard


def run_all_tests():
    """Minimal runner matching the project's test_eval_fixes.py convention."""
    t = TestQuestionScorerNoneHandling()
    tests = [
        t.test_none_prediction_on_string_truth_is_false_not_raised,
        t.test_none_prediction_on_number_truth_is_false_not_raised,
        t.test_none_prediction_on_list_truth_is_false_not_raised,
        t.test_literal_none_prediction_matches_literal_none_truth,
        t.test_string_prediction_still_works_after_guard,
        t.test_number_prediction_still_works_after_guard,
        t.test_list_prediction_still_works_after_guard,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {test.__name__}: {e}")
    print(f"{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
