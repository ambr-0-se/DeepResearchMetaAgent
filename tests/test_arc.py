"""Tests for ARC-AGI evaluation support."""
import sys
import os
import json
import tempfile
import traceback
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)


# ============================================================================
# ARC Scorer
# ============================================================================

class TestARCScorer:
    """Test arc_question_scorer and helpers."""

    def test_exact_match(self):
        from src.metric.arc_scorer import arc_question_scorer
        assert arc_question_scorer("[[1,2],[3,4]]", "[[1,2],[3,4]]")

    def test_mismatch(self):
        from src.metric.arc_scorer import arc_question_scorer
        assert not arc_question_scorer("[[1,2],[3,4]]", "[[1,2],[3,5]]")

    def test_dimension_mismatch(self):
        from src.metric.arc_scorer import arc_question_scorer
        assert not arc_question_scorer("[[1,2]]", "[[1,2],[3,4]]")

    def test_column_mismatch(self):
        from src.metric.arc_scorer import arc_question_scorer
        assert not arc_question_scorer("[[1,2,3]]", "[[1,2]]")

    def test_empty_grid(self):
        from src.metric.arc_scorer import arc_question_scorer
        assert not arc_question_scorer("[]", "[[1]]")

    def test_extract_from_text_with_surrounding(self):
        from src.metric.arc_scorer import arc_question_scorer
        pred = "The answer is [[1,2],[3,4]] based on the pattern."
        assert arc_question_scorer(pred, "[[1,2],[3,4]]")

    def test_extract_from_code_fence(self):
        from src.metric.arc_scorer import arc_question_scorer
        pred = "```json\n[[1,2],[3,4]]\n```"
        assert arc_question_scorer(pred, "[[1,2],[3,4]]")

    def test_extract_from_code_fence_no_lang(self):
        from src.metric.arc_scorer import arc_question_scorer
        pred = "```\n[[5,6],[7,8]]\n```"
        assert arc_question_scorer(pred, "[[5,6],[7,8]]")

    def test_non_json_prediction(self):
        from src.metric.arc_scorer import arc_question_scorer
        assert not arc_question_scorer("I don't know", "[[1,2]]")

    def test_none_prediction(self):
        from src.metric.arc_scorer import arc_question_scorer
        assert not arc_question_scorer("None", "[[1,2]]")

    def test_single_cell_grid(self):
        from src.metric.arc_scorer import arc_question_scorer
        assert arc_question_scorer("[[0]]", "[[0]]")
        assert not arc_question_scorer("[[0]]", "[[1]]")

    def test_large_grid(self):
        from src.metric.arc_scorer import arc_question_scorer
        grid = [[i % 10 for i in range(30)] for _ in range(30)]
        grid_str = json.dumps(grid)
        assert arc_question_scorer(grid_str, grid_str)

    def test_nested_brackets_in_text(self):
        from src.metric.arc_scorer import extract_grid_from_text
        text = "I think the output is [[0,1],[2,3]] because of symmetry."
        grid = extract_grid_from_text(text)
        assert grid == [[0, 1], [2, 3]]

    def test_float_values_cast_to_int(self):
        from src.metric.arc_scorer import arc_question_scorer
        assert arc_question_scorer("[[1.0,2.0]]", "[[1,2]]")


# ============================================================================
# ARC Scorer - grids_equal
# ============================================================================

class TestGridsEqual:
    """Test the grids_equal helper."""

    def test_equal(self):
        from src.metric.arc_scorer import grids_equal
        assert grids_equal([[1, 2], [3, 4]], [[1, 2], [3, 4]])

    def test_different_values(self):
        from src.metric.arc_scorer import grids_equal
        assert not grids_equal([[1, 2]], [[1, 3]])

    def test_different_rows(self):
        from src.metric.arc_scorer import grids_equal
        assert not grids_equal([[1]], [[1], [2]])

    def test_different_cols(self):
        from src.metric.arc_scorer import grids_equal
        assert not grids_equal([[1, 2]], [[1]])

    def test_empty(self):
        from src.metric.arc_scorer import grids_equal
        assert grids_equal([], [])


# ============================================================================
# ARC Scorer - extract_grid_from_text
# ============================================================================

class TestExtractGrid:
    """Test the extract_grid_from_text helper."""

    def test_direct_json(self):
        from src.metric.arc_scorer import extract_grid_from_text
        assert extract_grid_from_text("[[1,2],[3,4]]") == [[1, 2], [3, 4]]

    def test_with_whitespace(self):
        from src.metric.arc_scorer import extract_grid_from_text
        assert extract_grid_from_text("  [[1, 2], [3, 4]]  ") == [[1, 2], [3, 4]]

    def test_in_text(self):
        from src.metric.arc_scorer import extract_grid_from_text
        result = extract_grid_from_text("Output: [[0,1]] and done")
        assert result == [[0, 1]]

    def test_code_fence(self):
        from src.metric.arc_scorer import extract_grid_from_text
        result = extract_grid_from_text("```json\n[[5]]\n```")
        assert result == [[5]]

    def test_no_grid(self):
        from src.metric.arc_scorer import extract_grid_from_text
        assert extract_grid_from_text("no grid here") is None

    def test_flat_list_rejected(self):
        from src.metric.arc_scorer import extract_grid_from_text
        assert extract_grid_from_text("[1,2,3]") is None

    def test_empty_outer_list(self):
        from src.metric.arc_scorer import extract_grid_from_text
        assert extract_grid_from_text("[]") is None

    def test_string_values_rejected(self):
        from src.metric.arc_scorer import extract_grid_from_text
        assert extract_grid_from_text('[["a","b"]]') is None


# ============================================================================
# ARC Dataset Loader
# ============================================================================

class TestARCDataset:
    """Test ARCDataset loading from temporary JSON files."""

    def _make_dataset_dir(self, tmpdir, split="evaluation"):
        split_dir = os.path.join(tmpdir, split)
        os.makedirs(split_dir, exist_ok=True)

        task1 = {
            "train": [
                {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]}
            ],
            "test": [
                {"input": [[4, 5], [6, 7]], "output": [[7, 6], [5, 4]]}
            ]
        }
        with open(os.path.join(split_dir, "task_abc.json"), "w") as f:
            json.dump(task1, f)

        task2_multi = {
            "train": [
                {"input": [[0]], "output": [[1]]}
            ],
            "test": [
                {"input": [[2]], "output": [[3]]},
                {"input": [[4]], "output": [[5]]}
            ]
        }
        with open(os.path.join(split_dir, "task_multi.json"), "w") as f:
            json.dump(task2_multi, f)

        return tmpdir

    def test_loads_correct_count(self):
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dataset_dir(tmpdir)
            ds = ARCDataset(path=tmpdir, split="evaluation")
            # task_abc: 1 test case, task_multi: 2 test cases = 3 total
            assert len(ds) == 3

    def test_fields_present(self):
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dataset_dir(tmpdir)
            ds = ARCDataset(path=tmpdir, split="evaluation")
            row = ds[0]
            for field in ("task_id", "question", "true_answer", "task", "file_name"):
                assert field in row.index, f"Missing field: {field}"

    def test_task_id_format_single_test(self):
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dataset_dir(tmpdir)
            ds = ARCDataset(path=tmpdir, split="evaluation")
            task_ids = ds.data["task_id"].tolist()
            # task_abc has 1 test case -> no suffix
            assert "task_abc" in task_ids

    def test_task_id_format_multi_test(self):
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dataset_dir(tmpdir)
            ds = ARCDataset(path=tmpdir, split="evaluation")
            task_ids = ds.data["task_id"].tolist()
            # task_multi has 2 test cases -> suffixed
            assert "task_multi_0" in task_ids
            assert "task_multi_1" in task_ids

    def test_true_answer_is_valid_json(self):
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dataset_dir(tmpdir)
            ds = ARCDataset(path=tmpdir, split="evaluation")
            for i in range(len(ds)):
                row = ds[i]
                parsed = json.loads(row["true_answer"])
                assert isinstance(parsed, list)
                assert all(isinstance(r, list) for r in parsed)

    def test_question_contains_training_examples(self):
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dataset_dir(tmpdir)
            ds = ARCDataset(path=tmpdir, split="evaluation")
            row = ds.data[ds.data["task_id"] == "task_abc"].iloc[0]
            q = row["question"]
            assert "Training Examples" in q
            assert "Test Input" in q
            assert "Grid (" in q

    def test_file_name_is_empty(self):
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dataset_dir(tmpdir)
            ds = ARCDataset(path=tmpdir, split="evaluation")
            for i in range(len(ds)):
                assert ds[i]["file_name"] == ""

    def test_missing_split_raises(self):
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            raised = False
            try:
                ARCDataset(path=tmpdir, split="nonexistent")
            except FileNotFoundError:
                raised = True
            assert raised, "Expected FileNotFoundError for missing split directory"

    def test_getitem_returns_series(self):
        import pandas as pd
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dataset_dir(tmpdir)
            ds = ARCDataset(path=tmpdir, split="evaluation")
            assert isinstance(ds[0], pd.Series)

    def test_to_dict_records(self):
        from src.dataset.arc import ARCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dataset_dir(tmpdir)
            ds = ARCDataset(path=tmpdir, split="evaluation")
            records = ds.data.to_dict(orient="records")
            assert len(records) == 3
            assert all(isinstance(r, dict) for r in records)


# ============================================================================
# ARC Dataset - format_grid
# ============================================================================

class TestFormatGrid:
    """Test the format_grid helper."""

    def test_basic_grid(self):
        from src.dataset.arc import format_grid
        result = format_grid([[0, 1], [2, 3]])
        assert "Grid (2x2):" in result
        assert "0 1" in result
        assert "2 3" in result

    def test_single_cell(self):
        from src.dataset.arc import format_grid
        result = format_grid([[5]])
        assert "Grid (1x1):" in result
        assert "5" in result

    def test_rectangular_grid(self):
        from src.dataset.arc import format_grid
        result = format_grid([[1, 2, 3]])
        assert "Grid (1x3):" in result


# ============================================================================
# Metric __init__ - get_scorer
# ============================================================================

class TestGetScorer:
    """Test the get_scorer factory."""

    def test_default_is_gaia(self):
        from src.metric import get_scorer, question_scorer
        assert get_scorer() is question_scorer

    def test_gaia_explicit(self):
        from src.metric import get_scorer, question_scorer
        assert get_scorer("gaia") is question_scorer

    def test_arc(self):
        from src.metric import get_scorer, arc_question_scorer
        assert get_scorer("arc") is arc_question_scorer

    def test_unknown_falls_back_to_gaia(self):
        from src.metric import get_scorer, question_scorer
        assert get_scorer("unknown_dataset") is question_scorer


# ============================================================================
# Analysis script - detect_dataset_info
# ============================================================================

class TestDetectDatasetInfo:
    """Test ARC detection in analyze_results.detect_dataset_info."""

    def test_detect_from_meta(self):
        sys.path.insert(0, os.path.join(root, "scripts"))
        from analyze_results import detect_dataset_info
        results = [{"task": "evaluation", "true_answer": "[[1,2]]"}]
        meta = {"config_file": "configs/config_arc.py"}
        info = detect_dataset_info(results, meta)
        assert info["dataset"] == "ARC-AGI"
        assert info["category_label"] == "Split"

    def test_detect_from_data_fallback(self):
        sys.path.insert(0, os.path.join(root, "scripts"))
        from analyze_results import detect_dataset_info
        results = [{"task": "evaluation", "true_answer": "[[1,2],[3,4]]"}]
        meta = {}
        info = detect_dataset_info(results, meta)
        assert info["dataset"] == "ARC-AGI"

    def test_gaia_not_detected_as_arc(self):
        sys.path.insert(0, os.path.join(root, "scripts"))
        from analyze_results import detect_dataset_info
        results = [{"task": "1", "true_answer": "42"}]
        meta = {"config_file": "configs/config_gaia.py"}
        info = detect_dataset_info(results, meta)
        assert info["dataset"] == "GAIA"


# ============================================================================
# Compare script - auto-detect scorer
# ============================================================================

class TestAutoDetectScorer:
    """Test _auto_detect_scorer in compare_results."""

    def test_detects_arc(self):
        sys.path.insert(0, os.path.join(root, "scripts"))
        from compare_results import _auto_detect_scorer
        from src.metric import arc_question_scorer
        results = [{"true_answer": "[[1,2]]"}]
        assert _auto_detect_scorer(results) is arc_question_scorer

    def test_detects_gaia(self):
        sys.path.insert(0, os.path.join(root, "scripts"))
        from compare_results import _auto_detect_scorer
        from src.metric import question_scorer
        results = [{"true_answer": "42"}]
        assert _auto_detect_scorer(results) is question_scorer

    def test_strips_whitespace(self):
        sys.path.insert(0, os.path.join(root, "scripts"))
        from compare_results import _auto_detect_scorer
        from src.metric import arc_question_scorer
        results = [{"true_answer": "  [[1,2]]  "}]
        assert _auto_detect_scorer(results) is arc_question_scorer


# ============================================================================
# Runner
# ============================================================================

def run_all_tests():
    test_classes = [
        TestARCScorer,
        TestGridsEqual,
        TestExtractGrid,
        TestARCDataset,
        TestFormatGrid,
        TestGetScorer,
        TestDetectDatasetInfo,
        TestAutoDetectScorer,
    ]
    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            test_name = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS  {test_name}")
            except Exception as e:
                failed += 1
                tb = traceback.format_exc()
                errors.append((test_name, tb))
                print(f"  FAIL  {test_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for name, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
