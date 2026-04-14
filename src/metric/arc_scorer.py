import json
import re
from typing import List, Optional


def extract_grid_from_text(text: str) -> Optional[List[List[int]]]:
    """Extract a 2D integer grid from free-form text.

    Tries multiple strategies:
    1. Direct JSON parse of the whole string
    2. JSON parse of content inside markdown code fences
    3. Regex extraction of the first [[...]] pattern
    """
    text = text.strip()

    # Strategy 1: direct parse
    grid = _try_parse_grid(text)
    if grid is not None:
        return grid

    # Strategy 2: inside markdown code fences (```json ... ``` or ``` ... ```)
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    for match in fence_pattern.finditer(text):
        grid = _try_parse_grid(match.group(1).strip())
        if grid is not None:
            return grid

    # Strategy 3: find the first [[...]] block via bracket matching
    start = text.find("[[")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    grid = _try_parse_grid(candidate)
                    if grid is not None:
                        return grid
                    break

    return None


def _try_parse_grid(text: str) -> Optional[List[List[int]]]:
    """Attempt to parse text as a JSON 2D integer grid."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(obj, list) or not obj:
        return None
    if not all(isinstance(row, list) for row in obj):
        return None
    try:
        return [[int(cell) for cell in row] for row in obj]
    except (TypeError, ValueError):
        return None


def grids_equal(grid_a: List[List[int]], grid_b: List[List[int]]) -> bool:
    """Check if two 2D grids are identical (dimensions and values)."""
    if len(grid_a) != len(grid_b):
        return False
    for row_a, row_b in zip(grid_a, grid_b):
        if len(row_a) != len(row_b):
            return False
        if row_a != row_b:
            return False
    return True


def arc_question_scorer(model_answer: str, ground_truth: str) -> bool:
    """Score an ARC-AGI prediction against the ground truth grid.

    Both arguments are strings. ``ground_truth`` is a JSON-encoded 2D list
    (e.g. ``"[[1,2],[3,4]]"``).  ``model_answer`` may contain surrounding
    text; the scorer will attempt to extract a grid from it.
    """
    gt_grid = extract_grid_from_text(str(ground_truth))
    if gt_grid is None:
        return False

    pred_grid = extract_grid_from_text(str(model_answer))
    if pred_grid is None:
        return False

    return grids_equal(pred_grid, gt_grid)
