import os
import json
import pandas as pd

from src.utils import assemble_project_path
from src.logger import logger
from src.registry import DATASET


ARC_QUESTION_TEMPLATE = """\
You are solving an ARC-AGI task. Each task shows training examples of \
input-output grid pairs. You must figure out the transformation pattern \
and apply it to the test input to produce the correct output grid.

Grids contain integers 0-9 representing colors.

{training_examples}
=== Test Input ===
{test_input}

Provide your answer as ONLY a JSON 2D array (list of lists of integers), e.g. [[1,2],[3,4]].
Do not include any explanation, just the JSON array."""


def format_grid(grid):
    """Render a 2D integer grid as a human-readable text block."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    lines = [f"Grid ({rows}x{cols}):"]
    for row in grid:
        lines.append(" ".join(str(cell) for cell in row))
    return "\n".join(lines)


def _build_question(task_data, test_index):
    """Build the full question string for a single test case."""
    parts = []
    parts.append("=== Training Examples ===")
    for i, example in enumerate(task_data["train"], 1):
        parts.append(f"-- Example {i} --")
        parts.append("Input:")
        parts.append(format_grid(example["input"]))
        parts.append("Output:")
        parts.append(format_grid(example["output"]))
        parts.append("")

    test_input = task_data["test"][test_index]["input"]
    test_section = "Input:\n" + format_grid(test_input)

    return ARC_QUESTION_TEMPLATE.format(
        training_examples="\n".join(parts),
        test_input=test_section,
    )


@DATASET.register_module(name="arc_dataset", force=True)
class ARCDataset():
    def __init__(self, path, split):
        self.path = path
        self.split = split

        abs_path = assemble_project_path(path)
        split_dir = os.path.join(abs_path, split)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"ARC split directory not found: {split_dir}. "
                f"Download the ARC-AGI dataset and place it under {abs_path}/ "
                f"with subdirectories like training/ and evaluation/."
            )

        records = []
        json_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".json"))
        logger.info(f"Loading {len(json_files)} ARC tasks from {split_dir}")

        for filename in json_files:
            task_name = filename.removesuffix(".json")
            filepath = os.path.join(split_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                task_data = json.load(f)

            for test_idx, test_case in enumerate(task_data["test"]):
                task_id = f"{task_name}_{test_idx}" if len(task_data["test"]) > 1 else task_name
                question = _build_question(task_data, test_idx)
                true_answer = json.dumps(test_case["output"])

                records.append({
                    "task_id": task_id,
                    "question": question,
                    "true_answer": true_answer,
                    "task": split,
                    "file_name": "",
                })

        self.data = pd.DataFrame(records)
        logger.info(f"Loaded {len(self.data)} ARC test cases from {len(json_files)} tasks")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]
