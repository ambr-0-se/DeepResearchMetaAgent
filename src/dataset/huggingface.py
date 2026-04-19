import os
import random
import pandas as pd
import datasets
import base64

from src.utils import assemble_project_path
from src.logger import logger
from src.registry import DATASET

@DATASET.register_module(name="gaia_dataset", force=True)
class GAIADataset():
    def __init__(
        self,
        path,
        name,
        split,
        task_ids=None,
        skip_file_attachments=False,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            path: filesystem path to the downloaded GAIA dataset
            name: GAIA subset name (e.g. "2023_all")
            split: "validation" | "test"
            task_ids: optional list of task_id strings to restrict to.
                      When set, all other questions are dropped (preserves
                      original order of the matches).
            skip_file_attachments: when True, drops every question with a
                      non-empty `file_name`. Useful for smoke validation in
                      environments where attachment handling is broken
                      (e.g. local dev path contains spaces; browser-use's
                      pdf download truncates at the first space).
            shuffle: when True, randomly permute the dataset with `seed`
                     BEFORE any filters (task_ids, skip_file_attachments)
                     or downstream slicing (e.g. run_gaia.py's max_samples).
                     Purpose: enable E0 random-subsample training that
                     preserves validation's natural difficulty distribution
                     (vs. the biased first-N order of the raw dataset).
                     See HANDOFF_TEST_EVAL.md §E0 methodology note.
            seed: random seed for shuffle; default 42 so runs are
                  reproducible. Different seeds produce different but
                  equally-valid subsamples.
        """
        self.path = path
        self.name = name
        self.split = split
        self.shuffle = shuffle
        self.seed = seed

        path = assemble_project_path(path)
        ds = datasets.load_dataset(path, name, trust_remote_code=True)[split]
        ds = ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
        ds = ds.map(self.preprocess_file_paths, load_from_cache_file=False, fn_kwargs={"split": split, "path": path})

        data = pd.DataFrame(ds)

        if shuffle:
            # Deterministic shuffle via random.Random(seed) so the same seed
            # always produces the same order. Applied BEFORE filters so that
            # max_samples slicing downstream in run_gaia.py produces a
            # uniform random subsample (not biased by file order).
            indices = list(range(len(data)))
            random.Random(seed).shuffle(indices)
            data = data.iloc[indices].reset_index(drop=True)
            logger.info(
                f"[GAIADataset] shuffled {len(data)} questions with seed={seed}"
            )

        if skip_file_attachments:
            before = len(data)
            data = data[data["file_name"].map(lambda s: not s)].reset_index(drop=True)
            logger.info(
                f"[GAIADataset] skip_file_attachments=True: {before} -> {len(data)} questions"
            )

        if task_ids:
            allowed = set(task_ids)
            before = len(data)
            data = data[data["task_id"].isin(allowed)].reset_index(drop=True)
            logger.info(
                f"[GAIADataset] task_ids filter applied: {before} -> {len(data)} questions"
            )

        self.data = data
        
    def preprocess_file_paths(self, row, path, split):
        save_path = assemble_project_path(os.path.join(path, "2023", split))
        os.makedirs(save_path, exist_ok=True)
        if len(row["file_name"]) > 0:
            row["file_name"] = os.path.join(save_path, row["file_name"])
        return row
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]

@DATASET.register_module(name="hle_dataset", force=True)
class HLEDataset():
    def __init__(self, path, name, split):
        self.path = path
        self.name = name
        self.split = split

        path = assemble_project_path(path)
        ds = datasets.load_dataset(path, trust_remote_code=True)[split]
        ds = ds.rename_columns({"answer": "true_answer", "id": "task_id"})
        ds = ds.map(self.preprocess_file_paths, load_from_cache_file=False, fn_kwargs={"split": split, "path": path})

        data = pd.DataFrame(ds)
        self.data = data
        
    def preprocess_file_paths(self, row, path, split):
        save_path = assemble_project_path(os.path.join(path, "images", split))
        os.makedirs(save_path, exist_ok=True)

        image_path = ""
        if len(row["image"]) > 0:
            image_string = row["image"]
            task_id = row["task_id"]
            if image_string.startswith('data:image'):
                image_type = image_string.split(';')[0].split('/')[1]
                image_base64 = image_string.split(',')[1]

                image_path = os.path.join(save_path, f"{task_id}.{image_type}")
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(image_base64))
                logger.info(f"Save image {task_id} to {image_path}")
            else:
                image_path = ""

        row["file_name"] = image_path
        return row
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]