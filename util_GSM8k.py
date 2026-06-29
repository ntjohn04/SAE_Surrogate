from __future__ import annotations

import gc
from pathlib import Path
from typing import Literal

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"

def load_gsm8k(
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Load GSM8K dataset and return (train_split, test_split)."""
    ds = load_dataset(DATASET_ID, DATASET_CONFIG)
    if isinstance(ds, DatasetDict) and "train" in ds and "test" in ds:
        return ds["train"], ds["test"]
    root = ds["train"] if isinstance(ds, DatasetDict) else list(ds.values())[0]
    splits = root.train_test_split(test_size=test_size, seed=seed)
    return splits["train"], splits["test"]


def print_rows(
    split,
    n: int = 5,
    label: Literal["train", "test"] | str = "",
) -> None:
    """Print the first n rows of a dataset split."""
    header = f"--- {label} ({len(split)} rows) ---" if label else f"--- {len(split)} rows ---"
    print(header)
    for i, row in enumerate(split.select(range(min(n, len(split))))):
        print(f"[{i}] {row}")
