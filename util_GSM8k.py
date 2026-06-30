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


def make_dataloaders(
    train_split,
    test_split,
    model,
    prepend_bos: bool,
    batch_size: int = 8,
    max_length: int = 512,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) for GSM8K.

    Each batch dict contains:
      input_ids  : (B, T) long  — left-padded question+answer tokens
      prompt_lens: (B,)   long  — index into input_ids where the answer begins

    To build a fine-tuning loss mask:
      labels = input_ids.clone()
      for i, pl in enumerate(prompt_lens):
          labels[i, :pl] = -100
    """
    def collate(batch):
        seqs, prompt_lens = [], []
        for row in batch:
            q = model.to_tokens(row["question"], prepend_bos=prepend_bos)[0].tolist()
            a = model.to_tokens(row["answer"], prepend_bos=False)[0].tolist()
            seq = (q + a)[:max_length]
            seqs.append(seq)
            prompt_lens.append(min(len(q), max_length))

        max_len = max(len(s) for s in seqs)
        padded = [[0] * (max_len - len(s)) + s for s in seqs]  # pad_id=0 for Gemma
        padded_prompt_lens = [(max_len - len(s)) + pl for s, pl in zip(seqs, prompt_lens)]
        attention_mask = [[0]*(max_len-len(s)) + [1]*len(s) for s in seqs]

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "prompt_lens": torch.tensor(padded_prompt_lens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    # num_workers=0 required: model.to_tokens can't be pickled across processes
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,  drop_last=True,  collate_fn=collate)
    test_loader  = DataLoader(test_split,  batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate)
    return train_loader, test_loader
