from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Literal

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"

def safe_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

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

@torch.no_grad()
def precompute(split, model, max_length, hook_name, sae, get_feature_acts):
    records = []
    for i, row in enumerate(tqdm(split)):
        prompt_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            tokenize=False, add_generation_prompt=True)
        prompt_ids = model.to_tokens(prompt_text, prepend_bos=False)[0]   # template adds BOS
        true_ids   = model.to_tokens(row["answer"], prepend_bos=False)[0]

        seq = torch.cat([prompt_ids, true_ids])[:max_length]
        # true-sequence acts: keep ONLY if you want true/gold acts on the shelf; else drop this line
        fa = get_feature_acts(sae, model, hook_name, seq[None])[0].half().cpu()

        seq_full = torch.cat([prompt_ids, true_ids])
        if len(seq_full) > max_length:
            print(f"[trunc] id {i}: {len(seq_full)} -> {max_length} (prompt={len(prompt_ids)})")
        seq = seq_full[:max_length]

        records.append({
            "id": i,
            "prompt": prompt_text,                     # templated chat prompt
            "true_answer": row["answer"],
            "input_ids": seq,                          # prompt + true (for true-acts/analysis)
            #"prompt_len": len(prompt_ids),             # boundary = templated prompt length
            "prompt_len": min(len(prompt_ids), len(seq)),
            "feature_acts_withtrue": fa.to_sparse(),            # true-sequence acts (optional)
        })
    return records

def collate_gen(batch):
    B = len(batch)
    pad_id = 0
    prompts = [r["input_ids"][:r["prompt_len"]] for r in batch]   # prompt only
    Pmax = max(len(p) for p in prompts)
    input_ids = torch.full((B, Pmax), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, Pmax), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, Pmax-len(p):]      = p            # LEFT pad
        attention_mask[i, Pmax-len(p):] = 1
    ids = torch.tensor([r["id"] for r in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "ids": ids}

def build_or_load(split, path, model, max_length, hook_name, sae, get_feature_acts, force=False):
    path = Path(path)
    if path.exists() and not force:
        return torch.load(path, weights_only=False)
    records = precompute(split, model, max_length, hook_name, sae, get_feature_acts)
    safe_save(records, path)
    return records

"""
def collate(batch):
    B = len(batch)
    max_len = max(len(r["input_ids"]) for r in batch)
    F = batch[0]["feature_acts_withtrue"].shape[1]
    input_ids      = torch.zeros(B, max_len, dtype=torch.long)
    feature_acts   = torch.zeros(B, max_len, F, dtype=torch.float16)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    prompt_lens    = torch.empty(B, dtype=torch.long)
    for i, r in enumerate(batch):
        T   = len(r["input_ids"])
        pad = max_len - T
        input_ids[i, pad:]      = r["input_ids"]
        feature_acts[i, pad:]   = r["feature_acts"].to_dense()   # <-- densify here
        attention_mask[i, pad:] = 1
        prompt_lens[i]          = pad + r["prompt_len"]
    return {"input_ids": input_ids, "prompt_lens": prompt_lens,
            "attention_mask": attention_mask, "feature_acts": feature_acts}

def make_dataloaders(
    train_split,
    test_split,
    model,
    prepend_bos: bool,
    batch_size: int = 8,
    max_length: int = 512,
) -> tuple[DataLoader, DataLoader]:

    Return (train_loader, test_loader) for GSM8K.

    Each batch dict contains:
      input_ids  : (B, T) long  — left-padded question+answer tokens
      prompt_lens: (B,)   long  — index into input_ids where the answer begins

    To build a fine-tuning loss mask:
      labels = input_ids.clone()
      for i, pl in enumerate(prompt_lens):
          labels[i, :pl] = -100

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
"""