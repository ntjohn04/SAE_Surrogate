from __future__ import annotations

import gc
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Batched feature extraction
# ---------------------------------------------------------------------------

def get_feature_sequences_batch(sae, model, prepend_bos, hook_name, prompts, max_length=None):
    """Run a list of prompts through model + SAE and return per-token feature activations.

    Returns
    -------
    tokens        : LongTensor  [batch, seq_len]
    feature_acts  : Tensor      [batch, seq_len, num_features]
    attention_mask: BoolTensor  [batch, seq_len]  (True = real token)
    """
    with torch.inference_mode():
        tokens = model.to_tokens(prompts, prepend_bos=prepend_bos)
        if max_length is not None:
            tokens = tokens[:, :max_length]

        pad_token_id = model.tokenizer.pad_token_id
        if pad_token_id is not None:
            attention_mask = tokens != pad_token_id
        else:
            attention_mask = torch.ones_like(tokens, dtype=torch.bool)

        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        activations = cache[hook_name]
        feature_acts = sae.encode(activations)

    return tokens, feature_acts, attention_mask


# ---------------------------------------------------------------------------
# GSM8K loading
# ---------------------------------------------------------------------------

def load_gsm8k_raw(split: str = "train"):
    """Load GSM8K from HuggingFace and return (questions, answers) lists."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    questions = [ex["question"] for ex in ds]
    answers = [ex["answer"] for ex in ds]
    return questions, answers


def parse_gsm8k_answer(answer: str) -> str | None:
    """Extract the final numeric answer after '####' from a GSM8K answer string."""
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return None


# ---------------------------------------------------------------------------
# Pre-computation / caching
# ---------------------------------------------------------------------------

def precompute_features(
    sae,
    model,
    prepend_bos: bool,
    hook_name: str,
    prompts: list[str],
    questions: list[str],
    answers: list[str],
    max_length: int = 256,
    batch_size: int = 8,
    save_path: str | None = None,
) -> dict:
    """Run all prompts through model + SAE in batches, returning variable-length
    tensors trimmed to each sample's actual token count (no padding stored).

    Returns a dict with keys: tokens, acts, masks, questions, answers.
    Optionally saves to a .pt file for later reuse.
    """
    all_tokens: list[torch.Tensor] = []
    all_acts: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Precomputing SAE features"):
        batch_prompts = prompts[i : i + batch_size]
        tokens, feature_acts, mask = get_feature_sequences_batch(
            sae, model, prepend_bos, hook_name, batch_prompts, max_length,
        )
        for j in range(tokens.shape[0]):
            seq_len = int(mask[j].sum().item())
            all_tokens.append(tokens[j, :seq_len].cpu())
            all_acts.append(feature_acts[j, :seq_len].cpu().float())
            all_masks.append(mask[j, :seq_len].cpu())

        del tokens, feature_acts, mask
        gc.collect()
        torch.cuda.empty_cache()

    data = {
        "tokens": all_tokens,
        "acts": all_acts,
        "masks": all_masks,
        "questions": questions,
        "answers": answers,
    }

    if save_path:
        torch.save(data, save_path)
        print(f"Saved {len(all_tokens)} samples to {save_path}")

    return data


def load_or_precompute(
    sae,
    model,
    prepend_bos: bool,
    hook_name: str,
    cache_path: str,
    split: str = "train",
    text_field: str = "question",
    max_length: int = 256,
    batch_size: int = 8,
) -> dict:
    """Load cached feature activations from disk, or precompute and save them."""
    if Path(cache_path).exists():
        print(f"Loading cached features from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        if "questions" not in data:
            questions, answers = load_gsm8k_raw(split=split)
            data["questions"] = questions
            data["answers"] = answers
        return data

    questions, answers = load_gsm8k_raw(split=split)
    prompts = questions if text_field == "question" else answers
    return precompute_features(
        sae, model, prepend_bos, hook_name, prompts,
        questions=questions, answers=answers,
        max_length=max_length, batch_size=batch_size, save_path=cache_path,
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset + DataLoader
# ---------------------------------------------------------------------------

class GSM8KFeatureDataset(Dataset):
    """Holds variable-length (token_ids, feature_acts, mask) tuples
    plus the original GSM8K question and answer strings for inspection."""

    def __init__(
        self,
        token_ids: list[torch.Tensor],
        feature_acts: list[torch.Tensor],
        masks: list[torch.Tensor],
        questions: list[str] | None = None,
        answers: list[str] | None = None,
    ):
        self.token_ids = token_ids
        self.feature_acts = feature_acts
        self.masks = masks
        self.questions = questions or [""] * len(token_ids)
        self.answers = answers or [""] * len(token_ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.token_ids[idx], self.feature_acts[idx], self.masks[idx]

    @property
    def num_features(self) -> int:
        return self.feature_acts[0].shape[-1]

    def get_example(self, idx: int) -> dict:
        """Return a full example including original text (not just tensors)."""
        return {
            "index": idx,
            "question": self.questions[idx],
            "answer": self.answers[idx],
            "final_answer": parse_gsm8k_answer(self.answers[idx]),
            "token_ids": self.token_ids[idx],
            "feature_acts": self.feature_acts[idx],
            "mask": self.masks[idx],
            "seq_len": int(self.masks[idx].sum().item()),
        }


def collate_fn(batch):
    """Pad variable-length samples to the longest sequence in the batch."""
    token_ids, feature_acts, masks = zip(*batch)

    max_len = max(t.shape[0] for t in token_ids)
    num_features = feature_acts[0].shape[-1]
    bs = len(batch)

    padded_tokens = torch.zeros(bs, max_len, dtype=token_ids[0].dtype)
    padded_acts = torch.zeros(bs, max_len, num_features, dtype=feature_acts[0].dtype)
    padded_masks = torch.zeros(bs, max_len, dtype=torch.bool)

    for i, (t, a, m) in enumerate(zip(token_ids, feature_acts, masks)):
        seq_len = t.shape[0]
        padded_tokens[i, :seq_len] = t
        padded_acts[i, :seq_len] = a
        padded_masks[i, :seq_len] = m

    return padded_tokens, padded_acts, padded_masks


def create_dataloaders_from_split(
    train_ds: GSM8KFeatureDataset,
    test_ds: GSM8KFeatureDataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test DataLoaders from separate datasets."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, test_loader


def create_dataloaders_random(
    token_ids: list[torch.Tensor],
    feature_acts: list[torch.Tensor],
    masks: list[torch.Tensor],
    train_ratio: float = 0.8,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test DataLoaders via random split of a single pool."""
    dataset = GSM8KFeatureDataset(token_ids, feature_acts, masks)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Visualization / summary
# ---------------------------------------------------------------------------

def print_dataset_summary(ds: GSM8KFeatureDataset, name: str = "Dataset"):
    """Print summary statistics for a GSM8KFeatureDataset."""
    seq_lens = np.array([int(m.sum().item()) for m in ds.masks])
    num_features = ds.num_features

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Samples:          {len(ds)}")
    print(f"  Feature dim:      {num_features}")
    print(f"  Seq length:       min={seq_lens.min()}, max={seq_lens.max()}, "
          f"mean={seq_lens.mean():.1f}, median={np.median(seq_lens):.0f}")
    clipped = (seq_lens == seq_lens.max()).sum()
    if clipped > 0:
        print(f"  At max length:    {clipped} ({100 * clipped / len(ds):.1f}%)")

    n_sample = min(500, len(ds))
    all_nonzero = []
    total_elements = 0
    total_zeros = 0
    active_counts = []
    global_max = 0.0

    for i in range(n_sample):
        acts = ds.feature_acts[i]
        total_elements += acts.numel()
        total_zeros += (acts == 0).sum().item()
        nz = acts[acts > 0]
        if nz.numel() > 0:
            all_nonzero.append(nz)
            global_max = max(global_max, nz.max().item())
        per_token = (acts > 0).sum(dim=-1).float()
        active_counts.append(per_token[per_token > 0])

    sparsity = total_zeros / total_elements if total_elements > 0 else 0.0
    print(f"\n  Feature activations (sampled from first {n_sample} examples):")
    print(f"    Sparsity:       {100 * sparsity:.1f}% zeros")
    if all_nonzero:
        combined = torch.cat(all_nonzero)
        print(f"    Non-zero mean:  {combined.mean().item():.4f}")
        print(f"    Non-zero max:   {global_max:.4f}")
    if active_counts:
        combined_active = torch.cat(active_counts)
        print(f"    Active features/token: mean={combined_active.mean().item():.1f}, "
              f"max={combined_active.max().item():.0f}")

    if ds.questions:
        q_lens = [len(q.split()) for q in ds.questions if q]
        if q_lens:
            print(f"\n  Question length (words): min={min(q_lens)}, max={max(q_lens)}, "
                  f"mean={np.mean(q_lens):.1f}")

    final_answers = [parse_gsm8k_answer(a) for a in ds.answers if a]
    valid_answers = [a for a in final_answers if a is not None]
    if valid_answers:
        print(f"  Parsed final answers: {len(valid_answers)}/{len(ds)}")

    print(f"{'=' * 60}\n")


def print_sample(ds: GSM8KFeatureDataset, idx: int, top_k: int = 10, feature_catalog=None):
    """Print a single example with its question, answer, and top active features."""
    ex = ds.get_example(idx)

    print(f"\n{'─' * 60}")
    print(f"  Sample {idx}  (seq_len={ex['seq_len']})")
    print(f"{'─' * 60}")
    print(f"\n  Question:\n    {ex['question']}")
    print(f"\n  Answer:\n    {ex['answer']}")
    if ex["final_answer"]:
        print(f"\n  Final answer: {ex['final_answer']}")

    acts = ex["feature_acts"]
    max_acts, _ = acts.max(dim=0)
    top_vals, top_ids = torch.topk(max_acts, k=min(top_k, (max_acts > 0).sum().item()))

    print(f"\n  Top {len(top_vals)} features (max activation across tokens):")
    for rank, (fid, val) in enumerate(zip(top_ids, top_vals), 1):
        fid_int = fid.item()
        desc = ""
        if feature_catalog is not None:
            row = feature_catalog[feature_catalog["feature_id"] == fid_int]
            if len(row) > 0 and not row.iloc[0].isna().get("feature_desc", True):
                desc = f"  — {row.iloc[0]['feature_desc']}"
        print(f"    {rank:>2}. feature={fid_int:<7} activation={val.item():>9.4f}{desc}")

    print(f"{'─' * 60}\n")


def print_samples(ds: GSM8KFeatureDataset, indices: list[int] | None = None,
                  n: int = 3, top_k: int = 10, feature_catalog=None):
    """Print multiple samples. Picks random indices if none provided."""
    if indices is None:
        indices = torch.randperm(len(ds))[:n].tolist()
    for idx in indices:
        print_sample(ds, idx, top_k=top_k, feature_catalog=feature_catalog)


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def load_gsm8k_dataloaders(
    sae,
    model,
    prepend_bos: bool,
    hook_name: str,
    cache_dir: str = ".",
    text_field: str = "question",
    max_length: int = 256,
    batch_size: int = 32,
    compute_batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, GSM8KFeatureDataset, GSM8KFeatureDataset]:
    """End-to-end: load GSM8K, precompute SAE features (cached to disk),
    and return train / test DataLoaders + Datasets using GSM8K's native splits.

    Returns
    -------
    train_loader, test_loader, train_dataset, test_dataset

    Usage
    -----
    >>> train_loader, test_loader, train_ds, test_ds = load_gsm8k_dataloaders(
    ...     sae, model, prepend_bos, hook_name,
    ...     cache_dir="/content/drive/MyDrive/sae_cache",
    ... )
    >>> # Inspect the raw data
    >>> print_dataset_summary(train_ds, "Train")
    >>> print_sample(train_ds, 0, feature_catalog=feature_catalog)
    >>>
    >>> # Train loop
    >>> for token_ids, feature_acts, mask in train_loader:
    ...     # token_ids:    [B, T]          int64
    ...     # feature_acts: [B, T, 16384]   float32
    ...     # mask:         [B, T]          bool
    ...     pass
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    train_data = load_or_precompute(
        sae, model, prepend_bos, hook_name,
        str(cache / "gsm8k_train_features.pt"),
        split="train", text_field=text_field,
        max_length=max_length, batch_size=compute_batch_size,
    )
    test_data = load_or_precompute(
        sae, model, prepend_bos, hook_name,
        str(cache / "gsm8k_test_features.pt"),
        split="test", text_field=text_field,
        max_length=max_length, batch_size=compute_batch_size,
    )

    train_ds = GSM8KFeatureDataset(
        train_data["tokens"], train_data["acts"], train_data["masks"],
        train_data["questions"], train_data["answers"],
    )
    test_ds = GSM8KFeatureDataset(
        test_data["tokens"], test_data["acts"], test_data["masks"],
        test_data["questions"], test_data["answers"],
    )

    train_loader, test_loader = create_dataloaders_from_split(
        train_ds, test_ds, batch_size=batch_size, num_workers=num_workers,
    )

    return train_loader, test_loader, train_ds, test_ds
