from __future__ import annotations

import gc
from pathlib import Path

import torch
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

def load_gsm8k(split: str = "train", text_field: str = "question") -> list[str]:
    """Load GSM8K from HuggingFace and return a list of prompt strings.

    Parameters
    ----------
    split : "train" (7473 examples) or "test" (1319 examples).
    text_field : which field to use as the prompt ("question" or "answer").
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [example[text_field] for example in ds]


# ---------------------------------------------------------------------------
# Pre-computation / caching
# ---------------------------------------------------------------------------

def precompute_features(
    sae,
    model,
    prepend_bos: bool,
    hook_name: str,
    prompts: list[str],
    max_length: int = 128,
    batch_size: int = 8,
    save_path: str | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Run all prompts through model + SAE in batches, returning variable-length
    tensors trimmed to each sample's actual token count (no padding stored).

    Returns lists of (token_ids, feature_acts, attention_mask) per sample,
    and optionally saves to a .pt file for later reuse.
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

    if save_path:
        torch.save({"tokens": all_tokens, "acts": all_acts, "masks": all_masks}, save_path)
        print(f"Saved {len(all_tokens)} samples to {save_path}")

    return all_tokens, all_acts, all_masks


def load_or_precompute(
    sae,
    model,
    prepend_bos: bool,
    hook_name: str,
    cache_path: str,
    split: str = "train",
    text_field: str = "question",
    max_length: int = 128,
    batch_size: int = 8,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Load cached feature activations from disk, or precompute and save them."""
    if Path(cache_path).exists():
        print(f"Loading cached features from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data["tokens"], data["acts"], data["masks"]

    prompts = load_gsm8k(split=split, text_field=text_field)
    return precompute_features(
        sae, model, prepend_bos, hook_name, prompts,
        max_length=max_length, batch_size=batch_size, save_path=cache_path,
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset + DataLoader
# ---------------------------------------------------------------------------

class GSM8KFeatureDataset(Dataset):
    """Holds variable-length (token_ids, feature_acts, mask) tuples."""

    def __init__(self, token_ids: list[torch.Tensor], feature_acts: list[torch.Tensor], masks: list[torch.Tensor]):
        self.token_ids = token_ids
        self.feature_acts = feature_acts
        self.masks = masks

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.token_ids[idx], self.feature_acts[idx], self.masks[idx]


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
    train_data: tuple[list, list, list],
    test_data: tuple[list, list, list],
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test DataLoaders from separate pre-computed splits."""
    train_ds = GSM8KFeatureDataset(*train_data)
    test_ds = GSM8KFeatureDataset(*test_data)

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
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def load_gsm8k_dataloaders(
    sae,
    model,
    prepend_bos: bool,
    hook_name: str,
    cache_dir: str = ".",
    text_field: str = "question",
    max_length: int = 128,
    batch_size: int = 32,
    compute_batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """End-to-end: load GSM8K, precompute SAE features (cached to disk),
    and return train / test DataLoaders using GSM8K's native splits.

    Usage
    -----
    >>> train_loader, test_loader = load_gsm8k_dataloaders(
    ...     sae, model, prepend_bos, hook_name,
    ...     cache_dir="/content/drive/MyDrive/sae_cache",
    ...     max_length=128, batch_size=32,
    ... )
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

    return create_dataloaders_from_split(train_data, test_data, batch_size=batch_size, num_workers=num_workers)
