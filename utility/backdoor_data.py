"""BadNets data wrappers for the split-learning backdoor experiment.

Trigger pattern, PoisonedSubset, TriggeredDataset, and get_backdoor_loaders
ported from a reference badnets_attack implementation.
Reuses dirichlet_split, make_client_loaders, get_datasets, get_targets,
DATASET_STATS, MODEL_CONFIGS from train_split.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from train_split import (
    DATASET_STATS,
    MODEL_CONFIGS,
    dirichlet_split,
    get_datasets,
    get_targets,
    make_client_loaders,
)


def make_trigger_pattern(
    pattern: str = "square",
    size: int = 3,
    img_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (mask [1,H,W], pattern [3,H,W]) in [0,1] pixel space."""
    mask = torch.zeros(1, img_size, img_size)
    pat = torch.zeros(3, img_size, img_size)

    if pattern == "square":
        mask[:, -size:, -size:] = 1.0
        pat[:, -size:, -size:] = 1.0
    elif pattern == "cross":
        cx, cy = img_size - size // 2 - 1, img_size - size // 2 - 1
        for i in range(size):
            mask[:, cx - size // 2 + i, cy] = 1.0
            mask[:, cx, cy - size // 2 + i] = 1.0
            pat[:, cx - size // 2 + i, cy] = 1.0
            pat[:, cx, cy - size // 2 + i] = 1.0
    elif pattern == "l_shape":
        mask[:, -size:, -1] = 1.0
        mask[:, -1, -size:] = 1.0
        pat[:, -size:, -1] = 1.0
        pat[:, -1, -size:] = 1.0
    else:
        raise ValueError(f"Unknown trigger pattern: {pattern}")

    return mask, pat


def apply_trigger(
    x: torch.Tensor,
    mask: torch.Tensor,
    pattern: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> torch.Tensor:
    device = x.device
    m = mask.to(device)
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t = torch.tensor(std, device=device).view(3, 1, 1)
    p_norm = (pattern.to(device) - mean_t) / std_t
    return x * (1 - m) + p_norm * m


class PoisonedSubset(Dataset):
    """Wraps a Subset; deterministically poisons `poison_ratio` of its samples."""

    def __init__(
        self,
        subset: Subset,
        poison_ratio: float,
        target_label: int,
        trigger_mask: torch.Tensor,
        trigger_pattern: torch.Tensor,
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
        seed: int = 42,
    ):
        self.subset = subset
        self.target_label = target_label
        self.trigger_mask = trigger_mask
        self.trigger_pattern = trigger_pattern
        self.mean = mean
        self.std = std

        n = len(subset)
        rng = np.random.default_rng(seed)
        n_poison = int(n * poison_ratio)
        all_idx = np.arange(n)
        rng.shuffle(all_idx)
        self.poison_set = set(all_idx[:n_poison].tolist())

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if idx in self.poison_set:
            x = apply_trigger(x, self.trigger_mask, self.trigger_pattern,
                              self.mean, self.std)
            y = self.target_label
        return x, y


class TriggeredDataset(Dataset):
    """All non-target-class test samples with trigger applied (used for ASR)."""

    def __init__(
        self,
        dataset: Dataset,
        target_label: int,
        trigger_mask: torch.Tensor,
        trigger_pattern: torch.Tensor,
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
    ):
        self.dataset = dataset
        self.target_label = target_label
        self.trigger_mask = trigger_mask
        self.trigger_pattern = trigger_pattern
        self.mean = mean
        self.std = std

        targets = get_targets(dataset) if hasattr(dataset, "targets") else None
        if targets is not None:
            self.valid_indices = [i for i, t in enumerate(targets) if t != target_label]
        else:
            self.valid_indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x, _y = self.dataset[real_idx]
        x = apply_trigger(x, self.trigger_mask, self.trigger_pattern,
                          self.mean, self.std)
        return x, self.target_label


def get_backdoor_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    num_clients: int,
    alpha: float,
    seed: int,
    dataset: str,
    malicious_ratio: float,
    poison_rate: float,
    target_label: int,
    trigger_pattern: str,
    trigger_size: int,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, List[int]]:
    """Build per-client train loaders (poisoned for malicious clients), per-client
    clean test loaders, and an ASR loader (all non-target test samples + trigger).
    """
    stats = DATASET_STATS[dataset]
    img_size = MODEL_CONFIGS[dataset]["img_size"]
    trigger_mask, trigger_pat = make_trigger_pattern(
        trigger_pattern, trigger_size, img_size)

    train_set, test_set = get_datasets(data_root, dataset)

    train_subsets = dirichlet_split(train_set, num_clients, alpha, seed=seed)
    test_subsets = dirichlet_split(test_set, num_clients, alpha, seed=seed + 1)

    num_malicious = max(1, int(num_clients * malicious_ratio))
    rng = np.random.default_rng(seed + 100)
    malicious_ids = sorted(rng.choice(num_clients, num_malicious, replace=False).tolist())

    train_datasets = []
    for cid, subset in enumerate(train_subsets):
        if cid in malicious_ids:
            train_datasets.append(PoisonedSubset(
                subset, poison_rate, target_label,
                trigger_mask, trigger_pat,
                stats["mean"], stats["std"],
                seed=seed + cid,
            ))
        else:
            train_datasets.append(subset)

    train_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, drop_last=True)
        for ds in train_datasets
    ]

    test_loaders = make_client_loaders(
        test_subsets, batch_size=batch_size, num_workers=num_workers, train=False)

    asr_dataset = TriggeredDataset(
        test_set, target_label, trigger_mask, trigger_pat,
        stats["mean"], stats["std"])
    asr_loader = DataLoader(
        asr_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loaders, test_loaders, asr_loader, malicious_ids
