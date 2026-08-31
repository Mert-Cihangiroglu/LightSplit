"""
ImageNet-1k data loading for the U-shaped split-learning utility comparison.

Expects an ImageFolder layout on the (remote) data server:

    <data-root>/train/<wnid>/*.JPEG
    <data-root>/val/<wnid>/*.JPEG        (val pre-sorted into class folders)

Provides:
  * IID client partition (default) or Dirichlet non-IID partition of the train set
    across `num_clients`, matching the utility harness protocol.
  * a single full validation loader -> the headline top-1 / top-5 metric is the
    standard ImageNet val accuracy, not a per-client average.
  * a `--fake-data` synthetic path (random tensors) so the full train/eval pipeline
    can be smoke-tested without the dataset present.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Synthetic dataset (smoke test only) ─────────────────────────────────────
class FakeImageNet(Dataset):
    """Random 224x224 images with random labels; lets the pipeline run with no data."""
    def __init__(self, n: int, num_classes: int = 1000, seed: int = 0):
        self.n = n
        self.num_classes = num_classes
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(int(self.labels[i]) * 100003 + i)
        x = torch.randn(3, 224, 224, generator=g)
        return x, int(self.labels[i])


# ── Partitioning ────────────────────────────────────────────────────────────
def iid_split(dataset, num_clients: int, seed: int = 42) -> List[Subset]:
    """Random equal partition of the dataset across clients (IID)."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    parts = np.array_split(perm, num_clients)
    return [Subset(dataset, idxs.tolist()) for idxs in parts]


def dirichlet_split(dataset, num_clients: int, alpha: float, seed: int = 42) -> List[Subset]:
    """Dirichlet(alpha)-over-classes partition (non-IID).  Higher alpha -> more IID.

    Falls back to a label-free IID split if per-sample targets are not cheaply
    available (ImageFolder exposes `.targets`, so this is used only for non-IID).
    """
    targets = np.asarray(getattr(dataset, "targets", None))
    if targets is None or targets.ndim == 0:
        return iid_split(dataset, num_clients, seed=seed)
    num_classes = int(targets.max()) + 1
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        splits = np.clip((np.cumsum(proportions) * len(class_idx)).astype(int), 0, len(class_idx))
        for client_id, part in enumerate(np.split(class_idx, splits[:-1])):
            client_indices[client_id].extend(part.tolist())
    return [Subset(dataset, idxs) for idxs in client_indices]


def make_client_loaders(subsets: List[Subset], batch_size: int, num_workers: int,
                        train: bool) -> List[DataLoader]:
    return [
        DataLoader(s, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                   pin_memory=True, drop_last=train, persistent_workers=num_workers > 0)
        for s in subsets
    ]


# IID alpha sentinel: alpha >= this is treated as IID (matches the utility harness's 1e7).
IID_ALPHA = 1e7


def subsample_per_class(dataset, per_class: int, seed: int = 42):
    """Keep at most `per_class` training images per class, in place.

    Edits ImageFolder's own sample lists rather than wrapping in a Subset, so `.targets`
    survives and dirichlet_split still sees per-sample labels. Deterministic given `seed`.
    """
    targets = list(dataset.targets)
    by_cls = {}
    for i, t in enumerate(targets):
        by_cls.setdefault(int(t), []).append(i)
    rng = np.random.default_rng(seed)
    keep = []
    for t in sorted(by_cls):
        idxs = np.asarray(by_cls[t])
        if len(idxs) > per_class:
            idxs = rng.choice(idxs, per_class, replace=False)
        keep.extend(int(i) for i in idxs)
    keep.sort()
    dataset.samples = [dataset.samples[i] for i in keep]
    dataset.imgs = dataset.samples
    dataset.targets = [dataset.targets[i] for i in keep]
    return dataset


def get_loaders(data_root: str, batch_size: int, num_workers: int, num_clients: int,
                alpha: float = IID_ALPHA, seed: int = 42, fake: bool = False,
                train_subdir: str = "train", val_subdir: str = "val",
                train_per_class: int = 0,
                ) -> Tuple[List[DataLoader], DataLoader]:
    """Return (per-client train loaders, single full-val loader)."""
    if fake:
        train_set = FakeImageNet(n=max(num_clients * batch_size * 2, 256), seed=seed)
        val_set = FakeImageNet(n=batch_size * 2, seed=seed + 1)
    else:
        root = Path(data_root)
        train_set = datasets.ImageFolder(str(root / train_subdir), build_transforms(train=True))
        val_set = datasets.ImageFolder(str(root / val_subdir), build_transforms(train=False))
        if train_per_class > 0:
            n_before = len(train_set)
            subsample_per_class(train_set, train_per_class, seed=seed)
            print(f"[data] train subsampled to <={train_per_class}/class: "
                  f"{n_before:,} -> {len(train_set):,} images (val kept full: {len(val_set):,})")

    if alpha >= IID_ALPHA:
        train_subsets = iid_split(train_set, num_clients, seed=seed)
    else:
        train_subsets = dirichlet_split(train_set, num_clients, alpha, seed=seed)

    train_loaders = make_client_loaders(train_subsets, batch_size, num_workers, train=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
    return train_loaders, val_loader
