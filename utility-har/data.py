"""
UCI HAR (Human Activity Recognition Using Smartphones) loader for LightSplit.

Tabular representation: 561 precomputed time/frequency-domain features per
2.56 s sliding window. 6 activity classes, 30 subjects. The official split is
subject-disjoint (21 train subjects / 9 test subjects), which gives us a clean
generalization measurement and, separately, a per-subject label for the
re-identification attack.

The dataset is downloaded and cached under <data_root> on first use.
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# The classic UCI ML-repo mirror extracts straight to "UCI HAR Dataset/".
# The newer /static/public/ endpoint wraps it in an outer zip (handled below).
_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
]

ACTIVITY_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


@dataclass
class HARData:
    X_train: np.ndarray      # (N_tr, 561) float32
    y_train: np.ndarray      # (N_tr,)     int64 in [0, 5]
    subj_train: np.ndarray   # (N_tr,)     int64 subject id (1..30)
    X_test: np.ndarray
    y_test: np.ndarray
    subj_test: np.ndarray
    feature_names: List[str]

    @property
    def num_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def num_classes(self) -> int:
        return 6


def _default_root() -> Path:
    """HAR_DATA, if set, else ./data next to this file. Used by every script that omits --data_root."""
    return Path(os.environ["HAR_DATA"]) if os.environ.get("HAR_DATA") else Path(__file__).resolve().parent / "data"


def _extract(raw: bytes, data_root: Path) -> None:
    """Extract a (possibly nested) zip blob into data_root."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extractall(data_root)
    # Some mirrors wrap the dataset in an outer zip; unwrap any inner copy.
    for inner in data_root.rglob("UCI HAR Dataset.zip"):
        with zipfile.ZipFile(inner) as z:
            z.extractall(data_root)
        inner.unlink(missing_ok=True)


def _download(data_root: Path) -> Path:
    root = data_root / "UCI HAR Dataset"
    if (root / "train" / "X_train.txt").exists():
        return root
    data_root.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in _URLS:
        try:
            print(f"[data] downloading {url}")
            raw = urllib.request.urlopen(url, timeout=120).read()
            _extract(raw, data_root)
            if (root / "train" / "X_train.txt").exists():
                return root
        except Exception as e:  # noqa: BLE001 - try next mirror
            last_err = e
            print(f"[data] mirror failed: {e}")
    raise RuntimeError(f"Could not obtain UCI HAR dataset: {last_err}")


def load_har(data_root: str | None = None) -> HARData:
    """Load UCI HAR (raw, unstandardized features). Cached as .npz after first read."""
    root_dir = Path(data_root) if data_root else _default_root()
    cache = root_dir / "har_cache.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return HARData(
            d["X_train"], d["y_train"], d["subj_train"],
            d["X_test"], d["y_test"], d["subj_test"],
            list(d["feature_names"]),
        )

    base = _download(root_dir)

    def mat(p: Path) -> np.ndarray:
        return np.loadtxt(p, dtype=np.float32)

    def vec(p: Path) -> np.ndarray:
        return np.loadtxt(p, dtype=np.int64)

    X_train = mat(base / "train" / "X_train.txt")
    y_train = vec(base / "train" / "y_train.txt") - 1          # 1..6 -> 0..5
    subj_train = vec(base / "train" / "subject_train.txt")
    X_test = mat(base / "test" / "X_test.txt")
    y_test = vec(base / "test" / "y_test.txt") - 1
    subj_test = vec(base / "test" / "subject_test.txt")
    feature_names = [
        line.split(maxsplit=1)[1] if " " in line else line
        for line in (base / "features.txt").read_text().splitlines()
    ]

    np.savez(
        cache,
        X_train=X_train, y_train=y_train, subj_train=subj_train,
        X_test=X_test, y_test=y_test, subj_test=subj_test,
        feature_names=np.array(feature_names, dtype=object),
    )
    return HARData(X_train, y_train, subj_train, X_test, y_test, subj_test, feature_names)


def fit_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std fit on the data the model is allowed to see (D_in)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def member_split(
    stratify: np.ndarray, holdout_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices of the training pool into D_in (members) / D_out (non-members).

    Stratified by `stratify` (activity label) so both halves share the same
    class distribution; members and non-members are then drawn from the SAME
    distribution, which isolates *memorization* in the membership attack rather
    than train/test distribution shift.
    """
    rng = np.random.default_rng(seed)
    in_idx: List[int] = []
    out_idx: List[int] = []
    for c in np.unique(stratify):
        idx = np.where(stratify == c)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * (1.0 - holdout_frac)))
        in_idx.extend(idx[:cut].tolist())
        out_idx.extend(idx[cut:].tolist())
    return np.array(sorted(in_idx)), np.array(sorted(out_idx))


def dirichlet_partition(
    y: np.ndarray, num_clients: int, alpha: float, seed: int
) -> List[np.ndarray]:
    """Partition sample indices across clients by a Dirichlet(alpha) over labels.

    For each class, draw a proportion vector from Dir(alpha) and split that class's
    samples accordingly. Small alpha -> highly non-IID (each client sees few classes);
    large alpha -> near-IID. Verbatim logic of dirichlet_split in train_split.py.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(y.max()) + 1
    client_idx: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx, cuts)):
            client_idx[cid].extend(part.tolist())
    return [np.array(sorted(c)) for c in client_idx]


# ── Canonical tabular MIA benchmarks (Purchase-100 / Texas-100) ───────────────
# Both from the Shokri lab dataset repo (https://github.com/privacytrustlab/datasets),
# the same lab as Privacy Meter. Binary features, 100 classes, models overfit hard
# (unlike HAR), so membership inference is a real threat here. No subject / re-id
# axis (records are not per-person sequences); the MIA + reconstruction attacks apply.
_PURCHASE_URL = "https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_purchase.tgz"
_TEXAS_URL = "https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_texas.tgz"


@dataclass
class TabularData:
    """Dataset-agnostic container (mirrors HARData's X/y fields; no subjects)."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    name: str = "tabular"
    # HAR-compatibility shims so attack/train code can be written once.
    subj_train: np.ndarray | None = None
    subj_test: np.ndarray | None = None

    @property
    def num_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def num_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_test.max())) + 1


def _split_train_test(X, y, test_frac, seed) -> Tuple[np.ndarray, ...]:
    """Stratified random split (no canonical split ships with these datasets)."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * (1.0 - test_frac)))
        tr.extend(idx[:cut].tolist())
        te.extend(idx[cut:].tolist())
    tr, te = np.array(sorted(tr)), np.array(sorted(te))
    return X[tr], y[tr], X[te], y[te]


def _download_tgz(url: str, marker: Path, data_root: Path) -> None:
    """Download+extract a .tgz from the Shokri datasets repo if `marker` is absent."""
    if marker.exists():
        return
    import tarfile
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {url}")
    raw = urllib.request.urlopen(url, timeout=300).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
        t.extractall(data_root)


def load_purchase100(data_root: str | None = None, test_frac: float = 0.2,
                     seed: int = 0) -> TabularData:
    """Purchase-100: 197,324 records, 600 binary features, 100 classes."""
    root = Path(data_root) if data_root else _default_root()
    cache = root / "purchase100_cache.npz"
    if cache.exists():
        d = np.load(cache)
        return TabularData(d["X_train"], d["y_train"], d["X_test"], d["y_test"], "purchase100")
    _download_tgz(_PURCHASE_URL, root / "dataset_purchase", root)
    arr = pd.read_csv(root / "dataset_purchase", header=None, dtype=np.int16).values
    y = (arr[:, 0] - 1).astype(np.int64)              # 1..100 -> 0..99
    X = arr[:, 1:].astype(np.float32)                 # 600 binary features
    Xtr, ytr, Xte, yte = _split_train_test(X, y, test_frac, seed)
    np.savez_compressed(cache, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    return TabularData(Xtr, ytr, Xte, yte, "purchase100")


def load_texas100(data_root: str | None = None, test_frac: float = 0.2,
                  seed: int = 0) -> TabularData:
    """Texas-100: 67,330 records, 6,169 binary features, 100 classes."""
    root = Path(data_root) if data_root else _default_root()
    cache = root / "texas100_cache.npz"
    if cache.exists():
        d = np.load(cache)
        return TabularData(d["X_train"], d["y_train"], d["X_test"], d["y_test"], "texas100")
    _download_tgz(_TEXAS_URL, root / "texas" / "100" / "feats", root)
    X = pd.read_csv(root / "texas" / "100" / "feats", header=None, dtype=np.int8).values.astype(np.float32)
    y = (pd.read_csv(root / "texas" / "100" / "labels", header=None, dtype=np.int64).values.ravel() - 1)
    Xtr, ytr, Xte, yte = _split_train_test(X, y, test_frac, seed)
    np.savez_compressed(cache, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    return TabularData(Xtr, ytr, Xte, yte, "texas100")


def load_tabular(dataset: str, data_root: str | None = None):
    """Dispatch to a dataset loader by name (har | purchase100 | texas100)."""
    d = dataset.lower()
    if d in ("har", "uci_har"):
        return load_har(data_root)
    if d in ("purchase100", "purchase"):
        return load_purchase100(data_root)
    if d in ("texas100", "texas"):
        return load_texas100(data_root)
    raise ValueError(f"unknown dataset: {dataset}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        d = load_tabular(sys.argv[1])
        print(f"{sys.argv[1]}: train {d.X_train.shape} test {d.X_test.shape} "
              f"feats {d.num_features} classes {d.num_classes}")
        raise SystemExit
    d = load_har()
    print(f"train {d.X_train.shape}  test {d.X_test.shape}  "
          f"features {d.num_features}  classes {d.num_classes}")
    print(f"train subjects {sorted(set(d.subj_train.tolist()))}")
    print(f"test  subjects {sorted(set(d.subj_test.tolist()))}")
