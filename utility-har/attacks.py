"""
Privacy attacks against a trained LightSplit-HAR run.

Threat model: honest-but-curious server. It observes the transmitted payload
z~ = R^T f(x) for every sample (and, for LightSplit-L, owns the lift-back +
backbone weights). It does NOT own the client head f, the projection R, or the
labels. Each attack is the tabular analog of an image-pipeline attack:

  1. Membership inference (MIA)  -- "was this sample in training?"
       loss-based scoring on members vs non-members, drawn from the SAME
       distribution so the AUC reflects memorization. Reported alongside a
       no-defense control so resistance is not falsely credited to LightSplit.
       NOTE: loss-based MIA is a *worst-case* adversary that also needs labels
       and the full model; the server-only analog is weaker.
  2. Subject re-identification     -- "whose data is this?"
       a probe z~ -> subject id. Train/test split is GROUP-AWARE over contiguous
       (subject, activity) recording blocks, because UCI HAR windows overlap
       50% and a random split would leak near-duplicate windows.
  3. Input reconstruction          -- "recover the 561 features from z~"
       a decoder trained on auxiliary data (non-members), scored by per-feature
       normalized RMSE and the full R^2 distribution (not just the mean).

Usage:
    python attacks.py --run results/k32_w0.1            # all three
    python attacks.py --run results/k32_w0.1 --attack mia --seed 1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from data import load_har
from lightsplit import HARSplitModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Load a trained run ───────────────────────────────────────────────────────
def load_run(run_dir: Path):
    ckpt = torch.load(run_dir / "ckpt.pt", map_location=DEVICE, weights_only=False)
    model = HARSplitModel(**ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = np.load(run_dir / "meta.npz")
    return model, meta, ckpt["config"]


def split_views(meta) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return standardized (X, y, subject) for members / non-members / test."""
    har = load_har()
    mean, std = meta["mean"], meta["std"]
    m_idx, n_idx = meta["member_idx"], meta["nonmember_idx"]

    def std_x(X):
        return ((X - mean) / std).astype(np.float32)

    return {
        "in": (std_x(har.X_train[m_idx]), har.y_train[m_idx], har.subj_train[m_idx]),
        "out": (std_x(har.X_train[n_idx]), har.y_train[n_idx], har.subj_train[n_idx]),
        "test": (std_x(har.X_test), har.y_test, har.subj_test),
    }


def encode(model: HARSplitModel, X: np.ndarray) -> np.ndarray:
    """z~ that the server observes."""
    t = torch.from_numpy(X).to(DEVICE)
    return model.encode(t).cpu().numpy()


# ── A tiny supervised probe / decoder (shared by re-id and reconstruction) ───
def _mlp(sizes, out_act=None) -> nn.Module:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.BatchNorm1d(sizes[i + 1]), nn.ReLU(inplace=True)]
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


def _train_net(net, Xtr, Ttr, loss_fn, epochs=300, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)            # reproducible probe/decoder init + shuffling
    net = net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    Ttr = torch.as_tensor(Ttr).to(DEVICE)
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        net.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss_fn(net(Xtr[idx]), Ttr[idx]).backward()
            opt.step()
    net.eval()
    return net


def _tpr_at_fpr(labels, scores, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = fpr <= target_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


# ── 1. Membership inference ──────────────────────────────────────────────────
def attack_membership(model, views) -> Dict[str, float]:
    """Loss-based MIA: lower per-sample CE => more likely a member.

    AUC is threshold-free. We also report TPR at low FPR (the modern MIA
    standard); balanced accuracy is intentionally omitted because any single
    global threshold would have to be calibrated on held-out data.
    """
    def per_sample_loss(X, y):
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(X).to(DEVICE))
            return F.cross_entropy(
                logits, torch.from_numpy(y).long().to(DEVICE), reduction="none"
            ).cpu().numpy()

    Xi, yi, _ = views["in"]
    Xo, yo, _ = views["out"]
    loss_in = per_sample_loss(Xi, yi)
    loss_out = per_sample_loss(Xo, yo)

    scores = np.concatenate([-loss_in, -loss_out])           # high => member
    labels = np.concatenate([np.ones_like(loss_in), np.zeros_like(loss_out)])
    return {
        "mia_auc": float(roc_auc_score(labels, scores)),
        "mia_tpr@1%fpr": _tpr_at_fpr(labels, scores, 0.01),
        "mia_tpr@0.1%fpr": _tpr_at_fpr(labels, scores, 0.001),
        "mean_loss_member": float(loss_in.mean()),
        "mean_loss_nonmember": float(loss_out.mean()),
    }


# ── 2. Subject re-identification ─────────────────────────────────────────────
def _group_aware_split(subj, y, frac_train=0.7) -> Tuple[np.ndarray, np.ndarray]:
    """Split members into train/test by contiguous (subject, activity) blocks.

    Rows arrive in original (time) order, so consecutive same-(subject,activity)
    rows are a recording run. We send the first `frac_train` of each run to train
    and the rest to test, so 50%-overlapping neighbors do not straddle the split
    (only one boundary pair per run, vs. ~half of all rows under a random split).
    """
    groups: Dict[tuple, list] = defaultdict(list)
    for pos, (s, a) in enumerate(zip(subj.tolist(), y.tolist())):
        groups[(s, a)].append(pos)
    tr, te = [], []
    for idxs in groups.values():
        cut = int(round(frac_train * len(idxs)))
        tr += idxs[:cut]
        te += idxs[cut:]
    return np.array(tr), np.array(te)


def attack_reid(model, views, seed=0) -> Dict[str, float]:
    """Probe z~ -> subject id, over the members (all training subjects)."""
    X, y, subj = views["in"]
    Z = encode(model, X)
    uniq = np.unique(subj)
    remap = {s: i for i, s in enumerate(uniq)}
    t = np.array([remap[s] for s in subj])

    tr, te = _group_aware_split(subj, y, frac_train=0.7)
    net = _mlp([Z.shape[1], 256, 256, len(uniq)])
    net = _train_net(net, Z[tr], t[tr], nn.CrossEntropyLoss(), epochs=300, seed=seed)
    with torch.no_grad():
        pred = net(torch.as_tensor(Z[te], dtype=torch.float32, device=DEVICE)).argmax(1)
    acc = float((pred.cpu().numpy() == t[te]).mean())
    chance = 1.0 / len(uniq)
    return {"reid_acc": acc, "reid_chance": chance,
            "reid_lift_over_chance": acc / chance, "reid_num_subjects": int(len(uniq))}


# ── 3. Input reconstruction ──────────────────────────────────────────────────
def attack_reconstruction(model, views, config, seed=0) -> Dict[str, float]:
    """Decoder z~ -> x, trained on auxiliary data (non-members), tested on members."""
    Xo, _, _ = views["out"]          # attacker's auxiliary data
    Xi, _, _ = views["in"]           # target (members)
    Zo, Zi = encode(model, Xo), encode(model, Xi)

    # decoder input = the actual observed payload dim (k for projection/codec,
    # but d for raw/topk_sparse which transmit a sparse d-dim vector)
    dec = _mlp([Zo.shape[1], 256, 512, config["in_features"]])
    dec = _train_net(dec, Zo, Xo, nn.MSELoss(), epochs=400, seed=seed)
    with torch.no_grad():
        xhat = dec(torch.as_tensor(Zi, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    # Features are standardized (per-feature std ~= 1), so RMSE is in std units;
    # the "predict the mean" baseline gives RMSE 1.0 and R^2 0.0.
    err = xhat - Xi
    rmse = float(np.sqrt((err ** 2).mean()))
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((Xi - Xi.mean(axis=0)) ** 2).sum(axis=0) + 1e-8
    per_feat_r2 = 1.0 - ss_res / ss_tot
    return {
        "recon_rmse_std_units": rmse,
        "recon_r2_mean": float(per_feat_r2.mean()),
        "recon_r2_median": float(np.median(per_feat_r2)),
        "recon_r2_p25": float(np.percentile(per_feat_r2, 25)),
        "recon_r2_p75": float(np.percentile(per_feat_r2, 75)),
        "recon_frac_feats_r2>0.5": float((per_feat_r2 > 0.5).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="run dir from train_har.py")
    ap.add_argument("--attack", choices=["all", "mia", "reid", "recon"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run)
    model, meta, config = load_run(run_dir)
    views = split_views(meta)

    results: Dict[str, float] = {}
    chosen = ["mia", "reid", "recon"] if args.attack == "all" else [args.attack]
    for name in chosen:
        print(f"[attack] {name} ...")
        if name == "mia":
            results.update(attack_membership(model, views))
        elif name == "reid":
            results.update(attack_reid(model, views, seed=args.seed))
        elif name == "recon":
            results.update(attack_reconstruction(model, views, config, seed=args.seed))

    (run_dir / "attacks.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
