"""
Multi-client, non-IID utility sweep for tabular HAR (the accuracy/communication
comparison table). Same U-shaped split-learning model as the privacy experiments;
the data is Dirichlet-partitioned across clients by activity label, and a single
shared model is trained round-robin over the clients (this is what makes split
learning robust to non-IID: the server backbone aggregates all clients' data).

Methods (--liftback): none (vanilla SL) | codec (learned codec) | topk_sparse
| fixed (LightSplit-F) | learned (LightSplit-L). WCC (--wcc_lambda) applies to
the LightSplit modes only.

Utility is reported on the official subject-disjoint test set at the epoch chosen
by a validation split carved from the training pool (no test-set selection).

Example:
    python train_noniid.py --liftback learned --wcc_lambda 0.1 --alpha 0.1 \
        --num_clients 5 --out results_noniid/learned_w0.1_a0.1
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import dirichlet_partition, fit_standardizer, load_har, member_split
from lightsplit import HARSplitModel, wcc_loss

WCC_MODES = {"fixed", "learned"}   # WCC is a LightSplit component only


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, X, y) -> float:
    model.eval()
    return float((model(X)[0].argmax(1) == y).float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--liftback", choices=["none", "fixed", "learned", "codec", "topk_sparse"],
                    default="learned")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--wcc_lambda", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.1, help="Dirichlet non-IID concentration")
    ap.add_argument("--num_clients", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.liftback not in WCC_MODES and args.wcc_lambda > 0:
        raise SystemExit(f"WCC only applies to {WCC_MODES}, not '{args.liftback}'")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Data: carve a val set, Dirichlet-partition the rest across clients ──
    har = load_har(args.data_root)
    fit_idx, val_idx = member_split(har.y_train, args.val_frac, args.seed + 1)
    mean, std = fit_standardizer(har.X_train[fit_idx])

    def prep(X):
        return torch.from_numpy(((X - mean) / std).astype(np.float32)).to(device)

    X_fit = prep(har.X_train[fit_idx]); y_fit = torch.from_numpy(har.y_train[fit_idx]).long().to(device)
    X_val = prep(har.X_train[val_idx]); y_val = torch.from_numpy(har.y_train[val_idx]).long().to(device)
    X_te = prep(har.X_test); y_te = torch.from_numpy(har.y_test).long().to(device)

    parts = dirichlet_partition(har.y_train[fit_idx], args.num_clients, args.alpha, args.seed)
    client_loaders = []
    for pos in parts:
        if len(pos) < 2:
            continue
        ds = TensorDataset(X_fit[pos], y_fit[pos])
        client_loaders.append(DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=True, drop_last=len(pos) >= args.batch_size))
    sizes = [len(p) for p in parts]
    print(f"[noniid] {args.liftback} wcc={args.wcc_lambda} alpha={args.alpha} "
          f"clients={len(client_loaders)} sizes={sizes}")

    # ── Model (shared across clients) ──
    model = HARSplitModel(in_features=har.num_features, d=args.d, k=args.k,
                          num_classes=har.num_classes, liftback=args.liftback,
                          hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    comm = model.comm_floats_per_sample
    print(f"         comm={comm:.0f} floats/sample  CR={model.compression_ratio:.1f}x")

    best_val, best_state, best_epoch = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for loader in client_loaders:                 # round-robin over clients
            for xb, yb in loader:
                logits, ztilde = model(xb)
                loss = F.cross_entropy(logits, yb)
                if args.wcc_lambda > 0:
                    loss = loss + args.wcc_lambda * wcc_loss(ztilde, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        val_acc = evaluate(model, X_val, y_val)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, ep
            best_state = copy.deepcopy(model.state_dict())
        if ep % 20 == 0 or ep == args.epochs:
            print(f"  epoch {ep:3d}  val {val_acc:.4f}  best {best_val:.4f}")

    final_test = evaluate(model, X_te, y_te)
    model.load_state_dict(best_state)
    test_acc = evaluate(model, X_te, y_te)

    summary = {
        "liftback": args.liftback, "wcc_lambda": args.wcc_lambda,
        "alpha": args.alpha, "num_clients": args.num_clients,
        "k": model.k, "comm_floats_per_sample": comm,
        "compression_ratio": model.compression_ratio,
        "test_acc": test_acc, "final_test_acc": final_test,
        "best_val_acc": best_val, "best_epoch": best_epoch, "seed": args.seed,
        "client_sizes": sizes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[noniid] done: test(val-sel) {test_acc:.4f}  comm {comm:.0f}  -> {out}")


if __name__ == "__main__":
    main()
