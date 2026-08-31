"""
Train LightSplit on tabular UCI HAR.

Data discipline (so the attacks downstream are sound):
  * The training pool is split into D_in (members) and D_out (non-members),
    stratified by activity so both are the SAME distribution. The membership
    attack then measures memorization, not distribution shift.
  * D_in is split again into a fit set (actually trained on) and a validation
    set used ONLY for model selection / early stopping. The fit set is the
    member set the attacks use.
  * Utility is reported on the official, subject-disjoint test set, at the epoch
    chosen by validation. The test set is NEVER used for selection (no oracle).

Example:
    python train_har.py --k 32 --liftback learned --wcc_lambda 0.1 --out results/k32_w0.1
    python train_har.py --liftback none --out results/control     # no-defense control
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

from data import fit_standardizer, load_har, member_split
from lightsplit import HARSplitModel, wcc_loss


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: HARSplitModel, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    logits, _ = model(X)
    return float((logits.argmax(1) == y).float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32, help="compressed dim (payload size)")
    ap.add_argument("--d", type=int, default=256, help="cut-layer / smashed-data dim")
    ap.add_argument("--liftback",
                    choices=["fixed", "learned", "none", "codec", "topk_sparse"], default="learned",
                    help="'none' = no-defense control; 'codec'/'topk_sparse' = compression baselines")
    ap.add_argument("--hidden", type=int, default=256, help="learned lift-back width")
    ap.add_argument("--wcc_lambda", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--holdout_frac", type=float, default=0.5,
                    help="fraction of train pool held out as non-members")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="fraction of members held out for model selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--dp_clip", type=float, default=0.0,
                    help="per-sample L2 clip on the transmitted payload; 0 disables DP")
    ap.add_argument("--dp_sigma", type=float, default=0.0,
                    help="noise multiplier; payload noise std = dp_sigma * dp_clip")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── 3-way split of the training pool: members (fit) / val / non-members ──
    har = load_har(args.data_root)
    in_idx, out_idx = member_split(har.y_train, args.holdout_frac, args.seed)
    # carve a validation set out of the members; the rest are actually trained on
    pos_fit, pos_val = member_split(har.y_train[in_idx], args.val_frac, args.seed + 1)
    member_idx = in_idx[pos_fit]      # trained on  -> the member set for MIA
    val_idx = in_idx[pos_val]         # model selection only (never an MIA member)
    nonmember_idx = out_idx           # never trained on -> non-members for MIA

    mean, std = fit_standardizer(har.X_train[member_idx])   # fit on what we train on

    def prep(X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((X - mean) / std).astype(np.float32)).to(device)

    X_fit = prep(har.X_train[member_idx])
    y_fit = torch.from_numpy(har.y_train[member_idx]).long().to(device)
    X_val = prep(har.X_train[val_idx])
    y_val = torch.from_numpy(har.y_train[val_idx]).long().to(device)
    X_te = prep(har.X_test)
    y_te = torch.from_numpy(har.y_test).long().to(device)

    loader = DataLoader(
        TensorDataset(X_fit, y_fit),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    # ── Model ──
    model = HARSplitModel(
        in_features=har.num_features, d=args.d, k=args.k,
        num_classes=har.num_classes, liftback=args.liftback, hidden=args.hidden,
        dp_clip=args.dp_clip, dp_sigma=args.dp_sigma,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cr = model.compression_ratio
    print(f"[train] device={device} d={args.d} k={model.k} (CR={cr:.1f}x) "
          f"liftback={args.liftback} wcc={args.wcc_lambda} "
          f"dp_clip={args.dp_clip} dp_sigma={args.dp_sigma} | "
          f"members={len(member_idx)} val={len(val_idx)} non-members={len(nonmember_idx)}")

    best_val, best_state, best_epoch = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
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
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  epoch {ep:3d}  val_acc {val_acc:.4f}  best_val {best_val:.4f} @ {best_epoch}")

    final_test = evaluate(model, X_te, y_te)
    # Deploy / report the validation-selected model (selection never saw the test set).
    model.load_state_dict(best_state)
    val_selected_test = evaluate(model, X_te, y_te)
    train_acc = evaluate(model, X_fit, y_fit)

    # ── Persist the val-selected model + everything the attacks need ──
    torch.save(
        {"state_dict": model.state_dict(),
         "config": {"in_features": har.num_features, "d": args.d, "k": model.k,
                    "num_classes": har.num_classes, "liftback": args.liftback,
                    "hidden": args.hidden,
                    "dp_clip": args.dp_clip, "dp_sigma": args.dp_sigma}},
        out / "ckpt.pt",
    )
    np.savez(out / "meta.npz", mean=mean, std=std,
             member_idx=member_idx, nonmember_idx=nonmember_idx, val_idx=val_idx)
    summary = {
        "k": model.k, "d": args.d, "compression_ratio": cr,
        "liftback": args.liftback, "wcc_lambda": args.wcc_lambda,
        "train_acc": train_acc,
        "test_acc": val_selected_test,     # headline utility (val-selected, no test peeking)
        "final_test_acc": final_test,      # for reference / collapse detection
        "best_val_acc": best_val, "best_epoch": best_epoch,
        "gen_gap": train_acc - val_selected_test, "seed": args.seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done: train {train_acc:.4f}  test(val-sel) {val_selected_test:.4f}  "
          f"test(final) {final_test:.4f}  gap {train_acc - val_selected_test:.4f}  -> {out}")


if __name__ == "__main__":
    main()
