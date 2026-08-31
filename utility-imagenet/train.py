#!/usr/bin/env python3
"""
Train one U-shaped split ResNet-50 on ImageNet-1k with one compression method.

Headline metric = standard ImageNet val top-1 / top-5 on the full validation set.
Recipe = the canonical ImageNet ResNet-50 recipe (SGD + momentum, cosine LR, label
smoothing, AMP), NOT the CIFAR Adam-1e-3 recipe -- an A* ImageNet number needs it.

Resume is two-level:
  * checkpoint-level: a per-run .pt (model/opt/sched/epoch/best) is saved each epoch
    and reloaded on restart, so a 12-22h run survives interruption (launch in tmux).
  * row-level: if the summary CSV already has a finished row for this run, skip it.

Usage (one method, one GPU):
  python train.py --data-root /srv/imagenet --mode fixed_random \
      --compression-ratio 8 --deep-head --epochs 90 --gpu 0

Smoke test the plumbing with no dataset:
  python train.py --fake-data --mode fixed_random --epochs 1 --num-clients 2 \
      --batch-size 8 --num-workers 0 --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import IID_ALPHA, get_loaders
from models import IMAGENET_CONFIG, activation_dim_for, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def topk_correct(logits: torch.Tensor, y: torch.Tensor, ks=(1, 5)) -> Dict[int, float]:
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.t().eq(y.view(1, -1))
    return {k: correct[:min(k, maxk)].reshape(-1).float().sum().item() for k in ks}


# ── one training epoch (round-robin over clients) ───────────────────────────
def train_epoch(model, train_loaders, optimizer, scaler, device,
                recon_lambda: float, wcc_lambda: float, label_smoothing: float,
                use_amp: bool) -> Dict[str, float]:
    model.train()
    tot_loss = tot_correct = tot = 0.0
    for client_id, loader in enumerate(train_loaders):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, info = model(x, client_id=client_id)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                if recon_lambda > 0 and "recon_loss" in info:
                    loss = loss + recon_lambda * info["recon_loss"]
                if wcc_lambda > 0 and "bottleneck_z" in info:
                    z_bn = info["bottleneck_z"]
                    wcc = torch.zeros((), device=device)
                    for c in y.unique():
                        m = (y == c)
                        if m.sum() > 1:
                            z_c = z_bn[m]
                            wcc = wcc + (z_c - z_c.mean(0, keepdim=True)).pow(2).sum(1).mean()
                    loss = loss + wcc_lambda * wcc
            scaler.scale(loss).backward()
            _opts = ([optimizer.base, optimizer.adapter]
                     if optimizer.__class__.__name__ == "_DualOptimizer" else [optimizer])
            for _o in _opts:
                scaler.unscale_(_o)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for _o in _opts:
                scaler.step(_o)
            scaler.update()
            bsz = x.size(0)
            tot_loss += loss.item() * bsz
            tot_correct += topk_correct(logits, y, ks=(1,))[1]
            tot += bsz
    return {"loss": tot_loss / tot, "acc": tot_correct / tot}


@torch.no_grad()
def evaluate(model, val_loader, device, use_amp: bool) -> Dict[str, float]:
    model.eval()
    c1 = c5 = tot = 0.0
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(x, client_id=0)
        cc = topk_correct(logits, y, ks=(1, 5))
        c1 += cc[1]; c5 += cc[5]; tot += y.size(0)
    return {"top1": c1 / tot, "top5": c5 / tot}


def already_done(summary_csv: Path, key: dict) -> bool:
    if not summary_csv.exists():
        return False
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if all(str(row.get(k)) == str(v) for k, v in key.items()):
                return True
    return False


class _DualOptimizer:
    """Drives the backbone optimizer and a separate adapter optimizer as one object.

    Only the backbone optimizer is handed to the LR scheduler, so the adapter keeps the
    constant LR it was given -- mirroring the CIFAR harness, where every LS-L result was
    produced with Adam at a constant 1e-3.
    """
    def __init__(self, base, adapter):
        self.base, self.adapter = base, adapter
        self.param_groups = base.param_groups          # scheduler drives the backbone only

    def zero_grad(self, set_to_none: bool = True):
        self.base.zero_grad(set_to_none=set_to_none)
        self.adapter.zero_grad(set_to_none=set_to_none)

    def step(self, *a, **k):
        self.base.step(*a, **k); self.adapter.step(*a, **k)

    def state_dict(self):
        return {"base": self.base.state_dict(), "adapter": self.adapter.state_dict()}

    def load_state_dict(self, sd):
        self.base.load_state_dict(sd["base"]); self.adapter.load_state_dict(sd["adapter"])


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(row.keys())
    if exists:                       # never let a schema change shift existing columns
        with path.open(newline="") as f:
            hdr = next(csv.reader(f), None)
        if hdr:
            fields = hdr
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data/imagenet")
    p.add_argument("--train-subdir", type=str, default="train")
    p.add_argument("--val-subdir", type=str, default="val")
    p.add_argument("--fake-data", action="store_true", help="random tensors; smoke test only")
    p.add_argument("--mode", type=str, required=True,
                   choices=["raw", "topk_sparse", "learned_1x1", "fixed_random",
                            "fixed_random_ll", "block_random"])
    p.add_argument("--compression-ratio", type=int, default=8)
    p.add_argument("--cut-hw", type=int, default=8, help="head pools to cut_hw^2; d=64*cut_hw^2")
    p.add_argument("--ll-hidden", type=int, default=512, help="FR-LL (fixed_random_ll) learned-liftback MLP hidden dim")
    p.add_argument("--ll-depth", type=int, default=2, help="FR-LL adapter depth (2 = original single hidden layer)")
    p.add_argument("--train-per-class", type=int, default=0, help="cap training images per class (0 = full ImageNet); val always full")
    p.add_argument("--adapter-opt", type=str, default="same", choices=["same", "adam"],
                   help="optimizer for the FR-LL adapter only; backbone recipe is untouched either way")
    p.add_argument("--adapter-lr", type=float, default=None, help="absolute LR for the adapter group (default: backbone LR)")
    p.add_argument("--adapter-wd", type=float, default=None, help="weight decay for the adapter group (default: backbone wd)")
    p.add_argument("--block-size", type=int, default=2, help="block_random tile size (1=per-position .. cut_hw=global dense)")
    p.add_argument("--deep-head", action="store_true", help="L2H: extra client-head block")
    p.add_argument("--client-heads", action="store_true", help="PCH: per-client heads (default SCH)")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--alpha", type=float, default=IID_ALPHA, help=">=1e7 = IID")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    p.add_argument("--lr", type=float, default=0.1, help="base LR for batch 256 (SGD)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    # Exploratory; not reported in the paper.
    p.add_argument("--recon-lambda", type=float, default=0.0)
    p.add_argument("--wcc-lambda", type=float, default=0.0)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--summary-csv", type=str, default="results/csv/imagenet_summary.csv")
    p.add_argument("--epoch-csv", type=str, default="results/csv/imagenet_epochs.csv")
    p.add_argument("--ckpt-dir", type=str, default="results/ckpt")
    p.add_argument("--resume", action="store_true", help="skip if a finished summary row exists")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    head = "PCH" if args.client_heads else "SCH"
    depth = "L2H" if args.deep_head else "L1H"
    mode_tag = f"{args.mode}b{args.block_size}" if args.mode == "block_random" else args.mode
    if args.mode == "fixed_random_ll":
        mode_tag = f"{mode_tag}h{args.ll_hidden}"   # widths must not share a checkpoint
        if args.ll_depth != 2:
            mode_tag = f"{mode_tag}d{args.ll_depth}"
        if args.adapter_opt != "same":
            mode_tag = f"{mode_tag}{args.adapter_opt}"
        if args.adapter_lr is not None:
            mode_tag = f"{mode_tag}lr{args.adapter_lr:g}"
        if args.adapter_wd is not None:
            mode_tag = f"{mode_tag}wd{args.adapter_wd:g}"
    cfg_tag = f"{mode_tag}_cr{args.compression_ratio}_{depth}_{head}_a{args.alpha:g}"
    if args.train_per_class > 0:
        cfg_tag = f"{cfg_tag}_pc{args.train_per_class}"
    summary_csv = Path(args.summary_csv)
    run_key = {"mode": args.mode, "compression_ratio": args.compression_ratio,
               "deep_head": args.deep_head, "client_heads": args.client_heads,
               "alpha": f"{args.alpha:g}", "epochs": args.epochs, "cut_hw": args.cut_hw,
               "train_per_class": args.train_per_class}
    if args.mode == "fixed_random_ll":
        run_key["ll_hidden"] = args.ll_hidden
        run_key["ll_depth"] = args.ll_depth
        run_key["adapter_opt"] = args.adapter_opt
        run_key["adapter_lr"] = "-" if args.adapter_lr is None else f"{args.adapter_lr:g}"
        run_key["adapter_wd"] = "-" if args.adapter_wd is None else f"{args.adapter_wd:g}"
    if args.resume and already_done(summary_csv, run_key):
        print(f"[skip] {cfg_tag}: finished row already in {summary_csv}")
        return

    act_dim = activation_dim_for(args.cut_hw)
    k = act_dim // args.compression_ratio
    print(f"=== {cfg_tag} | d={act_dim} k={k} ({4*act_dim*k/1e6:.1f} MB R) "
          f"clients={args.num_clients} epochs={args.epochs} amp={use_amp} ===")

    train_loaders, val_loader = get_loaders(
        args.data_root, args.batch_size, args.num_workers, args.num_clients,
        alpha=args.alpha, seed=args.seed, fake=args.fake_data,
        train_subdir=args.train_subdir, val_subdir=args.val_subdir,
        train_per_class=args.train_per_class)

    model = build_model(args.mode, args.compression_ratio, args.deep_head,
                        args.client_heads, args.num_clients, cut_hw=args.cut_hw,
                        wcc_lambda=args.wcc_lambda, ll_hidden=args.ll_hidden,
                        ll_depth=args.ll_depth,
                        block_size=args.block_size).to(device)
    print(f"payload_dim={model.payload_dim} floats/sample "
          f"(raw={model.activation_dim}; ratio={model.activation_dim/model.payload_dim:.1f}x)")

    lr = args.lr * (args.batch_size / 256) if args.optimizer == "sgd" else args.lr
    ad_params = []
    if hasattr(model.bottleneck, "learned_lift"):
        ad_params = [p for p in model.bottleneck.learned_lift.parameters() if p.requires_grad]
    ad_ids = {id(p) for p in ad_params}
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in ad_ids]
    a_lr = args.adapter_lr if args.adapter_lr is not None else lr
    a_wd = args.adapter_wd if args.adapter_wd is not None else args.weight_decay
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD([{"params": base_params, "lr": lr,
                                      "weight_decay": args.weight_decay}],
                                    lr=lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
        if ad_params and args.adapter_opt == "same":
            optimizer.add_param_group({"params": ad_params, "lr": a_lr, "weight_decay": a_wd,
                                       "momentum": args.momentum, "nesterov": True})
    else:
        optimizer = torch.optim.Adam([{"params": base_params, "lr": lr,
                                       "weight_decay": args.weight_decay}], lr=lr)
        if ad_params and args.adapter_opt == "same":
            optimizer.add_param_group({"params": ad_params, "lr": a_lr, "weight_decay": a_wd})
    adapter_opt = None
    if ad_params and args.adapter_opt == "adam":
        adapter_opt = torch.optim.Adam(ad_params, lr=a_lr, weight_decay=a_wd)
        optimizer = _DualOptimizer(optimizer, adapter_opt)
    print(f"adapter: n_params={sum(p.numel() for p in ad_params):,} opt={args.adapter_opt} "
          f"lr={a_lr:g} wd={a_wd:g} depth={args.ll_depth}")
    _sched_target = optimizer.base if isinstance(optimizer, _DualOptimizer) else optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_sched_target, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── checkpoint-level resume ──
    ckpt_path = Path(args.ckpt_dir) / f"{cfg_tag}.pt"
    start_epoch, best_top1, best_top5 = 0, 0.0, 0.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_top1, best_top5 = ck["best_top1"], ck["best_top5"]
        print(f"[resume] {cfg_tag} from epoch {start_epoch} (best top1 {best_top1:.4f})")

    epoch_csv = Path(args.epoch_csv)
    for epoch in range(start_epoch, args.epochs):
        tr = train_epoch(model, train_loaders, optimizer, scaler, device,
                         args.recon_lambda, args.wcc_lambda, args.label_smoothing, use_amp)
        scheduler.step()
        ev = evaluate(model, val_loader, device, use_amp)
        best_top1 = max(best_top1, ev["top1"]); best_top5 = max(best_top5, ev["top5"])
        print(f"[{cfg_tag}] epoch {epoch+1}/{args.epochs} "
              f"train_loss {tr['loss']:.3f} train_acc {tr['acc']:.4f} | "
              f"val_top1 {ev['top1']:.4f} val_top5 {ev['top5']:.4f} "
              f"(best {best_top1:.4f}) lr {scheduler.get_last_lr()[0]:.4g}")
        append_row(epoch_csv, {**run_key, "epoch": epoch + 1, "train_loss": tr["loss"],
                               "train_acc": tr["acc"], "val_top1": ev["top1"],
                               "val_top5": ev["top5"], "payload_dim": model.payload_dim})
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_top1": best_top1, "best_top5": best_top5}, ckpt_path)

    append_row(summary_csv, {**run_key, "k": k, "activation_dim": act_dim,
                             "payload_dim": model.payload_dim,
                             "num_clients": args.num_clients, "batch_size": args.batch_size,
                             "optimizer": args.optimizer, "lr": lr,
                             "best_top1": best_top1, "best_top5": best_top5,
                             "final_top1": ev["top1"], "final_top5": ev["top5"]})
    print(f"=== DONE {cfg_tag}: best top1 {best_top1:.4f} top5 {best_top5:.4f} -> {summary_csv}")


if __name__ == "__main__":
    main()
