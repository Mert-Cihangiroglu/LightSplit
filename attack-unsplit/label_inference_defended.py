"""UnSplit label-inference attack with defense support.

Mirrors the structure of `label_inference_demo.py` but lets us insert a
compression bottleneck between the server's output (post-fc2_act) and the
client's classifier head (fc3). The attack still works by enumerating
candidate labels and matching the gradient at fc3.weight; the bottleneck
sits between server_out and the gradient computation, so it is part of the
chain the attacker has to traverse.

Threat model: the server (attacker) knows the architecture of both the
classifier head and the bottleneck (compression-family known, weights
unknown), and observes server_out + the gradient that flows back at the
cut layer. As in the upstream demo, this is vanilla SL only -- in
U-shaped SL the attacker has no gradient access to fc3.

Setup (mirroring upstream defaults):
- MNIST/F-MNIST: split_layer=9, grad_index=8 (fc3.weight)
- CIFAR:        split_layer=16, grad_index=14 (fc3.weight)

Usage:
  python label_inference_defended.py mnist --defense none
  python label_inference_defended.py mnist --defense fixed_random --wcc-lambda 0.1 --cr 2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import torch
from torchvision import datasets, transforms

import unsplit.attacks as unsplit_attacks
from unsplit.bottlenecks import (
    FixedRandomBottleneck, Learned1x1Bottleneck,
)
from unsplit.models import CifarNet, MnistNet


DEFAULTS = {
    "mnist":   {"model": MnistNet, "split_layer": 9, "grad_index": 8, "act_dim": 84},
    "f_mnist": {"model": MnistNet, "split_layer": 9, "grad_index": 8, "act_dim": 84},
    "cifar":   {"model": CifarNet, "split_layer": 16, "grad_index": 14, "act_dim": 512},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["mnist", "f_mnist", "cifar"])
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--defense", choices=["none", "learned_1x1", "fixed_random"], default="none")
    p.add_argument("--cr", type=int, default=2,
                   help="Compression ratio: k = activation_dim / cr (84-dim -> k=42 at cr=2).")
    p.add_argument("--wcc-lambda", type=float, default=0.0)
    p.add_argument("--pretrain-epochs", type=int, default=0,
                   help="Pre-train both the (server, bottleneck, client) stack AND the "
                        "(clone, clone_bottleneck) stack on disjoint slices of the trainset "
                        "before launching the per-sample label-inference attack. 0 = no "
                        "pre-training (matches upstream demo's online-attack setup).")
    p.add_argument("--pretrain-batch", type=int, default=64)
    p.add_argument("--clone-pretrain-fraction", type=float, default=0.2,
                   help="Fraction of trainset reserved for attacker-side (clone) pre-training. "
                        "The remaining (1 - fraction) is used for victim pre-training. Splits "
                        "are disjoint by index.")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def build_bottleneck(defense: str, k: int, activation_dim: int):
    """The label-inference cut is post-fc2_act, which is a 1D vector. None of
    our bottlenecks need spatial information here; fixed_random works
    on flat features natively, learned_1x1 needs to be wrapped."""
    if defense == "fixed_random":
        return FixedRandomBottleneck(activation_dim=activation_dim, k=k)
    if defense == "learned_1x1":
        # Reshape 1D activation to (1, dim, 1, 1) and use a 1x1 conv.
        # We treat each activation dim as a "channel" with 1x1 spatial.
        return Learned1x1Bottleneck(in_channels=activation_dim, k_channels=k)
    raise ValueError(defense)


def through_bottleneck(bottleneck, smashed: torch.Tensor):
    """Apply compress + lift_back. Handles the spatial (Learned 1x1) and
    flat (FR) cases: for Learned 1x1 we synthesise a (B, dim, 1, 1)
    layout."""
    if isinstance(bottleneck, Learned1x1Bottleneck):
        # smashed: (B, dim) -> (B, dim, 1, 1) -> conv1x1 -> (B, k, 1, 1)
        sp = smashed.unsqueeze(-1).unsqueeze(-1)
        payload, info = bottleneck(sp)
        recon = bottleneck.lift_back(payload, info)
        return recon.squeeze(-1).squeeze(-1), info
    payload, info = bottleneck(smashed)
    recon = bottleneck.lift_back(payload, info)
    return recon, info


def append_csv_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    cfg = DEFAULTS[args.dataset]
    Model = cfg["model"]
    split_layer = cfg["split_layer"]
    grad_index = cfg["grad_index"]
    activation_dim = cfg["act_dim"]
    k = activation_dim // args.cr if args.defense != "none" else activation_dim

    if args.dataset == "mnist":
        trainset = datasets.MNIST("data/mnist", download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.MNIST("data/mnist", download=True, train=False, transform=transforms.ToTensor())
    elif args.dataset == "f_mnist":
        trainset = datasets.FashionMNIST("data/f_mnist", download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.FashionMNIST("data/f_mnist", download=True, train=False, transform=transforms.ToTensor())
    else:
        trainset = datasets.CIFAR10("data/cifar", download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.CIFAR10("data/cifar", download=True, train=False, transform=transforms.ToTensor())
    # Disjoint split for victim vs attacker pretraining.
    n_total = len(trainset)
    n_clone = int(n_total * args.clone_pretrain_fraction)
    g = torch.Generator().manual_seed(args.seed if args.seed is not None else 0)
    perm = torch.randperm(n_total, generator=g).tolist()
    clone_indices = perm[:n_clone]
    victim_indices = perm[n_clone:]
    victim_set = torch.utils.data.Subset(trainset, victim_indices)
    clone_set = torch.utils.data.Subset(trainset, clone_indices)
    victim_loader = torch.utils.data.DataLoader(victim_set, batch_size=args.pretrain_batch, shuffle=True)
    clone_loader = torch.utils.data.DataLoader(clone_set, batch_size=args.pretrain_batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True)

    server, client, clone = Model(), Model(), Model()
    bottleneck = build_bottleneck(args.defense, k, activation_dim) if args.defense != "none" else None
    clone_bottleneck = build_bottleneck(args.defense, k, activation_dim) if args.defense != "none" else None

    server_opt = torch.optim.Adam(server.parameters(), lr=0.001, amsgrad=True)
    client_opt = torch.optim.Adam(client.parameters(), lr=0.001, amsgrad=True)
    clone_opt = torch.optim.Adam(clone.parameters(), lr=0.001, amsgrad=True)
    if bottleneck is not None:
        bottleneck_opt = torch.optim.Adam(bottleneck.parameters(), lr=0.001, amsgrad=True) \
            if list(bottleneck.parameters()) else None
        clone_bottleneck_opt = torch.optim.Adam(clone_bottleneck.parameters(), lr=0.001, amsgrad=True) \
            if list(clone_bottleneck.parameters()) else None

    criterion = torch.nn.CrossEntropyLoss()

    print(f"Setup: dataset={args.dataset} split_layer={split_layer} defense={args.defense} k={k} "
          f"wcc_lambda={args.wcc_lambda} pretrain_epochs={args.pretrain_epochs}")

    # ---- Optional pre-training (batch=64, WCC active) ----
    # Without this, WCC at batch=1 produces zero per-class scatter and has
    # no effect on the trained representation. Pre-training lets WCC
    # actually shape the model before we mount the per-sample attack.
    if args.pretrain_epochs > 0:
        print(f"Pre-training: {len(victim_set)} victim samples, {len(clone_set)} clone "
              f"samples, batch={args.pretrain_batch}, {args.pretrain_epochs} epochs each.")

        def _pretrain(loader, srv, btl, cls, srv_opt, btl_opt, cls_opt, tag):
            for epoch in range(args.pretrain_epochs):
                rl = 0.0; rw = 0.0; nb = 0
                for images, labels in loader:
                    cls_opt.zero_grad(); srv_opt.zero_grad()
                    if btl is not None and btl_opt is not None:
                        btl_opt.zero_grad()
                    so = srv(images, end=split_layer)
                    if btl is not None:
                        rec, info = through_bottleneck(btl, so)
                    else:
                        rec, info = so, {}
                    pr = cls(rec, start=split_layer + 1)
                    loss = criterion(pr, labels)
                    if args.wcc_lambda > 0 and "bottleneck_z" in info:
                        z = info["bottleneck_z"]
                        wcc = torch.zeros((), device=z.device, dtype=z.dtype)
                        for c in labels.unique():
                            mask = (labels == c)
                            if mask.sum() > 1:
                                zc = z[mask]
                                mu = zc.mean(dim=0, keepdim=True)
                                wcc = wcc + (zc - mu).pow(2).sum(dim=1).mean()
                        loss = loss + args.wcc_lambda * wcc
                        rw += float(wcc)
                    rl += float(loss); nb += 1
                    loss.backward()
                    cls_opt.step(); srv_opt.step()
                    if btl is not None and btl_opt is not None:
                        btl_opt.step()
                msg = f"  {tag} ep{epoch} loss={rl/nb:.4f}"
                if args.wcc_lambda > 0:
                    msg += f" wcc={rw/nb:.4f}"
                print(msg)

        # Victim half (the real client/server/bottleneck).
        _pretrain(victim_loader, server, bottleneck, client,
                  server_opt, bottleneck_opt if bottleneck is not None else None,
                  client_opt, "victim")
        # Attacker half. The clone needs its own (server', bottleneck') stack
        # to forward through during pre-training, so reuse a copy of the
        # victim's server architecture (random init, attacker doesn't know
        # the victim's server weights either) and its own clone_bottleneck.
        att_server = Model()
        att_server_opt = torch.optim.Adam(att_server.parameters(), lr=0.001, amsgrad=True)
        clone_btl_opt = clone_bottleneck_opt if clone_bottleneck is not None else None
        _pretrain(clone_loader, att_server, clone_bottleneck, clone,
                  att_server_opt, clone_btl_opt, clone_opt, "clone ")
        # The attack feeds server_out to the clone (server is the attacker
        # itself), so we leave att_server unused after pre-training -- the
        # clone has been shaped by it but the attack uses the victim's
        # server output as input to the clone (worst case for defense).
        print("Done pre-training.\n")

    results = []
    for idx, (image, label) in enumerate(testloader):
        if idx == args.count:
            break

        # Step 1: client + server + (optional) bottleneck forward, then back.
        client_opt.zero_grad(); server_opt.zero_grad()
        if bottleneck is not None and bottleneck_opt is not None:
            bottleneck_opt.zero_grad()

        server_out = server(image, end=split_layer)            # (B, activation_dim)
        if bottleneck is not None:
            recon, info = through_bottleneck(bottleneck, server_out)
        else:
            recon, info = server_out, {}
        pred = client(recon, start=split_layer + 1)            # (B, num_classes)
        loss = criterion(pred, label)
        if args.wcc_lambda > 0 and "bottleneck_z" in info:
            # WCC on the bottleneck output (single sample, so per-class scatter
            # is 0 here; included for symmetry with the inversion experiments).
            z = info["bottleneck_z"]
            mu = z.mean(dim=0, keepdim=True)
            wcc_loss = (z - mu).pow(2).sum(dim=1).mean()
            loss = loss + args.wcc_lambda * wcc_loss
        loss.backward(retain_graph=True)

        # The attacker (server) observes the gradient at fc3.weight on the
        # client side. In the original demo this is read directly via
        # client.parameters()[grad_index]; we keep the same convention.
        target_grad = list(client.parameters())[grad_index].grad.detach().clone()

        # Step 2: clone forward (same architecture, random init) + label inference.
        clone_opt.zero_grad()
        if clone_bottleneck is not None and clone_bottleneck_opt is not None:
            clone_bottleneck_opt.zero_grad()
        if clone_bottleneck is not None:
            clone_recon, _ = through_bottleneck(clone_bottleneck, server_out.detach())
        else:
            clone_recon = server_out.detach()
        clone_pred = clone(clone_recon, start=split_layer + 1)

        label_vals = [c * torch.ones(len(label)).long() for c in range(10)]
        pred_label = unsplit_attacks.label_inference(
            clone_pred, clone, target_grad, label_vals, grad_index)

        correct = label.item() == pred_label.item()
        results.append(correct)
        if idx % 10 == 0 or idx < 5:
            print(f"  [{idx:3d}] true={label.item()} pred={pred_label.item()} {'OK' if correct else 'X'}")

        # Step 3: training updates (same as upstream demo).
        clone_loss = criterion(clone_pred, pred_label)
        clone_loss.backward()
        client_opt.step()
        server_opt.step()
        clone_opt.step()
        if bottleneck is not None and bottleneck_opt is not None:
            bottleneck_opt.step()
        if clone_bottleneck is not None and clone_bottleneck_opt is not None:
            clone_bottleneck_opt.step()

    acc = sum(results) / len(results)
    print()
    print(f"Label inference accuracy: {acc:.4f}  ({sum(results)}/{len(results)})")
    append_csv_row(Path("results/label_inference.csv"), {
        "dataset": args.dataset, "defense": args.defense, "k": k,
        "split": split_layer, "wcc_lambda": args.wcc_lambda,
        "seed": (args.seed if args.seed is not None else ""),
        "count": args.count, "label_inference_acc": f"{acc:.4f}",
    })


if __name__ == "__main__":
    main()
