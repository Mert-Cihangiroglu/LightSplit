#!/usr/bin/env python3
"""Visualize how WCC affects the privileged known-R reconstruction.

For each (block, wcc) cell it loads the sweep's victim checkpoint, retrains the
cheap known-R decoder, and renders the SAME 12 test columns so grids are
comparable across lambda. Two artifacts per request:
  * per-cell 4-row grid  results/instance_leak_grid_b{B}_w{lam}.png
        rows: true | known-R recon | class-mean(prototype) | nearest same-class real
  * per-block progression results/wcc_progression_b{B}_cr{cr}.png
        rows: true | recon@lam0 | recon@lam1 | ... | class-mean
        -> watch the recon drift from instance-specific toward the smooth prototype.

Console also prints matched / proto / id_rank per cell for context.
Reuses helpers from instance_leak.py; FIXES the lambda tag (handles 1.0/3.0/10.0).
"""
import sys, argparse
from pathlib import Path
import torch
from torchvision.utils import save_image

PA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PA)
from instance_leak import gauss_window, ssim_pairwise, build_loaders, batched_pair  # noqa: E402
from train_split import SplitModel                                                   # noqa: E402
from privileged_attack import (collect_activations, KnownRImageDecoder,             # noqa: E402
                               _train_image_decoder)

ACT_DIM = 4096
NS = 12  # columns (fixed test samples, deterministic order)


def lam_dir(w):   # sweep dir suffix: l0, l0.01, l0.1, l1.0, l10.0 ...
    return "0" if w == 0 else str(float(w))


def lam_file(w):  # checkpoint wcc tag: "", _wcc0.01, _wcc1.0, _wcc10.0 ...
    return "" if w == 0 else f"_wcc{float(w)}"


def reconstruct(block, wcc, cr, aux_loader, test_loader, device, dataset, epochs):
    k = ACT_DIM // cr
    ckpt = (Path(PA) / f"results/wb_block_b{block}_l{lam_dir(wcc)}" /
            "checkpoints" / f"block_random_k{k}_b{block}_deep{lam_file(wcc)}.pt")
    if not ckpt.exists():
        print(f"  [skip] missing checkpoint: {ckpt}")
        return None
    model = SplitModel(dataset=dataset, mode="block_random", k=k,
                       deep_head=True, block_size=block).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    R = model.bottleneck.R
    d, k = R.shape[0], R.shape[1]
    aux = collect_activations(model, aux_loader, device, dataset, max_samples=50000)
    test = collect_activations(model, test_loader, device, dataset, max_samples=10000)
    dec = KnownRImageDecoder(R.detach().cpu(), d, k)
    dec = _train_image_decoder(dec, aux["z"], aux["x01"], device,
                               epochs=epochs, label=f"b{block}_w{wcc:g}_knownR")
    recon = []
    with torch.no_grad():
        for i in range(0, test["z"].size(0), 256):
            recon.append(dec(test["z"][i:i + 256].to(device)).cpu())
    recon = torch.cat(recon)
    true, y = test["x01"], test["y"]
    protos = torch.stack([true[y == c].mean(0) for c in range(int(y.max()) + 1)])
    return dict(recon=recon, true=true, y=y, proto_t=protos[y])


def quick_metrics(d, device, qpc=25):
    win = gauss_window(device=device)
    recon, true, y, proto_t = d["recon"], d["true"], d["y"], d["proto_t"]
    matched = batched_pair(recon, true, win, device).mean().item()
    proto = batched_pair(recon, proto_t, win, device).mean().item()
    by_cls = {c: (y == c).nonzero(as_tuple=True)[0] for c in range(int(y.max()) + 1)}
    g = torch.Generator().manual_seed(0)
    ranks = []
    for c, idxs in by_cls.items():
        cand = true[idxs].to(device)
        for qi in idxs[torch.randperm(len(idxs), generator=g)[:qpc]].tolist():
            r = recon[qi:qi + 1].to(device).expand(len(idxs), -1, -1, -1).contiguous()
            s = ssim_pairwise(r, cand, win)
            pos = (idxs == qi).nonzero(as_tuple=True)[0].item()
            rest = torch.cat([s[:pos], s[pos + 1:]])
            ranks.append((rest < s[pos]).float().mean().item())
    return matched, proto, sum(ranks) / len(ranks)


def four_row_grid(d, device, path):
    win = gauss_window(device=device)
    recon, true, y, proto_t = d["recon"], d["true"], d["y"], d["proto_t"]
    by_cls = {c: (y == c).nonzero(as_tuple=True)[0] for c in range(int(y.max()) + 1)}
    nn_imgs = []
    for qi in range(NS):
        idxs = by_cls[int(y[qi])]
        cand = true[idxs].to(device)
        r = recon[qi:qi + 1].to(device).expand(len(idxs), -1, -1, -1).contiguous()
        s = ssim_pairwise(r, cand, win)
        nn_imgs.append(true[idxs[s.argmax().item()]])
    grid = torch.cat([true[:NS], recon[:NS], proto_t[:NS], torch.stack(nn_imgs)], 0)
    save_image(grid, path, nrow=NS, padding=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--cr", type=int, default=8)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--data-root", default=f"{PA}/data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cells", default=("2:0 2:0.01 2:0.1 2:1.0 2:3.0 2:10.0 "
                                       "1:0 1:0.01 1:0.1 1:1.0 8:0 8:0.1"))
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(args.dataset, args.data_root, args.seed, 256, 2)

    cells = [(int(b), float(w)) for b, w in (tok.split(":") for tok in args.cells.split())]
    print(f"\nWCC VISUAL GRIDS  (privileged known-R, CR{args.cr}, {args.epochs}ep decoder)")
    print(f"{'cell':>10} | {'matched':>7} {'proto':>6} {'id_rank':>7}   (matched=recon vs OWN; "
          "proto=recon vs class-mean; id_rank ->1 = instance-identifiable)\n")

    per_block = {}
    for b, w in cells:
        d = reconstruct(b, w, args.cr, *loaders, device, args.dataset, args.epochs)
        if d is None:
            continue
        m, pr, idr = quick_metrics(d, device)
        print(f"{'b%d w%g' % (b, w):>10} | {m:7.3f} {pr:6.3f} {idr:7.3f}", flush=True)
        gpath = f"{PA}/results/instance_leak_grid_b{b}_w{lam_dir(w)}.png"
        four_row_grid(d, device, gpath)
        per_block.setdefault(b, []).append((w, d["recon"][:NS].clone(),
                                            d["true"][:NS].clone(), d["proto_t"][:NS].clone()))

    print()
    for b, items in sorted(per_block.items()):
        items.sort(key=lambda t: t[0])
        true12, proto12 = items[0][2], items[0][3]
        rows = [true12] + [r for (_, r, _, _) in items] + [proto12]
        path = f"{PA}/results/wcc_progression_b{b}_cr{args.cr}.png"
        save_image(torch.cat(rows, 0), path, nrow=NS, padding=2)
        legend = "true | " + " | ".join(f"w{w:g}" for (w, _, _, _) in items) + " | class-mean"
        print(f"progression b{b} -> {path}\n   rows(top->bottom): {legend}")


if __name__ == "__main__":
    main()
