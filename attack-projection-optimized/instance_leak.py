#!/usr/bin/env python3
"""Instance-leak test for the privileged KNOWN-R attack.

Settles whether the ~0.5 known-R SSIM is a pure class prototype (no instance
leak) or carries instance-specific detail, and whether WCC shrinks the instance
part even when it barely moves raw SSIM.

For each test image x_i (class c) the known-R decoder yields recon x_hat_i. We
decompose the reconstruction's similarity, CONTROLLING FOR CLASS:
  matched = SSIM(x_hat_i, x_i)                       recon vs its OWN sample
  within  = mean_j SSIM(x_hat_i, x_j), y_j=c, j!=i   recon vs OTHER same-class reals
  cross   = mean_j SSIM(x_hat_i, x_j), y_j!=c        recon vs other-class reals (ref floor)
  proto   = SSIM(x_hat_i, classmean_img(c))          recon vs the class-mean image
  GAP     = matched - within                         == INSTANCE LEAKAGE (>0 => leak)
  id_rank = percentile of x_i among same-class reals, ranked by SSIM to x_hat_i
            0.5 = chance (recon equidistant to all classmates = pure class);
            ->1 = the attacker can pick your specific image out of your class.
Also: proto_floor = SSIM(x_i, classmean_img(c)) = the unavoidable class floor (~0.37).

Loads the victim checkpoints the sweep already trained; retrains only the cheap
known-R image decoder (exp3's). Faithfully reuses the attack pipeline.

Usage: python instance_leak.py --gpu 3 --epochs 200 --cells "1:0 2:0 8:0 2:0.01 2:0.1 8:0.1"
"""
import sys, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

PA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PA)
from train_split import SplitModel, DATASET_STATS                       # noqa: E402
from privileged_attack import (collect_activations, KnownRImageDecoder, # noqa: E402
                               _train_image_decoder)

ACT_DIM = 4096  # CIFAR head cut = 64*8*8


def gauss_window(ws=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(ws, dtype=torch.float32) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    g1d = g.unsqueeze(0).unsqueeze(0)
    w = g1d.unsqueeze(-1) * g1d.unsqueeze(-2)
    return w.expand(channels, 1, ws, ws).contiguous().to(device)


def ssim_pairwise(a, b, window, ws=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """Per-sample SSIM (N,) for aligned [0,1] batches. Mirrors sdar_attack.compute_ssim."""
    C = a.shape[1]; pad = ws // 2
    mu1 = F.conv2d(a, window, padding=pad, groups=C)
    mu2 = F.conv2d(b, window, padding=pad, groups=C)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(a * a, window, padding=pad, groups=C) - mu1s
    s2 = F.conv2d(b * b, window, padding=pad, groups=C) - mu2s
    s12 = F.conv2d(a * b, window, padding=pad, groups=C) - mu12
    smap = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1s + mu2s + C1) * (s1 + s2 + C2))
    return smap.flatten(1).mean(dim=1)


def build_loaders(dataset, data_root, seed, batch_size, num_workers):
    stats = DATASET_STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(stats["mean"], stats["std"])])
    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    full = cls(root=data_root, train=True, download=True, transform=tf)
    n = len(full)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    attacker_pool = perm[n - n // 10:]                 # same 5k aux split as the attack
    aux = DataLoader(Subset(full, attacker_pool), batch_size=batch_size,
                     shuffle=False, num_workers=num_workers, pin_memory=True)
    test = cls(root=data_root, train=False, download=True, transform=tf)
    test = DataLoader(test, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
    return aux, test


def batched_pair(a, b, window, device, bs=1000):
    out = []
    for i in range(0, a.size(0), bs):
        out.append(ssim_pairwise(a[i:i + bs].to(device), b[i:i + bs].to(device), window).cpu())
    return torch.cat(out)


def analyze(block, wcc, cr, aux_loader, test_loader, device, dataset, epochs, draws=3,
            save_grid=None):
    k = ACT_DIM // cr
    ltag = "0" if wcc == 0 else f"{wcc:g}"
    out_dir = Path(PA) / f"results/wb_block_b{block}_l{ltag}"
    tag = f"block_random_k{k}_b{block}_deep" + ("" if wcc == 0 else f"_wcc{wcc:g}")
    ckpt = out_dir / "checkpoints" / f"{tag}.pt"
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
    N, ncls = recon.size(0), int(y.max()) + 1
    win = gauss_window(device=device)

    protos = torch.stack([true[y == c].mean(0) for c in range(ncls)])
    proto_t = protos[y]

    matched = batched_pair(recon, true, win, device)
    proto_s = batched_pair(recon, proto_t, win, device)
    proto_floor = batched_pair(true, proto_t, win, device)   # real vs own class-mean

    by_cls = {c: (y == c).nonzero(as_tuple=True)[0] for c in range(ncls)}
    g2 = torch.Generator().manual_seed(0)
    within_d, cross_d = [], []
    for _ in range(draws):
        wi = torch.empty(N, dtype=torch.long)
        ci = torch.empty(N, dtype=torch.long)
        for c, idxs in by_cls.items():
            wi[idxs] = idxs[torch.randperm(len(idxs), generator=g2)]      # random same-class partner
            others = (y != c).nonzero(as_tuple=True)[0]
            ci[idxs] = others[torch.randint(len(others), (len(idxs),), generator=g2)]
        within_d.append(batched_pair(recon, true[wi], win, device))
        cross_d.append(batched_pair(recon, true[ci], win, device))
    within = torch.stack(within_d).mean(0)
    cross = torch.stack(cross_d).mean(0)

    # instance identifiability: rank own image among same-class reals by SSIM-to-recon
    id_ranks = []
    for c, idxs in by_cls.items():
        cand = true[idxs].to(device)
        q = idxs[torch.randperm(len(idxs), generator=g2)[:50]]
        for qi in q.tolist():
            r = recon[qi:qi + 1].to(device).expand(len(idxs), -1, -1, -1).contiguous()
            s = ssim_pairwise(r, cand, win)
            pos = (idxs == qi).nonzero(as_tuple=True)[0].item()
            own = s[pos]
            rest = torch.cat([s[:pos], s[pos + 1:]])
            id_ranks.append((rest < own).float().mean().item())
    id_rank = sum(id_ranks) / len(id_ranks)

    if save_grid:
        ns = 12
        # for the grid: nearest same-class neighbor of the recon (by SSIM)
        rows = [true[:ns], recon[:ns], proto_t[:ns]]
        nn_imgs = []
        for qi in range(ns):
            c = int(y[qi]); idxs = by_cls[c]
            cand = true[idxs].to(device)
            r = recon[qi:qi + 1].to(device).expand(len(idxs), -1, -1, -1).contiguous()
            s = ssim_pairwise(r, cand, win)
            nn_imgs.append(true[idxs[s.argmax().item()]])
        rows.append(torch.stack(nn_imgs))
        grid = torch.cat(rows, dim=0)
        save_image(grid, save_grid, nrow=ns, padding=2)
        print(f"  saved grid -> {save_grid}  (rows: true | recon | class-mean | NN-same-class)")

    res = dict(block=block, wcc=wcc, cr=cr, matched=matched.mean().item(),
               within=within.mean().item(), cross=cross.mean().item(),
               proto=proto_s.mean().item(), gap=(matched - within).mean().item(),
               id_rank=id_rank, proto_floor=proto_floor.mean().item())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--cr", type=int, default=8)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--data-root", default=f"{PA}/data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cells", default="1:0 2:0 8:0 2:0.01 2:0.1 8:0.1",
                   help="space-separated block:wcc cells")
    p.add_argument("--out", default=f"{PA}/results/instance_leak.csv")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    aux_loader, test_loader = build_loaders(args.dataset, args.data_root, args.seed, 256, 2)

    cells = []
    for tok in args.cells.split():
        b, w = tok.split(":")
        cells.append((int(b), float(w)))

    print(f"\nINSTANCE-LEAK TEST  (privileged known-R, CR{args.cr}, {args.dataset}, "
          f"decoder {args.epochs}ep)")
    print("gap = matched - within  =>  INSTANCE leakage (>0).  "
          "id_rank: 0.5=chance/pure-class, ->1=instance-identifiable.\n")
    hdr = f"{'cell':>10} | {'matched':>7} {'within':>7} {'cross':>7} {'proto':>6} | {'GAP':>6} {'id_rank':>7} | {'floor':>6}"
    print(hdr); print("-" * len(hdr))

    rows = []
    for i, (b, w) in enumerate(cells):
        grid = f"{PA}/results/instance_leak_grid_b{b}_w{w:g}.png" if i == 0 else None
        r = analyze(b, w, args.cr, aux_loader, test_loader, device, args.dataset,
                    args.epochs, save_grid=grid)
        if r is None:
            continue
        rows.append(r)
        print(f"{'b%d w%g' % (b, w):>10} | {r['matched']:7.3f} {r['within']:7.3f} "
              f"{r['cross']:7.3f} {r['proto']:6.3f} | {r['gap']:6.3f} {r['id_rank']:7.3f} | "
              f"{r['proto_floor']:6.3f}", flush=True)

    if rows:
        import csv
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print(f"\nwrote {args.out}")
    print("\nREAD: GAP>0 and id_rank>0.5 => real instance leakage WCC must remove.")
    print("      GAP~0 and id_rank~0.5 => the ~0.5 SSIM is pure class (prototype); "
          "WCC is redundant vs known-R because there is no instance info to strip.")


if __name__ == "__main__":
    main()
