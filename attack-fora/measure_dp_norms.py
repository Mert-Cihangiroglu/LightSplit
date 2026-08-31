#!/usr/bin/env python
"""Median L2 norm of the transmitted payload, the value used as the DP clip (noise std = sigma x clip).

Measures the payload of the untrained client stack (MobileNet level 2 followed by the defense) on
CIFAR-10 training batches and reports percentiles of its per-sample L2 norm; the clip is set at the
median, so about half of the samples are clipped. Run from a directory holding ./data (CIFAR-10 is
downloaded there if absent).

  python measure_dp_norms.py                                  # raw payload (16384-d activation)
  python measure_dp_norms.py --defense fixed_random --k 256
  python measure_dp_norms.py --defense block_random --k 256 --block-size 2
"""
import argparse
import torch, torchvision
import torchvision.transforms as transforms
from model import cifar_mobilenet
from defenses import DefenseWrapper

LAYER = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", default="none", choices=["none", "fixed_random", "block_random", "topk_sparse", "learned_1x1"])
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--gid", type=int, default=0)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gid}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = torchvision.datasets.CIFAR10(root="./data", train=True, transform=tf, download=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=4)
    bottom, _ = cifar_mobilenet(level=LAYER); bottom = bottom.to(dev).eval()
    defense = DefenseWrapper(a.defense, in_channels=64, H=16, W=16, k=a.k, block=a.block_size).to(dev).eval()
    norms = []
    with torch.no_grad():
        for i, (x, _) in enumerate(dl):
            if i >= a.batches:
                break
            h = bottom(x.to(dev))
            if defense.module is None:                      # raw: the activation itself crosses the cut
                z = h.reshape(h.size(0), -1)
            elif a.defense == "learned_1x1":
                z = defense.module(h)[0]
            else:                                           # projections and top-k sparse act on the flat activation
                z = defense.module(h.reshape(h.size(0), -1))[0]
            norms.append(z.reshape(z.size(0), -1).norm(dim=1).cpu()); dim = z.reshape(z.size(0), -1).shape[1]
    v = torch.cat(norms)
    q = lambda p: torch.quantile(v, p).item()
    med = round(q(0.5), 3)
    print(f"{a.defense} k={a.k}{' b=' + str(a.block_size) if a.defense == 'block_random' else ''}: n={v.numel()} payload_dim={dim}  "
          f"p10={q(0.1):.3f} median={q(0.5):.3f} mean={v.mean():.3f} p90={q(0.9):.3f}  clip={med}  clipped={(v > med).float().mean():.1%}")


if __name__ == "__main__":
    main()
