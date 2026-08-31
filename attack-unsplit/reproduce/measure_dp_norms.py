#!/usr/bin/env python
"""Clip calibration for the UnSplit DP cells: clip = median L2 norm of the transmitted payload
over the first 256 test images, measured on the freshly built (untrained) victim that
model_inversion_demo.py constructs for the same --seed. Run from the directory that holds data/.

  python measure_dp_norms.py --dataset mnist --split 2 --defense fixed_random --cr 8 --seed 0
"""
import argparse, os, sys
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unsplit.defended_models import DefendedSplitModel  # noqa: E402
from unsplit.models import MnistNet  # noqa: E402
from model_inversion_demo import detect_split_shape  # noqa: E402

SETS = {"mnist": datasets.MNIST, "f_mnist": datasets.FashionMNIST}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=SETS, default="mnist")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--defense", default="fixed_random")
    ap.add_argument("--cr", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    trainset = SETS[a.dataset](f"data/{a.dataset}", download=True, train=True, transform=transforms.ToTensor())
    testset = SETS[a.dataset](f"data/{a.dataset}", download=True, train=False, transform=transforms.ToTensor())
    torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64)  # the demo builds this first; it consumes RNG
    ch, H, W = detect_split_shape(MnistNet, a.split, (1, 28, 28))
    k = (ch * H * W) // a.cr
    victim = DefendedSplitModel(MnistNet, split_layer=a.split, defense=a.defense, k=k, in_channels=ch,
                                spatial_hw=(H, W), **({"block_size": a.block_size} if a.defense == "block_random" else {}))
    victim.eval()
    x = torch.stack([testset[i][0] for i in range(a.n)])
    with torch.no_grad():
        norms = victim.compress_payload(x).flatten(1).norm(dim=1)
    clip = float(norms.median())
    print(f"{a.dataset} split{a.split} {a.defense} k={k}: clip={clip:.3f}  fraction clipped={(norms > clip).float().mean():.1%}")


if __name__ == "__main__":
    main()
