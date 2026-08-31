#!/usr/bin/env python3
"""FSHA runner for the DP and block_random cells.

Bypasses build_cases and constructs explicit case dicts, then reuses run_bottleneck_grid.run_one
(shared training loop, reconstruction saving) so the outputs are byte-compatible with the rest of
the FSHA results. Scoring (SSIM/LPIPS) is done afterward from reconstructions.npz.

  --which dp     : raw+DP and fixed_random(LS-F)+DP at cr8            (table rows for fsha_active)
  --which block  : block_random b in {1,2,4} at cr {8,16,32} + raw   (block columns of block_privacy_attacks)
  --which blockdp: block_random b in {1,2,4} at cr8 with DP           (its +DP rows)
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import run_bottleneck_grid as rbg


def build_args(a):
    return SimpleNamespace(
        dataset=a.dataset, seed=a.seed, iterations=a.iterations, log_frequency=a.log_frequency,
        batch_size=64, id_setup=3, shuffle_buffer=10000, sample_count=20,
        score_examples=0, accuracy_examples=0, accuracy_batch_size=128,
        wgan=True, gradient_penalty=500.0, lr_f=1e-5, lr_tilde=1e-5, lr_D=1e-4,
        progress=False,
    )


def dp_cases():
    # raw+DP (Identity payload) and LS-F+DP (fixed_random wcc0) at the 8x operating point.
    return [
        {"mode": "none", "compression_ratio": 1.0, "wcc_lambda": 0.0, "dp": True},
        {"mode": "fixed_random", "compression_ratio": 8.0, "wcc_lambda": 0.0, "dp": True},
    ]


def block_cases():
    cases = [{"mode": "none", "compression_ratio": 1.0, "wcc_lambda": 0.0}]  # raw reference
    for cr in (8.0, 16.0, 32.0):
        for b in (1, 2, 4):  # b=4 is the global dense R (== LS-F) at this 4x4 cut
            cases.append({"mode": "block_random", "compression_ratio": cr, "wcc_lambda": 0.0, "block_size": b})
    return cases


def blockdp_cases():
    # block_random b in {1,2,4} at cr8 with DP (sigma = 0.02): the +DP rows of the block table
    return [{"mode": "block_random", "compression_ratio": 8.0, "wcc_lambda": 0.0, "block_size": b, "dp": True} for b in (1, 2, 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["dp", "block", "blockdp"], required=True)
    ap.add_argument("--dataset", choices=["mnist", "fashion_mnist"], required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--log-frequency", type=int, default=500, help="must divide --iterations")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-root", default=str(rbg.REPO_DIR / "results" / "bottleneck_grid"))
    a = ap.parse_args()

    args = build_args(a)
    x_train, y_train, x_test, y_test, parser = rbg.load_arrays(a.dataset, 0, a.seed)
    arrays = (x_train, y_train, x_test, y_test)
    root = Path(a.output_root) / a.run_name
    root.mkdir(parents=True, exist_ok=True)

    cases = {"dp": dp_cases, "block": block_cases, "blockdp": blockdp_cases}[a.which]()
    print("[run_dpblock] %s / %s / seed %d : %d cases -> %s" % (a.which, a.dataset, a.seed, len(cases), root))
    for case in cases:
        rbg.run_one(args, arrays, parser, case, root, accuracy_classifier=None)


if __name__ == "__main__":
    main()
