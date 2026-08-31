#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import FSHA


def make_synthetic_dataset(num_examples, image_shape, num_classes, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(num_examples,) + tuple(image_shape)).astype("float32")
    y = rng.integers(0, num_classes, size=(num_examples,), dtype=np.int32)
    return tf.data.Dataset.from_tensor_slices((x, y))


def run_case(args, mode, wcc_lambda):
    xpriv = make_synthetic_dataset(args.examples, args.image_shape, args.num_classes, args.seed)
    xpub = make_synthetic_dataset(args.examples, args.image_shape, args.num_classes, args.seed + 1)

    hparams = {
        "lr_f": args.lr_f,
        "lr_tilde": args.lr_tilde,
        "lr_D": args.lr_D,
        "WGAN": False,
        "bottleneck_mode": mode,
        "compression_ratio": args.compression_ratio,
        "bottleneck_k": args.bottleneck_k,
        "wcc_lambda": wcc_lambda,
    }
    print("\n[SMOKE] mode=%s wcc_lambda=%s" % (mode, wcc_lambda))
    attack = FSHA.FSHA(xpriv, xpub, args.id_setup, args.batch_size, hparams)
    log = attack(args.iterations, log_frequency=args.log_frequency, verbose=True, progress_bar=False)

    sample = next(iter(xpriv.batch(args.batch_size, drop_remainder=True)))[0]
    recon, control = attack.attack(sample)
    print("[SMOKE] log_shape=%s recon_shape=%s control_shape=%s" % (log.shape, recon.shape, control.shape))


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-run FSHA bottleneck defenses on synthetic data.")
    parser.add_argument(
        "--bottleneck-mode",
        default="all",
        choices=["all", "none", "fixed_random", "learned_1x1"],
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--log-frequency", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--id-setup", type=int, default=0)
    parser.add_argument("--compression-ratio", type=float, default=16.0)
    parser.add_argument("--bottleneck-k", type=int, default=2)
    parser.add_argument("--wcc-lambda", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr-f", type=float, default=1e-4)
    parser.add_argument("--lr-tilde", type=float, default=1e-4)
    parser.add_argument("--lr-D", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    args.image_shape = (32, 32, 3)

    if args.bottleneck_mode == "all":
        cases = [
            ("none", 0.0),
            ("fixed_random", 0.0),
            ("fixed_random", args.wcc_lambda),
            ("learned_1x1", 0.0),
        ]
    elif args.bottleneck_mode == "fixed_random":
        cases = [("fixed_random", args.wcc_lambda)]
    else:
        cases = [(args.bottleneck_mode, 0.0)]

    for mode, wcc_lambda in cases:
        run_case(args, mode, wcc_lambda)


if __name__ == "__main__":
    main()
