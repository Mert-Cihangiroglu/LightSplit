#!/usr/bin/env python3
"""Clean (victim) accuracy for the FSHA DP and block_random cells.

Mirrors run_dpblock.py but drives run_clean_accuracy_grid.run_one so the DP/block rows get a
victim test-accuracy number consistent with the rest of the FSHA clean grid.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import run_clean_accuracy_grid as rcg
from run_dpblock import dp_cases, block_cases  # reuse the exact same case definitions


def build_args(a):
    return SimpleNamespace(
        dataset=a.dataset, seed=a.seed, epochs=a.epochs, batch_size=128, id_setup=3,
        learning_rate=0.001, classifier_hidden_units=256, shuffle_buffer=10000,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["dp", "block"], required=True)
    ap.add_argument("--dataset", choices=["mnist", "fashion_mnist"], required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-root", default=str(rcg.REPO_DIR / "results" / "clean_accuracy_grid"))
    a = ap.parse_args()

    args = build_args(a)
    x_train, y_train, x_test, y_test, parser = rcg.load_arrays(a.dataset, 0, a.seed)
    arrays = (x_train, y_train, x_test, y_test)
    root = Path(a.output_root) / a.run_name
    root.mkdir(parents=True, exist_ok=True)

    cases = dp_cases() if a.which == "dp" else block_cases()
    print("[run_clean_dpblock] %s / %s / seed %d : %d cases -> %s" % (a.which, a.dataset, a.seed, len(cases), root))
    for case in cases:
        rcg.run_one(args, arrays, parser, case, root)


if __name__ == "__main__":
    main()
