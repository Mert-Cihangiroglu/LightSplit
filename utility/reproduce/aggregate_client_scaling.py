#!/usr/bin/env python3
"""Aggregate the client-count scaling ablation (tab:client_scaling_ablation).

  python aggregate_client_scaling.py --results DIR [...] --fullsweep DIR_OR_CSV [...] --out CSV

--results holds the n in {25,50,100} runs (CR 8, L2H-SCH, IID); the n=10 column
comes from the full utility sweep (--fullsweep). One row per dataset x method
with best test accuracy (%) per client count and the n10 -> n100 delta.
"""
import argparse, csv, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_full_sweep import collect, BOOL

NS = [10, 25, 50, 100]
METHODS = [("raw", "Raw"), ("fixed_random", "LightSplit-F"), ("fixed_random_ll", "LightSplit-L")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--fullsweep", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cells = {}
    files = []
    for p in a.results:
        files += sorted(glob.glob(os.path.join(p, "**", "*summary*.csv"), recursive=True)) \
            if os.path.isdir(p) else [p]
    for fn in files:
        for r in csv.DictReader(open(fn)):
            ds, mode = r.get("dataset"), r.get("mode")
            if ds not in ("cifar10", "cifar100") or mode not in dict(METHODS):
                continue
            deep = BOOL.get(str(r.get("deep_head")).lower())
            ch = BOOL.get(str(r.get("client_heads", "0")).lower())
            if deep is not True or ch is not False:
                continue
            try:
                acc = float(r["best_global_test_acc"]) * 100
                n = int(float(r["num_clients"]))
                if float(r["alpha"]) < 1e6 or float(r.get("wcc_lambda") or 0):
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            if n in NS:
                cells[(ds, mode, n)] = acc

    ref = collect(a.fullsweep) if a.fullsweep else {}
    for ds in ("cifar10", "cifar100"):
        for mode, _ in METHODS:
            v = ref.get((mode, ds, "iid", "L2H", "SCH", 1 if mode == "raw" else 8))
            if v is not None:
                cells.setdefault((ds, mode, 10), v)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "method"] + [f"n{n}" for n in NS] + ["delta"])
        for ds in ("cifar10", "cifar100"):
            for mode, name in METHODS:
                vs = [cells.get((ds, mode, n)) for n in NS]
                row = [ds, name] + ["" if v is None else f"{v:.3f}" for v in vs]
                row.append("" if (vs[0] is None or vs[-1] is None) else f"{vs[-1] - vs[0]:+.3f}")
                w.writerow(row)
    print(f"[aggregate_client_scaling] -> {a.out}")


if __name__ == "__main__":
    main()
