#!/usr/bin/env python3
"""Aggregate the WCC-regularisation ablation (tab:wcc_ablation).

  python aggregate_wcc_ablation.py --results DIR [...] --fullsweep DIR_OR_CSV [...] --out CSV

--results holds the ablation runs: fixed_random / fixed_random_ll (global) and
block_random / block_random_ll with b=2, each at CR 8, CIFAR-10, IID, over
lambda_WCC {0, 1e-3, 1e-2, 1e-1}, plus one raw run per head config for the
b=2 rows' reference. Global rows read against the full utility sweep's Raw
cells (--fullsweep), matching the paper. Cells are the relative accuracy
change vs the reference, in percent.
"""
import argparse, csv, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_full_sweep import collect, BOOL

LAMBDAS = [0.0, 0.001, 0.01, 0.1]
CFGS = [("L1H", "SCH"), ("L2H", "SCH"), ("L1H", "PCH"), ("L2H", "PCH")]
ROWS = [("LightSplit-F", "global", "fixed_random"), ("LightSplit-F", "b2", "block_random"),
        ("LightSplit-L", "global", "fixed_random_ll"), ("LightSplit-L", "b2", "block_random_ll")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--fullsweep", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cells, raws = {}, {}
    files = []
    for p in a.results:
        files += sorted(glob.glob(os.path.join(p, "**", "*summary*.csv"), recursive=True)) \
            if os.path.isdir(p) else [p]
    for fn in files:
        for r in csv.DictReader(open(fn)):
            if r.get("dataset") != "cifar10":
                continue
            deep = BOOL.get(str(r.get("deep_head")).lower())
            ch = BOOL.get(str(r.get("client_heads", "0")).lower())
            if deep is None or ch is None:
                continue
            try:
                acc = float(r["best_global_test_acc"]) * 100
                w = float(r.get("wcc_lambda") or 0)
                if float(r["alpha"]) < 1e6:
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            cfg = ("L2H" if deep else "L1H", "PCH" if ch else "SCH")
            mode = r.get("mode")
            if mode == "raw":
                raws.setdefault(cfg, []).append(acc)
            elif mode in ("fixed_random", "fixed_random_ll", "block_random", "block_random_ll") \
                    and w in LAMBDAS:
                cells.setdefault((mode,) + cfg + (w,), []).append(acc)

    ref = collect(a.fullsweep) if a.fullsweep else {}

    with open(a.out, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["method", "config", "proj"] + [f"w{w:g}" for w in LAMBDAS])
        for name, proj, mode in ROWS:
            for depth, head in CFGS:
                if proj == "global":
                    rr = ref.get(("raw", "cifar10", "iid", depth, head, 1))
                else:
                    rl = raws.get((depth, head))
                    rr = sum(rl) / len(rl) if rl else None
                row = [name, f"{depth}-{head}", proj]
                for w in LAMBDAS:
                    vs = cells.get((mode, depth, head, w))
                    v = sum(vs) / len(vs) if vs else None
                    row.append("" if (v is None or rr is None)
                               else f"{(v - rr) / rr * 100:+.2f}")
                wtr.writerow(row)
    print(f"[aggregate_wcc_ablation] -> {a.out}")


if __name__ == "__main__":
    main()
