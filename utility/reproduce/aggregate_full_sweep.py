#!/usr/bin/env python3
"""Aggregate train_split.py summary CSVs into the full utility sweep table
(tab:full_sweep_baseline_appendix) and the utility columns of the compact
comparison (tab:main-compact).

  python aggregate_full_sweep.py --results DIR [--results DIR ...] --out tab_full_sweep.csv

Every --results argument is a summary CSV or a directory searched recursively
for *summary*.csv. Rows are keyed by (method, CR, depth, head, dataset,
distribution); when the same cell appears more than once the run with the
larger epoch budget wins (ties: the row read last).

Main output: one row per (method, CR, depth, head) with best test accuracy (%)
and the delta vs the matched Raw cell, per dataset x {non-IID, IID}.
Also writes <out stem>_main_utility.csv: the L2H-SCH cells as percent change
relative to Raw (IID per dataset, plus the mean over the non-IID splits).
"""
import argparse, csv, glob, os

DATASETS = ["cifar10", "cifar100", "fmnist", "gtsrb"]
NONIID_ALPHA = {"cifar10": 0.1, "cifar100": 0.5, "fmnist": 0.1, "gtsrb": 0.1}
METHODS = [("raw", "Raw"), ("fixed_random", "LightSplit-F"), ("fixed_random_ll", "LightSplit-L"),
           ("learned_1x1", "Learned1x1"), ("topk_sparse", "topk_sparse")]
MODE_OF = {m: n for m, n in METHODS}
CRS = [8, 16, 32]
BOOL = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*summary*.csv"), recursive=True))
        else:
            files.append(p)
    cells = {}   # (mode, ds, dist, depth, head, cr) -> (epochs, acc)
    for fn in files:
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                mode = r.get("mode")
                if mode not in MODE_OF or not r.get("best_global_test_acc"):
                    continue
                deep = BOOL.get(str(r.get("deep_head")).lower())
                ch = BOOL.get(str(r.get("client_heads", "0")).lower())
                if deep is None or ch is None:
                    continue
                try:
                    acc = float(r["best_global_test_acc"]) * 100
                    alpha = float(r["alpha"])
                    ep = int(float(r.get("epochs", 0)))
                    if float(r.get("wcc_lambda") or 0) or float(r.get("recon_lambda") or 0):
                        continue
                    k = int(float(r.get("k") or 0))
                    act = int(float(r.get("activation_dim") or 0))
                except (KeyError, ValueError):
                    continue
                ds = r["dataset"]
                if ds not in DATASETS:
                    continue
                if alpha >= 1e6:
                    dist = "iid"
                elif abs(alpha - NONIID_ALPHA[ds]) < 1e-9:
                    dist = "noniid"
                else:
                    continue
                if mode == "raw":
                    cr = 1
                elif k > 0 and act > 0:
                    cr = min(CRS, key=lambda c: abs(act / c - k))
                else:
                    continue
                key = (mode, ds, dist, "L2H" if deep else "L1H", "PCH" if ch else "SCH", cr)
                if key not in cells or ep >= cells[key][0]:
                    cells[key] = (ep, acc)
    return {k: v[1] for k, v in cells.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cells = collect(args.results)

    def get(mode, ds, dist, depth, head, cr):
        return cells.get((mode, ds, dist, depth, head, cr))

    cols = [f"{ds}_{dist}" for ds in DATASETS for dist in ("noniid", "iid")]
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "cr", "depth", "head"] + [c for col in cols for c in (col, col + "_delta")])
        for mode, name in METHODS:
            for cr in ([1] if mode == "raw" else CRS):
                for depth in ("L1H", "L2H"):
                    for head in ("SCH", "PCH"):
                        row = [name, cr, depth, head]
                        for ds in DATASETS:
                            for dist in ("noniid", "iid"):
                                a = get(mode, ds, dist, depth, head, cr)
                                raw = get("raw", ds, dist, depth, head, 1)
                                row.append("" if a is None else f"{a:.2f}")
                                row.append("" if (a is None or raw is None or mode == "raw")
                                           else f"{a - raw:+.2f}")
                        w.writerow(row)

    # tab:main-compact utility columns: L2H-SCH cells as percent change vs Raw.
    out2 = os.path.splitext(args.out)[0] + "_main_utility.csv"
    raw_iid = {ds: get("raw", ds, "iid", "L2H", "SCH", 1) for ds in DATASETS}
    raw_nn = [get("raw", ds, "noniid", "L2H", "SCH", 1) for ds in DATASETS]
    raw_nn_mean = sum(raw_nn) / len(raw_nn) if all(v is not None for v in raw_nn) else None
    with open(out2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "cr"] + [f"{ds}_iid_pct" for ds in DATASETS] + ["noniid_mean_pct"])
        w.writerow(["Raw", 1]
                   + ["" if raw_iid[ds] is None else f"{raw_iid[ds]:.2f}" for ds in DATASETS]
                   + ["" if raw_nn_mean is None else f"{raw_nn_mean:.2f}"])
        for mode, name in METHODS[1:]:
            for cr in CRS:
                row = [name, cr]
                for ds in DATASETS:
                    a = get(mode, ds, "iid", "L2H", "SCH", cr)
                    row.append("" if (a is None or raw_iid[ds] is None)
                               else f"{(a - raw_iid[ds]) / raw_iid[ds] * 100:+.1f}")
                nn = [get(mode, ds, "noniid", "L2H", "SCH", cr) for ds in DATASETS]
                if all(v is not None for v in nn) and raw_nn_mean:
                    d = sum(nn) / len(nn) - raw_nn_mean
                    row.append(f"{d / raw_nn_mean * 100:+.1f}")
                else:
                    row.append("")
                w.writerow(row)
    print(f"[aggregate_full_sweep] {sum(1 for v in cells.values() if v is not None)} cells -> {args.out}, {out2}")


if __name__ == "__main__":
    main()
