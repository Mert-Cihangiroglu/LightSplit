#!/usr/bin/env python3
"""Aggregate the block-local utility grid (tab:block_util_F/L) and the merged
DP-composition utility table (tab:dp_utility).

  python aggregate_block_dp.py --util2k DIR [--dpraw DIR --dpblk DIR]
      --fullsweep DIR_OR_CSV [...] [--out-block CSV] [--out-dp CSV]

--util2k dirs hold util2k_<ds>_<depth>_<head>_b<B>[_a05]_summary.csv files
(block_random = transpose lift-back F, block_random_ll = learned lift-back L).
--dpraw / --dpblk hold the +DP twins (dpraw_*_summary.csv, dpblk_*_<arm>_summary.csv
with arm in b1/b2/b4/fr/frll). --fullsweep feeds the no-DP Raw / LightSplit-F /
LightSplit-L reference cells (same inputs as aggregate_full_sweep.py).
Every cell is best test accuracy; deltas are accuracy points vs the matched
reference. Duplicate cells: the run with the larger epoch budget wins.
"""
import argparse, csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_full_sweep import collect, BOOL, DATASETS, NONIID_ALPHA

CRS = [8, 16, 32]
CFGS = [("L1H", "SCH"), ("L1H", "PCH"), ("L2H", "SCH"), ("L2H", "PCH")]
COLS = [("cifar10", "noniid", "c10_a01"), ("cifar10", "iid", "c10_iid"),
        ("cifar100", "noniid", "c100_a05"), ("cifar100", "iid", "c100_iid"),
        ("fmnist", "noniid", "fm_a01"), ("fmnist", "iid", "fm_iid"),
        ("gtsrb", "noniid", "gt_a01"), ("gtsrb", "iid", "gt_iid")]


def scan(dirs, pattern, arm_of, dedup="mean"):
    """Collect cells; a duplicated cell is averaged (dedup="mean", the block
    utility grid) or resolved to the last run read (dedup="last", the DP arms).
    The requested CR is recovered from k (the compression_ratio column holds
    the realised ratio, which differs when the tile cannot shrink further)."""
    cells = {}
    for d in dirs or []:
        files = sorted(glob.glob(os.path.join(d, pattern))) if os.path.isdir(d) else [d]
        for fn in files:
            arm = arm_of(os.path.basename(fn))
            if arm is None:
                continue
            for r in csv.DictReader(open(fn)):
                try:
                    acc = float(r["best_global_test_acc"]) * 100
                    alpha = float(r["alpha"])
                    k = int(float(r.get("k") or 0))
                    act = int(float(r.get("activation_dim") or 0))
                except (KeyError, ValueError, TypeError):
                    continue
                ds, mode = r.get("dataset"), r.get("mode")
                if ds not in DATASETS:
                    continue
                deep = BOOL.get(str(r.get("deep_head")).lower())
                ch = BOOL.get(str(r.get("client_heads", "0")).lower())
                if deep is None or ch is None:
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
                key = (arm, mode, ds, dist, "L2H" if deep else "L1H",
                       "PCH" if ch else "SCH", cr)
                cells.setdefault(key, []).append(acc)
    if dedup == "last":
        return {k: v[-1] for k, v in cells.items()}
    return {k: sum(v) / len(v) for k, v in cells.items()}


def arm_util2k(name):
    m = re.search(r"_b([124])(_a05)?_summary", name)
    return f"b{m.group(1)}" if m else None


def arm_dpblk(name):
    m = re.search(r"_(b1|b2|b4|fr|frll)_summary", name)
    return m.group(1) if m else None


def header():
    cols = ["method", "cr", "depth", "head"]
    for _, _, c in COLS:
        cols += [c + "_f", c + "_l"]
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util2k", action="append", default=[])
    ap.add_argument("--dpraw", action="append", default=[])
    ap.add_argument("--dpblk", action="append", default=[])
    ap.add_argument("--fullsweep", action="append", default=[])
    ap.add_argument("--out-block")
    ap.add_argument("--out-dp")
    a = ap.parse_args()

    u2k = scan(a.util2k, "util2k_*summary*.csv", arm_util2k)
    ref = collect(a.fullsweep) if a.fullsweep else {}

    def fmt(v, plus=True, nd=3):
        return "" if v is None else (f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}")

    def block_cell(arm, mode, ds, dist, depth, head, cr):
        v = u2k.get((arm, mode, ds, dist, depth, head, cr))
        r = ref.get(("raw", ds, dist, depth, head, 1))
        return None if v is None or r is None else v - r

    if a.out_block:
        with open(a.out_block, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header())
            for depth, head in CFGS:
                row = ["Raw", 1, depth, head]
                for ds, dist, _ in COLS:
                    row += [fmt(ref.get(("raw", ds, dist, depth, head, 1)), False), ""]
                w.writerow(row)
            for b in ("b1", "b2", "b4"):
                for cr in CRS:
                    for depth, head in CFGS:
                        row = [b, cr, depth, head]
                        for ds, dist, _ in COLS:
                            row.append(fmt(block_cell(b, "block_random", ds, dist, depth, head, cr)))
                            row.append(fmt(block_cell(b, "block_random_ll", ds, dist, depth, head, cr)))
                        w.writerow(row)
        print(f"[aggregate_block_dp] block table -> {a.out_block}")

    if a.out_dp:
        dpr = scan(a.dpraw, "dpraw_*summary*.csv", lambda n: "raw", dedup="last")
        dpb = scan(a.dpblk, "dpblk_*summary*.csv", arm_dpblk, dedup="last")

        def dp_delta(cell, refv):
            return None if cell is None or refv is None else cell - refv

        with open(a.out_dp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header())
            for depth, head in CFGS:
                row = ["Raw+DP", 1, depth, head]
                for ds, dist, _ in COLS:
                    d = dp_delta(dpr.get(("raw", "raw", ds, dist, depth, head, 1)),
                                 ref.get(("raw", ds, dist, depth, head, 1)))
                    row += [fmt(d, nd=2), ""]
                w.writerow(row)
            for cr in CRS:
                for depth, head in CFGS:
                    row = ["LS+DP", cr, depth, head]
                    for ds, dist, _ in COLS:
                        f = dp_delta(dpb.get(("fr", "fixed_random", ds, dist, depth, head, cr)),
                                     ref.get(("fixed_random", ds, dist, depth, head, cr)))
                        l = dp_delta(dpb.get(("frll", "fixed_random_ll", ds, dist, depth, head, cr)),
                                     ref.get(("fixed_random_ll", ds, dist, depth, head, cr)))
                        row += [fmt(f, nd=2), fmt(l, nd=2)]
                    w.writerow(row)
            for b in ("b1", "b2", "b4"):
                for cr in CRS:
                    for depth, head in CFGS:
                        row = [b + "+DP", cr, depth, head]
                        for ds, dist, _ in COLS:
                            d = dp_delta(dpb.get((b, "block_random", ds, dist, depth, head, cr)),
                                         u2k.get((b, "block_random", ds, dist, depth, head, cr)))
                            row += [fmt(d, nd=2), ""]
                        w.writerow(row)
        print(f"[aggregate_block_dp] dp table -> {a.out_dp}")


if __name__ == "__main__":
    main()
