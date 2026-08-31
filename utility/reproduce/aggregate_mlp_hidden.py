#!/usr/bin/env python3
"""Aggregate the LightSplit-L hidden-width ablation (tab:mlp-hidden-ablation).

  python aggregate_mlp_hidden.py --results DIR [...] --fullsweep DIR_OR_CSV [...] --out CSV

--results holds the ablation grid summary CSVs (fixed_random_ll at CR 8, IID,
several --mlp-hidden widths, plus Raw baselines). The Raw reference of each
(dataset, head config) is the grid's own raw run when present, else the full
utility sweep's cell. One output row per (dataset, config): the raw reference,
then acc/delta per hidden width.
"""
import argparse, csv, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_full_sweep import collect, BOOL

WIDTHS = [64, 128, 256, 512, 1024, 2048]
DATASETS = ["cifar10", "cifar100", "fmnist", "gtsrb"]
CFGS = [("L2H", "SCH"), ("L1H", "SCH"), ("L2H", "PCH"), ("L1H", "PCH")]


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
            ds, mode = r.get("dataset"), r.get("mode")
            deep = BOOL.get(str(r.get("deep_head")).lower())
            ch = BOOL.get(str(r.get("client_heads", "0")).lower())
            if ds not in DATASETS or deep is None or ch is None:
                continue
            try:
                acc = float(r["best_global_test_acc"]) * 100
                if float(r["alpha"]) < 1e6 or float(r.get("wcc_lambda") or 0):
                    continue
                h = int(float(r.get("mlp_hidden") or 0))
            except (KeyError, ValueError, TypeError):
                continue
            cfg = ("L2H" if deep else "L1H", "PCH" if ch else "SCH")
            if mode == "raw":
                raws[(ds,) + cfg] = acc
            elif mode == "fixed_random_ll" and h in WIDTHS:
                cells[(ds,) + cfg + (h,)] = acc

    ref = collect(a.fullsweep) if a.fullsweep else {}
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "depth", "head", "raw"]
                   + [c for h in WIDTHS for c in (f"h{h}", f"h{h}_delta")])
        for ds in DATASETS:
            for depth, head in CFGS:
                rr = raws.get((ds, depth, head))
                if rr is None:
                    rr = ref.get(("raw", ds, "iid", depth, head, 1))
                row = [ds, depth, head, "" if rr is None else f"{rr:.3f}"]
                for h in WIDTHS:
                    v = cells.get((ds, depth, head, h))
                    row.append("" if v is None else f"{v:.3f}")
                    row.append("" if (v is None or rr is None) else f"{v - rr:+.3f}")
                w.writerow(row)
    print(f"[aggregate_mlp_hidden] {len(cells)} cells -> {a.out}")


if __name__ == "__main__":
    main()
