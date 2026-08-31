#!/usr/bin/env python
"""Aggregate the HAR label-heterogeneity sweep into tab:har_noniid.

Each run directory holds summary.json (train_noniid.py); a cell is the mean over seeds of test_acc
(test accuracy at the best-validation epoch). The table prints the difference in points to the Raw
cell of the same alpha; the Raw row prints its accuracy. --exclude alpha:seed drops one seed at one
alpha (the paper drops seed 2 at alpha 0.1, where every arm including Raw collapses in that seed).
--iid-csv adds the alpha -> infinity column from the single-client IID sweep (fig_har.csv: mean
test_acc per method and k).

  python aggregate_har_noniid.py --results results/reproduce/tab_har_noniid --iid-csv results/reproduce/fig_har.csv \
      --exclude 0.1:2 --out results/reproduce/tab_har_noniid.csv
"""
import argparse, csv, glob, json, statistics as st
from pathlib import Path

METHODS = [("fixed", 0.0, "LS-F"), ("fixed", 0.1, "LS-F+WCC"), ("learned", 0.0, "LS-L"), ("learned", 0.1, "LS-L+WCC"),
           ("codec", 0.0, "learned_1x1"), ("topk_sparse", 0.0, "topk_sparse")]
CR = {32: 8, 16: 16, 8: 32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True, help="directory of run folders (repeatable)")
    ap.add_argument("--iid-csv", help="fig_har.csv of the IID sweep, for the alpha -> infinity column")
    ap.add_argument("--exclude", action="append", default=[], help="alpha:seed to leave out, e.g. 0.1:2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    excl = {(float(x.split(":")[0]), int(x.split(":")[1])) for x in a.exclude}
    runs = [json.load(open(p)) for d in a.results for p in glob.glob(str(Path(d) / "*" / "summary.json"))]
    cells = {}
    for r in runs:
        if (r["alpha"], r["seed"]) in excl:
            continue
        key = (r["liftback"], float(r["wcc_lambda"]), r["k"] if r["liftback"] != "none" else 0, r["alpha"])
        cells.setdefault(key, {})[r["seed"]] = 100 * r["test_acc"]
    alphas = sorted({k[3] for k in cells}, reverse=True)
    iid = {}
    if a.iid_csv:
        for r in csv.DictReader(open(a.iid_csv)):
            iid[(r["method"], r["k"])] = float(r["test_acc"])
    mean = lambda d: st.mean(d.values()) if d else float("nan")
    raw = {al: mean(cells.get(("none", 0.0, 0, al), {})) for al in alphas}
    raw_iid = iid.get(("raw", "256"), float("nan"))
    cols = ["method", "cr", "k", "iid_delta"] + [f"a{al:g}_delta" for al in alphas] + [f"a{al:g}_n" for al in alphas]
    out = [{"method": "raw", "cr": 1, "k": "", "iid_delta": f"{raw_iid:.2f}", **{f"a{al:g}_delta": f"{raw[al]:.2f}" for al in alphas},
            **{f"a{al:g}_n": len(cells.get(("none", 0.0, 0, al), {})) for al in alphas}}]
    for lb, w, name in METHODS:
        for k in (32, 16, 8):
            row = {"method": name, "cr": CR[k], "k": k}
            v = iid.get((name, str(k)))
            row["iid_delta"] = f"{v - raw_iid:.2f}" if v is not None and raw_iid == raw_iid else ""
            for al in alphas:
                d = cells.get((lb, w, k, al), {})
                row[f"a{al:g}_delta"] = f"{mean(d) - raw[al]:.2f}" if d else ""; row[f"a{al:g}_n"] = len(d)
            out.append(row)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(out)
    print(f"wrote {a.out} ({len(out)} rows; Raw row carries accuracies, other rows differences in points)")


if __name__ == "__main__":
    main()
