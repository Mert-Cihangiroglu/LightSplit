#!/usr/bin/env python3
"""Turn the HAR reproduction runs into the values plotted in fig:har_cr8 / fig:har_generality / fig:har_privacy.

Per cell (method x CR) over seeds: test_acc (best-checkpoint test accuracy, %), recon_r2 (median per-feature
R^2 of the decoder probe) and reid (subject re-identification accuracy), each with the population std
(_pstd) and the sample std (_std); the paper's bars show _pstd for accuracy and _std for the two privacy metrics.
Reads $ROOT/fig_har/<liftback>_w<wcc>_(k<k>|none)_s<seed>/{summary.json,attacks.json}.
--runs-map JSON overrides cell directories: {"<liftback>_w<wcc>_k<k>": {seed: dir}}.
"""
import argparse, csv, json, statistics as st
from pathlib import Path

METHODS = [("raw", "none", "0"), ("LS-F", "fixed", "0"), ("LS-F+WCC", "fixed", "0.1"), ("LS-L", "learned", "0"),
           ("LS-L+WCC", "learned", "0.1"), ("learned_1x1", "codec", "0"), ("topk_sparse", "topk_sparse", "0")]
KS = {8: 32, 16: 16, 32: 8}

def load(d):
    d = Path(d)
    if not (d / "attacks.json").exists(): return None
    s, a = json.load(open(d / "summary.json")), json.load(open(d / "attacks.json"))
    return {"test_acc": 100 * s["test_acc"], "recon_r2": a["recon_r2_median"], "reid": a["reid_acc"]}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs-map", type=Path); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); rmap = json.load(open(a.runs_map)) if a.runs_map else {}
    rows = []
    for method, lb, w in METHODS:
        for cr, k in ([(1, None)] if lb == "none" else KS.items()):
            tag = f"{lb}_w{w}_" + ("none" if k is None else f"k{k}")
            vals = []
            for s in a.seeds:
                d = rmap.get(tag, {}).get(str(s), a.root / "fig_har" / f"{tag}_s{s}")
                v = load(d)
                if v is None: print(f"[missing] {d}")
                else: vals.append(v)
            row = {"method": method, "cr": cr, "k": k or 256, "n_seeds": len(vals)}
            for m in ("test_acc", "recon_r2", "reid"):
                x = [v[m] for v in vals]
                row[m] = st.mean(x) if x else float("nan")
                row[m + "_pstd"] = st.pstdev(x) if x else float("nan")
                row[m + "_std"] = st.stdev(x) if len(x) > 1 else float("nan")
            rows.append(row)
    fields = ["method", "cr", "k", "n_seeds"] + [f"{m}{s}" for m in ("test_acc", "recon_r2", "reid") for s in ("", "_pstd", "_std")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=fields); wri.writeheader(); wri.writerows(rows)
    for r in rows: print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
