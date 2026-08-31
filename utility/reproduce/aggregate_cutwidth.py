#!/usr/bin/env python3
"""Cut-width utility panel (fig:cutwidth b): mean best held-out test accuracy over seeds per (cut width, arm).

Reads $ROOT/fig_cutwidth_utility/cut<C>_<arm>_seed<S>_summary.csv (arm = raw | cr8 | cr16 | cr32), column
best_global_test_acc. Writes method, cut_channels, d (= 64*C), cr, test_acc (%), test_acc_std (sample std), n_seeds.
"""
import argparse, csv, statistics as st
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 123])
    ap.add_argument("--cuts", type=int, nargs="+", default=[8, 16, 32, 64, 128]); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); rows = []
    for C in a.cuts:
        for arm, method, cr in (("raw", "raw", 1), ("cr8", "LS-F", 8), ("cr16", "LS-F", 16), ("cr32", "LS-F", 32)):
            vals = []
            for s in a.seeds:
                f = a.root / "fig_cutwidth_utility" / f"cut{C}_{arm}_seed{s}_summary.csv"
                if not f.exists(): print(f"[missing] {f}"); continue
                r = list(csv.DictReader(open(f)))[-1]; v = float(r["best_global_test_acc"]); vals.append(100 * v if v <= 1.5 else v)
            rows.append({"method": method, "cut_channels": C, "d": 64 * C, "cr": cr, "test_acc": st.mean(vals) if vals else float("nan"),
                         "test_acc_std": st.stdev(vals) if len(vals) > 1 else float("nan"), "n_seeds": len(vals)})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows: print("  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
