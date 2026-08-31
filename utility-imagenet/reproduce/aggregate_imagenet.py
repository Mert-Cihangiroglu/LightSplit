#!/usr/bin/env python3
"""Turn the ImageNet reproduction runs into the paper's figure values.

  --figure imagenet_generality -> method, cr, best_top1, pct_of_raw
  --figure imagenet_lsl_lsf    -> pair, lsl_best_top1, lsf_best_top1, ratio_best_pct, ratio_final_tenth_pct, ratio_final_pct
                                  (+ a block_b2 reference row: block b=2 vs fixed_random at CR 8)

best_top1 is the best validation top-1 over the run (the summary CSV's best_top1, else the maximum
of the epoch CSV). pct_of_raw = 100 * best_top1 / best_top1(raw). ratio_best_pct = 100 * LS-L / LS-F
best top-1; ratio_final_tenth_pct = mean over the last 10% of epochs of the per-epoch LS-L / LS-F ratio.

--runs-map JSON overrides where an arm is read from: {arm: {"summary": path, "epochs": path,
"filter": {column: value, ...}}}; arms not listed use $ROOT/<figure>/<arm>_{summary,epochs}.csv.
"""
import argparse, csv, json, statistics as st, sys
from pathlib import Path

GEN_ARMS = ["raw"] + [f"{m}_cr{cr}" for cr in (8, 16, 32) for m in ("block_b2", "topk_sparse", "learned_1x1", "fixed_random")]
PAIRS = ["adam30", "adam100", "sgd90", "pc200"]

def div(a, b):
    return a / b if b else float("nan")

def rows(path, flt=None):
    if not Path(path).exists(): return []
    flt = flt or {}
    return [r for r in csv.DictReader(open(path)) if all(r.get(k) == str(v) for k, v in flt.items())]

def best_top1(summary, epochs, flt=None):
    s = rows(summary, flt)
    if s: return max(float(r["best_top1"]) for r in s)
    e = rows(epochs, flt)
    return max(float(r["val_top1"]) for r in e) if e else float("nan")

def final_top1(summary, epochs, flt=None):
    s = rows(summary, flt)
    if s: return max(float(r["final_top1"]) for r in s)
    c = curve(epochs, flt)
    return c[-1] if c else float("nan")

def curve(epochs, flt=None):
    e = rows(epochs, flt)
    return [float(r["val_top1"]) for r in sorted(e, key=lambda r: int(float(r["epoch"])))]

def paths(root, fig, arm, rmap):
    m = rmap.get(arm, {})
    return (m.get("summary", root / fig / f"{arm}_summary.csv"), m.get("epochs", root / fig / f"{arm}_epochs.csv"), m.get("filter"))

def generality(root, rmap):
    best = {arm: best_top1(*paths(root, "fig_imagenet_generality", arm, rmap)) for arm in GEN_ARMS}
    raw = best["raw"]; out = []
    for arm in GEN_ARMS:
        method, _, cr = arm.rpartition("_cr") if arm != "raw" else ("raw", "", "")
        out.append({"method": method, "cr": cr or 1, "best_top1": 100 * best[arm], "pct_of_raw": 100 * div(best[arm], raw)})
    return out, ["method", "cr", "best_top1", "pct_of_raw"]

def lsl_lsf(root, rmap):
    out = []
    for pair in PAIRS:
        L = paths(root, "fig_imagenet_lsl_lsf", f"{pair}_lsl", rmap); F = paths(root, "fig_imagenet_lsl_lsf", f"{pair}_lsf", rmap)
        bl, bf = best_top1(*L), best_top1(*F)
        cl, cf = curve(L[1], L[2]), curve(F[1], F[2]); n = min(len(cl), len(cf))
        ratios = [div(cl[i], cf[i]) for i in range(n)]
        tenth = st.mean(ratios[int(0.9 * n):]) if n else float("nan")
        out.append({"pair": pair, "lsl_best_top1": 100 * bl, "lsf_best_top1": 100 * bf, "ratio_best_pct": 100 * div(bl, bf),
                    "ratio_final_tenth_pct": 100 * tenth, "ratio_final_pct": 100 * div(final_top1(*L), final_top1(*F)), "epochs_compared": n})
    B, F = paths(root, "fig_imagenet_generality", "block_b2_cr8", rmap), paths(root, "fig_imagenet_generality", "fixed_random_cr8", rmap)
    b2, f8 = best_top1(*B), best_top1(*F)
    out.append({"pair": "block_b2_vs_lsf_sgd90", "lsl_best_top1": 100 * b2, "lsf_best_top1": 100 * f8, "ratio_best_pct": 100 * div(b2, f8),
                "ratio_final_tenth_pct": float("nan"), "ratio_final_pct": 100 * div(final_top1(*B), final_top1(*F)), "epochs_compared": 0})
    return out, ["pair", "lsl_best_top1", "lsf_best_top1", "ratio_best_pct", "ratio_final_tenth_pct", "ratio_final_pct", "epochs_compared"]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=["imagenet_generality", "imagenet_lsl_lsf"], required=True)
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--runs-map", type=Path); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); rmap = json.load(open(a.runs_map)) if a.runs_map else {}
    out, fields = (generality if a.figure == "imagenet_generality" else lsl_lsf)(a.root, rmap)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    for r in out: print("  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
