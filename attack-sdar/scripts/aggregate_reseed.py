#!/usr/bin/env python3
"""
Aggregate the re-seeded SDAR privacy cells (8x+wcc0.1, 32x+wcc0.01, 32x+wcc0.1).
For each cell, collect SSIM + victim accuracy from run1 (original) + runs 2,3,4 (reseed),
report per-run values and mean+/-std over HEALTHY victims (label acc >= 0.10; a victim that
collapsed below chance has a meaningless reconstruction and is excluded + flagged).

Prints a summary and (with --write) updates ssim in wcc_cr_grid.csv with the healthy mean.
"""
import argparse
import csv
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
# fine WCC grid (0.001/0.003/0.03 added so the 0->0.01 accuracy cliff is resolved)
CELLS = [(cr, w) for cr in (8, 16, 32) for w in ("0", "0.001", "0.003", "0.01", "0.03", "0.1")]
CHANCE = 0.40        # victim must classify well above chance to count as healthy (diverged ~0.05-0.10)
SSIM_RE = re.compile(r"Average SSIM over all client's images:\s*([-\d.]+)")
ACC_RE = re.compile(r"Average label accuracy over all client's images:\s*([-\d.]+)")


def orig_log(cr, wcc, level):
    # blackbox single-run original; exists only for the coarse lambda {0,0.01,0.1}
    return HERE / f"blackbox_usl_l{level}_cr{cr}_logs/usl_resnet_cifar10_l{level}_run1_fixed_random_cr{cr}_wcc{wcc}.log"


def reseed_log(cr, wcc, run, level):
    return HERE / f"reseed_logs/usl_l{level}_cr{cr}_wcc{wcc}_run{run}.log"


def parse(path):
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    s = SSIM_RE.findall(txt); a = ACC_RE.findall(txt)
    if not s or not a:
        return None
    return float(s[-1].rstrip(".")), float(a[-1].rstrip("."))


def main(write, level):
    summary = {}
    for cr, wcc in CELLS:
        runs = []
        v = parse(orig_log(cr, wcc, level))                  # blackbox original (coarse lambda only)
        if v is not None:
            runs.append(("orig", v[0], v[1]))
        for r in (1, 2, 3, 4):
            v = parse(reseed_log(cr, wcc, r, level))
            if v is not None:
                runs.append((r, v[0], v[1]))                 # (run, ssim, acc)
        print(f"\n=== L{level}  CR{cr}x  wcc{wcc} ===")
        if not runs:
            print("  (no results yet)"); continue
        for run, ssim, acc in runs:
            tag = "DIVERGED" if acc < CHANCE else "ok"
            print(f"  run{run}: SSIM={ssim:.3f}  victim_acc={acc:.3f}  [{tag}]")
        healthy = [(ssim, acc) for _, ssim, acc in runs if acc >= CHANCE]
        if healthy:
            ms = st.mean([s for s, _ in healthy]); sd = st.pstdev([s for s, _ in healthy]) if len(healthy) > 1 else 0.0
            ma = st.mean([a for _, a in healthy])
            print(f"  -> healthy mean SSIM = {ms:.3f} +/- {sd:.3f}  (victim acc {ma:.2f}, n={len(healthy)} of {len(runs)})")
            summary[(cr, wcc)] = (ms, sd, ma, len(healthy), len(runs))
        else:
            print(f"  -> ALL {len(runs)} victims diverged: reconstruction not measurable (flag, do not plot)")
            summary[(cr, wcc)] = (None, None, None, 0, len(runs))

    if write:
        out = HERE / f"wcc_cr_grid_l{level}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["level", "cr", "wcc", "ssim", "ssim_std", "victim_acc", "n_healthy", "n_total"])
            for (cr, wcc), (ms, sd, ma, nh, nt) in summary.items():
                w.writerow([level, cr, wcc,
                            "" if ms is None else f"{ms:.6f}",
                            "" if sd is None else f"{sd:.6f}",
                            "" if ma is None else f"{ma:.6f}", nh, nt])
        print(f"\nwrote healthy-mean SSIM grid into {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write healthy means to wcc_cr_grid_l<level>.csv")
    ap.add_argument("--level", type=int, default=7, help="SDAR split level to aggregate (default 7 = deployed cut)")
    args = ap.parse_args()
    main(args.write, args.level)
