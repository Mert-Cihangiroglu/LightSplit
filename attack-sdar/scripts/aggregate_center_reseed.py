#!/usr/bin/env python3
"""
Aggregate the SDAR CENTER-LOSS privacy cells (Center loss vs WCC).

Reads reseed_logs/usl_l<level>_cr<cr>_center<c>_run<r>.log, prints per-run SSIM +
attacker label accuracy, and the UNGATED mean SSIM over all seeds. We report ungated
to match the X1 WCC trade-off figure: a low SSIM under the defense is the ATTACK
failing (the intended effect), not the victim collapsing, so gating it out would hide
the defense's success. A gated mean (attacker-label acc >= 0.40) is also printed for
reference. With --write, dumps center_cr<cr>_l<level>.csv.

Mirrors aggregate_reseed.py (WCC) so the figure can overlay the two arms directly.
"""
import argparse
import csv
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANCE = 0.40
SSIM_RE = re.compile(r"Average SSIM over all client's images:\s*([-\d.]+)")
ACC_RE = re.compile(r"Average label accuracy over all client's images:\s*([-\d.]+)")


def reseed_log(cr, c, run, level):
    return HERE / f"reseed_logs/usl_l{level}_cr{cr}_center{c}_run{run}.log"


def parse(path):
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    s = SSIM_RE.findall(txt)
    a = ACC_RE.findall(txt)
    if not s or not a:
        return None
    return float(s[-1].rstrip(".")), float(a[-1].rstrip("."))


def main(write, level, cr, centers, runs):
    rows = []
    for c in centers:
        cell_runs = []
        for r in runs:
            v = parse(reseed_log(cr, c, r, level))
            if v is not None:
                cell_runs.append((r, v[0], v[1]))
        print(f"\n=== L{level}  CR{cr}x  center{c} ===")
        if not cell_runs:
            print("  (no results yet)")
            continue
        for r, ssim, acc in cell_runs:
            tag = "DIVERGED" if acc < CHANCE else "ok"
            print(f"  run{r}: SSIM={ssim:.3f}  attacker_label_acc={acc:.3f}  [{tag}]")
        all_ssim = [s for _, s, _ in cell_runs]
        un_m = st.mean(all_ssim)
        un_sd = st.pstdev(all_ssim) if len(all_ssim) > 1 else 0.0
        healthy = [s for _, s, a in cell_runs if a >= CHANCE]
        g_m = st.mean(healthy) if healthy else None
        msg = f"  -> UNGATED mean SSIM = {un_m:.3f} +/- {un_sd:.3f} (n={len(all_ssim)})"
        msg += f" | gated mean = {g_m:.3f} (n={len(healthy)})" if g_m is not None else " | gated: none healthy"
        print(msg)
        rows.append((c, un_m, un_sd, g_m, len(all_ssim), len(healthy)))

    if write and rows:
        out = HERE / f"center_cr{cr}_l{level}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["level", "cr", "center", "ssim_ungated", "ssim_ungated_std",
                        "ssim_gated", "n_total", "n_healthy"])
            for c, un_m, un_sd, g_m, nt, nh in rows:
                w.writerow([level, cr, c, f"{un_m:.6f}", f"{un_sd:.6f}",
                            "" if g_m is None else f"{g_m:.6f}", nt, nh])
        print(f"\nwrote {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write center_cr<cr>_l<level>.csv")
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--cr", type=int, default=8)
    ap.add_argument("--centers", nargs="+", default=["0.01", "0.1"])
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    main(args.write, args.level, args.cr, args.centers, args.runs)
