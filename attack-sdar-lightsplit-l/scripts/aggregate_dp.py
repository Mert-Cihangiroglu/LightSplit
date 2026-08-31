#!/usr/bin/env python3
"""Aggregate the X4-privacy DP cells: SDAR reconstruction SSIM vs a CORRECT health gate,
for {raw, raw+DP, fixed_random CR8, fixed_random CR8+DP} at a given cut level.

Health gate (this is the subtle part):
  * VICTIM health   = fg_acc  (the victim's OWN classification accuracy, sdar.py:301).
                      A collapsed victim has garbage z, so its SSIM is meaningless.
  * ATTACK health   = e_dis_loss  (the simulator/discriminator loss). When the SDAR GAN
                      simulator collapses it goes ~0 (discriminator perfectly separates
                      real vs simulated z) and the reconstruction SSIM is an ARTIFACT that
                      UNDER-reports leakage. Excluding these is the conservative direction.
  * We DO NOT gate on `label_acc` (the ATTACKER's label-inference head, sdar.py:298) — a
    low label_acc is a privacy win, not a victim/attack failure, so gating on it would
    wrongly drop the most-private runs.

SSIM is read from the run's text log (reseed_logs/usl_l<L>_<tag>_run<r>.log); fg_acc and
e_dis_loss are read from the saved history .npy. Prints the deliverable table and (with
--write) dumps dp_privacy_l<level>.csv.
"""
import argparse
import csv
import re
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
SSIM_RE = re.compile(r"Average SSIM over all client's images:\s*([-\d.]+)")

FG_MIN = 0.50     # victim must classify well above chance (healthy ~0.95+, collapsed <0.2)
EDIS_MIN = 0.12   # attack simulator must not have collapsed (healthy ~0.2-1.0, collapsed <0.08)

# (cell key, human label, log tag, npy path relative to scripts/)
# {L}=level, {r}=run, {sig}=sigma tag
def cell_defs(sig):
    return [
        ("raw",   "raw (undefended)",      "raw",
         "sdar_results/usl/cifar10/output/resnet_l{L}_run{r}.npy"),
        ("rawdp", "raw + DP",              f"raw_dps{sig}",
         "defense_results/dp/usl/cifar10/output/raw_dps%s_resnet_l{L}_run{r}.npy" % sig),
        ("cr8",   "fixed_random CR8",      "cr8",
         "defense_results/fixed_random/usl/cifar10/output/cr8_resnet_l{L}_run{r}.npy"),
        ("cr8dp", "fixed_random CR8 + DP", f"cr8_dps{sig}",
         "defense_results/fixed_random/usl/cifar10/output/cr8_dps%s_resnet_l{L}_run{r}.npy" % sig),
    ]


def ssim_from_log(level, tag, r):
    p = HERE / f"reseed_logs/usl_l{level}_{tag}_run{r}.log"
    if not p.exists():
        return None
    m = SSIM_RE.findall(p.read_text(errors="ignore"))
    return float(m[-1].rstrip(".")) if m else None


def tail(hist, key, n=500):
    a = hist.get(key)
    if a is None:
        return float("nan")
    a = np.asarray(a)
    return float(np.mean(a[-n:]))


def health(level, npy_pat, r):
    p = HERE / npy_pat.format(L=level, r=r)
    if not p.exists():
        return None, None
    h = np.load(p, allow_pickle=True).item()
    return tail(h, "fg_acc"), tail(h, "e_dis_loss")


def main(write, level, sig, runs):
    rows = []
    for key, label, tag, npy_pat in cell_defs(sig):
        kept, dropped = [], []
        for r in runs:
            s = ssim_from_log(level, tag, r)
            if s is None:
                continue
            fg, ed = health(level, npy_pat, r)
            if fg is None:
                dropped.append((r, s, None, None, "no-history"))
                continue
            if fg > FG_MIN and ed > EDIS_MIN:
                kept.append((r, s, fg, ed))
            else:
                why = "victim-collapsed" if fg <= FG_MIN else "attack-collapsed"
                dropped.append((r, s, fg, ed, why))
        print(f"\n=== {label}  (tag={tag}) ===")
        for r, s, fg, ed in kept:
            print(f"  run{r}: SSIM={s:.3f}  fg_acc={fg:.3f}  e_dis={ed:.3f}  [ok]")
        for r, s, fg, ed, why in dropped:
            fgs = "-" if fg is None else f"{fg:.3f}"
            eds = "-" if ed is None else f"{ed:.3f}"
            print(f"  run{r}: SSIM={s:.3f}  fg_acc={fgs}  e_dis={eds}  [DROP: {why}]")
        ss = [s for _, s, _, _ in kept]
        if ss:
            ms = float(np.mean(ss)); sd = float(np.std(ss)) if len(ss) > 1 else 0.0
            ma = float(np.mean([fg for _, _, fg, _ in kept]))
            print(f"  -> SSIM = {ms:.3f} +/- {sd:.3f}  (victim acc {ma:.2f}, n={len(kept)} kept, {len(dropped)} dropped)")
            rows.append((key, label, ms, sd, ma, len(kept), len(dropped)))
        else:
            print(f"  -> no healthy runs ({len(dropped)} dropped)")
            rows.append((key, label, None, None, None, 0, len(dropped)))

    print(f"\n================  DP privacy table (L{level}, sigma={sig})  ================")
    print(f"{'cell':28s} {'SSIM':>16s} {'victim_acc':>11s}   n_kept")
    for key, label, ms, sd, ma, nk, nd in rows:
        ss = "n/a" if ms is None else f"{ms:.3f} +/- {sd:.3f}"
        aa = "-" if ma is None else f"{ma:.3f}"
        print(f"{label:28s} {ss:>16s} {aa:>11s}   {nk}")

    if write:
        out = HERE / f"dp_privacy_l{level}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["level", "cell", "label", "sigma", "ssim", "ssim_std", "victim_acc", "n_kept", "n_dropped"])
            for key, label, ms, sd, ma, nk, nd in rows:
                w.writerow([level, key, label, sig,
                            "" if ms is None else f"{ms:.6f}",
                            "" if sd is None else f"{sd:.6f}",
                            "" if ma is None else f"{ma:.6f}", nk, nd])
        print(f"\nwrote {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--sigma", default="0.02")
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = ap.parse_args()
    main(args.write, args.level, args.sigma, args.runs)
