#!/usr/bin/env python3
"""Aggregate the SDAR cut-width (d) sweep with victim AND attacker health gating.

One row per (d, CR, lambda_WCC) cell, pooled over seeds.

Health gating -- two separate things can go wrong in an SDAR run and they mean opposite things:

  * victim collapse  -> the split model never learned the task. fg_loss stays high. The cell is
    meaningless (a useless model is not a private one). Gate: final fg_loss <= VICTIM_MAX.

  * attacker non-convergence -> SDAR's simulator/discriminator GAN diverged, so the decoder was
    trained against a simulator that never matched the victim's activations. attack_loss stays flat
    or rises across training and e_dis_loss collapses (the discriminator has won). SSIM then lands
    near 0.05-0.11 for reasons that have nothing to do with the defense. Reporting those cells as
    "private" is exactly backwards, so they are flagged.

The attacker flag is deliberately advisory rather than a hard filter: under a genuinely strong
defense the attacker also fails to improve, and we must not confuse "defense worked" with "GAN
diverged". The tell that separates them is e_dis_loss -- whether the SIMULATOR matched the victim's
activation distribution, which is upstream of whether the decoder can invert it. So the table
reports the median over all victim-healthy seeds (robust to a minority of diverged seeds, matching
the paper's protocol) alongside the max (strongest observed attack) and the healthy-seed count.

Usage: python3 aggregate_dsweep.py [--logdir DIR] [--csv OUT.csv]
"""
import argparse
import os
import re
import statistics
import sys
from collections import defaultdict

# width key -> cut-layer dimensionality d at level 7 (8x8 spatial)
WIDTH_D = {"cut8": 512, "cut16": 1024, "cut32": 2048, "cut64": 4096, "cut128": 8192,
           "standard": 4096}

VICTIM_MAX = 0.60      # final fg_loss above this => victim never learned; drop the seed
EDIS_MIN = 0.20        # final e_dis_loss below this => simulator discriminator won; flag it
ATTACK_DROP_MIN = 0.10  # attack_loss must fall >=10% from early to final; else flag it

ITER_RE = re.compile(r"^\[(\d+)\]:.*?fg_loss: ([0-9.]+).*?e_dis_loss: ([0-9.]+).*?attack_loss: ([0-9.]+)")
SSIM_RE = re.compile(r"Average SSIM over all client's images: (\d+\.\d+)")
TAG_RE = re.compile(r"^(?:fr_cr(?P<cr>\d+)_wcc(?P<wcc>[0-9.]+)_(?P<w1>\w+?)|raw_(?P<w2>\w+?)|parity_cr8_(?P<w3>standard))_run(?P<seed>\d+)$")


def parse_log(path):
    """-> dict with ssim + health telemetry, or None if the run has not finished."""
    ssim = None
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = ITER_RE.match(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
                continue
            m = SSIM_RE.search(line)
            if m:
                ssim = float(m.group(1))
    if ssim is None or not rows:
        return None
    rows.sort()
    early = rows[len(rows) // 4]          # ~25% through training
    final = rows[-1]
    drop = (early[3] - final[3]) / early[3] if early[3] > 0 else 0.0
    return {
        "ssim": ssim,
        "fg_loss": final[1],
        "e_dis_loss": final[2],
        "attack_loss_early": early[3],
        "attack_loss_final": final[3],
        "attack_drop": drop,
        "victim_ok": final[1] <= VICTIM_MAX,
        "attacker_ok": final[2] >= EDIS_MIN and drop >= ATTACK_DROP_MIN,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cells = defaultdict(list)   # (d, cr, wcc) -> [(seed, rec)]
    for fn in sorted(os.listdir(args.logdir)):
        if not fn.endswith(".log"):
            continue
        tag = fn[:-4]
        m = TAG_RE.match(tag)
        if not m:
            continue
        rec = parse_log(os.path.join(args.logdir, fn))
        if rec is None:
            continue
        w = m.group("w1") or m.group("w2") or m.group("w3")
        if m.group("w3"):
            key = (WIDTH_D[w], 8.0, 0.0, "parity")
        elif m.group("w2"):
            key = (WIDTH_D[w], 1.0, 0.0, "raw")
        else:
            key = (WIDTH_D[w], float(m.group("cr")), float(m.group("wcc")), "proj")
        cells[key].append((int(m.group("seed")), rec))

    out = []
    for key in sorted(cells):
        d, cr, wcc, kind = key
        runs = sorted(cells[key])
        healthy = [r for _, r in runs if r["victim_ok"]]
        conv = [r for r in healthy if r["attacker_ok"]]
        if not healthy:
            continue
        ss = [r["ssim"] for r in healthy]
        out.append({
            "kind": kind, "d": d, "k": int(round(d / cr)) if kind != "raw" else d,
            "cr": cr, "wcc": wcc,
            "n": len(runs), "n_victim_ok": len(healthy), "n_attacker_conv": len(conv),
            "ssim_median": statistics.median(ss),
            "ssim_max": max(ss),
            "ssim_median_conv": statistics.median([r["ssim"] for r in conv]) if conv else float("nan"),
            "seeds": ",".join(f"{s}{'' if r['attacker_ok'] else '*'}" for s, r in runs),
        })

    hdr = f"{'kind':6s} {'d':>5s} {'CR':>4s} {'k':>5s} {'wcc':>5s} {'n':>3s} {'conv':>4s} {'med':>6s} {'medC':>6s} {'max':>6s}  seeds"
    print(hdr)
    print("-" * len(hdr))
    for r in out:
        mc = "  n/a " if r["ssim_median_conv"] != r["ssim_median_conv"] else f"{r['ssim_median_conv']:6.3f}"
        print(f"{r['kind']:6s} {r['d']:5d} {r['cr']:4.0f} {r['k']:5d} {r['wcc']:5.3g} "
              f"{r['n_victim_ok']:3d} {r['n_attacker_conv']:4d} {r['ssim_median']:6.3f} {mc} {r['ssim_max']:6.3f}  {r['seeds']}")
    print("\n* on a seed = attacker did not converge (e_dis_loss < %.2f or attack_loss fell < %d%%);"
          " 'medC' is the median over converged seeds only." % (EDIS_MIN, int(ATTACK_DROP_MIN * 100)))

    if args.csv:
        import csv
        if not out:
            print("no complete cell found in", args.logdir, "(a run counts once its log has the full iteration trace and an SSIM line)")
            return 0
        with open(args.csv, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            wtr.writeheader()
            wtr.writerows(out)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
