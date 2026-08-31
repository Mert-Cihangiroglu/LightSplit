#!/usr/bin/env python3
"""Aggregate the HAR DP probe cells of the privacy overview (tab:privacy_all).

  python aggregate_har_dp.py --dp DIR --fig-har CSV --out CSV

--dp holds the run_dp_har-style output (none_w0_dp_s{0,1,2}/attacks.json and
fixed_w0_dp_s{0,1,2}/attacks.json). Each arm's cell is the MEDIAN over seeds of
the per-run median per-feature reconstruction R^2 and the subject
re-identification accuracy; the percent change reads against the Raw row of
the HAR figure CSV (fig_har.csv, produced by fig_har.sh).
"""
import argparse, csv, json, os, statistics as st


def arm(root, pfx):
    r2, reid = [], []
    for s in range(3):
        p = os.path.join(root, f"{pfx}_s{s}", "attacks.json")
        if os.path.exists(p):
            d = json.load(open(p))
            r2.append(d["recon_r2_median"])
            reid.append(d["reid_acc"])
    return (st.median(r2), st.median(reid)) if r2 else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", required=True)
    ap.add_argument("--fig-har", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = next(r for r in csv.DictReader(open(a.fig_har)) if r["method"] == "raw")
    r2b, reidb = float(raw["recon_r2"]), float(raw["reid"])
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "recon_r2", "reid", "r2_pct", "reid_pct"])
        for name, pfx in (("Raw+DP", "none_w0_dp"), ("LS-F+DP", "fixed_w0_dp")):
            r2, reid = arm(a.dp, pfx)
            w.writerow([name] + ([""] * 4 if r2 is None else
                       [f"{r2:.4f}", f"{reid:.4f}",
                        f"{(r2 - r2b) / r2b * 100:+.2f}", f"{(reid - reidb) / reidb * 100:+.2f}"]))
    print(f"[aggregate_har_dp] -> {a.out}")


if __name__ == "__main__":
    main()
