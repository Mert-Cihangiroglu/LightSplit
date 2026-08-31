#!/usr/bin/env python3
"""Emit the split-point communication-cost table (tab:comm_cost).

Pure arithmetic: each sample crosses the cut twice per iteration (forward
payload and returned gradient, k float32 each), CIFAR-10 has 50,000 training
samples per epoch across the client fleet, and the table charges 100 epochs.

  python tab_comm_cost.py --out tab_comm_cost.csv
"""
import argparse, csv

D = 4096
SAMPLES, EPOCHS = 50000, 100

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["cr", "k", "floats_per_sample", "total_gb"])
    for cr in (1, 8, 16, 32):
        k = D // cr
        fl = 2 * k
        gb = fl * 4 * SAMPLES * EPOCHS / 2**30
        w.writerow([cr, k, fl, f"{gb:.2f}"])
print(f"[tab_comm_cost] -> {a.out}")
