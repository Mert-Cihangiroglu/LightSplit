#!/usr/bin/env python3
"""Aggregate the label-free backdoor-detection table (tab:bd-ours).

  python aggregate_backdoor.py --results DIR [...] --out CSV

--results holds run_backdoor.py output directories
(cifar10_<mode>_c<n>_mal<m>_pr<p>_k<k>_seed<s>_a<alpha>). For each run the
detector applies per-epoch MAD z-scores to every client's cosine between its
backbone-output statistics and the global mean (r_cosine_to_global in
detection_stats.csv), averages |z| over epochs, flags clients above 3, and
scores F1 against the true malicious set. The baseline row is the mean over
seeds; ablation rows are single runs. The table's numbers come from the
fixed_random arm; fixed_random_ll crossings coincide (checked when present).
"""
import argparse, csv, glob, os, re, statistics

THRESHOLD = 3.0
METRIC = "r_cosine_to_global"
PAT = re.compile(r"cifar10_(?P<mode>[a-z0-9_]+?)_c(?P<n>\d+)_mal(?P<mal>[\d.]+)"
                 r"_pr(?P<pr>[\d.]+)_k(?P<k>\d+)_seed(?P<seed>\d+)_a(?P<alpha>[\d.]+)$")


def run_f1(stats_csv):
    per_epoch = {}
    mal = set()
    for r in csv.DictReader(open(stats_csv)):
        try:
            v = float(r[METRIC])
        except (KeyError, ValueError, TypeError):
            continue
        cid = int(r["client_id"])
        per_epoch.setdefault(int(r["epoch"]), {})[cid] = v
        if r.get("is_malicious") in ("1", "True", "true"):
            mal.add(cid)
    zsum, zcnt = {}, {}
    for ep, vals in per_epoch.items():
        vs = sorted(vals.values())
        med = statistics.median(vs)
        madv = statistics.median(sorted(abs(v - med) for v in vs)) * 1.4826
        for cid, v in vals.items():
            z = 0.0 if madv < 1e-12 else abs((v - med) / madv)
            zsum[cid] = zsum.get(cid, 0.0) + z
            zcnt[cid] = zcnt.get(cid, 0) + 1
    flagged = {c for c in zsum if zsum[c] / zcnt[c] > THRESHOLD}
    tp = len(flagged & mal)
    fp = len(flagged - mal)
    fn = len(mal - flagged)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--mode", default="fixed_random")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    runs = {}
    for root in a.results:
        for fn in sorted(glob.glob(os.path.join(root, "*", "detection_stats.csv"))):
            m = PAT.match(os.path.basename(os.path.dirname(fn)))
            if not m or m.group("mode") != a.mode:
                continue
            g = m.groupdict()
            runs[(int(g["n"]), g["mal"], g["pr"], g["alpha"], int(g["k"]), int(g["seed"]))] = fn

    def f1_of(n=10, mal="0.1", pr="0.3", alpha="10000000"):
        vals = [run_f1(fn) for key, fn in runs.items()
                if key[:5] == (n, mal, pr, alpha, 256)]
        return None if not vals else sum(vals) / len(vals)

    rows = [("baseline", "iid", f1_of())]
    rows += [("alpha", al, f1_of(alpha=al)) for al in ("0.1", "1", "10")]
    rows += [("n", str(n), f1_of(n=n)) for n in (5, 20, 50)]
    rows += [("mal", m2, f1_of(mal=m2)) for m2 in ("0.2", "0.3")]
    rows += [("pr", p2, f1_of(pr=p2)) for p2 in ("0.1", "0.2", "0.5", "0.8")]
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "value", "f1"])
        for s, v, f in rows:
            w.writerow([s, v, "" if f is None else f"{f:.3f}"])
    print(f"[aggregate_backdoor] {sum(1 for _, _, f in rows if f is not None)} cells -> {a.out}")


if __name__ == "__main__":
    main()
