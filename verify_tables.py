#!/usr/bin/env python3
"""Compare a reproduced exhibit CSV with the numbers printed in the paper.

  python verify_tables.py expected/tab_fsha_active.json attack-fsha/results/tab_fsha_active.csv

Rows are matched on the exhibit's key columns; a cell agrees when it is within half a unit (plus 6% of a unit for cells the paper rounded twice, e.g. 4 decimals then 3) of the
paper's last printed decimal. Exit status 1 if any cell differs or any expected row is missing.
"""
import csv, json, math, sys

def main(expected_path, csv_path):
    exp = json.load(open(expected_path))
    key, dec = exp["key"], exp["decimals"]
    def norm(v):
        try: return f"{float(v):g}"
        except ValueError: return str(v)
    got = {tuple(norm(r[k]) for k in key): r for r in csv.DictReader(open(csv_path))}
    bad = 0
    for e in exp["rows"]:
        k = tuple(norm(e[c]) for c in key)
        r = got.get(k)
        if r is None:
            print(f"MISSING  {k}"); bad += 1; continue
        for col, nd in dec.items():
            if col not in e: continue
            try: v = float(r[col])
            except (KeyError, ValueError): v = float("nan")
            if math.isnan(v) or abs(v - float(e[col])) > 0.56 * 10 ** -nd:
                print(f"DIFF     {k} {col}: paper {e[col]} vs reproduced {v:.{nd + 1}f}"); bad += 1
    print(f"{exp['table']}: {len(exp['rows'])} rows checked, {bad} problem(s)")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
