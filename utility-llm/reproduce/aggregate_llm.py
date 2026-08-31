#!/usr/bin/env python3
"""Turn the GPT-2 reproduction runs into the values plotted in fig:llm_generality.

Reads $ROOT/fig_llm_generality/<model>_<mode>_cr<cr>.csv (last row of each summary CSV) and writes
model, params_m, mode, cr, cut, best_val_ppl, inflation, where inflation = best_val_ppl / best_val_ppl
of the same model's Raw run. --runs-map JSON overrides file locations: {"<model>_<mode>_cr<cr>": path}.
"""
import argparse, csv, json
from pathlib import Path

PARAMS_M = {"gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774, "gpt2-xl": 1500}
ARMS = [("raw", 1)] + [(m, cr) for cr in (8, 16, 32) for m in ("fixed_random", "fixed_random_ll")]

def last_row(path):
    if not Path(path).exists(): return None
    rows = list(csv.DictReader(open(path)))
    return rows[-1] if rows else None

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--models", nargs="+", default=list(PARAMS_M))
    ap.add_argument("--runs-map", type=Path); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); rmap = json.load(open(a.runs_map)) if a.runs_map else {}
    out = []
    for model in a.models:
        got = {}
        for mode, cr in ARMS:
            tag = f"{model}_{mode}_cr{cr}"
            r = last_row(rmap.get(tag, a.root / "fig_llm_generality" / f"{tag}.csv"))
            if r is None: print(f"[missing] {tag}"); continue
            got[(mode, cr)] = r
        raw = float(got[("raw", 1)]["best_val_ppl"]) if ("raw", 1) in got else float("nan")
        for (mode, cr), r in got.items():
            ppl = float(r["best_val_ppl"])
            out.append({"model": model, "params_m": PARAMS_M.get(model, ""), "mode": mode, "cr": cr, "cut": r.get("cut", ""),
                        "best_val_ppl": ppl, "inflation": ppl / raw if raw == raw and raw else float("nan")})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "params_m", "mode", "cr", "cut", "best_val_ppl", "inflation"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    for r in out: print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
