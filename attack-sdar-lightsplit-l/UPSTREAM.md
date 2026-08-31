# Upstream provenance — SDAR (LightSplit-L fork)

Same upstream as `attack-sdar/`:

- **Repository:** https://github.com/zhxchd/SDAR_SplitNN
- **Base commit:** `3f0a0a8e119c9c12113b538bf3c120bd183de771`

A variant of `attack-sdar/` that adds the **LightSplit-L**
bottleneck (`fixed_random_ll`) — the fixed random projection followed by a
server-side learned MLP lift-back.

It is kept as a separate folder rather than merged into `attack-sdar/` because
its changes to `src/sdar/sdar.py` and `scripts/run_sdar.py` land on the same lines as those
of `attack-sdar-cutwidth/`. Keeping the forks apart means each folder is exactly
the code that produced its numbers.

## Difference from `attack-sdar/`

- `src/sdar/bottlenecks.py` — adds the `fixed_random_ll` bottleneck and its
  learned lift-back MLP.
- `src/sdar/sdar.py`, `scripts/run_sdar.py` — register the new mode and expose
  it on the CLI.
- `scripts/run_fixedrandom_ll_usl_l7.sh` — the LightSplit-L sweep driver.
