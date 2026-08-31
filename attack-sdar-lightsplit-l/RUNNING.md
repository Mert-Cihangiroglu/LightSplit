# attack-sdar-lightsplit-l — SDAR against LightSplit-L

A variant of `attack-sdar/` that adds the **LightSplit-L** bottleneck: the fixed
random projection followed by a server-side learned MLP lift-back. It answers
whether making the lift-back learned rather than analytic changes the privacy
picture. See `UPSTREAM.md` for what differs from `attack-sdar/`.

Everything in `attack-sdar/RUNNING.md` applies. The addition:

```bash
cd scripts
python run_sdar.py usl resnet cifar10 7 1 \
  --bottleneck_mode fixed_random_ll --compression_ratio 8
```

`scripts/run_fixedrandom_ll_usl_l7.sh` is the sweep driver used for the paper.

Use `attack-sdar/` for LightSplit-F and every other mode — this fork exists only
because its changes conflict with the cut-width fork, not because it is a
different attack.

## Reproducing the paper's table

`reproduce/tab_sdar_lsl.sh` re-runs the LightSplit-L rows of the SDAR LS-L table;
see `../REPRODUCE.md` for commands, cost, and verification.
