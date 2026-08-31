# Upstream provenance — SDAR (cut-width fork)

Same upstream as `attack-sdar/`:

- **Repository:** https://github.com/zhxchd/SDAR_SplitNN
- **Base commit:** `3f0a0a8e119c9c12113b538bf3c120bd183de771`

A variant of `attack-sdar/` for the cut-width sweep: the client
head emits a narrower cut tensor, and compression is applied on top of the
narrowed cut. Cut width is a separate axis from the compression ratio.

Kept separate from `attack-sdar-lightsplit-l/` because both modify
`src/sdar/sdar.py` and `scripts/run_sdar.py` on the same lines.

## Difference from `attack-sdar/`

- `src/models/resnet.py` — narrowed cut-layer widths.
- `src/sdar/sdar.py`, `scripts/run_sdar.py` — plumb the cut width through and
  expose it on the CLI.
