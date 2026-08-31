# Upstream provenance — UnSplit

Derived from the authors' reference implementation of UnSplit:

- **Repository:** https://github.com/ege-erdogan/unsplit
- **Last upstream commit:** `4d972bf` ("Update README.md")

The UnSplit model-inversion and label-inference attacks are upstream's. We did
not change the attack's optimisation procedure.

## Files we added

- `unsplit/bottlenecks.py` — the defense bottlenecks.
- `unsplit/defended_models.py`, `unsplit/defended_clone.py` — the defended split
  model and the matching attacker-side clone.
- `label_inference_defended.py` — label inference against a defended cut.
- `reproduce/` — the run script and aggregator for the paper's UnSplit tables.
- `test_*.py` — unit checks for each defense.

## Files we modified

- `model_inversion_demo.py` — accept a defended model and expose the defense
  flags, and score reconstruction on foreground-masked as well as whole-image
  metrics.
