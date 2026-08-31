# attack-projection-optimized — attackers built around the projection

Because the server applies the lift-back $z \mapsto zR^{\top}$ on every forward
pass, the projection $R$ is public. This folder holds the attackers that exploit
that directly, and the rank analysis explaining why it does not help them.

## Entry points

- `privileged_attack.py` — the known-$R$ white-box suite.
- `test_grad_fusion.py` — the gradient-fusion attacker.
- `train_split.py`, `client_models.py` — the training harness these attacks
  import, a reduced copy of `utility/train_split.py` kept here so the folder runs
  standalone. Accuracy results come from `utility/`.

## The five experiments

```bash
python privileged_attack.py --cr-list 8 16 32 --experiments 1 2 3 4 5
python privileged_attack.py --experiments 4        # null-space analysis, fast
```

1. Pseudoinverse recovery baseline (linear, no training)
2. Oracle nonlinear decoder given $R$
3. Known-$R$ vs unknown-$R$ image reconstruction
4. Null-space energy analysis — the rank-deficiency numbers
5. Per-class reconstruction breakdown

Useful flags: `--mode {fixed_random, block_random, ...}`, `--block-size`,
`--wcc-lambda`, `--clip-norm` / `--noise-multiplier`, `--fuse-pinv`,
`--load-split-checkpoint` to attack an already-trained split model instead of
retraining.

## Gradient fusion

`test_grad_fusion.py` trains decoders that consume the projected activation
alone, the activation plus the returned cut-layer gradient, and the maximal
concatenation of activation, gradient and analytic lift-back.

## Aggregation

`aggregate_seeds.py` builds the multi-seed master CSV and `wcc_grids.py` the
WCC grids. Sweep drivers are the `run_wb_*.sh` scripts.

## Reproducing the paper table

`reproduce/tab_proj_optimized.sh` reruns the two gradient-fusion passes and
aggregates the signal-fusion SSIM table; see the top-level `REPRODUCE.md`.
