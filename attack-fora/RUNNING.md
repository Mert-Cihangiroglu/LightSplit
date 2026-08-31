# attack-fora — FORA feature-oriented reconstruction

FORA trains a surrogate client model and a decoder against the captured
cut-layer payload. Upstream attribution is in `UPSTREAM.md`; upstream's own
README is kept as `README_upstream_FORA.md`.

PyTorch. Needs `piq` and `tensorboard` on top of the base environment.

## Entry point

`main.py`, run from a directory of your choice: it writes `log/`, `model/` and
`recon_pics/` there and reads CIFAR-10 from `./data` (downloaded if absent).

```bash
python main.py \
  --dataset cifar --layer_id 2 --iteration 40000 \
  --defense fixed_random --defense-k 256 --wcc-lambda 0.01 \
  --lr 1e-4 --dlr 3e-4 --a 0.8 --batch_size 64 --gid 0 \
  --save_path cifar_level2_fixedrandom.pth
```

`--defense-k` sets the transmitted payload size directly (channels for
`learned_1x1`). At the level-2 cut (d = 16 384) the paper's 8x/16x/32x rows use
`k = 2048/1024/512` (8/4/2 channels for `learned_1x1`). `--seed N` offsets the
fixed seeds; the paper averages several attacker seeds per cell.

## Defense flags

| Flag | Effect |
| --- | --- |
| `--defense` | `none`, `fixed_random` (**LightSplit-F**), `block_random`, `topk_sparse`, `learned_1x1` |
| `--defense-k K` | transmitted payload size |
| `--block-size B` | tile size for `block_random` |
| `--wcc-lambda L` | within-class compaction |
| `--clip-norm C`, `--noise-multiplier S` | DP on the payload |
| `--seed N` | offset on the fixed seeds; 0 is the paper's setting |
| `--inv-lift`, `--inv-classprior`, `--inv-control-labelonly` | attacker-side ablations |

`measure_dp_norms.py` reports payload norms, used to calibrate `--clip-norm`
per configuration. `test_*.py` are unit checks for each defense.

## Reproducing the paper's tables

`reproduce/tab_fora.sh` runs every cell behind the paper's FORA tables (61
runs) and writes `results/reproduce/tab_fora.csv`; see the top-level
`REPRODUCE.md`. `reproduce/aggregate_fora.py` scores each run's saved 32-image
reconstruction grid (SSIM, PSNR, LPIPS for both decoders), averages the runs of
a cell and reads the victim accuracy from the logs.
