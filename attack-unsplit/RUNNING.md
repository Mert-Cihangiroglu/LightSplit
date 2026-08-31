# attack-unsplit — UnSplit model inversion and label inference

UnSplit is the weakest of the reconstruction attacks we run, included for
coverage: it recovers inputs by alternating optimisation over the attacker's
clone and the candidate input. Upstream attribution is in `UPSTREAM.md`;
upstream's own README is preserved as `README.md`.

PyTorch.

## Model inversion

```bash
python model_inversion_demo.py <dataset> <split_layer> [num_images] [flags]

python model_inversion_demo.py mnist 2 10 \
  --defense fixed_random --cr 8 --epochs 20
```

Datasets `mnist`, `f_mnist`, `cifar`. The paper reports split-2 and split-4.

| Flag | Effect |
| --- | --- |
| `--defense` | `none`/`raw`, `fixed_random` (**LightSplit-F**), `block_random`, `topk_sparse`, `learned_1x1` |
| `--cr R` / `--k K` | compression ratio, or payload size directly |
| `--block-size B` | tile size for `block_random` |
| `--wcc-lambda L` | within-class compaction |
| `--clip-norm C`, `--noise-multiplier S` | DP on the payload |
| `--main-iters`, `--input-iters`, `--model-iters` | attack optimisation budget |
| `--seed S` | multi-seed runs |

## Label inference

`label_inference_defended.py` runs the label-inference attack against a defended
cut; `label_inference_demo.py` is upstream's undefended version.

## Reproducing the paper's tables

`reproduce/tab_unsplit.sh` runs every cell behind the four UnSplit tables and
writes `results/reproduce/tab_unsplit.csv`; see the top-level `REPRODUCE.md`.
`reproduce/aggregate_unsplit.py` scores the saved reconstruction pairs
(whole-image SSIM/PSNR/LPIPS and the foreground-masked MSE) and forms the
ratios to Raw; `reproduce/measure_dp_norms.py` calibrates the DP clip.
