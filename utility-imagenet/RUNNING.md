# utility-imagenet — ImageNet-1k scaling

Backs the paper's ImageNet-1k result: a ResNet split across 10 IID clients,
90 epochs, 1000 classes. The client adaptively pools the cut to an
`8x8x64` map, so the cut dimension stays `d = 4096` regardless of input
resolution and matches the CIFAR cut.

At this scale the cut is a spatial feature map, so the projection is applied
**block-locally**: block size `b` tiles the `8x8` cut into `b x b` patches at the
same k-scalar budget, and `b = 8` is the global dense R. Here `b` is a utility
dial, not a privacy dial.

## Entry point

```bash
python train.py \
  --data-root /path/to/imagenet --mode block_random --block-size 2 \
  --compression-ratio 8 --deep-head --num-clients 10 --epochs 90 \
  --batch-size 256 --num-workers 16 --gpu 0 \
  --summary-csv results/csv/imagenet_block_b2_cr8.csv
```

`--mode` accepts `raw`, `fixed_random` (**LightSplit-F**), `fixed_random_ll`
(**LightSplit-L**, width via `--ll-hidden`), `block_random` (+ `--block-size`),
`topk_sparse` and `learned_1x1`. `--fake-data` runs the pipeline on synthetic
tensors for a fast smoke test.

`data.py` expects the standard ImageNet train/val directory layout. ImageNet-1k
is not included and must be obtained separately.

## Reproducing the paper's figures

`reproduce/fig_imagenet_generality.sh` (Fig. imagenet_generality) and
`reproduce/fig_imagenet_lsl_lsf.sh` (Fig. imagenet_lsl_lsf) run every arm of the
two ImageNet figures and write one CSV each; `../REPRODUCE.md` gives the commands,
their cost and the column meanings.
