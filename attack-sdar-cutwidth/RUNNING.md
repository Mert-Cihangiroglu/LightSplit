# attack-sdar-cutwidth — SDAR at reduced cut-layer width

A variant of `attack-sdar/` in which the client head emits a **narrower cut
tensor**, with compression applied on top. Cut width is a separate axis from the
compression ratio: the uncompressed row at each width isolates the effect of
narrowing by itself. See `UPSTREAM.md` for what differs from `attack-sdar/`.

This supports the paper's finding that privacy tracks the transmitted payload
size $k$ rather than the compression ratio, so matched-$k$ pairs are the honest
comparison across widths.

`src/models/resnet.py` carries the narrowed widths and `scripts/run_sdar.py`
exposes them (`--width cut8 ... cut128`, `--server_width`, `--attacker_width`,
`--force_cut_adapter`); run it from inside `scripts/`, `python run_sdar.py --help`.
`scripts/aggregate_dsweep.py` turns a directory of run logs into the per-cell
table with its victim- and attacker-health gates.

## Reproducing the paper's figure

`reproduce/fig_cutwidth_privacy.sh` runs panel (a) of the cut-width figure and
writes `results/reproduce/fig_cutwidth_privacy.csv`; panel (b) is
`utility/reproduce/fig_cutwidth_utility.sh`. `../REPRODUCE.md` gives the
commands, their cost and the column meanings.
