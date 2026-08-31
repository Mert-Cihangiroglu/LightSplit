# utility-llm — language-modelling generality (GPT-2)

Backs the paper's language-modelling result: the projection carries over from
vision and tabular classifiers to a transformer language model, where the
quantity of interest is validation perplexity rather than accuracy.

## Entry point

```bash
python train_split.py --mode fixed_random --compression-ratio 8 ...
```

`split_model.py` defines the GPT-2 split and the cut, `bottlenecks.py` the
defenses (shared vocabulary with the vision harness: `fixed_random`,
`fixed_random_ll`, `block_random`, `topk_sparse`, `learned_1x1`), and `data.py`
the corpus loading and tokenisation.

`reproduce/fig_llm_generality.sh` runs every arm of the paper's GPT-2 figure and
writes `results/reproduce/fig_llm_generality.csv`; `../REPRODUCE.md` gives the
command, its cost and the column meanings. `smoke_test.py` checks shapes, gradient flow and raw parity of the split model.

Model weights and the tokenised corpus cache are not included — both are
re-downloaded and rebuilt on first run.
