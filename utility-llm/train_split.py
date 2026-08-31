"""Causal-LM utility harness for LightSplit on a generative decoder.

Fine-tunes the split model (client blocks + LightSplit bottleneck + server blocks +
LM head) end-to-end on WikiText and reports validation perplexity. `raw` is the
utility ceiling; fixed_random (F) / fixed_random_ll (L) are measured as perplexity
inflation vs raw at each compression ratio. Utility-only: no attack, no WCC.
"""
import argparse
import csv
import math
import os

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from data import get_lm_dataloaders
from split_model import SplitCausalLM


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, val_dl, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for (ids,) in val_dl:
        ids = ids.to(device)
        out = model(input_ids=ids, labels=ids)
        n_tok = (ids.size(0) * (ids.size(1) - 1))  # shifted targets
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    mean_loss = total_loss / max(1, total_tokens)
    return mean_loss, math.exp(mean_loss)


def sample(model, tokenizer, device,
           prompts=("The city of", "In recent years,", "Scientists have discovered"),
           max_new=40):
    """Greedy sample through the split model (hook active) as a coherence check."""
    model.eval()
    outs = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             repetition_penalty=1.3, pad_token_id=tokenizer.eos_token_id)
        outs.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--mode", default="raw",
                   choices=["raw", "fixed_random", "fixed_random_ll"])
    p.add_argument("--compression-ratio", type=float, default=8.0)
    p.add_argument("--cut", type=int, default=0, help="0 -> n_layer//2")
    p.add_argument("--mlp-hidden", type=int, default=512)
    p.add_argument("--norm", default="batch", choices=["batch", "layer"])
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--block-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-train-blocks", type=int, default=0)
    p.add_argument("--max-val-blocks", type=int, default=0)
    p.add_argument("--dataset-repo", default="Salesforce/wikitext")
    p.add_argument("--text-col", default="text")
    p.add_argument("--dataset-name", default="", help="label for output (defaults to config/repo)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-csv", default="")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cut = args.cut if args.cut > 0 else None
    model = SplitCausalLM(args.model_name, mode=args.mode, cr=args.compression_ratio,
                          cut=cut, mlp_hidden=args.mlp_hidden, norm=args.norm).to(device)
    print(f"[cfg] model={args.model_name} mode={args.mode} CR={args.compression_ratio} "
          f"k={model.k} cut={model.cut}/{model.n_layer} d={model.d}")

    ds_label = args.dataset_name or args.dataset_config or args.dataset_repo
    train_dl, val_dl = get_lm_dataloaders(
        tokenizer, dataset_repo=args.dataset_repo, dataset_config=args.dataset_config,
        text_col=args.text_col, block_len=args.block_len, batch_size=args.batch_size,
        max_train_blocks=args.max_train_blocks, max_val_blocks=args.max_val_blocks,
        cache_name=ds_label)
    print(f"[data] {ds_label} train_blocks={len(train_dl.dataset)} "
          f"val_blocks={len(val_dl.dataset)} block_len={args.block_len}")

    opt = AdamW(model.parameters(), lr=args.lr)

    init_loss, init_ppl = evaluate(model, val_dl, device)
    print(f"[eval] pre-train val ppl={init_ppl:.3f}")

    best_ppl = init_ppl
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (ids,) in enumerate(train_dl):
            ids = ids.to(device)
            out = model(input_ids=ids, labels=ids)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += out.loss.item()
            if (step + 1) % 100 == 0:
                print(f"  ep{epoch} step{step+1}/{len(train_dl)} "
                      f"loss={running/(step+1):.4f}")
        val_loss, val_ppl = evaluate(model, val_dl, device)
        best_ppl = min(best_ppl, val_ppl)
        print(f"[eval] ep{epoch} val ppl={val_ppl:.3f} (best {best_ppl:.3f})")

    if args.sample:
        for s in sample(model, tokenizer, device):
            print("[sample]", s.replace("\n", " "))

    if args.summary_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_csv)), exist_ok=True)
        write_header = not os.path.exists(args.summary_csv)
        with open(args.summary_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["model", "mode", "cr", "k", "cut", "n_layer", "seed",
                            "epochs", "dataset", "pretrain_ppl", "best_val_ppl",
                            "final_val_ppl"])
            w.writerow([args.model_name, args.mode, args.compression_ratio, model.k,
                        model.cut, model.n_layer, args.seed, args.epochs,
                        ds_label, f"{init_ppl:.4f}", f"{best_ppl:.4f}",
                        f"{val_ppl:.4f}"])
        print(f"[write] {args.summary_csv}")


if __name__ == "__main__":
    main()
