"""LM data: concatenate -> tokenize -> fixed-length blocks, dataset-agnostic + cached.

Handles WikiText (standard val split), Enron (LLM-PBE/enron-email, single split -> held-out
val slice), and any HF text dataset with a `text` column. Tokenization is capped
(`max_train_blocks`) and cached to disk, so a whole raw/F/L x CR sweep tokenizes each dataset
ONCE. Same tokenizer/block_len/val split across arms -> perplexity is comparable.
"""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, get_dataset_split_names

CACHE = os.environ.get("SLLM_TOK_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tok_cache"))


def _chunk_ids(text_iter, tokenizer, block_len, max_blocks):
    """Tokenize documents incrementally, stopping once we have max_blocks blocks."""
    target = max_blocks * block_len if max_blocks else None
    ids, buf = [], []
    for t in text_iter:
        if not t or not t.strip():
            continue
        buf.append(t)
        if len(buf) >= 2000:
            ids.extend(tokenizer("\n\n".join(buf), add_special_tokens=False)["input_ids"])
            buf = []
            if target and len(ids) >= target:
                break
    if buf and (not target or len(ids) < target):
        ids.extend(tokenizer("\n\n".join(buf), add_special_tokens=False)["input_ids"])
    if target:
        ids = ids[:target]
    n = (len(ids) // block_len) * block_len
    return torch.tensor(ids[:n], dtype=torch.long).view(-1, block_len)


def _cached(path, build):
    if os.path.exists(path):
        return torch.load(path)
    ids = build()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ids, path)
    return ids


def get_lm_dataloaders(tokenizer, dataset_repo="Salesforce/wikitext",
                       dataset_config="wikitext-2-raw-v1", text_col="text",
                       block_len=256, batch_size=8, max_train_blocks=0, max_val_blocks=0,
                       cache_name=None, n_val_holdout=5000):
    tokenizer.model_max_length = int(1e9)
    dataset_config = dataset_config or None
    key = cache_name or f"{dataset_repo.replace('/', '_')}_{dataset_config}"
    tr_path = os.path.join(CACHE, f"{key}_train_bl{block_len}_cap{max_train_blocks}.pt")
    va_path = os.path.join(CACHE, f"{key}_val_bl{block_len}_cap{max_val_blocks}.pt")

    splits = set(get_dataset_split_names(dataset_repo, dataset_config))
    has_val = "validation" in splits or "test" in splits
    val_split = "validation" if "validation" in splits else ("test" if "test" in splits else "train")

    def build_train():
        ds = load_dataset(dataset_repo, dataset_config, split="train")
        if not has_val:  # single-split dataset: hold out the first n_val_holdout docs for val
            ds = ds.select(range(n_val_holdout, len(ds)))
        return _chunk_ids((ex[text_col] for ex in ds), tokenizer, block_len, max_train_blocks)

    def build_val():
        ds = load_dataset(dataset_repo, dataset_config, split=val_split)
        if not has_val:
            ds = ds.select(range(min(n_val_holdout, len(ds))))
        return _chunk_ids((ex[text_col] for ex in ds), tokenizer, block_len, max_val_blocks)

    train_ids = _cached(tr_path, build_train)
    val_ids = _cached(va_path, build_val)

    train_dl = DataLoader(TensorDataset(train_ids), batch_size=batch_size,
                          shuffle=True, drop_last=True)
    val_dl = DataLoader(TensorDataset(val_ids), batch_size=batch_size, shuffle=False)
    return train_dl, val_dl
