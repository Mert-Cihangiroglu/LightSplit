"""SplitCausalLM: a decoder LLM split at `cut`, with a LightSplit bottleneck at the cut.

We do NOT reimplement the backbone's forward. Instead we register a forward hook on
the cut block that rewrites its output hidden state h -> bottleneck(h), and delegate
the whole forward/loss/generation to the stock HF model. Consequences:
  - raw mode (identity bottleneck) reproduces the stock model *exactly* (parity),
  - all causal-mask / position / KV-cache plumbing stays the backbone's own code,
  - generation (base.generate) also flows through the bottleneck.

The bottleneck is a submodule, so its params (LightSplit-L's learned lift) are
trainable and move with .to(device). R is a no-grad buffer.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from bottlenecks import build_bottleneck


def _get_blocks(base):
    if hasattr(base, "transformer") and hasattr(base.transformer, "h"):   # GPT-2 family
        return base.transformer.h
    if hasattr(base, "model") and hasattr(base.model, "layers"):          # Llama / Qwen family
        return base.model.layers
    raise ValueError(f"don't know where the decoder blocks live for {type(base).__name__}")


class SplitCausalLM(nn.Module):
    def __init__(self, model_name="gpt2", mode="raw", cr=8, cut=None,
                 mlp_hidden=512, norm="batch"):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        self.config = self.base.config
        self.d = self.config.hidden_size
        self.blocks = _get_blocks(self.base)
        self.n_layer = len(self.blocks)
        self.cut = cut if cut is not None else self.n_layer // 2
        self.mode = mode
        self.bottleneck = build_bottleneck(mode, self.d, cr, mlp_hidden=mlp_hidden, norm=norm)
        self.k = self.bottleneck.k
        self.compression_ratio = self.bottleneck.compression_ratio
        self._register_hook()

    def _register_hook(self):
        target = self.blocks[self.cut - 1]

        def hook(module, inputs, output):
            # block output is a Tensor (GPT-2, transformers 5.x) or a tuple (Llama-style)
            if isinstance(output, tuple):
                return (self.bottleneck(output[0]),) + tuple(output[1:])
            return self.bottleneck(output)

        self._handle = target.register_forward_hook(hook)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)
