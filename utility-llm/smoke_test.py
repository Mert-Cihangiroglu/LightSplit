"""Verification 1-2: shapes, gradient flow, raw-parity, and payload/lift wiring."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from split_model import SplitCausalLM

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ids = tok(["the quick brown fox jumps over the lazy dog",
           "language models split at the cut layer"],
          return_tensors="pt", padding=True).input_ids.to(dev)
B, T = ids.shape
print(f"batch: B={B} T={T}")

# --- capture payload/lifted shapes at the cut via a probe hook on the bottleneck ---
def check_mode(mode, cr=8.0):
    m = SplitCausalLM("gpt2", mode=mode, cr=cr).to(dev)
    captured = {}
    if mode != "raw":
        bn = m.bottleneck
        orig_proj, orig_lift = bn.project, bn.lift
        def proj(h):
            z = orig_proj(h); captured["payload"] = tuple(z.shape); return z
        def lift(z):
            o = orig_lift(z); captured["lifted"] = tuple(o.shape); return o
        bn.project, bn.lift = proj, lift
    out = m(input_ids=ids, labels=ids)
    print(f"[{mode}] k={m.k} cr={m.compression_ratio:.1f} cut={m.cut} "
          f"logits={tuple(out.logits.shape)} loss={out.loss.item():.4f} {captured}")
    # gradient flow
    m.zero_grad(); out.loss.backward()
    g_client = m.blocks[0].mlp.c_fc.weight.grad
    g_server = m.blocks[m.cut].mlp.c_fc.weight.grad
    ok_client = g_client is not None and g_client.abs().sum().item() > 0
    ok_server = g_server is not None and g_server.abs().sum().item() > 0
    print(f"       grad client_block0={ok_client} server_block{m.cut}={ok_server}", end="")
    if mode == "fixed_random_ll":
        g_lift = m.bottleneck.learned_lift[0].weight.grad
        print(f" learned_lift={g_lift is not None and g_lift.abs().sum().item() > 0}", end="")
        print(f" R.requires_grad={m.bottleneck.R.requires_grad}")
    elif mode == "fixed_random":
        print(f" R.requires_grad={m.bottleneck.R.requires_grad}")
    else:
        print()
    return out.loss.item()

raw_loss = check_mode("raw")
check_mode("fixed_random", 8.0)
check_mode("fixed_random_ll", 8.0)

# --- raw parity vs stock model ---
stock = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").to(dev)
with torch.no_grad():
    stock_loss = stock(input_ids=ids, labels=ids).loss.item()
print(f"\nRAW PARITY: split_raw={raw_loss:.6f}  stock={stock_loss:.6f}  "
      f"|diff|={abs(raw_loss - stock_loss):.2e}  -> "
      f"{'PASS' if abs(raw_loss - stock_loss) < 1e-4 else 'FAIL'}")
