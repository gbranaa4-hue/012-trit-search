#!/usr/bin/env python3
"""MY ternary LM -- a tiny char-level GPT that is yours end to end.

  * layers   : tritkit's alpha-scaled TernaryLinear (your toolkit)
  * training : your ternary QAT (float warmup -> quantized), your loop
  * data     : YOUR research writing (paper/*.md, FINDINGS.txt, READMEs)
  * weights  : trained here, from scratch -> yours (not adopted)
  * runtime  : your ternary weights

Unlike running Qwen on llama.cpp or BitNet on bitnet.cpp (someone else's model
on someone else's engine), every layer of this is yours. It is tiny and
from-scratch, so expect research-flavoured but not-truly-coherent text -- the
point is that YOUR ternary stack trains and runs a real autoregressive LM.
"""
import os
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import tritkit as tk

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- model / train config (tiny, minutes on a GPU) ----
D_MODEL, N_HEAD, N_LAYER, BLOCK = 192, 6, 4, 128
STEPS, WARMUP, BATCH, LR = 4000, 600, 64, 3e-4


def load_corpus():
    """Concatenate the user's own writing into one text corpus."""
    root = os.path.dirname(os.path.abspath(__file__))
    files = (glob.glob(os.path.join(root, "paper", "*.md"))
             + glob.glob(os.path.join(root, "paper", "*.txt"))
             + [os.path.join(root, f) for f in ("FINDINGS.txt", "README.md", "DOCS.md",
                                                "reservoir_computing/FINDINGS.txt")])
    text = []
    for f in files:
        try:
            text.append(open(f, encoding="utf-8", errors="ignore").read())
        except FileNotFoundError:
            pass
    return "\n\n".join(text)


class CausalSelfAttention(nn.Module):
    def __init__(self, d, nh, T):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.nh = nh
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.nh
        q = q.view(B, T, self.nh, hd).transpose(1, 2)
        k = k.view(B, T, self.nh, hd).transpose(1, 2)
        v = v.view(B, T, self.nh, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * hd ** -0.5
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, nh, T):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = CausalSelfAttention(d, nh, T)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, nh, nl, T):
        super().__init__()
        self.T = T
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        self.blocks = nn.Sequential(*[Block(d, nh, T) for _ in range(nl)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos[:, :T]
        x = self.ln_f(self.blocks(x))
        return self.head(x)


@torch.no_grad()
def generate(model, idx, n, T):
    tk.set_quant(model, True); model.eval()
    for _ in range(n):
        logits = model(idx[:, -T:])[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx


def main():
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    vocab = len(chars)
    print(f"corpus: {len(text):,} chars, vocab {vocab} | your own writing", flush=True)

    model = TinyGPT(vocab, D_MODEL, N_HEAD, N_LAYER, BLOCK)
    tk.ternarize(model, keep_first_last=True)          # <- YOUR ternary layers
    model.to(DEV)
    _, fkb = tk.size_kb(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyGPT: {n_params/1e6:.2f}M params, float {fkb/1024:.1f} MB", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def batch():
        ix = torch.randint(0, len(data) - BLOCK - 1, (BATCH,))
        x = torch.stack([data[i:i + BLOCK] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
        return x.to(DEV), y.to(DEV)

    print("training (float warmup -> ternary QAT)...", flush=True)
    t0 = time.time()
    for step in range(STEPS):
        tk.set_quant(model, step >= WARMUP)
        model.train()
        x, y = batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            ph = "TERNARY" if step >= WARMUP else "warmup"
            print(f"  step {step:>4}/{STEPS} [{ph}] loss {loss.item():.3f}", flush=True)
    print(f"trained in {time.time()-t0:.0f}s", flush=True)

    # ternary deployed size
    _, tkb = tk.size_kb(model)
    print(f"\nternary size: {tkb/1024:.2f} MB  ({fkb/tkb:.1f}x smaller than float)", flush=True)

    # generate from a seed drawn from your own text
    seed = "The "
    idx = torch.tensor([[stoi.get(c, 0) for c in seed]], device=DEV)
    out = generate(model, idx, 400, BLOCK)
    sample = "".join(itos[i] for i in out[0].tolist())
    print("\n" + "=" * 70)
    print("SAMPLE from YOUR ternary LM (trained on your writing):")
    print("=" * 70)
    print(sample)

    # save the weights -- yours
    os.makedirs("results", exist_ok=True)
    tk.save_packed(model, "results/trit_lm_mine.tt")
    print("\nsaved packed ternary weights -> results/trit_lm_mine.tt")


if __name__ == "__main__":
    main()
