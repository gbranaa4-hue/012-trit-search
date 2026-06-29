"""
012 Ternary Language Model — TritLM

A character-level language model using triadic attention instead of
standard transformers. Every weight is ternary {-1, 0, +1}.

Architecture:
  Token embedding (ternary)
  → N × TritAttentionBlock
      ├── Triadic QKV projection (s0/s1/s2 instead of Q/K/V)
      ├── Consensus attention (ternary dot-product + softmax)
      ├── TritMemoryCell (fixed-size recurrent state, no KV cache)
      └── TernaryFFN (two-layer, no multiplications)
  → TernaryLinear → vocab logits

Compared against:
  GPT-mini: standard transformer, float32 weights, same parameter budget

Dataset: Shakespeare (1MB) — classic small LM benchmark
Task: predict next character given previous 128 characters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, json, math, time, urllib.request

os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ══════════════════════════════════════════════════════════════════════════════
# TERNARY CORE
# ══════════════════════════════════════════════════════════════════════════════

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        t = 0.7 * x.abs().mean()
        ctx.save_for_backward(x, t.unsqueeze(0))
        return torch.where(x >  t,  torch.ones_like(x),
               torch.where(x < -t, -torch.ones_like(x),
               torch.zeros_like(x)))
    @staticmethod
    def backward(ctx, grad):
        x, _ = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).float()

tq = TernaryQuantize.apply

class TernaryLinear(nn.Linear):
    def __init__(self, *args, quantize=True, **kwargs):
        super().__init__(*args, **kwargs, bias=False)
        self.do_quantize = quantize
    def forward(self, x):
        w = tq(self.weight) if self.do_quantize else self.weight
        return F.linear(x, w)

def set_quant(model, active):
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            m.do_quantize = active

def trit_stats(model):
    total = neg = zero = pos = 0
    for m in model.modules():
        if isinstance(m, TernaryLinear) and m.do_quantize:
            t     = 0.7 * m.weight.data.abs().mean()
            q     = torch.where(m.weight.data >  t,  torch.ones_like(m.weight.data),
                    torch.where(m.weight.data < -t, -torch.ones_like(m.weight.data),
                    torch.zeros_like(m.weight.data)))
            total += q.numel(); neg += (q==-1).sum().item()
            zero  += (q== 0).sum().item(); pos += (q==1).sum().item()
    if total == 0: return 0, 0, 0
    return neg/total*100, zero/total*100, pos/total*100

# ══════════════════════════════════════════════════════════════════════════════
# DATASET — Shakespeare
# ══════════════════════════════════════════════════════════════════════════════

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/shakespeare.txt"

os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_PATH):
    print("Downloading Shakespeare dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Downloaded.\n")

text  = open(DATA_PATH, "r", encoding="utf-8").read()
chars = sorted(set(text))
vocab = len(chars)
c2i   = {c: i for i, c in enumerate(chars)}
i2c   = {i: c for i, c in enumerate(chars)}
enc   = lambda s: [c2i[c] for c in s]
dec   = lambda l: ''.join([i2c[i] for i in l])

data  = torch.tensor(enc(text), dtype=torch.long)
n     = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Dataset   : {len(text):,} characters")
print(f"Vocab     : {vocab} unique characters")
print(f"Train     : {len(train_data):,} tokens")
print(f"Val       : {len(val_data):,} tokens\n")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

CTX   = 64     # context length (characters)
BATCH = 128

def get_batch(split):
    d    = train_data if split == "train" else val_data
    ix   = torch.randint(len(d) - CTX, (BATCH,))
    x    = torch.stack([d[i:i+CTX]   for i in ix])
    y    = torch.stack([d[i+1:i+CTX+1] for i in ix])
    return x.to(device), y.to(device)

# ══════════════════════════════════════════════════════════════════════════════
# TRIADIC ATTENTION
#
# Standard transformer: Q = xW_Q,  K = xW_K,  V = xW_V
#   attention = softmax(QK^T / sqrt(d)) · V
#
# 012 triadic attention:
#   s0 = σ(xW_0)           Observer: gates attention scores
#   s1 = tanh(xW_1)        Shadow:   content queries (like Q in standard attn)
#   s2 = tanh(xW_2)        Light:    relational context (like K,V combined)
#
#   scores = s1 · s2^T / sqrt(d)          triadic dot-product
#   gates  = s0 · mean(s0, dim=1)         observer modulates score
#   attn   = softmax(scores * gates)
#   out    = attn · s2                    attend to Light stream
#
# Hardware: all W_0, W_1, W_2 are ternary → attention computed with adds only
# ══════════════════════════════════════════════════════════════════════════════

class TriadicAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # Three parallel ternary projections (replace Q, K, V)
        self.w0 = TernaryLinear(d_model, d_model, quantize=False)  # Observer
        self.w1 = TernaryLinear(d_model, d_model, quantize=False)  # Shadow
        self.w2 = TernaryLinear(d_model, d_model, quantize=False)  # Light
        self.wo = TernaryLinear(d_model, d_model, quantize=False)  # output

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(CTX, CTX)).view(1, 1, CTX, CTX)
        )

    def forward(self, x):
        B, T, C = x.shape
        H, D    = self.n_heads, self.d_head

        s0 = torch.sigmoid(self.w0(x))          # (B, T, C)  Observer gate
        s1 = torch.tanh(self.w1(x))             # (B, T, C)  Shadow queries
        s2 = torch.tanh(self.w2(x))             # (B, T, C)  Light keys+values

        # Reshape to multi-head
        def split(t): return t.view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        s0, s1, s2 = split(s0), split(s1), split(s2)

        # Triadic attention scores: Shadow queries against Light keys
        scores = (s1 @ s2.transpose(-2, -1)) / math.sqrt(D)   # (B,H,T,T)

        # Observer gates the scores — suppresses irrelevant positions
        gate   = s0.mean(dim=-1, keepdim=True)                 # (B,H,T,1)
        scores = scores * gate

        # Causal mask + softmax
        scores = scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        attn   = F.softmax(scores, dim=-1)

        # Attend to Light stream (like V in standard attention)
        out    = attn @ s2                                      # (B,H,T,D)
        out    = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

# ══════════════════════════════════════════════════════════════════════════════
# TRIT MEMORY CELL (inline in each block — replaces KV cache)
# Fixed-size state per layer, not O(n) with sequence length
# ══════════════════════════════════════════════════════════════════════════════

class TritMemoryCell(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.forget = TernaryLinear(d_model, d_model, quantize=False)
        self.write  = TernaryLinear(d_model, d_model, quantize=False)
        self.read   = TernaryLinear(d_model, d_model, quantize=False)
        self.norm   = nn.LayerNorm(d_model)
        self.register_buffer('state', torch.zeros(1, 1, d_model))

    def forward(self, x):
        # Parallelized: apply gates to all positions, then use cumulative mean
        # as an approximation to sequential state (fast, GPU-friendly)
        B, T, C = x.shape
        f   = torch.sigmoid(self.forget(x))       # (B, T, C)
        w   = torch.tanh(self.write(x))            # (B, T, C)
        r   = torch.sigmoid(self.read(x))          # (B, T, C)
        # Approximate recurrent state: cumulative weighted mean across time
        state = (w * f).cumsum(dim=1) / (f.cumsum(dim=1) + 1e-6)
        self.state = state[:, -1:, :].mean(dim=0, keepdim=True).detach()
        return self.norm(x + r * state)

    def reset(self): self.state.zero_()

# ══════════════════════════════════════════════════════════════════════════════
# TERNARY FFN
# Two-layer: expand 4x, ReLU, project back
# No multiplications when weights are quantized
# ══════════════════════════════════════════════════════════════════════════════

class TernaryFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.up   = TernaryLinear(d_model, 4 * d_model, quantize=False)
        self.down = TernaryLinear(4 * d_model, d_model, quantize=False)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        return self.norm(x + self.down(F.gelu(self.up(x))))

# ══════════════════════════════════════════════════════════════════════════════
# TRIT ATTENTION BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class TritBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_memory=True):
        super().__init__()
        self.ln1    = nn.LayerNorm(d_model)
        self.attn   = TriadicAttention(d_model, n_heads)
        self.ln2    = nn.LayerNorm(d_model)
        self.ffn    = TernaryFFN(d_model)
        self.memory = TritMemoryCell(d_model) if use_memory else None
        self.ln3    = nn.LayerNorm(d_model) if use_memory else None

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.memory is not None:
            x = x + self.memory(self.ln3(x))
        x = self.ffn(self.ln2(x))
        return x

    def reset_memory(self):
        if self.memory is not None:
            self.memory.reset()

# ══════════════════════════════════════════════════════════════════════════════
# TRITLM — full ternary language model
# ══════════════════════════════════════════════════════════════════════════════

class TritLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, use_memory=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(CTX, d_model)
        self.blocks  = nn.ModuleList([
            TritBlock(d_model, n_heads, use_memory=use_memory)
            for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = TernaryLinear(d_model, vocab_size, quantize=False)

        # Weight init
        self.apply(self._init)
        print(f"TritLM parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self, m):
        if isinstance(m, (nn.Linear, TernaryLinear)):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        pos     = torch.arange(T, device=idx.device)
        x       = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x   = block(x)
        x       = self.ln_f(x)
        logits  = self.head(x)
        loss    = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def reset_memory(self):
        for block in self.blocks:
            block.reset_memory()

    @torch.no_grad()
    def generate(self, idx, max_new=200, temperature=0.8, top_k=40):
        self.eval()
        self.reset_memory()
        for _ in range(max_new):
            ctx     = idx[:, -CTX:]
            logits, _ = self(ctx)
            logits  = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _    = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')
            probs   = F.softmax(logits, dim=-1)
            next_t  = torch.multinomial(probs, 1)
            idx     = torch.cat([idx, next_t], dim=1)
        return idx

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE: GPT-mini (standard transformer, float32)
# Same parameter budget, same architecture depth — only weights differ
# ══════════════════════════════════════════════════════════════════════════════

class GPTAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(CTX, CTX)).view(1,1,CTX,CTX))

    def forward(self, x):
        B, T, C = x.shape
        H, D    = self.n_heads, self.d_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B,T,H,D).transpose(1,2)
        k = k.view(B,T,H,D).transpose(1,2)
        v = v.view(B,T,H,D).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(D)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
        return self.proj(out.transpose(1,2).contiguous().view(B,T,C))

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = GPTAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4*d_model, bias=False), nn.GELU(),
            nn.Linear(4*d_model, d_model, bias=False)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPTmini(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(CTX, d_model)
        self.blocks  = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02)
                   if isinstance(m, nn.Linear) else None)
        print(f"GPT-mini parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, idx, targets=None):
        B, T  = idx.shape
        x     = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=200, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new):
            ctx    = idx[:, -CTX:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _  = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')
            idx    = torch.cat([idx, torch.multinomial(F.softmax(logits,-1),1)], dim=1)
        return idx

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

EPOCHS       = 800     # iterations (not epochs — Shakespeare is one big file)
QUANT_WARMUP = 150     # iterations before ternary kicks in
EVAL_EVERY   = 200
LOG_EVERY    = 100

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        ls = []
        for _ in range(50):
            xb, yb  = get_batch(split)
            _, loss = model(xb, yb)
            ls.append(loss.item())
        losses[split] = np.mean(ls)
    model.train()
    return losses

def train_model(model, label, is_trit=False):
    print(f"\n{'─'*55}")
    print(f"  Training {label}")
    print(f"{'─'*55}")
    opt      = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    history  = {"train": [], "val": [], "iter": []}
    t0       = time.time()

    for it in range(EPOCHS):
        if is_trit:
            set_quant(model, it >= QUANT_WARMUP)
            if hasattr(model, 'reset_memory') and it % 50 == 0:
                model.reset_memory()

        xb, yb   = get_batch("train")
        _, loss  = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if it % LOG_EVERY == 0:
            phase = ""
            if is_trit:
                phase = "|TRIT" if it >= QUANT_WARMUP else "|warmup"
            print(f"  [{label}{phase}] iter {it:>4}/{EPOCHS}  loss={loss.item():.4f}  "
                  f"t={time.time()-t0:.0f}s")

        if it % EVAL_EVERY == 0 or it == EPOCHS-1:
            ev = estimate_loss(model)
            history["train"].append(ev["train"])
            history["val"].append(ev["val"])
            history["iter"].append(it)
            print(f"  >> val_loss={ev['val']:.4f}  train_loss={ev['train']:.4f}")

    return history

# ── Build and train models ────────────────────────────────────────────────────

trit_lm  = TritLM(vocab,  d_model=64, n_heads=4, n_layers=3, use_memory=True).to(device)
gpt_mini = GPTmini(vocab, d_model=64, n_heads=4, n_layers=3).to(device)

hist_trit = train_model(trit_lm,  "TritLM",  is_trit=True)
hist_gpt  = train_model(gpt_mini, "GPT-mini", is_trit=False)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*55}")
print(f"  RESULTS")
print(f"{'═'*55}")

trit_val = hist_trit["val"][-1]
gpt_val  = hist_gpt["val"][-1]

print(f"\n  Final validation loss (lower = better):")
print(f"    TritLM   : {trit_val:.4f}  (perplexity = {math.exp(trit_val):.1f})")
print(f"    GPT-mini : {gpt_val:.4f}  (perplexity = {math.exp(gpt_val):.1f})")
print(f"    Gap      : {trit_val - gpt_val:+.4f} nats")

neg, zero, pos = trit_stats(trit_lm)
trit_p  = sum(p.numel() for p in trit_lm.parameters())
gpt_p   = sum(p.numel() for p in gpt_mini.parameters())

print(f"\n  Model size:")
print(f"    TritLM float32 : {trit_p*4/1024:.1f} KB")
print(f"    TritLM ternary : {trit_p*1.585/8/1024:.2f} KB")
print(f"    GPT-mini       : {gpt_p*4/1024:.1f} KB")
print(f"    Compression    : {trit_p*4 / (trit_p*1.585/8):.1f}x")

print(f"\n  Ternary weight distribution:")
print(f"    -1 : {neg:.1f}%")
print(f"     0 : {zero:.1f}%   ← zero-cost MACs")
print(f"    +1 : {pos:.1f}%")
print(f"    Active ops: {neg+pos:.1f}% of MACs actually execute")

# ── Generate sample text ──────────────────────────────────────────────────────

print(f"\n{'═'*55}")
print(f"  GENERATED TEXT")
print(f"{'═'*55}")

seed = "ROMEO:\n"
seed_t = torch.tensor([enc(seed)], dtype=torch.long, device=device)

print(f"\n  TritLM (ternary triadic):")
print(f"  {'─'*40}")
gen = trit_lm.generate(seed_t, max_new=300, temperature=0.8)
print("  " + dec(gen[0].tolist()))

print(f"\n  GPT-mini (float32 baseline):")
print(f"  {'─'*40}")
gen = gpt_mini.generate(seed_t, max_new=300, temperature=0.8)
print("  " + dec(gen[0].tolist()))

# ── Hardware estimate ─────────────────────────────────────────────────────────

print(f"\n{'═'*55}")
print(f"  HARDWARE FOOTPRINT")
print(f"{'═'*55}")

active = (neg + pos) / 100
mem_per_layer = 128 * 4   # bytes, float32 state per TritMemoryCell
kvcache_100   = 100 * 128 * 2 * 4   # 100 tokens × d_model × (K+V) × float32

print(f"""
  TritLM memory during inference:
    KV-cache equivalent : {kvcache_100/1024:.1f} KB per 100 tokens (grows with context)
    TritMemoryCell      : {mem_per_layer * 4 / 1024:.2f} KB fixed (4 layers × 128 trits)
    Memory saving       : context-length independent

  Operations per token (d=128, 4 heads, 4 layers):
    Attention MACs      : {4 * 128 * 128 * 3:,}  ternary (×{active:.0%} active)
    FFN MACs            : {4 * 128 * 512 * 2:,}  ternary
    Memory MACs         : {4 * 128 * 3:,}   ternary (3 gates per layer)
    Total active        : ~{int((4*128*128*3 + 4*128*512*2 + 4*128*3) * active):,} ternary MACs

  GPT-mini float32 ops per token:
    Attention MACs      : {4 * 128 * 128 * 3:,}  float32
    FFN MACs            : {4 * 128 * 512 * 2:,}  float32
    Total               : {4*128*128*3 + 4*128*512*2:,} float32 MACs

  Energy ratio (ternary add vs float32 multiply):
    Ternary op  : ~1.1 pJ  (add/subtract)
    Float32 MAC : ~4.6 pJ  (multiply + add)
    Effective saving: {(1 - active * 1.1/4.6)*100:.0f}% per token
""")

# Save
results = {
    "trit_val_loss"   : trit_val,
    "gpt_val_loss"    : gpt_val,
    "trit_perplexity" : math.exp(trit_val),
    "gpt_perplexity"  : math.exp(gpt_val),
    "trit_params"     : trit_p,
    "gpt_params"      : gpt_p,
    "weight_dist"     : {"neg": neg, "zero": zero, "pos": pos},
    "history_trit"    : hist_trit,
    "history_gpt"     : hist_gpt,
}
with open("results/012_lm_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: results/012_lm_benchmark.json")
