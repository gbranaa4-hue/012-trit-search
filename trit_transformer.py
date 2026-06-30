"""
012 Triadic Transformer — TritGPT

Replaces standard QKV attention with triadic streams {s0, s1, s2}.
Adds predictive coding auxiliary loss (each layer predicts its input).
All projection weights are ternary {-1, 0, +1}.

Standard transformer:          012 TritGPT:
  Q = xW_Q                       s0 = σ(xW_0)    Observer gate   [0,1]
  K = xW_K                       s1 = tanh(xW_1)  Shadow stream  [-1,1]
  V = xW_V                       s2 = tanh(xW_2)  Light stream   [-1,1]
  attn = softmax(QK^T/√d)V       gate = s1*(1-s0) + s2*s0
                                 attn = softmax(gate · gate^T / √d) · gate

Hypothesis: triadic gating + predictive loss improves:
  - Sample efficiency (learns faster per token)
  - Robustness (less sensitive to input noise/corruption)
  - Compression (ternary weights = 16x smaller)

Benchmark: TritGPT vs GPT-mini on Shakespeare next-token prediction.
Same parameter budget, same data, same compute.

Install:
  pip install torch datasets

Usage:
  python trit_transformer.py --train         Train both models, save results
  python trit_transformer.py --compare       Load saved results, print table
  python trit_transformer.py --chat          Chat with trained TritGPT
  python trit_transformer.py --ablation      Train all ablation variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, argparse, json, os, urllib.request
from pathlib import Path

device = torch.device("cuda")
print(f"Device: {device}")

os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

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
        super().__init__(*args, **kwargs)
        self.do_quantize = quantize
    def forward(self, x):
        w = tq(self.weight) if self.do_quantize else self.weight
        return F.linear(x, w, self.bias)

def set_quant(model, active):
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            m.do_quantize = active

def trit_dist(model):
    total = neg = zero = pos = 0
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            t     = 0.7 * m.weight.data.abs().mean()
            q     = torch.where(m.weight.data >  t,  torch.ones_like(m.weight.data),
                    torch.where(m.weight.data < -t, -torch.ones_like(m.weight.data),
                    torch.zeros_like(m.weight.data)))
            total += q.numel()
            neg   += (q == -1).sum().item()
            zero  += (q ==  0).sum().item()
            pos   += (q ==  1).sum().item()
    if total == 0: return 0, 0, 0
    return neg/total*100, zero/total*100, pos/total*100

# ══════════════════════════════════════════════════════════════════════════════
# DATASET — Shakespeare
# ══════════════════════════════════════════════════════════════════════════════

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("data/shakespeare.txt")

def get_data():
    DATA_PATH.parent.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        print("Downloading Shakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    text  = DATA_PATH.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for c, i in stoi.items()}
    data  = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n     = int(0.9 * len(data))
    return data[:n], data[n:], len(chars), stoi, itos

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i+block_size]   for i in ix]).to(device)
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# ══════════════════════════════════════════════════════════════════════════════
# GPT-MINI — standard transformer baseline
# ══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.n_head  = n_head
        self.n_embd  = n_embd
        self.qkv     = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj    = nn.Linear(n_embd, n_embd,     bias=False)
        self.drop    = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                           .view(1, 1, block_size, block_size))
    def forward(self, x):
        B, T, C = x.shape
        H       = self.n_head
        d       = C // H
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, T, H, d).transpose(1, 2)
        v = v.view(B, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(d)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y   = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.proj(y), None

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd, bias=False),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd, dropout)
    def forward(self, x):
        a, _ = self.attn(self.ln1(x))
        x    = x + a
        x    = x + self.mlp(self.ln2(x))
        return x, None

class GPTMini(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb    = nn.Embedding(vocab_size, n_embd)
        self.pos_emb    = nn.Embedding(block_size, n_embd)
        self.drop       = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([GPTBlock(n_embd, n_head, block_size, dropout)
                                         for _ in range(n_layer)])
        self.ln_f       = nn.LayerNorm(n_embd)
        self.head       = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x, _ = block(x)
        return self.head(self.ln_f(x)), []

# ══════════════════════════════════════════════════════════════════════════════
# TRITGPT — triadic attention + predictive coding
# ══════════════════════════════════════════════════════════════════════════════

class TriadicAttention(nn.Module):
    """
    Replaces Q/K/V with three triadic streams:
      s0 = σ(xW_0)    Observer gate:  [0,1]  — controls how much context vs detail
      s1 = tanh(xW_1) Shadow stream: [-1,1]  — fine-grained local features
      s2 = tanh(xW_2) Light stream:  [-1,1]  — broad contextual features

    Gate output: gate = s1*(1-s0) + s2*s0
    Attention:   softmax(gate · gate^T / √d) · gate

    Hardware: consensus(trit(s0), trit(s1), trit(s2)) = sign(s0+s1+s2)
    — majority vote, implementable with just adders and a comparator
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.w0     = TernaryLinear(n_embd, n_embd, bias=False)
        self.w1     = TernaryLinear(n_embd, n_embd, bias=False)
        self.w2     = TernaryLinear(n_embd, n_embd, bias=False)
        self.proj   = TernaryLinear(n_embd, n_embd, bias=False)
        self.drop   = nn.Dropout(dropout)
        self.pred   = TernaryLinear(n_embd, n_embd, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                           .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        H       = self.n_head
        d       = C // H

        s0 = torch.sigmoid(self.w0(x))
        s1 = torch.tanh(self.w1(x))
        s2 = torch.tanh(self.w2(x))

        gate = s1 * (1 - s0) + s2 * s0

        g = gate.view(B, T, H, d).transpose(1, 2)
        att = (g @ g.transpose(-2,-1)) / math.sqrt(d)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y   = (att @ g).transpose(1,2).contiguous().view(B, T, C)
        out = self.proj(y)

        pred = self.pred(out)
        return out, (pred, x.detach())

class TritMLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            TernaryLinear(n_embd, 4*n_embd, bias=False),
            nn.GELU(),
            TernaryLinear(4*n_embd, n_embd, bias=False),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class TritBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = TriadicAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = TritMLP(n_embd, dropout)
    def forward(self, x):
        a, pred_pair = self.attn(self.ln1(x))
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, pred_pair

class TritGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb    = nn.Embedding(vocab_size, n_embd)
        self.pos_emb    = nn.Embedding(block_size, n_embd)
        self.drop       = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([TritBlock(n_embd, n_head, block_size, dropout)
                                         for _ in range(n_layer)])
        self.ln_f       = nn.LayerNorm(n_embd)
        self.head       = TernaryLinear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        pred_pairs = []
        for block in self.blocks:
            x, pair = block(x)
            if pair is not None:
                pred_pairs.append(pair)
        return self.head(self.ln_f(x)), pred_pairs

# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(logits, targets, pred_pairs, pred_weight=0.01):
    B, T, V = logits.shape
    ce   = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
    pred = sum(F.mse_loss(p, a) for p, a in pred_pairs) if pred_pairs else 0.0
    return ce + pred_weight * pred, ce.item()

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    "n_embd"    : 256,
    "n_head"    : 8,
    "n_layer"   : 6,
    "block_size": 128,
    "batch_size": 64,
    "lr"        : 3e-4,
    "epochs"    : 5000,
    "quant_warmup": 500,
    "pred_weight" : 0.01,
    "eval_every"  : 250,
    "dropout"     : 0.1,
}

def train(model, train_data, val_data, label, use_pred=False, is_trit=False):
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    log = {"label": label, "train_loss": [], "val_loss": [], "step": [], "time": []}
    t0  = time.time()

    print(f"\n── Training {label} ──")

    for step in range(CFG["epochs"]):
        if is_trit:
            set_quant(model, step >= CFG["quant_warmup"])

        model.train()
        x, y          = get_batch(train_data, CFG["block_size"], CFG["batch_size"])
        logits, pairs = model(x)

        if use_pred:
            loss, ce_val = compute_loss(logits, y, pairs, CFG["pred_weight"])
        else:
            B, T, V = logits.shape
            loss    = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
            ce_val  = loss.item()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if step % CFG["eval_every"] == 0 or step == CFG["epochs"] - 1:
            model.eval()
            with torch.no_grad():
                vx, vy        = get_batch(val_data, CFG["block_size"], CFG["batch_size"] * 4)
                vl, vp        = model(vx)
                B, T, V       = vl.shape
                val_loss      = F.cross_entropy(vl.view(B*T, V), vy.view(B*T)).item()
            elapsed = time.time() - t0
            phase   = "TERNARY" if (is_trit and step >= CFG["quant_warmup"]) else "warmup"
            print(f"  step {step:>5}/{CFG['epochs']}  train={ce_val:.4f}  val={val_loss:.4f}"
                  f"  [{phase}]  {elapsed:.0f}s")
            log["train_loss"].append(ce_val)
            log["val_loss"].append(val_loss)
            log["step"].append(step)
            log["time"].append(elapsed)

    return log

# ══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, itos, stoi, prompt="HAMLET:", max_new=300, temperature=0.8):
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    for _ in range(max_new):
        idx_cond    = idx[:, -CFG["block_size"]:]
        logits, _   = model(idx_cond)
        logits      = logits[:, -1, :] / temperature
        probs       = F.softmax(logits, dim=-1)
        next_tok    = torch.multinomial(probs, num_samples=1)
        idx         = torch.cat([idx, next_tok], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_models(vocab_size):
    gpt = GPTMini(
        vocab_size  = vocab_size,
        n_embd      = CFG["n_embd"],
        n_head      = CFG["n_head"],
        n_layer     = CFG["n_layer"],
        block_size  = CFG["block_size"],
        dropout     = CFG["dropout"],
    ).to(device)

    trit = TritGPT(
        vocab_size  = vocab_size,
        n_embd      = CFG["n_embd"],
        n_head      = CFG["n_head"],
        n_layer     = CFG["n_layer"],
        block_size  = CFG["block_size"],
        dropout     = CFG["dropout"],
    ).to(device)

    return gpt, trit

def print_results(gpt_log, trit_log):
    print("\n" + "═"*65)
    print("  RESULTS — GPT-mini vs TritGPT")
    print("═"*65)
    print(f"\n  Final validation loss:")
    print(f"    GPT-mini : {gpt_log['val_loss'][-1]:.4f}")
    print(f"    TritGPT  : {trit_log['val_loss'][-1]:.4f}")
    delta = gpt_log['val_loss'][-1] - trit_log['val_loss'][-1]
    if delta > 0:
        print(f"    TritGPT wins by {delta:.4f} nats ({delta/gpt_log['val_loss'][-1]*100:.1f}%)")
    else:
        print(f"    GPT-mini wins by {-delta:.4f} nats ({-delta/trit_log['val_loss'][-1]*100:.1f}%)")

    print(f"\n  Training time:")
    print(f"    GPT-mini : {gpt_log['time'][-1]:.0f}s")
    print(f"    TritGPT  : {trit_log['time'][-1]:.0f}s")

    print(f"\n  Early convergence (loss at step 500):")
    idx500_g = next((i for i, s in enumerate(gpt_log['step'])  if s >= 500), -1)
    idx500_t = next((i for i, s in enumerate(trit_log['step']) if s >= 500), -1)
    if idx500_g >= 0 and idx500_t >= 0:
        print(f"    GPT-mini : {gpt_log['val_loss'][idx500_g]:.4f}")
        print(f"    TritGPT  : {trit_log['val_loss'][idx500_t]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--compare",  action="store_true")
    parser.add_argument("--chat",     action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    train_data, val_data, vocab_size, stoi, itos = get_data()
    print(f"Vocab: {vocab_size}  Train: {len(train_data):,}  Val: {len(val_data):,}")

    if args.train or (not args.compare and not args.chat and not args.ablation):
        gpt, trit = build_models(vocab_size)
        gpt_p  = sum(p.numel() for p in gpt.parameters())
        trit_p = sum(p.numel() for p in trit.parameters())
        print(f"\nParameters:")
        print(f"  GPT-mini : {gpt_p:,}")
        print(f"  TritGPT  : {trit_p:,}")

        gpt_log  = train(gpt,  train_data, val_data, "GPT-mini",  use_pred=False, is_trit=False)
        trit_log = train(trit, train_data, val_data, "TritGPT",   use_pred=True,  is_trit=True)

        torch.save(trit.state_dict(), "checkpoints/tritgpt.pt")
        torch.save(gpt.state_dict(),  "checkpoints/gptmini.pt")
        with open("results/tritgpt_vs_gpt.json", "w") as f:
            json.dump({"gpt": gpt_log, "trit": trit_log, "cfg": CFG}, f, indent=2)

        print_results(gpt_log, trit_log)

        neg, zero, pos = trit_dist(trit)
        print(f"\n  TritGPT weight distribution: -1:{neg:.1f}%  0:{zero:.1f}%  +1:{pos:.1f}%")

        print("\n── Sample from TritGPT ──")
        print(generate(trit, itos, stoi))

    if args.ablation:
        print("\n── Ablation variants ──")
        variants = {
            "TritGPT-Full"      : {"use_pred": True,  "is_trit": True},
            "TritGPT-NoPred"    : {"use_pred": False, "is_trit": True},
            "TritGPT-Float32"   : {"use_pred": True,  "is_trit": False},
            "GPT-mini"          : {"use_pred": False, "is_trit": False},
        }
        logs = {}
        for name, flags in variants.items():
            _, trit = build_models(vocab_size)
            if "GPT" in name and "Trit" not in name:
                gpt, _ = build_models(vocab_size)
                model  = gpt
            else:
                model  = trit
            logs[name] = train(model, train_data, val_data, name, **flags)

        print("\n" + "═"*65)
        print("  ABLATION RESULTS")
        print("═"*65)
        print(f"{'Model':<25} | {'Final Val Loss':>14} | {'Step 500 Val':>12}")
        print("-"*55)
        for name, log in logs.items():
            idx500 = next((i for i, s in enumerate(log['step']) if s >= 500), -1)
            s500   = f"{log['val_loss'][idx500]:.4f}" if idx500 >= 0 else "—"
            print(f"{name:<25} | {log['val_loss'][-1]:>14.4f} | {s500:>12}")

        with open("results/tritgpt_ablation.json", "w") as f:
            json.dump(logs, f, indent=2)

    if args.compare:
        with open("results/tritgpt_vs_gpt.json") as f:
            data = json.load(f)
        print_results(data["gpt"], data["trit"])

    if args.chat:
        _, trit = build_models(vocab_size)
        trit.load_state_dict(torch.load("checkpoints/tritgpt.pt", map_location=device))
        set_quant(trit, True)
        print("TritGPT Chat — type a prompt, Enter to generate, 'q' to quit")
        while True:
            prompt = input("\n> ")
            if prompt.lower() == "q": break
            print(generate(trit, itos, stoi, prompt=prompt))
