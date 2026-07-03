"""
012 Triadic Sentence Encoder — TritEncoder

A from-scratch sentence embedding model using triadic attention
(Observer/Shadow/Light gating) instead of standard Q/K/V attention.
Trained with contrastive loss on code pairs, same data pipeline as
trit_embed_train.py, then benchmarked against MiniLM baseline and
your existing fine-tuned MiniLM using the same triples.

Architecture:
  Token embedding (learned, char/subword via simple BPE-free tokenizer)
  → N x TriadicAttention blocks (Observer/Shadow/Light gating)
  → mean pooling over tokens
  → L2 normalize → 256-dim sentence embedding

This is NOT pretrained — it only knows what it learns from your
contrastive training pairs. Expect lower accuracy than MiniLM
(which had ~1B pretraining pairs) unless trained on a lot of data.

Usage:
  python trit_triadic_encoder.py --train       Train on code pairs
  python trit_triadic_encoder.py --benchmark   Compare vs MiniLM baseline + fine-tuned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re, json, time, random, argparse, os
from pathlib import Path
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# TERNARY CORE (same as trit_transformer.py)
# ══════════════════════════════════════════════════════════════════════════════

# Original bare ternary quantizer -- kept for reference / easy revert, but
# SUPERSEDED by ternary_quantize_scaled below. It returned raw {-1,0,+1}
# with no magnitude scale, throwing away how big the weights actually were.
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


def ternary_quantize_scaled(x, frac=0.7):
    """Ternary quantize WITH an optimal per-tensor magnitude scale.

    Measured win (trit_residual_ab.py, 3-arm component-isolated A/B on this
    exact encoder): adding this scale to the bare {-1,0,+1} quantizer raised
    held-out retrieval accuracy from 5.1% to 15.2% (+10.1pp) -- by far the
    biggest lever tested, and larger than the "shadow"/residual second trit,
    which added +0.0pp at 2x storage and so was dropped. The bare version
    threw away all magnitude information: sign only.

    Scale a = <x, t> / <t, t> is the least-squares optimal multiplier for
    the ternary mask t (minimizes ||x - a*t||). Straight-through estimator:
    forward returns a*t, backward passes the gradient straight to x. This is
    the EXACT configuration that produced the 15.2% measurement -- both the
    scale and the STE gradient path, not the old backward mask -- so what
    ships here is what was actually measured, not an approximation of it.
    """
    thresh = frac * x.abs().mean()
    t = torch.where(x > thresh, torch.ones_like(x),
        torch.where(x < -thresh, -torch.ones_like(x), torch.zeros_like(x)))
    a = (x * t).sum() / (t * t).sum().clamp(min=1e-8)
    approx = a * t
    return x + (approx - x).detach()   # STE: forward a*t, backward identity to x


tq = ternary_quantize_scaled

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

# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE WORD-LEVEL TOKENIZER (built from training data vocabulary)
# ══════════════════════════════════════════════════════════════════════════════

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\S")

def tokenize(text, max_len=64):
    toks = TOKEN_RE.findall(text)[:max_len]
    return toks

class Vocab:
    def __init__(self, texts, max_vocab=20000):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for tok, _ in counter.most_common(max_vocab - 2):
            self.stoi[tok] = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text, max_len=64):
        toks = tokenize(text, max_len)
        ids  = [self.stoi.get(t, 1) for t in toks]
        ids  = ids[:max_len] + [0] * (max_len - len(ids))
        return ids

    def __len__(self): return len(self.stoi)

    def save(self, path):
        json.dump(self.stoi, open(path, "w", encoding="utf-8"))

    @classmethod
    def load(cls, path):
        v = cls.__new__(cls)
        v.stoi = json.load(open(path, encoding="utf-8"))
        v.itos = {i: s for s, i in v.stoi.items()}
        return v

# ══════════════════════════════════════════════════════════════════════════════
# TRIADIC ENCODER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TriadicSelfAttention(nn.Module):
    """Observer/Shadow/Light gating — bidirectional (no causal mask, for encoding)."""
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.w0     = TernaryLinear(n_embd, n_embd, bias=False)
        self.w1     = TernaryLinear(n_embd, n_embd, bias=False)
        self.w2     = TernaryLinear(n_embd, n_embd, bias=False)
        self.proj   = TernaryLinear(n_embd, n_embd, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        B, T, C = x.shape
        H = self.n_head
        d = C // H

        s0 = torch.sigmoid(self.w0(x))
        s1 = torch.tanh(self.w1(x))
        s2 = torch.tanh(self.w2(x))
        gate = s1 * (1 - s0) + s2 * s0

        g = gate.view(B, T, H, d).transpose(1, 2)
        att = (g @ g.transpose(-2, -1)) / (d ** 0.5)
        att = att.masked_fill(pad_mask[:, None, None, :] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att)
        att = self.drop(att)
        y = (att @ g).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class TriadicMLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            TernaryLinear(n_embd, 4*n_embd, bias=False),
            nn.GELU(),
            TernaryLinear(4*n_embd, n_embd, bias=False),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class TriadicBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = TriadicSelfAttention(n_embd, n_head, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = TriadicMLP(n_embd, dropout)
    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TritSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=4, max_len=64, dropout=0.1):
        super().__init__()
        self.max_len  = max_len
        self.tok_emb  = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, n_embd)
        self.drop     = nn.Dropout(dropout)
        self.blocks   = nn.ModuleList([TriadicBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f     = nn.LayerNorm(n_embd)
        self.out_dim  = n_embd

    def forward(self, ids):
        pad_mask = (ids != 0).float()
        B, T = ids.shape
        pos  = torch.arange(T, device=ids.device)
        x    = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.ln_f(x)
        mask = pad_mask.unsqueeze(-1)
        summed  = (x * mask).sum(1)
        counts  = mask.sum(1).clamp(min=1e-6)
        pooled  = summed / counts
        return F.normalize(pooled, dim=-1)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA — reuse pair extraction logic similar to trit_embed_train.py
# ══════════════════════════════════════════════════════════════════════════════

# Edit this list (or set TRIT_SCAN_DIRS env var, os.pathsep-separated) to
# point at your own codebase(s) for training data.
LOCAL_PROJECT_DIRS = os.environ.get("TRIT_SCAN_DIRS", "").split(os.pathsep) if os.environ.get("TRIT_SCAN_DIRS") else [
    str(Path(__file__).resolve().parent),
]
SKIP_DIRS = {".git","__pycache__","node_modules",".venv","venv","dist","build",
             "target","models","search_index","terrain_3d","addons","demo"}
EXTS = {".py",".gd",".js",".ts",".cs",".rs",".go",".c",".cpp",".h",".java"}

FUNC_RE    = re.compile(r"func\s+(\w+)\s*\([^)]*\)[^:{]*[:{]\s*\n((?:[ \t]+.+\n?)+)", re.M)
PYFUNC_RE  = re.compile(r"def\s+(\w+)\s*\([^)]*\)\s*:\s*\n((?:[ \t]+.+\n?)+)", re.M)
COMMENT_RE = re.compile(r"#\s*(.+)\n(.{20,400})", re.M)

def extract_pairs_from_code(text):
    pairs = []
    for rx in (FUNC_RE, PYFUNC_RE):
        for m in rx.finditer(text):
            name, body = m.group(1), m.group(2)
            if len(body.strip()) > 20:
                query = " ".join(re.findall(r"[A-Za-z][a-z]*", name)).lower()
                if query:
                    pairs.append((query, body.strip()[:400]))
    for m in COMMENT_RE.finditer(text):
        comment, code = m.group(1).strip(), m.group(2).strip()
        if len(comment) > 5 and len(code) > 20:
            pairs.append((comment, code[:400]))
    return pairs

def collect_local_pairs():
    pairs = []
    for base in LOCAL_PROJECT_DIRS:
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in fnames:
                if Path(fname).suffix.lower() not in EXTS:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    text = open(fpath, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                pairs.extend(extract_pairs_from_code(text))
    return pairs

# ══════════════════════════════════════════════════════════════════════════════
# CONTRASTIVE LOSS — MultipleNegativesRankingLoss equivalent
# ══════════════════════════════════════════════════════════════════════════════

def contrastive_loss(anchor_emb, positive_emb, temperature=0.05):
    sims = anchor_emb @ positive_emb.T / temperature
    labels = torch.arange(sims.size(0), device=sims.device)
    return F.cross_entropy(sims, labels)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    "n_embd": 256, "n_head": 8, "n_layer": 4, "max_len": 64,
    "batch_size": 32, "lr": 1e-4, "epochs": 2000,
    "quant_warmup": 200, "eval_every": 100,
}

def train():
    print("Collecting training pairs from local codebase...")
    pairs = collect_local_pairs()
    print(f"  Found {len(pairs):,} pairs")
    if len(pairs) < 100:
        print("  WARNING: very few pairs — accuracy will be poor")

    all_texts = [a for a, b in pairs] + [b for a, b in pairs]
    vocab = Vocab(all_texts)
    print(f"  Vocab size: {len(vocab):,}")

    model = TritSentenceEncoder(
        vocab_size=len(vocab), n_embd=CFG["n_embd"], n_head=CFG["n_head"],
        n_layer=CFG["n_layer"], max_len=CFG["max_len"]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])

    t0 = time.time()
    for step in range(CFG["epochs"]):
        set_quant(model, step >= CFG["quant_warmup"])
        model.train()

        batch = random.sample(pairs, min(CFG["batch_size"], len(pairs)))
        anchors    = torch.tensor([vocab.encode(a, CFG["max_len"]) for a, b in batch], device=device)
        positives  = torch.tensor([vocab.encode(b, CFG["max_len"]) for a, b in batch], device=device)

        a_emb = model(anchors)
        p_emb = model(positives)
        loss  = contrastive_loss(a_emb, p_emb)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if step % CFG["eval_every"] == 0 or step == CFG["epochs"] - 1:
            phase = "TERNARY" if step >= CFG["quant_warmup"] else "warmup"
            elapsed = time.time() - t0
            print(f"  step {step:>5}/{CFG['epochs']}  loss={loss.item():.4f}  [{phase}]  {elapsed:.0f}s")

    os.makedirs("models/triadic-encoder", exist_ok=True)
    torch.save(model.state_dict(), "models/triadic-encoder/model.pt")
    vocab.save("models/triadic-encoder/vocab.json")
    json.dump(CFG, open("models/triadic-encoder/config.json", "w"))
    print("\nSaved: models/triadic-encoder/")
    return model, vocab

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK — same triples as trit_benchmark.py
# ══════════════════════════════════════════════════════════════════════════════

def load_triadic_encoder():
    cfg = json.load(open("models/triadic-encoder/config.json"))
    vocab = Vocab.load("models/triadic-encoder/vocab.json")
    model = TritSentenceEncoder(
        vocab_size=len(vocab), n_embd=cfg["n_embd"], n_head=cfg["n_head"],
        n_layer=cfg["n_layer"], max_len=cfg["max_len"]
    ).to(device)
    model.load_state_dict(torch.load("models/triadic-encoder/model.pt", map_location=device))
    set_quant(model, True)
    model.eval()
    return model, vocab, cfg["max_len"]

@torch.no_grad()
def triadic_encode(model, vocab, max_len, texts):
    ids = torch.tensor([vocab.encode(t, max_len) for t in texts], device=device)
    return model(ids)

def benchmark():
    import trit_benchmark as bench
    triples = bench.BENCHMARK

    model, vocab, max_len = load_triadic_encoder()

    correct = 0
    total_margin = 0.0
    t0 = time.time()
    for query, right, wrong in triples:
        embs = triadic_encode(model, vocab, max_len, [query, right, wrong])
        q, r, w = embs[0], embs[1], embs[2]
        score_r = (q @ r).item()
        score_w = (q @ w).item()
        if score_r > score_w:
            correct += 1
        total_margin += (score_r - score_w)
    dt = (time.time() - t0) * 1000

    n = len(triples)
    print("\n" + "="*60)
    print(f"  Model    : TriadicEncoder (from-scratch, trained on local code)")
    print(f"  Accuracy : {correct/n*100:.1f}%  ({correct}/{n} correct)")
    print(f"  Margin   : {total_margin/n:+.3f}  (correct score - wrong score)")
    print(f"  Time     : {dt:.0f}ms for {n} queries")
    print("="*60)
    print("\n  Compare against:")
    print("    Microsoft baseline : 92.0% accuracy, +0.190 margin, 252ms")
    print("    Your fine-tuned    : 96.0% accuracy, +0.215 margin, 31ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    if args.benchmark:
        benchmark()
    if not args.train and not args.benchmark:
        train()
        benchmark()
