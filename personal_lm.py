"""
012 Personal TritLM — trained on YOUR writing

Ingests any text files you point it at:
  - emails (.txt, .eml)
  - notes (.txt, .md)
  - code (.py, .gd, .js, .cs)
  - chat logs (.txt)
  - anything text-based

Learns your vocabulary, sentence patterns, code style.
Runs entirely on your machine. 38 KB model. No API calls.

Usage:
  1. Put your text files in a folder (or point DATA_DIRS at existing folders)
  2. python personal_lm.py --train
  3. python personal_lm.py --chat
  4. python personal_lm.py --complete "start typing and it finishes"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, glob, math, time, json, re
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these to point at your writing
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIRS = [
    r"C:\Users\gbran\OneDrive\Documents",          # your documents
    r"C:\Users\gbran\OneDrive\Documents\012-ternary",  # your code
    # Add more folders here:
    # r"C:\Users\gbran\Desktop\notes",
    # r"C:\path\to\emails",
]

# File types to ingest
EXTENSIONS = [".txt", ".md", ".py", ".gd", ".js", ".cs", ".json", ".eml", ".log"]

# Folders to skip
SKIP_DIRS  = {"__pycache__", ".git", "node_modules", ".venv", "venv", "data", "results"}

MODEL_PATH = "results/personal_tritlm.pt"
VOCAB_PATH = "results/personal_vocab.json"

# Model size — keep small so it trains fast and fits in RAM
CTX        = 128   # characters of context
D_MODEL    = 128   # embedding dimension
N_HEADS    = 4
N_LAYERS   = 4
BATCH      = 64

# Training
EPOCHS       = 3000   # iterations
QUANT_WARMUP = 500
LR           = 3e-4

# ══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def collect_files(dirs, extensions, skip_dirs):
    files = []
    for d in dirs:
        if not os.path.exists(d):
            print(f"  Skipping (not found): {d}")
            continue
        for ext in extensions:
            for f in Path(d).rglob(f"*{ext}"):
                if not any(s in f.parts for s in skip_dirs):
                    files.append(str(f))
    return sorted(set(files))

def load_text(files, min_chars=100):
    chunks = []
    total  = 0
    for f in files:
        try:
            text = open(f, "r", encoding="utf-8", errors="ignore").read().strip()
            if len(text) >= min_chars:
                # Add a separator so the model knows files are distinct
                chunks.append(f"\n\n--- {Path(f).name} ---\n\n" + text)
                total += len(text)
        except Exception:
            pass
    print(f"  Loaded {len(chunks)} files, {total:,} characters")
    return "\n\n".join(chunks)

# ══════════════════════════════════════════════════════════════════════════════
# VOCABULARY
# Build from your actual writing — not a fixed 65-char set
# Captures your specific symbols, emoji, code tokens
# ══════════════════════════════════════════════════════════════════════════════

def build_vocab(text, max_vocab=512):
    """
    Character-level vocab built from your text.
    Keeps the most frequent characters up to max_vocab.
    512 covers all ASCII + common unicode in most writing.
    """
    freq   = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    # Always include basic ASCII
    must_have = set(chr(i) for i in range(32, 127)) | {'\n', '\t'}
    top       = sorted(freq.keys(), key=lambda c: -freq[c])
    vocab_set = must_have | set(top[:max_vocab])
    chars     = sorted(vocab_set)
    c2i       = {c: i for i, c in enumerate(chars)}
    i2c       = {i: c for i, c in enumerate(chars)}
    return chars, c2i, i2c

def save_vocab(chars, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(chars, open(path, "w", encoding="utf-8"), ensure_ascii=False)

def load_vocab(path):
    chars = json.load(open(path, "r", encoding="utf-8"))
    c2i   = {c: i for i, c in enumerate(chars)}
    i2c   = {i: c for i, c in enumerate(chars)}
    return chars, c2i, i2c

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
    def __init__(self, *a, quantize=True, **kw):
        super().__init__(*a, **kw, bias=False)
        self.do_quantize = quantize
    def forward(self, x):
        return F.linear(x, tq(self.weight) if self.do_quantize else self.weight)

def set_quant(model, active):
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            m.do_quantize = active

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class TriadicAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads; self.dh = d // n_heads
        self.w0 = TernaryLinear(d, d, quantize=False)
        self.w1 = TernaryLinear(d, d, quantize=False)
        self.w2 = TernaryLinear(d, d, quantize=False)
        self.wo = TernaryLinear(d, d, quantize=False)
        self.register_buffer("mask", torch.tril(torch.ones(CTX,CTX)).view(1,1,CTX,CTX))

    def forward(self, x):
        B,T,C = x.shape; H,D = self.h, self.dh
        s0 = torch.sigmoid(self.w0(x))
        s1 = torch.tanh(self.w1(x))
        s2 = torch.tanh(self.w2(x))
        def sp(t): return t.view(B,T,H,D).transpose(1,2)
        s0,s1,s2 = sp(s0),sp(s1),sp(s2)
        sc = (s1 @ s2.transpose(-2,-1)) / math.sqrt(D) * s0.mean(-1,keepdim=True)
        sc = sc.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        return self.wo((F.softmax(sc,-1) @ s2).transpose(1,2).contiguous().view(B,T,C))

class TritMemoryCell(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.forget = TernaryLinear(d, d, quantize=False)
        self.write  = TernaryLinear(d, d, quantize=False)
        self.read   = TernaryLinear(d, d, quantize=False)
        self.norm   = nn.LayerNorm(d)
        self.register_buffer('state', torch.zeros(1, 1, d))

    def forward(self, x):
        B,T,C = x.shape
        f = torch.sigmoid(self.forget(x))
        w = torch.tanh(self.write(x))
        r = torch.sigmoid(self.read(x))
        s = self.state.expand(B,1,C).clamp(-1,1)
        outs = []
        for t in range(T):
            s = s*(1-f[:,t:t+1]) + w[:,t:t+1]*f[:,t:t+1]
            outs.append(s)
        self.state = s.mean(0,keepdim=True).detach()
        return self.norm(x + r * torch.cat(outs,dim=1))

    def reset(self): self.state.zero_()

class TritBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = TriadicAttention(d, n_heads)
        self.ln2  = nn.LayerNorm(d)
        self.mem  = TritMemoryCell(d)
        self.ln3  = nn.LayerNorm(d)
        self.up   = TernaryLinear(d, 4*d, quantize=False)
        self.down = TernaryLinear(4*d, d, quantize=False)
        self.ln4  = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.mem(self.ln2(x))
        x = x + self.down(F.gelu(self.up(self.ln3(x))))
        return x

    def reset_memory(self): self.mem.reset()

class PersonalTritLM(nn.Module):
    def __init__(self, vocab_size, d=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(CTX, d)
        self.blocks  = nn.ModuleList([TritBlock(d, n_heads) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(d)
        self.head    = TernaryLinear(d, vocab_size, quantize=False)
        self.vocab   = vocab_size
        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters())
        print(f"  Parameters : {n:,}")
        print(f"  Ternary size: {n*1.585/8/1024:.1f} KB")

    def _init(self, m):
        if isinstance(m, (nn.Linear, TernaryLinear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        x   = self.tok_emb(idx) + self.pos_emb(torch.arange(T,device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss   = F.cross_entropy(logits.view(-1,self.vocab), targets.view(-1)) if targets is not None else None
        return logits, loss

    def reset_memory(self):
        for b in self.blocks: b.reset_memory()

    @torch.no_grad()
    def complete(self, text, c2i, i2c, max_new=200, temperature=0.8, top_k=40):
        """Complete a text prompt in your style"""
        self.eval()
        self.reset_memory()
        unk  = c2i.get(' ', 0)
        idx  = torch.tensor([[c2i.get(c, unk) for c in text[-CTX:]]],
                             dtype=torch.long, device=device)
        for _ in range(max_new):
            ctx    = idx[:, -CTX:]
            logits, _ = self(ctx)
            logits = logits[:,-1,:] / temperature
            if top_k:
                v,_ = torch.topk(logits, min(top_k,logits.size(-1)))
                logits[logits < v[:,-1:]] = float('-inf')
            next_t = torch.multinomial(F.softmax(logits,-1), 1)
            idx    = torch.cat([idx, next_t], dim=1)
        return text + ''.join(i2c.get(i.item(),'?') for i in idx[0, len(text[-CTX:]):])

    @torch.no_grad()
    def memory_state(self):
        """Return human-readable trit state of all memory cells"""
        states = {}
        for i, block in enumerate(self.blocks):
            s   = block.mem.state.squeeze()
            t   = 0.7 * s.abs().mean()
            q   = torch.where(s>t, torch.ones_like(s),
                  torch.where(s<-t, -torch.ones_like(s), torch.zeros_like(s)))
            neg = (q==-1).sum().item()
            zer = (q== 0).sum().item()
            pos = (q== 1).sum().item()
            states[f"layer_{i}"] = {"neg": neg, "zero": zer, "pos": pos,
                                    "pattern": ''.join('-' if v<0 else ('+' if v>0 else '0')
                                                       for v in q[:32].tolist())}
        return states

    def save(self, path, vocab_path, chars):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.state_dict(), "vocab_size": self.vocab}, path)
        save_vocab(chars, vocab_path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  Saved model: {path}  ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, path, vocab_path):
        chars, c2i, i2c = load_vocab(vocab_path)
        ck    = torch.load(path, map_location=device)
        model = cls(ck["vocab_size"]).to(device)
        model.load_state_dict(ck["model"])
        set_quant(model, True)
        return model, chars, c2i, i2c

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def get_batch(data, batch_size, ctx):
    ix = torch.randint(len(data)-ctx, (batch_size,))
    x  = torch.stack([data[i:i+ctx]   for i in ix])
    y  = torch.stack([data[i+1:i+ctx+1] for i in ix])
    return x.to(device), y.to(device)

def train(model, data, c2i, chars):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_loss = float('inf')
    t0 = time.time()

    print(f"\nTraining for {EPOCHS} iterations on your writing...")
    print(f"  Warmup: {QUANT_WARMUP} iters (float) → ternary")

    for it in range(EPOCHS):
        set_quant(model, it >= QUANT_WARMUP)
        if it % 100 == 0: model.reset_memory()

        xb, yb  = get_batch(data, BATCH, CTX)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()

        if it % 500 == 0 or it == EPOCHS-1:
            phase = "TRIT" if it >= QUANT_WARMUP else "warmup"
            elapsed = time.time()-t0
            print(f"  [{phase}] {it:>4}/{EPOCHS}  loss={loss.item():.4f}  "
                  f"t={elapsed:.0f}s  eta={elapsed/(it+1)*(EPOCHS-it-1):.0f}s")

        if loss.item() < best_loss and it >= QUANT_WARMUP:
            best_loss = loss.item()
            model.save(MODEL_PATH, VOCAB_PATH, chars)

    print(f"\nBest loss: {best_loss:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CHAT
# ══════════════════════════════════════════════════════════════════════════════

def chat(model, c2i, i2c, chars):
    print("\n" + "═"*55)
    print("  Personal TritLM — trained on your writing")
    print("  Commands: :quit  :memory  :reset  :temp 0.8")
    print("═"*55 + "\n")

    temperature = 0.8
    model.eval()
    set_quant(model, True)
    model.reset_memory()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not prompt: continue

        if prompt == ":quit":
            break
        elif prompt == ":memory":
            print("\nMemory state:")
            for layer, state in model.memory_state().items():
                print(f"  {layer}: -{state['neg']} 0:{state['zero']} +{state['pos']}")
                print(f"    [{state['pattern']}...]")
            print()
            continue
        elif prompt == ":reset":
            model.reset_memory()
            print("Memory reset.\n")
            continue
        elif prompt.startswith(":temp"):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}\n")
            except:
                print("Usage: :temp 0.8\n")
            continue

        result = model.complete(prompt, c2i, i2c,
                                max_new=200, temperature=temperature)
        completion = result[len(prompt):]
        print(f"012 : {prompt}\033[32m{completion}\033[0m\n")

# ══════════════════════════════════════════════════════════════════════════════
# WRITING STYLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_style(text, c2i):
    """Show what the model learned about your writing"""
    print("\n" + "═"*55)
    print("  YOUR WRITING PROFILE")
    print("═"*55)

    total  = len(text)
    lines  = text.split('\n')
    words  = re.findall(r'\b\w+\b', text.lower())
    freq   = {}
    for w in words: freq[w] = freq.get(w,0)+1
    top20  = sorted(freq.items(), key=lambda x:-x[1])[:20]

    avg_line  = sum(len(l) for l in lines) / max(len(lines),1)
    avg_word  = sum(len(w) for w in words) / max(len(words),1)
    code_lines = sum(1 for l in lines if l.strip().startswith(
        ('def ', 'class ', 'import ', 'func ', 'var ', 'if ', 'for ')))

    print(f"\n  Total characters  : {total:,}")
    print(f"  Total words       : {len(words):,}")
    print(f"  Unique words      : {len(freq):,}")
    print(f"  Avg line length   : {avg_line:.1f} chars")
    print(f"  Avg word length   : {avg_word:.1f} chars")
    print(f"  Code lines        : {code_lines:,} ({code_lines/max(len(lines),1)*100:.1f}%)")
    print(f"  Vocab size (chars): {len(c2i)}")

    print(f"\n  Your 20 most common words:")
    for i, (w, c) in enumerate(top20):
        bar = '█' * min(int(c/max(top20[0][1],1)*30), 30)
        print(f"    {w:<15} {bar} {c}")

    # Punctuation style
    puncts = {c: text.count(c) for c in '.,!?;:()[]{}"\'-'}
    print(f"\n  Punctuation style:")
    for p, c in sorted(puncts.items(), key=lambda x:-x[1])[:8]:
        if c > 0: print(f"    '{p}'  {c:,}x")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true", help="Train on your writing")
    parser.add_argument("--chat",     action="store_true", help="Interactive chat")
    parser.add_argument("--analyze",  action="store_true", help="Analyze your writing style")
    parser.add_argument("--complete", type=str,            help="Complete a prompt")
    parser.add_argument("--dirs",     nargs="+",           help="Override data directories")
    args = parser.parse_args()

    dirs = args.dirs if args.dirs else DATA_DIRS

    if args.train:
        print("\n" + "═"*55)
        print("  Collecting your writing...")
        print("═"*55)
        files = collect_files(dirs, EXTENSIONS, SKIP_DIRS)
        print(f"  Found {len(files)} files")
        text  = load_text(files)

        if len(text) < 1000:
            print(f"\n  Only {len(text)} characters found.")
            print(f"  Add more text files to DATA_DIRS in personal_lm.py")
            print(f"  Minimum ~10,000 characters recommended.")
            return

        chars, c2i, i2c = build_vocab(text)
        data  = torch.tensor([c2i.get(c, 0) for c in text], dtype=torch.long)

        if args.analyze:
            analyze_style(text, c2i)

        print(f"\n  Building model (vocab={len(chars)})...")
        model = PersonalTritLM(len(chars)).to(device)
        train(model, data, c2i, chars)

        print("\nDone. Run with --chat to talk to your model.")

    elif args.chat:
        if not os.path.exists(MODEL_PATH):
            print("No trained model found. Run --train first.")
            return
        print("Loading your personal TritLM...")
        model, chars, c2i, i2c = PersonalTritLM.load(MODEL_PATH, VOCAB_PATH)
        model = model.to(device)
        chat(model, c2i, i2c, chars)

    elif args.complete:
        if not os.path.exists(MODEL_PATH):
            print("No trained model found. Run --train first.")
            return
        model, chars, c2i, i2c = PersonalTritLM.load(MODEL_PATH, VOCAB_PATH)
        model = model.to(device)
        result = model.complete(args.complete, c2i, i2c)
        print(result)

    elif args.analyze:
        print("Collecting writing for analysis...")
        files = collect_files(dirs, EXTENSIONS, SKIP_DIRS)
        text  = load_text(files)
        chars, c2i, i2c = build_vocab(text)
        analyze_style(text, c2i)

    else:
        print("""
Personal TritLM — 012 Ternary Language Model trained on your writing

Commands:
  python personal_lm.py --train              Train on your documents
  python personal_lm.py --train --analyze    Train + show writing profile
  python personal_lm.py --chat               Interactive chat with your model
  python personal_lm.py --analyze            Analyze writing style (no training)
  python personal_lm.py --complete "text"    Complete a prompt

  python personal_lm.py --train --dirs "C:/path/to/folder" "C:/another"
                                             Train on specific folders

Edit DATA_DIRS in personal_lm.py to set your default folders.
        """)

if __name__ == "__main__":
    main()
