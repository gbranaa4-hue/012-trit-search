"""
012 Ternary Distillation
Qwen2.5:7b (teacher via Ollama) → TritLM (student, your architecture)

Pipeline:
  1. Generate diverse text from Qwen via Ollama API
  2. Save as training corpus
  3. Train TritLM on that corpus with ternary-aware training
  4. Result: your triadic architecture trained on quality data

Ollama must be running: ollama run qwen2.5:7b

Usage:
  python trit_distill.py --generate        Generate training data from Qwen
  python trit_distill.py --train           Train TritLM on generated data
  python trit_distill.py --all             Generate + train in one shot
  python trit_distill.py --chat            Chat with trained model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, json, time, urllib.request, urllib.error
from pathlib import Path

os.makedirs("results",   exist_ok=True)
os.makedirs("data",      exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b"
DATA_PATH   = "data/distill_corpus.txt"
MODEL_PATH  = "results/trit_distilled.pt"
VOCAB_PATH  = "results/trit_vocab.json"

# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS — diverse topics so TritLM learns broad knowledge
# ══════════════════════════════════════════════════════════════════════════════

SEED_PROMPTS = [
    # Science
    "Explain how black holes form in simple terms.",
    "What is quantum entanglement?",
    "How does DNA replication work?",
    "Explain the theory of relativity simply.",
    "What causes the northern lights?",
    "How do vaccines work?",
    "What is dark matter?",
    "Explain photosynthesis step by step.",
    "How does the immune system fight viruses?",
    "What is the difference between fission and fusion?",
    # Computing
    "Explain how neural networks learn.",
    "What is the difference between RAM and storage?",
    "How does encryption work?",
    "Explain recursion with an example.",
    "What is a binary search tree?",
    "How does the internet actually work?",
    "What is an API?",
    "Explain object oriented programming.",
    "What is the difference between a process and a thread?",
    "How does garbage collection work?",
    # Math
    "Explain calculus derivatives intuitively.",
    "What is the Pythagorean theorem and why does it work?",
    "Explain prime numbers and why they matter.",
    "What is a Fourier transform in simple terms?",
    "How does probability relate to real life?",
    # History
    "What caused World War 1?",
    "Explain the Renaissance in 3 paragraphs.",
    "What was the significance of the printing press?",
    "How did the Roman Empire fall?",
    "What started the space race?",
    # Game development
    "How do game physics engines work?",
    "Explain pathfinding algorithms used in games.",
    "What is a game loop?",
    "How does procedural generation work in games?",
    "What is the difference between a shader and a script in game dev?",
    "Explain how collision detection works.",
    "What makes a good enemy AI in games?",
    "How do multiplayer games handle lag?",
    # Godot specific
    "Explain Godot's scene tree and node system.",
    "How do signals work in GDScript?",
    "What is the difference between _process and _physics_process in Godot?",
    "How do you handle state machines in Godot?",
    "Explain Godot's resource system.",
    # 012 / AI research
    "What is the difference between a transformer and an RNN?",
    "Explain attention mechanisms in neural networks.",
    "What is quantization in machine learning?",
    "How does knowledge distillation work?",
    "What is the vanishing gradient problem?",
    "Explain reinforcement learning simply.",
    "What is the difference between supervised and unsupervised learning?",
    "What is transfer learning?",
    "Explain what a loss function does.",
    "What is backpropagation?",
    # Philosophy / reasoning
    "What is the scientific method?",
    "Explain Occam's razor.",
    "What is the difference between correlation and causation?",
    "How should you approach a problem you've never seen before?",
    "What makes a good explanation?",
    # Writing / language
    "What makes writing clear and easy to understand?",
    "Explain the difference between active and passive voice.",
    "How do you structure a good argument?",
    "What is the difference between denotation and connotation?",
    # Practical
    "How do you debug code systematically?",
    "What is version control and why use it?",
    "How do you estimate how long a project will take?",
    "What is technical debt?",
    "How do you read someone else's code?",
]

# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def ollama_generate(prompt, max_tokens=400, temperature=0.7):
    """Call Ollama API and return generated text."""
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")
    except urllib.error.URLError:
        return None

def check_ollama():
    """Check if Ollama is running."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return models
    except:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_corpus(n_prompts=None, append=False):
    """Generate training corpus from Qwen via Ollama."""
    models = check_ollama()
    if models is None:
        print("Ollama is not running.")
        print("Start it with: ollama run qwen2.5:7b")
        return False

    print(f"Ollama running. Models: {models}")
    if not any("qwen2.5" in m for m in models):
        print(f"qwen2.5:7b not found. Run: ollama run qwen2.5:7b")
        return False

    prompts = SEED_PROMPTS[:n_prompts] if n_prompts else SEED_PROMPTS
    mode    = "a" if append else "w"
    total   = 0

    print(f"\nGenerating {len(prompts)} responses from {OLLAMA_MODEL}...")
    print(f"Saving to {DATA_PATH}\n")

    with open(DATA_PATH, mode, encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")
            t0       = time.time()
            response = ollama_generate(prompt, max_tokens=400)
            elapsed  = time.time() - t0

            if response is None:
                print(f"    FAILED — skipping")
                continue

            # Write as Q&A pair
            text = f"Q: {prompt}\nA: {response}\n\n"
            f.write(text)
            f.flush()
            total += len(response)
            print(f"    {len(response)} chars in {elapsed:.1f}s")

    size_kb = os.path.getsize(DATA_PATH) / 1024
    print(f"\nCorpus: {DATA_PATH} ({size_kb:.1f} KB, {total:,} chars)\n")
    return True

# ══════════════════════════════════════════════════════════════════════════════
# TRITLM — scaled up from trit_lm.py
# ══════════════════════════════════════════════════════════════════════════════

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        t = 0.7 * x.abs().mean()
        ctx.save_for_backward(x)
        return torch.where(x > t, torch.ones_like(x),
               torch.where(x < -t, -torch.ones_like(x),
               torch.zeros_like(x)))
    @staticmethod
    def backward(ctx, grad): return grad

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

class TritMemCell(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Linear(d, d, bias=False)
        self.w = nn.Linear(d, d, bias=False)
        self.r = nn.Linear(d, d, bias=False)
        self.n = nn.LayerNorm(d)
        self.register_buffer('s', torch.zeros(1,1,d))

    def forward(self, x):
        B,T,C = x.shape
        f = torch.sigmoid(self.f(x))
        w = torch.tanh(self.w(x))
        r = torch.sigmoid(self.r(x))
        s = self.s.expand(B,1,C).clamp(-1,1)
        outs = []
        for t in range(T):
            s = s*(1-f[:,t:t+1]) + w[:,t:t+1]*f[:,t:t+1]
            outs.append(s)
        self.s = s.mean(0,keepdim=True).detach()
        return self.n(x + r * torch.cat(outs,dim=1))

    def reset(self): self.s.zero_()

class TritBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)
        self.ln3  = nn.LayerNorm(d)
        self.mem  = TritMemCell(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.0)
        self.ff1  = TernaryLinear(d, 4*d)
        self.ff2  = TernaryLinear(4*d, d)
        self.register_buffer("mask", ~torch.tril(torch.ones(CTX, CTX, dtype=torch.bool)))

    def forward(self, x):
        B,T,_ = x.shape
        a,_   = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                          attn_mask=self.mask[:T,:T])
        x = x + a
        x = self.mem(self.ln2(x))
        x = x + self.ff2(F.gelu(self.ff1(self.ln3(x))))
        return x

    def reset(self): self.mem.reset()

# Model config — bigger than trit_lm.py
CTX     = 512
D_MODEL = 256
N_HEADS = 8
N_LAYER = 8

class TritLM(nn.Module):
    def __init__(self, vocab_size, d=D_MODEL, heads=N_HEADS, layers=N_LAYER):
        super().__init__()
        self.vocab  = vocab_size
        self.emb    = nn.Embedding(vocab_size, d)
        self.pos    = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList([TritBlock(d, heads) for _ in range(layers)])
        self.ln_f   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab_size, bias=False)
        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters())
        print(f"TritLM: {n:,} params  ({n*1.585/8/1024:.1f} KB ternary)")

    def _init(self, m):
        if isinstance(m, (nn.Linear, TernaryLinear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B,T  = idx.shape
        x    = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss   = F.cross_entropy(logits.view(-1,self.vocab),
                                  targets.view(-1)) if targets is not None else None
        return logits, loss

    def reset_memory(self):
        for b in self.blocks: b.reset()

    @torch.no_grad()
    def generate(self, idx, max_new=300, temperature=0.8, top_k=50):
        self.eval(); self.reset_memory()
        for _ in range(max_new):
            ctx    = idx[:, -CTX:]
            logits, _ = self(ctx)
            logits = logits[:,-1,:] / max(temperature, 1e-5)
            if top_k:
                v,_ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,-1:]] = float('-inf')
            nxt = torch.multinomial(F.softmax(logits,-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == self.vocab - 1: break  # eos
        return idx

# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER — character level on actual corpus vocab
# ══════════════════════════════════════════════════════════════════════════════

def build_vocab(text):
    chars = sorted(set(text))
    stoi  = {c:i for i,c in enumerate(chars)}
    itos  = {i:c for i,c in enumerate(chars)}
    return stoi, itos

def encode(text, stoi):
    return [stoi.get(c, 0) for c in text]

def decode(ids, itos):
    return ''.join(itos.get(i,'?') for i in ids)

# ══════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# Stores a random sample of past training data.
# Mixed into every new training cycle to prevent forgetting.
# Stays small (~1-2 MB) regardless of how much data you've trained on.
# ══════════════════════════════════════════════════════════════════════════════

REPLAY_PATH    = "data/replay_buffer.txt"
REPLAY_MAX_KB  = 1024   # 1 MB max — enough to remember everything important
REPLAY_MIX     = 0.3    # 30% of each batch comes from replay, 70% from new data

class ReplayBuffer:
    def __init__(self, path=REPLAY_PATH, max_kb=REPLAY_MAX_KB):
        self.path   = path
        self.max_kb = max_kb
        self.lines  = []
        if os.path.exists(path):
            self.lines = open(path, "r", encoding="utf-8").readlines()
            print(f"  Replay buffer: {len(self.lines)} lines "
                  f"({os.path.getsize(path)//1024} KB)")

    def add(self, text, sample_rate=0.15):
        """
        Sample random lines from new text and add to replay.
        sample_rate=0.15 means keep 15% of new data in replay.
        Trims oldest lines if buffer exceeds max_kb.
        """
        new_lines = text.splitlines(keepends=True)
        import random
        sampled   = [l for l in new_lines if random.random() < sample_rate and len(l.strip()) > 20]
        self.lines.extend(sampled)

        # Trim to max size — keep most recent
        current_kb = sum(len(l) for l in self.lines) / 1024
        while current_kb > self.max_kb and len(self.lines) > 100:
            self.lines.pop(0)
            current_kb = sum(len(l) for l in self.lines) / 1024

        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(self.lines)

        print(f"  Replay buffer: {len(self.lines)} lines "
              f"({sum(len(l) for l in self.lines)//1024} KB)")

    def get_text(self):
        return "".join(self.lines)

    def empty(self):
        return len(self.lines) == 0

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def get_batch(data, batch_size=32, ctx=CTX):
    if len(data) <= ctx:
        return None, None
    ix = torch.randint(0, len(data)-ctx, (batch_size,))
    x  = torch.stack([data[i:i+ctx]     for i in ix])
    y  = torch.stack([data[i+1:i+ctx+1] for i in ix])
    return x.to(device), y.to(device)

def get_mixed_batch(new_data, replay_data, batch_size=32, ctx=CTX, mix=REPLAY_MIX):
    """
    Mix new data and replay data in each batch.
    mix=0.3 means 30% from replay, 70% from new.
    If no replay data, use 100% new data.
    """
    if replay_data is None or len(replay_data) <= ctx:
        return get_batch(new_data, batch_size, ctx)

    n_replay = max(1, int(batch_size * mix))
    n_new    = batch_size - n_replay

    xn, yn = get_batch(new_data,    n_new,    ctx)
    xr, yr = get_batch(replay_data, n_replay, ctx)

    if xn is None: return xr, yr
    if xr is None: return xn, yn

    return torch.cat([xn, xr], dim=0), torch.cat([yn, yr], dim=0)

def train(corpus_path=DATA_PATH, iters=3000, warmup=500, cycle=None):
    """
    Train TritLM on corpus_path.
    cycle: optional label for this training cycle (e.g. "wikipedia_chunk_1")
    Automatically loads existing model if one exists (continual learning).
    Saves replay buffer after training so next cycle remembers this one.
    """
    if not os.path.exists(corpus_path):
        print(f"No corpus found at {corpus_path}")
        print("Run: python trit_distill.py --generate")
        return

    text = open(corpus_path, "r", encoding="utf-8").read()
    print(f"Corpus: {len(text):,} chars  ({len(text)//1024} KB)")
    if cycle:
        print(f"Cycle : {cycle}")

    # Load or build vocab — extend existing vocab with new chars
    if os.path.exists(VOCAB_PATH):
        saved  = json.load(open(VOCAB_PATH))
        old_stoi = saved["stoi"]
        old_itos = saved["itos"]
        # Add any new characters from this corpus
        new_chars = set(text) - set(old_stoi.keys())
        if new_chars:
            print(f"  New chars in this corpus: {len(new_chars)} — extending vocab")
        all_chars = sorted(set(old_stoi.keys()) | set(text))
        stoi = {c:i for i,c in enumerate(all_chars)}
        itos = {i:c for i,c in enumerate(all_chars)}
    else:
        all_chars = sorted(set(text))
        stoi = {c:i for i,c in enumerate(all_chars)}
        itos = {i:c for i,c in enumerate(all_chars)}

    json.dump({"stoi": stoi, "itos": itos}, open(VOCAB_PATH, "w"))
    print(f"Vocab: {len(stoi)} unique characters")

    # Encode new corpus
    new_data  = torch.tensor(encode(text, stoi), dtype=torch.long)
    split     = int(0.9 * len(new_data))
    train_new = new_data[:split]
    val_data  = new_data[split:]

    # Load replay buffer
    replay       = ReplayBuffer()
    replay_data  = None
    if not replay.empty():
        replay_text = replay.get_text()
        replay_data = torch.tensor(encode(replay_text, stoi), dtype=torch.long)
        print(f"  Replay data: {len(replay_data):,} tokens")

    # Load existing model if available (continual learning)
    if os.path.exists(MODEL_PATH):
        print(f"\nLoading existing model for continual learning...")
        ckpt  = torch.load(MODEL_PATH, map_location=device)
        model = TritLM(vocab_size=len(stoi)).to(device)
        # Load what we can, ignore mismatched embedding sizes
        try:
            model.load_state_dict(ckpt["model"], strict=False)
            print(f"  Resumed from checkpoint")
        except:
            print(f"  Vocab changed — starting fresh with same architecture")
    else:
        print(f"\nNo existing model — training from scratch")
        model = TritLM(vocab_size=len(stoi)).to(device)

    # Lower LR for continual learning (don't overwrite old knowledge aggressively)
    lr  = 1e-4 if os.path.exists(MODEL_PATH) else 3e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    print(f"Training {iters} iters  lr={lr}  replay_mix={REPLAY_MIX}\n")
    best_val = float('inf')

    for it in range(iters):
        set_quant(model, it >= warmup)
        if it % 100 == 0:
            model.reset_memory()

        xb, yb  = get_mixed_batch(train_new, replay_data)
        _, loss = model(xb, yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()

        if (it+1) % 500 == 0 or it == iters-1:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data)
                if xv is not None:
                    _, vl = model(xv, yv)
                    val_loss = vl.item()
                else:
                    val_loss = loss.item()

            phase = "TRIT" if it >= warmup else "warm"
            print(f"  [{phase}] iter {it+1}/{iters}  "
                  f"train={loss.item():.4f}  val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(),
                            "stoi": stoi, "itos": itos,
                            "cycle": cycle}, MODEL_PATH)
                print(f"  Saved ({MODEL_PATH})")

            # Quick sample
            model.reset_memory()
            prompt = "Q: What is"
            idx    = torch.tensor([encode(prompt, stoi)],
                                   dtype=torch.long, device=device)
            out    = model.generate(idx, max_new=80, temperature=0.8)
            sample = decode(out[0].tolist(), itos)
            print(f"  Sample: {sample[:120]}\n")
            model.train()

    # Update replay buffer with this cycle's data
    print("Updating replay buffer...")
    replay.add(text, sample_rate=0.15)

    print(f"\nCycle done. Best val loss: {best_val:.4f}")
    print(f"Model : {MODEL_PATH}")
    print(f"Replay: {REPLAY_PATH}\n")

def train_file(path, iters=2000, cycle=None):
    """Train on any text file. Wrapper for feeding chunks."""
    train(corpus_path=path, iters=iters,
          cycle=cycle or Path(path).stem)

# ══════════════════════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════════════════════

def chat():
    if not os.path.exists(MODEL_PATH):
        print(f"No model found. Run: python trit_distill.py --train")
        return

    ckpt  = torch.load(MODEL_PATH, map_location=device)
    stoi  = ckpt["stoi"]
    itos  = ckpt["itos"]
    model = TritLM(vocab_size=len(stoi)).to(device)
    model.load_state_dict(ckpt["model"])
    set_quant(model, True)
    model.eval()
    print(f"Loaded TritLM  ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Vocab: {len(stoi)} chars\n")

    print("012 TritLM — trained on Qwen2.5 distillation data")
    print("Commands: :quit  :reset  :temp 0.8\n")

    temperature = 0.8
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q: continue
        if q == ":quit": break
        if q == ":reset": model.reset_memory(); print("  Memory reset."); continue
        if q.startswith(":temp"):
            try: temperature = float(q.split()[1]); print(f"  Temp: {temperature}")
            except: pass
            continue

        prompt = f"Q: {q}\nA:"
        # Filter prompt to known vocab
        prompt_clean = ''.join(c for c in prompt if c in stoi)
        idx    = torch.tensor([encode(prompt_clean, stoi)],
                               dtype=torch.long, device=device)
        t0     = time.perf_counter()
        out    = model.generate(idx, max_new=300, temperature=temperature)
        elapsed = time.perf_counter() - t0
        n_new  = out.shape[1] - idx.shape[1]

        reply  = decode(out[0][idx.shape[1]:].tolist(), itos)
        # Stop at next Q:
        if "\nQ:" in reply:
            reply = reply[:reply.index("\nQ:")]

        print(f"\n012: {reply.strip()}")
        print(f"     [{n_new} tokens, {n_new/elapsed:.1f} tok/s]\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate",   action="store_true", help="Generate corpus from Qwen")
    parser.add_argument("--train",      action="store_true", help="Train on current corpus")
    parser.add_argument("--all",        action="store_true", help="Generate + train")
    parser.add_argument("--chat",       action="store_true", help="Chat with trained model")
    parser.add_argument("--learn",      type=str, default=None,
                        help="Continual learning: path to new text file")
    parser.add_argument("--iters",      type=int, default=3000)
    parser.add_argument("--prompts",    type=int, default=None)
    parser.add_argument("--cycle",      type=str, default=None,
                        help="Label for this training cycle")
    args = parser.parse_args()

    if args.generate or args.all:
        ok = generate_corpus(args.prompts)
        if not ok: return

    if args.train or args.all:
        train(iters=args.iters, cycle=args.cycle or "distill")

    if args.learn:
        # Feed any new file — model remembers old knowledge via replay buffer
        if not os.path.exists(args.learn):
            print(f"File not found: {args.learn}")
        else:
            train_file(args.learn, iters=args.iters,
                       cycle=args.cycle or Path(args.learn).stem)

    if args.chat:
        chat()

    if not any([args.generate, args.train, args.all, args.chat, args.learn]):
        print(__doc__)
        print(f"Corpus exists : {os.path.exists(DATA_PATH)}")
        print(f"Model exists  : {os.path.exists(MODEL_PATH)}")
        print(f"Replay exists : {os.path.exists(REPLAY_PATH)}")

if __name__ == "__main__":
    main()
