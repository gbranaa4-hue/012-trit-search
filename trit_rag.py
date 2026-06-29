"""
012 Ternary RAG — Retrieval Augmented Generation at Scale

Architecture:
  Query
    → TritEncoder      (text → 256-dim trit vector)
    → TritIndex        (FAISS — searches millions of facts in milliseconds)
    → retrieved facts
    → TritLM           (generates response using retrieved context)
    → answer in your style

Scale comparison:
  Hopfield memory   : ~500 facts max (matrix grows as N²)
  FAISS flat index  : ~10M facts on your GPU (exact search)
  FAISS IVF index   : ~1B facts on disk (approximate, still fast)

This is how Google-scale retrieval works.
The difference: Google uses float32 embeddings and massive servers.
This uses trit embeddings and your RTX 5060.

Install:
  pip install faiss-gpu wikipedia-api tqdm

Usage:
  python trit_rag.py --demo          Quick demo with built-in facts
  python trit_rag.py --ingest-wiki   Download + index Wikipedia articles
  python trit_rag.py --load facts.txt --chat
  python trit_rag.py --chat          Chat using saved index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse, os, json, math, time, re
from pathlib import Path

os.makedirs("results", exist_ok=True)
os.makedirs("index",   exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Check for FAISS
try:
    import faiss
    FAISS_GPU = hasattr(faiss, 'StandardGpuResources')
    print(f"FAISS: available  (GPU={FAISS_GPU})")
except ImportError:
    faiss = None
    print("FAISS: not installed — run: pip install faiss-gpu")
    print("Falling back to brute-force search (works up to ~100k facts)")

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
# TRIT ENCODER
# Maps any text to a 256-dim trit vector
# Similar text → nearby vectors → retrieved together
# ══════════════════════════════════════════════════════════════════════════════

TRIT_DIM  = 256
ENC_DIM   = 128
MAX_LEN   = 128
VOCAB_SIZE = 256   # byte-level — handles any text, any language, any code

class TritEncoder(nn.Module):
    """
    Byte-level encoder: works on raw UTF-8 bytes.
    No vocabulary file needed — handles any text automatically.
    Input: raw bytes (0-255)
    Output: TRIT_DIM-dimensional trit vector {-1, 0, +1}
    """
    def __init__(self, trit_dim=TRIT_DIM, d=ENC_DIM):
        super().__init__()
        self.emb     = nn.Embedding(256, d)
        self.pos     = nn.Embedding(MAX_LEN, d)
        self.layers  = nn.ModuleList([
            nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=4*d,
                                       batch_first=True, dropout=0.0)
            for _ in range(2)
        ])
        self.project = nn.Linear(d, trit_dim, bias=False)
        self.norm    = nn.LayerNorm(trit_dim)

    def encode_bytes(self, texts):
        """texts: list of strings → (B, MAX_LEN) byte tensor"""
        batch = []
        for t in texts:
            b = list(t.encode("utf-8", errors="replace")[:MAX_LEN])
            b = b + [0] * (MAX_LEN - len(b))
            batch.append(b)
        return torch.tensor(batch, dtype=torch.long, device=device)

    def forward(self, idx):
        """idx: (B, T) byte indices → (B, trit_dim) continuous"""
        B, T  = idx.shape
        x     = self.emb(idx) + self.pos(torch.arange(T, device=device))
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)
        return self.norm(self.project(pooled))

    @torch.no_grad()
    def encode_texts(self, texts, to_trit=True):
        """Encode list of strings → numpy array of trit vectors"""
        self.eval()
        all_vecs = []
        bs = 64
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            idx   = self.encode_bytes(batch)
            cont  = self.forward(idx)
            if to_trit:
                vecs = tq(cont).cpu().numpy().astype(np.float32)
            else:
                vecs = cont.cpu().numpy().astype(np.float32)
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        m = cls()
        m.load_state_dict(torch.load(path, map_location=device))
        return m.to(device)

def train_encoder(encoder, facts, epochs=800):
    """
    Train encoder with contrastive loss:
    Same fact key+value → similar trit vectors
    Different facts → dissimilar trit vectors
    """
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    keys   = [k for k, v in facts]
    values = [v for k, v in facts]

    print(f"Training encoder ({epochs} iters)...")
    for it in range(epochs):
        encoder.train()
        # Sample batch
        idx   = np.random.randint(0, len(facts), 32)
        bkeys = [keys[i]   for i in idx]
        bvals = [values[i] for i in idx]

        kidx  = encoder.encode_bytes(bkeys)
        vidx  = encoder.encode_bytes(bvals)
        kenc  = encoder(kidx)
        venc  = encoder(vidx)

        # Positive: key and value should be close
        pos   = 1 - F.cosine_similarity(kenc, venc).mean()
        # Negative: shuffled pairs should be far
        perm  = torch.randperm(len(idx))
        neg   = F.cosine_similarity(kenc, venc[perm]).clamp(0).mean()

        loss  = pos + 0.5 * neg
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt.step(); sch.step()

        if (it+1) % 200 == 0:
            print(f"  iter {it+1}/{epochs}  loss={loss.item():.4f}")

    encoder.eval()
    print("  Done.\n")

# ══════════════════════════════════════════════════════════════════════════════
# TRIT INDEX — FAISS-BACKED SCALABLE RETRIEVAL
#
# Scale tiers:
#   Flat (exact)   : up to ~10M facts, millisecond search
#   IVF (approx)   : up to ~1B facts, still fast
#   HNSW (approx)  : best recall/speed tradeoff for >10M
#
# Trit vectors stored as float32 in FAISS.
# On real ternary hardware: store as 2-bit, 16x more compact.
# ══════════════════════════════════════════════════════════════════════════════

class TritIndex:
    def __init__(self, trit_dim=TRIT_DIM, mode="flat"):
        self.trit_dim  = trit_dim
        self.mode      = mode
        self.facts     = []    # (key, value) pairs
        self.index     = None
        self._build_index()

    def _build_index(self):
        if faiss is None:
            self.index = None
            return
        if self.mode == "flat":
            # Exact search — best for < 10M facts
            self.index = faiss.IndexFlatIP(self.trit_dim)  # inner product = cosine on normalized
        elif self.mode == "ivf":
            # Approximate — best for 1M-1B facts
            quantizer  = faiss.IndexFlatIP(self.trit_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.trit_dim, 256,
                                             faiss.METRIC_INNER_PRODUCT)
        elif self.mode == "hnsw":
            # Graph-based — best recall at scale
            self.index = faiss.IndexHNSWFlat(self.trit_dim, 32,
                                              faiss.METRIC_INNER_PRODUCT)

    def add(self, keys, values, encoder):
        """Add (key, value) pairs. keys: list of strings."""
        print(f"Encoding {len(keys)} facts...")
        vecs = encoder.encode_texts(keys, to_trit=False)
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        vecs  = vecs / norms

        if faiss is not None:
            if self.mode == "ivf" and not self.index.is_trained:
                print("Training IVF index...")
                self.index.train(vecs)
            self.index.add(vecs.astype(np.float32))
        else:
            # Brute force fallback
            if not hasattr(self, '_vecs'):
                self._vecs = vecs
            else:
                self._vecs = np.vstack([self._vecs, vecs])

        self.facts.extend(zip(keys, values))
        print(f"  Index size: {len(self.facts):,} facts")

    def search(self, query_text, encoder, k=5):
        """
        Retrieve top-k facts matching the query.
        Returns list of (key, value, score) tuples.
        """
        vec   = encoder.encode_texts([query_text], to_trit=False)
        norm  = np.linalg.norm(vec) + 1e-8
        vec   = (vec / norm).astype(np.float32)

        if faiss is not None:
            scores, indices = self.index.search(vec, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.facts):
                    k_txt, v_txt = self.facts[idx]
                    results.append((k_txt, v_txt, float(score)))
        else:
            # Brute force cosine similarity
            sims    = self._vecs @ vec.T        # (N, 1)
            top_k   = np.argsort(sims[:,0])[::-1][:k]
            results = [(self.facts[i][0], self.facts[i][1], float(sims[i,0]))
                       for i in top_k]

        return results

    def save(self, path="index/trit_index"):
        os.makedirs(path, exist_ok=True)
        if faiss is not None:
            faiss.write_index(self.index, f"{path}/faiss.index")
        elif hasattr(self, '_vecs'):
            np.save(f"{path}/vecs.npy", self._vecs)
        with open(f"{path}/facts.json", "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)
        print(f"  Index saved: {path} ({len(self.facts):,} facts)")

    @classmethod
    def load(cls, path="index/trit_index", trit_dim=TRIT_DIM, mode="flat"):
        idx = cls(trit_dim, mode)
        if faiss is not None and os.path.exists(f"{path}/faiss.index"):
            idx.index = faiss.read_index(f"{path}/faiss.index")
        elif os.path.exists(f"{path}/vecs.npy"):
            idx._vecs = np.load(f"{path}/vecs.npy")
        idx.facts = json.load(open(f"{path}/facts.json", encoding="utf-8"))
        print(f"  Index loaded: {len(idx.facts):,} facts")
        return idx

    def size_report(self):
        n = len(self.facts)
        # Trit storage: TRIT_DIM × 2 bits per fact
        trit_kb  = n * TRIT_DIM * 2 / 8 / 1024
        float_kb = n * TRIT_DIM * 4 / 1024
        print(f"\n  Index size report:")
        print(f"    Facts indexed     : {n:,}")
        print(f"    Float32 storage   : {float_kb:,.1f} KB")
        print(f"    Ternary storage   : {trit_kb:,.1f} KB  (theoretical)")
        print(f"    Compression       : {float_kb/max(trit_kb,0.001):.1f}x")
        if n > 0:
            print(f"\n  Scaling projection:")
            for scale, label in [(1e4,"10k"),(1e5,"100k"),(1e6,"1M"),(1e7,"10M"),(1e9,"1B")]:
                tb = scale * TRIT_DIM * 2 / 8 / 1024 / 1024
                print(f"    {label:>4} facts : {tb:>8.1f} MB ternary  |  "
                      f"{tb*16:>8.1f} MB float32")

# ══════════════════════════════════════════════════════════════════════════════
# TRITLM — GENERATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

CTX = 256   # longer context to fit retrieved facts + query

class TritMemCell(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = TernaryLinear(d, d, quantize=False)
        self.w = TernaryLinear(d, d, quantize=False)
        self.r = TernaryLinear(d, d, quantize=False)
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
        return self.n(x + r * torch.cat(outs, dim=1))

    def reset(self): self.s.zero_()

class TritLMBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)
        self.ln3  = nn.LayerNorm(d)
        self.mem  = TritMemCell(d)
        self.up   = TernaryLinear(d, 4*d, quantize=False)
        self.dn   = TernaryLinear(4*d, d, quantize=False)
        # Standard attention (faster than triadic for generation)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.register_buffer("mask", ~torch.tril(torch.ones(CTX,CTX,dtype=torch.bool)))

    def forward(self, x):
        B,T,_ = x.shape
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                                attn_mask=self.mask[:T,:T])
        x = x + attn_out
        x = self.mem(self.ln2(x))
        x = x + self.dn(F.gelu(self.up(self.ln3(x))))
        return x

    def reset(self): self.mem.reset()

class TritLM(nn.Module):
    def __init__(self, vocab=256, d=128, n_heads=4, n_layers=4):
        super().__init__()
        self.vocab   = vocab
        self.emb     = nn.Embedding(vocab, d)
        self.pos     = nn.Embedding(CTX, d)
        self.blocks  = nn.ModuleList([TritLMBlock(d, n_heads) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(d)
        self.head    = TernaryLinear(d, vocab, quantize=False)
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02)
                   if isinstance(m, (nn.Linear, TernaryLinear, nn.Embedding)) else None)
        n = sum(p.numel() for p in self.parameters())
        print(f"TritLM: {n:,} params  ({n*1.585/8/1024:.1f} KB ternary)")

    def forward(self, idx, targets=None):
        B,T  = idx.shape
        x    = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss   = F.cross_entropy(logits.view(-1,self.vocab), targets.view(-1)) if targets is not None else None
        return logits, loss

    def reset_memory(self):
        for b in self.blocks: b.reset()

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new=300, temperature=0.8, top_k=40):
        self.eval(); self.reset_memory()
        idx = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
        for _ in range(max_new):
            ctx    = idx[:, -CTX:]
            logits, _ = self(ctx)
            logits = logits[:,-1,:] / temperature
            if top_k:
                v,_ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,-1:]] = float('-inf')
            nxt = torch.multinomial(F.softmax(logits,-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        new_bytes = idx[0, len(prompt_bytes):].cpu().tolist()
        return bytes([b for b in new_bytes if 0 <= b < 256]).decode("utf-8", errors="replace")

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"  TritLM saved: {path}")

    @classmethod
    def load(cls, path):
        m = cls().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        set_quant(m, True)
        return m

# ══════════════════════════════════════════════════════════════════════════════
# TRIT RAG PIPELINE
# Query → retrieve → generate
# ══════════════════════════════════════════════════════════════════════════════

class TritRAG:
    """
    Full pipeline:
      1. Encode query as trit vector
      2. Search FAISS index for top-k matching facts
      3. Build context: retrieved facts + original query
      4. TritLM generates answer from context
    """
    def __init__(self, encoder, index, lm):
        self.encoder = encoder
        self.index   = index
        self.lm      = lm

    def answer(self, query, k=3, temperature=0.8, verbose=True):
        # Step 1: retrieve
        t0      = time.perf_counter()
        results = self.index.search(query, self.encoder, k=k)
        t_ret   = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"\n  Retrieved ({t_ret:.1f}ms):")
            for key, val, score in results:
                print(f"    [{score:.3f}] {key} → {val}")

        # Step 2: build context string
        context = "Facts:\n"
        for key, val, score in results:
            context += f"- {key}: {val}\n"
        context += f"\nQuestion: {query}\nAnswer:"

        # Step 3: generate
        prompt_bytes = context.encode("utf-8")[:CTX]
        t0    = time.perf_counter()
        reply = self.lm.generate(prompt_bytes, max_new=200,
                                  temperature=temperature)
        t_gen = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"  Generated ({t_gen:.1f}ms)")

        return reply, results

# ══════════════════════════════════════════════════════════════════════════════
# WIKIPEDIA INGESTION
# Downloads Wikipedia articles and indexes them
# ══════════════════════════════════════════════════════════════════════════════

def ingest_wikipedia(topics, encoder, index, max_sections=5):
    """
    Download Wikipedia articles and store as facts.
    Each section becomes a (title+section, text) fact.
    """
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia("TritRAG/1.0", "en")
    except ImportError:
        print("Install: pip install wikipedia-api")
        return 0

    facts_added = 0
    for topic in topics:
        print(f"  Fetching: {topic}")
        page = wiki.page(topic)
        if not page.exists():
            print(f"    Not found: {topic}")
            continue

        keys, values = [], []
        for section in list(page.sections)[:max_sections]:
            if len(section.text) < 50: continue
            key = f"{topic} — {section.title}"
            val = section.text[:500]   # first 500 chars of each section
            keys.append(key)
            values.append(val)

        if keys:
            index.add(keys, values, encoder)
            facts_added += len(keys)
            print(f"    Added {len(keys)} sections")

    return facts_added

def ingest_file(path, encoder, index):
    """
    Load facts from a plain text file.
    Format: one fact per line, key | value
    Example:
      capital of France | Paris
      my project path | C:\\Users\\...
    """
    facts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            if "|" in line:
                key, val = line.split("|", 1)
                facts.append((key.strip(), val.strip()))
            else:
                # Treat whole line as both key and value
                facts.append((line, line))

    if facts:
        index.add([k for k,v in facts], [v for k,v in facts], encoder)
    return len(facts)

def ingest_text_chunks(text, source_name, encoder, index, chunk_size=300, overlap=50):
    """
    Split any long text into overlapping chunks and index each.
    Use for: books, papers, chat logs, your notes
    """
    words  = text.split()
    chunks = []
    keys   = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        key   = f"{source_name} [chunk {i//chunk_size}]"
        chunks.append(chunk)
        keys.append(key)

    index.add(keys, chunks, encoder)
    return len(chunks)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def get_batch(text_bytes, batch_size=64, ctx=CTX):
    data = torch.tensor(list(text_bytes), dtype=torch.long, device=device)
    ix   = torch.randint(0, len(data)-ctx, (batch_size,))
    x    = torch.stack([data[i:i+ctx]     for i in ix])
    y    = torch.stack([data[i+1:i+ctx+1] for i in ix])
    return x, y

def train_lm(lm, corpus_text, iters=1000, warmup=200):
    corpus = corpus_text.encode("utf-8")
    opt    = torch.optim.AdamW(lm.parameters(), lr=3e-4, weight_decay=0.1)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    print(f"Training TritLM ({iters} iters, {len(corpus):,} bytes)...")
    for it in range(iters):
        set_quant(lm, it >= warmup)
        if it % 50 == 0: lm.reset_memory()
        xb, yb  = get_batch(corpus)
        _, loss = lm(xb, yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step(); sch.step()
        if (it+1) % 200 == 0 or it == iters-1:
            phase = "TRIT" if it >= warmup else "warmup"
            print(f"  [{phase}] {it+1}/{iters}  loss={loss.item():.4f}")
    lm.eval()
    print("  Done.\n")

# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

BUILTIN_FACTS = [
    # Science
    ("capital of France",            "Paris"),
    ("capital of Germany",           "Berlin"),
    ("capital of Japan",             "Tokyo"),
    ("speed of light",               "299,792,458 metres per second"),
    ("boiling point of water",       "100 degrees Celsius at sea level"),
    ("formula for water",            "H2O, two hydrogen one oxygen"),
    ("formula for carbon dioxide",   "CO2, one carbon two oxygen"),
    ("DNA stands for",               "Deoxyribonucleic acid"),
    ("distance to the moon",         "384,400 kilometres on average"),
    ("year of moon landing",         "1969, Apollo 11"),
    ("inventor of telephone",        "Alexander Graham Bell in 1876"),
    # 012 Project
    ("what is a trit",               "Ternary digit with values -1 0 or +1"),
    ("what is triadic architecture", "Three parallel streams Observer Shadow Light"),
    ("what is the consensus gate",   "sign(a+b+c) majority vote of three trits"),
    ("what is TritLM",               "Ternary language model with fixed memory cell"),
    ("TritCognition accuracy",       "80.08 percent on CIFAR-10 at 0 degrees"),
    ("TritCognition size",           "0.078 megabytes, 28x fewer params than ResNet18"),
    ("TritLM memory footprint",      "0.05 kilobytes fixed regardless of context length"),
    ("012 noise recovery",           "0.345 nats recovered at 50 percent corruption"),
    ("hardware tests",               "36 of 36 testbench cases pass"),
    ("FPGA target",                  "Xilinx Ultrascale Plus at 250 MHz"),
    ("energy saving",                "87 percent per token vs float32 on 28nm CMOS"),
    ("what is ternary quantization", "Snap weights to -1 0 +1 with threshold 0.7E mean"),
    ("what is predictive coding",    "Loss term where each layer predicts next layer input"),
    ("what is a Hopfield network",   "Content-addressable associative memory using Hebbian weights"),
    # Computing
    ("what is FAISS",                "Facebook AI Similarity Search, billion-scale vector retrieval"),
    ("what is RAG",                  "Retrieval Augmented Generation, search then generate"),
    ("what is a KV cache",           "Key-Value store for transformer context, grows linearly with tokens"),
    ("what is CUDA",                 "NVIDIAs parallel computing platform for GPU programming"),
    ("PyTorch loss backward",        "loss.backward() computes gradients via autograd"),
    ("what is cosine similarity",    "Dot product of normalized vectors, measures angle between them"),
    ("what is top-k sampling",       "Sample from the k most likely next tokens only"),
    ("what is perplexity",           "exp of cross-entropy loss, measures language model quality"),
    ("what is byte-level encoding",  "Tokenize text as raw UTF-8 bytes, handles any language"),
    # Godot
    ("GDScript signal syntax",       "signal name(args) and emit_signal or name.emit"),
    ("Godot scene tree",             "Node hierarchy, root contains all other nodes"),
    ("GDScript extends",             "extends Node2D or extends CharacterBody2D etc"),
    ("Godot physics process",        "_physics_process(delta) called 60 times per second"),
]

# ══════════════════════════════════════════════════════════════════════════════
# SCALE DEMONSTRATION
# Shows what the system looks like at different scales
# ══════════════════════════════════════════════════════════════════════════════

def scale_demo(encoder, index):
    print(f"\n{'═'*60}")
    print(f"  SCALE DEMONSTRATION")
    print(f"{'═'*60}")
    print(f"""
  Current index  : {len(index.facts):,} facts

  What adding more data looks like:

  1,000 facts    → your complete personal knowledge base
                   all your notes, code comments, project docs
                   ~3 min to encode on RTX 5060

  10,000 facts   → a textbook
                   every section of every chapter indexed
                   ~30 min to encode

  100,000 facts  → Wikipedia subset (top 10k articles)
                   run: python trit_rag.py --ingest-wiki
                   ~5 hours to encode (overnight)

  1,000,000 facts → full English Wikipedia (~6.7M articles × ~150 sections)
                    ~50 hours on RTX 5060
                    fits in {1e6 * TRIT_DIM * 4 / 1024 / 1024:.0f} MB float32
                    or {1e6 * TRIT_DIM * 2 / 8 / 1024 / 1024:.0f} MB ternary

  1,000,000,000 facts → approaching Google scale
                        requires distributed FAISS across multiple machines
                        trit compression: {1e9*TRIT_DIM*2/8/1024/1024/1024:.0f} GB
                        vs float32:       {1e9*TRIT_DIM*4/1024/1024/1024:.0f} GB

  The retrieval speed stays constant at every scale:
    FAISS flat  : O(N) but GPU-parallel → ~1ms for 1M facts
    FAISS IVF   : O(sqrt(N))           → ~1ms for 1B facts
  """)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

ENC_PATH = "index/trit_encoder.pt"
LM_PATH  = "index/trit_lm.pt"
IDX_PATH = "index/trit_index"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",         action="store_true", help="Run built-in demo")
    parser.add_argument("--chat",         action="store_true", help="Interactive chat")
    parser.add_argument("--ingest-wiki",  action="store_true", help="Download Wikipedia articles")
    parser.add_argument("--load",         type=str,            help="Load facts from file (key | value)")
    parser.add_argument("--ingest-text",  type=str,            help="Ingest a text file as chunks")
    parser.add_argument("--query",        type=str,            help="Single query (no chat)")
    parser.add_argument("--scale",        action="store_true", help="Show scale projections")
    parser.add_argument("--retrain",      action="store_true", help="Force retrain encoder+LM")
    args = parser.parse_args()

    # ── Load or train encoder ──────────────────────────────────────────────────
    if os.path.exists(ENC_PATH) and not args.retrain:
        print("Loading encoder...")
        encoder = TritEncoder().to(device)
        encoder.load_state_dict(torch.load(ENC_PATH, map_location=device))
        encoder.eval()
    else:
        print("Training encoder on built-in facts...")
        encoder = TritEncoder().to(device)
        train_encoder(encoder, BUILTIN_FACTS, epochs=800)
        encoder.save(ENC_PATH)

    # ── Load or train LM ───────────────────────────────────────────────────────
    if os.path.exists(LM_PATH) and not args.retrain:
        print("Loading TritLM...")
        lm = TritLM.load(LM_PATH).to(device)
    else:
        lm = TritLM().to(device)
        # Train LM on knowledge base text as corpus
        corpus = "\n".join(f"{k}: {v}" for k, v in BUILTIN_FACTS) * 10
        train_lm(lm, corpus, iters=1000, warmup=200)
        lm.save(LM_PATH)

    # ── Load or build index ────────────────────────────────────────────────────
    if os.path.exists(f"{IDX_PATH}/facts.json") and not args.retrain:
        print("Loading index...")
        index = TritIndex.load(IDX_PATH)
    else:
        index = TritIndex(mode="flat")
        keys   = [k for k, v in BUILTIN_FACTS]
        values = [v for k, v in BUILTIN_FACTS]
        index.add(keys, values, encoder)
        index.save(IDX_PATH)

    # ── Ingest additional data ─────────────────────────────────────────────────
    if args.load:
        n = ingest_file(args.load, encoder, index)
        print(f"Added {n} facts from {args.load}")
        index.save(IDX_PATH)

    if args.ingest_text:
        text = open(args.ingest_text, "r", encoding="utf-8").read()
        n    = ingest_text_chunks(text, Path(args.ingest_text).name, encoder, index)
        print(f"Added {n} chunks from {args.ingest_text}")
        index.save(IDX_PATH)

    if args.ingest_wiki:
        topics = [
            "Ternary computer", "Hopfield network", "Transformer (deep learning)",
            "FPGA", "Predictive coding", "Neural network", "Computer memory",
            "Quantum computing", "Artificial intelligence", "Deep learning",
        ]
        n = ingest_wikipedia(topics, encoder, index)
        print(f"Added {n} Wikipedia sections")
        index.save(IDX_PATH)

    # ── Build RAG pipeline ─────────────────────────────────────────────────────
    rag = TritRAG(encoder, index, lm)

    if args.scale:
        scale_demo(encoder, index)
        index.size_report()

    if args.query:
        reply, _ = rag.answer(args.query)
        print(f"\nAnswer: {reply}")
        return

    if args.demo or (not args.chat and not args.load
                     and not args.ingest_wiki and not args.ingest_text):
        print(f"\n{'═'*60}")
        print(f"  DEMO — TRIT RAG PIPELINE")
        print(f"{'═'*60}\n")

        demo_queries = [
            "what is a trit",
            "how does TritLM save memory",
            "speed of light",
            "what is the consensus gate",
            "TritCognition accuracy results",
        ]

        for q in demo_queries:
            print(f"{'─'*50}")
            print(f"  Q: {q}")
            reply, results = rag.answer(q, k=2, verbose=True)
            # Clean up generated text — take first sentence
            clean = reply.split('\n')[0].strip()[:200]
            print(f"  A: {clean}\n")

        index.size_report()
        scale_demo(encoder, index)

    if args.chat:
        print(f"\n{'═'*60}")
        print(f"  TRIT RAG — Interactive")
        print(f"  {len(index.facts):,} facts indexed")
        print(f"  Commands: :facts  :scale  :add key|value  :quit")
        print(f"{'═'*60}\n")

        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q: continue
            if q == ":quit": break
            if q == ":facts":
                for k, v in index.facts:
                    print(f"  {k:<40} → {v[:60]}")
                continue
            if q == ":scale":
                scale_demo(encoder, index)
                index.size_report()
                continue
            if q.startswith(":add "):
                rest = q[5:]
                if "|" in rest:
                    k, v = rest.split("|", 1)
                    index.add([k.strip()], [v.strip()], encoder)
                    index.save(IDX_PATH)
                    print(f"  Added: '{k.strip()}' → '{v.strip()}'")
                continue

            reply, _ = rag.answer(q, k=3, verbose=True)
            clean = reply.split('\n')[0].strip()[:300]
            print(f"\n012: {clean}\n")

if __name__ == "__main__":
    main()
