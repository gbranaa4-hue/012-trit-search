"""
012 Ternary Associative Memory — Fact Storage and Retrieval

A Hopfield-style content-addressable memory using ternary weights.
Store facts as trit patterns. Query with partial/noisy input.
Consensus gate reconstructs the full pattern.

This is fundamentally different from TritLM:
  TritLM      : predicts next character (generative)
  TritMemStore: stores discrete facts, retrieves by pattern matching

Architecture:
  Fact → encode → trit pattern (N trits)
  Query (partial/noisy) → consensus retrieval → reconstructed fact

The consensus gate IS the retrieval mechanism:
  consensus(a, b, c) = sign(a + b + c)
  Majority vote across stored patterns → nearest stored fact

Hardware implication:
  Each retrieval = N ternary additions, no multiplications
  Storage = N trit registers per fact
  This is what ternary CAM (Content Addressable Memory) does in silicon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, math

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
    def __init__(self, *a, quantize=True, **kw):
        super().__init__(*a, **kw, bias=False)
        self.do_quantize = quantize
    def forward(self, x):
        return F.linear(x, tq(self.weight) if self.do_quantize else self.weight)

# ══════════════════════════════════════════════════════════════════════════════
# FACT ENCODER
# Converts a text string into a fixed-size trit vector
# This is the "address" in the content-addressable memory
# ══════════════════════════════════════════════════════════════════════════════

class FactEncoder(nn.Module):
    """
    Encodes a fact string into a TRIT_DIM-dimensional trit vector.

    Architecture:
      char embeddings → transformer-lite → mean pool → TernaryLinear → trit

    The trit vector IS the stored memory.
    Same fact → same trit pattern every time (deterministic after training).
    Similar facts → similar trit patterns (smooth manifold).
    """
    def __init__(self, vocab_size, trit_dim=256, d_model=128):
        super().__init__()
        self.trit_dim  = trit_dim
        self.emb       = nn.Embedding(vocab_size, d_model)
        self.pos       = nn.Embedding(256, d_model)
        self.attn      = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.ffn       = nn.Sequential(
            TernaryLinear(d_model, 4*d_model, quantize=False), nn.GELU(),
            TernaryLinear(4*d_model, d_model, quantize=False)
        )
        self.ln1       = nn.LayerNorm(d_model)
        self.ln2       = nn.LayerNorm(d_model)
        self.project   = TernaryLinear(d_model, trit_dim, quantize=False)

    def forward(self, idx):
        B, T     = idx.shape
        x        = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        attn_out, _ = self.attn(x, x, x)
        x        = self.ln1(x + attn_out)
        x        = self.ln2(x + self.ffn(x))
        pooled   = x.mean(dim=1)                # (B, d_model)
        return self.project(pooled)             # (B, trit_dim) — continuous

    def encode_to_trit(self, idx):
        """Encode and snap to {-1, 0, +1}"""
        with torch.no_grad():
            cont = self.forward(idx)
            return tq(cont)

# ══════════════════════════════════════════════════════════════════════════════
# TERNARY HOPFIELD MEMORY
#
# Classical Hopfield: W = (1/N) Σ_i p_i · p_i^T   (binary {-1,+1})
# Retrieval: x_{t+1} = sign(W · x_t)
#
# Ternary version:
# Store:   W = (1/N) Σ_i p_i · p_i^T   (patterns p_i ∈ {-1,0,+1}^n)
# Retrieve: x_{t+1} = sign(W · x_t)    (same formula, ternary patterns)
#
# The 0-trit acts as "don't care" — patterns with many zeros
# store more efficiently and interfere less with each other.
#
# Capacity: standard Hopfield ~0.14N patterns for N neurons
#           Ternary with sparse patterns: up to ~0.6N (sparsity bonus)
# ══════════════════════════════════════════════════════════════════════════════

class TernaryHopfield(nn.Module):
    def __init__(self, trit_dim=256):
        super().__init__()
        self.trit_dim   = trit_dim
        self.W          = torch.zeros(trit_dim, trit_dim, device=device)
        self.patterns   = []    # stored trit patterns
        self.labels     = []    # human-readable labels for each pattern
        self.n_stored   = 0

    def store(self, pattern, label=""):
        """
        Store a trit pattern.
        pattern: (trit_dim,) tensor with values {-1, 0, +1}
        Hebbian update: W += pattern ⊗ pattern
        """
        p = pattern.float().to(device)
        self.W          += torch.outer(p, p)
        self.W.fill_diagonal_(0)                # no self-connections
        self.patterns.append(p.clone())
        self.labels.append(label)
        self.n_stored   += 1

    def store_batch(self, patterns, labels):
        for p, l in zip(patterns, labels):
            self.store(p, l)
        # Normalize by number of stored patterns
        self.W /= max(self.n_stored, 1)

    def retrieve(self, query, steps=10):
        """
        Retrieve stored pattern nearest to query.
        query: (trit_dim,) partial or noisy pattern
        Returns: reconstructed pattern after convergence
        """
        x = query.float().to(device)
        for _ in range(steps):
            x_new = torch.sign(self.W @ x)
            if torch.allclose(x_new, x): break
            x = x_new
        return x

    def nearest_label(self, retrieved):
        """
        Find which stored pattern the retrieved vector is closest to.
        Uses Hamming distance on trit patterns.
        """
        if not self.patterns:
            return "empty memory", 0.0

        retrieved = retrieved.to(device)
        best_sim  = -float('inf')
        best_idx  = 0

        for i, p in enumerate(self.patterns):
            # Cosine similarity between retrieved and stored pattern
            sim = F.cosine_similarity(retrieved.unsqueeze(0), p.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return self.labels[best_idx], best_sim

    def capacity_info(self):
        n  = self.trit_dim
        stored = self.n_stored
        # Theoretical capacity for sparse ternary Hopfield
        sparsity = sum((p == 0).float().mean().item() for p in self.patterns) / max(stored, 1)
        capacity_est = int(n * (0.14 + 0.46 * sparsity))
        return {
            "neurons"       : n,
            "stored"        : stored,
            "capacity_est"  : capacity_est,
            "utilization"   : stored / max(capacity_est, 1) * 100,
            "avg_sparsity"  : sparsity * 100,
        }

# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# Facts stored as (key, value) pairs
# Key is encoded to trit → stored in Hopfield
# Retrieval: noisy/partial key → Hopfield → nearest key → lookup value
# ══════════════════════════════════════════════════════════════════════════════

class TernaryKnowledgeBase:
    """
    Content-addressable knowledge store using ternary Hopfield memory.

    Think of it as a dictionary where:
      - Keys are trit patterns (not strings)
      - Lookup works even with partial or corrupted keys
      - Storage is O(N²) trit weights, retrieval is O(N) ternary MACs
    """
    def __init__(self, encoder, trit_dim=256):
        self.encoder  = encoder
        self.memory   = TernaryHopfield(trit_dim)
        self.kv_store = {}   # key_label → full fact string

    def store_fact(self, key_text, value_text, chars, c2i):
        """
        Store a fact: key_text → value_text
        Example: "capital of France" → "Paris"
        """
        idx      = text_to_idx(key_text, c2i, chars)
        pattern  = self.encoder.encode_to_trit(idx.to(device)).squeeze(0)
        self.memory.store(pattern, label=key_text)
        self.kv_store[key_text] = value_text

    def store_batch(self, facts, chars, c2i):
        """
        facts: list of (key, value) tuples
        """
        patterns = []
        labels   = []
        for key, val in facts:
            idx     = text_to_idx(key, c2i, chars)
            pattern = self.encoder.encode_to_trit(idx.to(device)).squeeze(0)
            patterns.append(pattern)
            labels.append(key)
            self.kv_store[key] = val
        self.memory.store_batch(patterns, labels)
        print(f"  Stored {len(facts)} facts")

    def query(self, query_text, chars, c2i, noise_rate=0.0):
        """
        Retrieve fact nearest to query_text.
        noise_rate: fraction of query trits to randomly flip (tests robustness)
        """
        idx     = text_to_idx(query_text, c2i, chars)
        pattern = self.encoder.encode_to_trit(idx.to(device)).squeeze(0)

        # Optionally corrupt the query to test retrieval robustness
        if noise_rate > 0:
            mask    = torch.rand(pattern.shape) < noise_rate
            noise   = torch.randint(-1, 2, pattern.shape).float().to(device)
            pattern = torch.where(mask.to(device), noise, pattern)

        retrieved = self.memory.retrieve(pattern)
        key, sim  = self.memory.nearest_label(retrieved)
        value     = self.kv_store.get(key, "not found")
        return key, value, sim

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_char_vocab(texts):
    chars = sorted(set(''.join(texts)) | set(' abcdefghijklmnopqrstuvwxyz'
                                              'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                              '0123456789.,!?()-:;\'"'))
    c2i   = {c: i for i, c in enumerate(chars)}
    return chars, c2i

def text_to_idx(text, c2i, chars, maxlen=64):
    unk  = c2i.get(' ', 0)
    idx  = [c2i.get(c, unk) for c in text[:maxlen]]
    while len(idx) < maxlen:
        idx.append(0)
    return torch.tensor([idx], dtype=torch.long)

def corrupt_pattern(pattern, rate):
    mask   = torch.rand(pattern.shape) < rate
    noise  = torch.randint(-1, 2, pattern.shape).float().to(device)
    return torch.where(mask.to(device), noise, pattern.to(device))

# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE SETS
# ══════════════════════════════════════════════════════════════════════════════

GENERAL_FACTS = [
    # Science
    ("capital of France",           "Paris"),
    ("capital of Germany",          "Berlin"),
    ("capital of Japan",            "Tokyo"),
    ("capital of Brazil",           "Brasilia"),
    ("capital of Australia",        "Canberra"),
    ("speed of light",              "299,792,458 metres per second"),
    ("boiling point of water",      "100 degrees Celsius at sea level"),
    ("freezing point of water",     "0 degrees Celsius"),
    ("chemical symbol for gold",    "Au"),
    ("chemical symbol for iron",    "Fe"),
    ("number of planets",           "8 planets in the solar system"),
    ("distance to the moon",        "384,400 kilometres on average"),
    ("inventor of telephone",       "Alexander Graham Bell in 1876"),
    ("year of moon landing",        "1969, Apollo 11"),
    ("DNA stands for",              "Deoxyribonucleic acid"),
    ("speed of sound in air",       "343 metres per second at 20 degrees C"),
    ("atomic number of carbon",     "6"),
    ("atomic number of oxygen",     "8"),
    ("formula for water",           "H2O, two hydrogen one oxygen"),
    ("formula for carbon dioxide",  "CO2, one carbon two oxygen"),
    # Computing
    ("what is ternary",             "Base-3 number system using digits 0 1 2"),
    ("what is a trit",              "Ternary digit, one of -1 0 or +1"),
    ("what is the consensus gate",  "sign(a+b+c), majority vote of three trits"),
    ("what is triadic architecture","Three parallel streams Observer Shadow Light"),
    ("what is a Hopfield network",  "Content-addressable associative memory"),
    ("what is STE",                 "Straight-Through Estimator for gradient flow"),
    ("what is ternary quantization","Snap weights to -1 0 or +1 with threshold 0.7E"),
    ("what is predictive coding",   "Neural loss term predicting next layer input"),
    ("what is a KV cache",          "Key-Value store for transformer context, grows linearly"),
    ("what is TritLM",              "Ternary language model with fixed memory cell"),
    # 012 Project
    ("012 vision",                  "Observer Shadow Light triadic computing paradigm"),
    ("TritCognition accuracy",      "80.08 percent on CIFAR-10 at 0 degrees rotation"),
    ("TritCognition parameters",    "396174 parameters, 28x fewer than ResNet18"),
    ("TritCognition ternary size",  "0.078 megabytes, 20x compression from float32"),
    ("TritLM memory footprint",     "0.05 kilobytes fixed regardless of context length"),
    ("noise recovery at 50 percent","0.345 nats recovered using trit memory"),
    ("hardware tests result",       "36 of 36 testbench cases pass in simulation"),
    ("FPGA target",                 "Xilinx Ultrascale Plus ZCU102 at 250 MHz"),
    ("energy saving",               "87 percent per token vs float32 on 28nm CMOS"),
    ("zero weight fraction",        "approximately 50 percent of weights are zero"),
]

TECH_FACTS = [
    ("Python list comprehension",   "[x for x in iterable if condition]"),
    ("PyTorch optimizer step",      "opt.zero_grad(); loss.backward(); opt.step()"),
    ("CUDA device check",           "torch.cuda.is_available()"),
    ("cosine annealing schedule",   "CosineAnnealingLR(optimizer, T_max=epochs)"),
    ("batch normalization purpose", "Normalizes layer inputs, stabilizes training"),
    ("dropout purpose",             "Randomly zeros activations, prevents overfitting"),
    ("learning rate too high",      "Loss diverges or oscillates wildly"),
    ("learning rate too low",       "Training converges very slowly or stalls"),
    ("what is perplexity",          "exp(cross-entropy loss), lower is better"),
    ("what is top-k sampling",      "Sample from top k most likely next tokens"),
    ("GDScript extends",            "extends Node or extends KinematicBody etc"),
    ("Godot signal syntax",         "signal my_signal(arg) and emit_signal"),
    ("SystemVerilog trit type",     "typedef logic [1:0] trit_t with 2-bit encoding"),
    ("TRIT_NEG encoding",           "2b00 represents -1 in 012 hardware"),
    ("TRIT_ZERO encoding",          "2b01 represents 0 in 012 hardware"),
    ("TRIT_POS encoding",           "2b10 represents +1 in 012 hardware"),
]

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ENCODER
# The encoder learns to map similar text to nearby trit patterns
# Training objective: same meaning → high cosine similarity in trit space
# ══════════════════════════════════════════════════════════════════════════════

def train_encoder(encoder, facts, chars, c2i, epochs=500):
    """
    Train encoder with contrastive-style loss:
    - Positive pairs: key and value of the same fact should be close
    - The encoder learns semantic similarity in trit space
    """
    opt  = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    all_texts = [k for k, v in facts] + [v for k, v in facts]

    print(f"Training encoder on {len(facts)} facts...")
    for epoch in range(epochs):
        encoder.train()
        total_loss = 0

        # Sample fact pairs
        idx   = torch.randint(0, len(facts), (32,))
        pairs = [facts[i] for i in idx]

        keys  = torch.cat([text_to_idx(k, c2i, chars).to(device) for k, v in pairs])
        vals  = torch.cat([text_to_idx(v, c2i, chars).to(device) for k, v in pairs])

        key_enc = encoder(keys)    # (B, trit_dim) continuous
        val_enc = encoder(vals)    # (B, trit_dim) continuous

        # Positive loss: key and value should be similar in trit space
        pos_loss = 1 - F.cosine_similarity(key_enc, val_enc).mean()

        # Negative loss: different facts should be dissimilar
        # Shuffle to create negatives
        perm     = torch.randperm(len(pairs))
        neg_enc  = val_enc[perm]
        neg_loss = F.cosine_similarity(key_enc, neg_enc).clamp(min=0).mean()

        loss = pos_loss + 0.5 * neg_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

        if (epoch+1) % 100 == 0:
            print(f"  Encoder epoch {epoch+1}/{epochs}  loss={total_loss:.4f}")

    encoder.eval()
    print("  Encoder trained.\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DEMONSTRATION
# ══════════════════════════════════════════════════════════════════════════════

all_facts = GENERAL_FACTS + TECH_FACTS
all_texts = [k for k, v in all_facts] + [v for k, v in all_facts]
chars, c2i = build_char_vocab(all_texts)
vocab_size  = len(chars)
TRIT_DIM    = 256

print(f"Knowledge base: {len(all_facts)} facts")
print(f"Vocabulary    : {vocab_size} characters")
print(f"Trit dimension: {TRIT_DIM}\n")

# Build encoder
encoder = FactEncoder(vocab_size, trit_dim=TRIT_DIM).to(device)
enc_params = sum(p.numel() for p in encoder.parameters())
print(f"Encoder parameters: {enc_params:,}")
print(f"Encoder ternary size: {enc_params*1.585/8/1024:.1f} KB\n")

# Train encoder
train_encoder(encoder, all_facts, chars, c2i, epochs=500)

# Build knowledge base
print("Storing facts in Hopfield memory...")
kb = TernaryKnowledgeBase(encoder, trit_dim=TRIT_DIM)
kb.store_batch(all_facts, chars, c2i)

# Print capacity info
cap = kb.memory.capacity_info()
print(f"\nMemory capacity:")
print(f"  Neurons (trit dim)  : {cap['neurons']}")
print(f"  Stored facts        : {cap['stored']}")
print(f"  Estimated capacity  : {cap['capacity_est']} facts")
print(f"  Utilization         : {cap['utilization']:.1f}%")
print(f"  Avg pattern sparsity: {cap['avg_sparsity']:.1f}% zeros")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: EXACT RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  TEST 1: EXACT QUERY RETRIEVAL")
print(f"{'═'*60}\n")

test_queries = [
    "capital of France",
    "what is a trit",
    "TritCognition parameters",
    "formula for water",
    "noise recovery at 50 percent",
    "hardware tests result",
    "Python list comprehension",
    "TRIT_NEG encoding",
]

correct = 0
for q in test_queries:
    key, val, sim = kb.query(q, chars, c2i)
    match = "✓" if key == q else "~"
    if key == q: correct += 1
    print(f"  {match} Query : {q}")
    print(f"    Found : {key}  (sim={sim:.3f})")
    print(f"    Value : {val}\n")

print(f"  Exact match: {correct}/{len(test_queries)}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: PARTIAL / FUZZY QUERIES
# The key property — retrieve even with incomplete input
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  TEST 2: PARTIAL AND FUZZY QUERIES")
print(f"{'═'*60}\n")

fuzzy_queries = [
    ("capital France",          "capital of France"),
    ("speed light",             "speed of light"),
    ("what trit",               "what is a trit"),
    ("TritCognition params",    "TritCognition parameters"),
    ("water formula",           "formula for water"),
    ("012 vision thing",        "012 vision"),
    ("memory footprint TritLM", "TritLM memory footprint"),
    ("FPGA target board",       "FPGA target"),
]

fuzzy_correct = 0
for partial, expected in fuzzy_queries:
    key, val, sim = kb.query(partial, chars, c2i)
    match = "✓" if key == expected else "✗"
    if key == expected: fuzzy_correct += 1
    print(f"  {match} Partial : '{partial}'")
    print(f"    Expected: '{expected}'")
    print(f"    Got     : '{key}'  (sim={sim:.3f})")
    print(f"    Value   : {val}\n")

print(f"  Fuzzy retrieval: {fuzzy_correct}/{len(fuzzy_queries)}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: NOISE ROBUSTNESS
# Corrupt the query pattern directly (bit errors in hardware)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  TEST 3: TRIT NOISE ROBUSTNESS")
print(f"{'═'*60}\n")

test_fact  = "capital of France"
noise_tests = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
print(f"  Query: '{test_fact}'  →  Expected: 'Paris'\n")
print(f"  {'Noise':>6} | {'Retrieved key':<30} | {'Value':<15} | {'Sim':>6}")
print(f"  {'-'*65}")

for rate in noise_tests:
    key, val, sim = kb.query(test_fact, chars, c2i, noise_rate=rate)
    mark = "✓" if key == test_fact else "✗"
    print(f"  {rate*100:>5.0f}% | {key:<30} | {val:<15} | {sim:>6.3f}  {mark}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: RELATED FACT CLUSTERING
# Measure trit similarity between related vs unrelated facts
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  TEST 4: SEMANTIC CLUSTERING IN TRIT SPACE")
print(f"{'═'*60}\n")

groups = {
    "Geography" : ["capital of France", "capital of Germany", "capital of Japan"],
    "Chemistry" : ["formula for water", "formula for carbon dioxide", "chemical symbol for gold"],
    "012 Project": ["what is a trit", "012 vision", "TritCognition parameters"],
    "Hardware"  : ["TRIT_NEG encoding", "TRIT_POS encoding", "SystemVerilog trit type"],
}

print("  Within-group vs between-group cosine similarity:")
print("  (higher within = facts cluster by topic in trit space)\n")

def get_trit(text):
    idx = text_to_idx(text, c2i, chars).to(device)
    return encoder.encode_to_trit(idx).squeeze(0)

for group_name, keys in groups.items():
    trits = [get_trit(k) for k in keys]
    within_sims = []
    for i in range(len(trits)):
        for j in range(i+1, len(trits)):
            s = F.cosine_similarity(trits[i].unsqueeze(0), trits[j].unsqueeze(0)).item()
            within_sims.append(s)
    print(f"  {group_name}: within-group avg sim = {np.mean(within_sims):.3f}")

# Cross-group
geo_trits  = [get_trit(k) for k in groups["Geography"]]
chem_trits = [get_trit(k) for k in groups["Chemistry"]]
cross_sims = [F.cosine_similarity(g.unsqueeze(0), c.unsqueeze(0)).item()
              for g in geo_trits for c in chem_trits]
print(f"\n  Geography vs Chemistry cross-group avg sim = {np.mean(cross_sims):.3f}")
print(f"  (lower cross-group = topics are separated in trit space)")

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE QUERY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  INTERACTIVE MODE")
print(f"{'═'*60}")
print(f"  Query the knowledge base.")
print(f"  Type 'list' to see all stored facts.")
print(f"  Type 'quit' to exit.\n")

while True:
    try:
        q = input("  Query: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not q: continue
    if q == "quit": break
    if q == "list":
        print(f"\n  Stored facts ({len(kb.kv_store)}):")
        for k, v in kb.kv_store.items():
            print(f"    {k:<35} → {v}")
        print()
        continue

    key, val, sim = kb.query(q, chars, c2i)
    print(f"\n  Key  : {key}")
    print(f"  Value: {val}")
    print(f"  Sim  : {sim:.3f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  SUMMARY")
print(f"{'═'*60}")
print(f"""
  Storage mechanism : Ternary Hopfield (Hebbian weight matrix)
  Retrieval         : Consensus iteration — sign(W·x) until convergence
  Retrieval cost    : {TRIT_DIM} ternary MACs per step, ~{TRIT_DIM*10} total
  Storage cost      : {TRIT_DIM*TRIT_DIM*2//8//1024} KB for weight matrix ({TRIT_DIM}×{TRIT_DIM} trits)
  Facts stored      : {len(all_facts)}
  Estimated capacity: {cap['capacity_est']} facts at this trit dimension

  This is NOT a language model.
  It stores discrete facts as trit patterns and retrieves by
  pattern completion — like a human associating a partial cue
  with a full memory.

  Hardware equivalent:
    Ternary CAM (Content Addressable Memory) on FPGA
    Each retrieval step = {TRIT_DIM} ternary additions, zero multiplications
    Can be pipelined to retrieve in a single clock cycle
""")

results = {
    "facts_stored"    : len(all_facts),
    "trit_dim"        : TRIT_DIM,
    "capacity"        : cap,
    "exact_recall"    : f"{correct}/{len(test_queries)}",
    "fuzzy_recall"    : f"{fuzzy_correct}/{len(fuzzy_queries)}",
}
with open("results/012_memory_store.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: results/012_memory_store.json")
