"""
Option B — Quantization-Aware Training (QAT) for Ternary MiniLM

Fine-tunes all-MiniLM-L6-v2 from scratch WITH ternary weights baked in
from the start, using a Straight-Through Estimator (STE) so gradients
flow through the discrete quantization step. Compares against:
  - Float32 (the existing code-minilm model, baseline)
  - Ternary PTQ (Option A result: 50% accuracy, -45pp drop)
  - Ternary QAT (this script)

QAT gives the model a chance to learn weight distributions that *survive*
ternary quantization at inference time, rather than having float32 weights
snapped post-hoc.

Training data: the same pairs_*.json files code-minilm was trained on
(already downloaded, no streaming needed).

Usage:
  python trit_qat_ternary_test.py
"""
import json, time, copy, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

MODEL_NAME   = "models/code-minilm"   # load from local path, no HF download
PAIRS_DIR    = Path("models/code-minilm")
SAVE_PATH    = Path("models/code-minilm-ternary-qat")
FLOAT32_PATH = "models/code-minilm"

# Local codebases to mine for training pairs
LOCAL_SCAN_DIRS = [
    Path(__file__).parent,                                          # 012-ternary itself
    Path(r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1"),  # Godot game
    Path(r"C:\Users\gbran\OneDrive\Documents\tribe"),               # tribe project
]

EPOCHS       = 3
BATCH_SIZE   = 32
LR           = 2e-5
WARMUP_FRAC  = 0.1
MAX_LEN      = 128
QUANT_WARMUP_FRAC = 0.15   # first 15% of steps: float32 warmup, then switch on ternary
MAX_PAIRS    = 80_000      # cap to keep runtime reasonable

# ── Benchmark triplets (same 20 as Option A) ──────────────────────────────────

TRIPLETS = [
    ("take damage from enemy",
     "func take_damage(amount: float):\n\thealth -= amount\n\thealth = clamp(health, 0, max_health)\n\tif health <= 0: _die()",
     "func heal(amount: float):\n\thealth += amount\n\thealth = clamp(health, 0, max_health)"),
    ("spawn wave of enemies",
     "func start_wave():\n\tfor i in wave_size:\n\t\tvar e = enemy_scene.instantiate()\n\t\tspawn_points[i % spawn_points.size()].add_child(e)",
     "func spawn_single_enemy(pos: Vector3):\n\tvar e = enemy_scene.instantiate()\n\te.global_position = pos\n\tadd_child(e)"),
    ("enemy chases player",
     "func _chase(delta):\n\tvar next = nav_agent.get_next_path_position()\n\tvelocity = (next - global_position).normalized() * chase_speed\n\tmove_and_slide()",
     "func _patrol(delta):\n\tif position.distance_to(patrol_target) < 1.0:\n\t\tpatrol_index = (patrol_index + 1) % patrol_points.size()\n\tvelocity = (patrol_target - position).normalized() * patrol_speed"),
    ("compute training loss",
     "model.train()\noptimizer.zero_grad()\noutput = model(input_ids, labels=labels)\nloss = output.loss\nloss.backward()\noptimizer.step()",
     "model.eval()\nwith torch.no_grad():\n\toutput = model(input_ids, labels=labels)\n\tval_loss = output.loss.item()"),
    ("contrastive learning loss",
     "logits = torch.matmul(q, k.T) / temperature\nlabels = torch.arange(q.size(0)).to(device)\nloss = F.cross_entropy(logits, labels)",
     "logits = model(input_ids).logits\nshift_logits = logits[..., :-1, :]\nshift_labels = labels[..., 1:]\nloss = F.cross_entropy(shift_logits.reshape(-1, vocab), shift_labels.reshape(-1))"),
    ("merge lora into base model",
     "model = PeftModel.from_pretrained(base_model, lora_path)\nmodel = model.merge_and_unload()\nmodel.save_pretrained(output_path)",
     "bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')\nmodel = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config)"),
    ("save game state to disk",
     "func save_game():\n\tvar data = {wave=current_wave, health=player.health, gold=gold}\n\tvar f = FileAccess.open(SAVE_PATH, FileAccess.WRITE)\n\tf.store_string(JSON.stringify(data))",
     "func load_game():\n\tvar f = FileAccess.open(SAVE_PATH, FileAccess.READ)\n\tvar data = JSON.parse_string(f.get_as_text())\n\tcurrent_wave = data.wave\n\tgold = data.gold"),
    ("open websocket connection",
     "ws = await websockets.connect(uri)\nasync for msg in ws:\n\tdata = json.loads(msg)\n\tawait handle(data)",
     "conn = await asyncpg.connect(dsn)\nrows = await conn.fetch('SELECT * FROM events WHERE ts > $1', cutoff)"),
    ("jwt authentication middleware",
     "def jwt_required(f):\n\t@wraps(f)\n\tdef wrapper(*args, **kwargs):\n\t\ttoken = request.headers.get('Authorization', '').split()[-1]\n\t\tpayload = jwt.decode(token, SECRET, algorithms=['HS256'])\n\t\tg.user_id = payload['sub']\n\t\treturn f(*args, **kwargs)\n\treturn wrapper",
     "def rate_limit(f):\n\t@wraps(f)\n\tdef wrapper(*args, **kwargs):\n\t\tkey = f'rl:{request.remote_addr}'\n\t\tcount = redis_client.incr(key)\n\t\tif count == 1: redis_client.expire(key, 60)\n\t\tif count > LIMIT: abort(429)\n\t\treturn f(*args, **kwargs)\n\treturn wrapper"),
    ("binary search in sorted array",
     "def binary_search(arr, target):\n\tlo, hi = 0, len(arr)-1\n\twhile lo <= hi:\n\t\tmid = (lo+hi)//2\n\t\tif arr[mid] == target: return mid\n\t\telif arr[mid] < target: lo = mid+1\n\t\telse: hi = mid-1\n\treturn -1",
     "def linear_search(arr, target):\n\tfor i, v in enumerate(arr):\n\t\tif v == target: return i\n\treturn -1"),
    ("sort list descending by score",
     "ranked = sorted(results, key=lambda x: x['score'], reverse=True)",
     "grouped = defaultdict(list)\nfor r in results:\n\tgrouped[r['category']].append(r)"),
    ("gradient clipping during training",
     "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\noptimizer.step()",
     "scheduler.step()\noptimizer.zero_grad()"),
    ("deploy docker container",
     "docker build -t myapp:latest .\ndocker run -d -p 8080:8080 --env-file .env myapp:latest",
     "kubectl apply -f deployment.yaml\nkubectl rollout status deployment/myapp"),
    ("parse json from http response",
     "resp = requests.get(url, headers=headers)\nresp.raise_for_status()\ndata = resp.json()",
     "with open('config.json') as f:\n\tdata = json.load(f)"),
    ("database connection pool",
     "pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)\nasync with pool.acquire() as conn:\n\trows = await conn.fetch(query)",
     "engine = create_engine(url, pool_size=5, max_overflow=10)\nwith engine.connect() as conn:\n\tresult = conn.execute(text(query))"),
    ("encrypt password with bcrypt",
     "hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))\nbcrypt.checkpw(password.encode(), hashed)",
     "signature = hmac.new(SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()"),
    ("resize image to thumbnail",
     "from PIL import Image\nimg = Image.open(path)\nimg.thumbnail((128, 128), Image.LANCZOS)\nimg.save(thumb_path)",
     "from PIL import Image\nimg = Image.open(path)\nimg = img.rotate(90, expand=True)\nimg.save(rotated_path)"),
    ("retry with exponential backoff",
     "for attempt in range(max_retries):\n\ttry:\n\t\treturn call()\n\texcept Exception:\n\t\tif attempt == max_retries-1: raise\n\t\ttime.sleep(2**attempt)",
     "with timeout(seconds=30):\n\tresult = call()"),
    ("pagination query with offset",
     "SELECT * FROM items WHERE user_id=$1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
     "SELECT COUNT(*) FROM items WHERE user_id=$1 AND status='active'"),
    ("calculate moving average",
     "def moving_avg(values, window):\n\treturn [sum(values[i:i+window])/window for i in range(len(values)-window+1)]",
     "def cumulative_sum(values):\n\ttotal = 0\n\treturn [total := total + v for v in values]"),
]

# ── STE Ternary Quantization ───────────────────────────────────────────────────

class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        t = 0.7 * x.abs().mean()
        ctx.save_for_backward(x)
        return torch.where(x > t,  torch.ones_like(x),
               torch.where(x < -t, -torch.ones_like(x),
                           torch.zeros_like(x)))
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).float()

ternary_ste = TernarySTE.apply

# ── Ternary-aware Linear wrapper ──────────────────────────────────────────────

class TernaryLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize = False

    def forward(self, x):
        w = ternary_ste(self.weight) if self.quantize else self.weight
        return F.linear(x, w, self.bias)


def replace_linears(model):
    """Recursively replace all nn.Linear with TernaryLinear in-place."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and not isinstance(child, TernaryLinear):
            dev = child.weight.device
            tl = TernaryLinear(child.in_features, child.out_features,
                               bias=child.bias is not None).to(dev)
            tl.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                tl.bias.data.copy_(child.bias.data)
            setattr(model, name, tl)
        else:
            replace_linears(child)

def set_quantize(model, active: bool):
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            m.quantize = active

def count_ternary_layers(model):
    return sum(1 for m in model.modules() if isinstance(m, TernaryLinear))

# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

def extract_pairs_from_code(code: str) -> list:
    """Extract (anchor, positive) pairs from a code file using function-body heuristic."""
    import re
    pairs = []
    lines = code.splitlines()
    func_pat = re.compile(
        r'(?:def|func|fn|function|fun|sub|proc|public|private|protected|static)?\s*'
        r'(?:void|int|float|str|bool|auto|var)?\s*(\w+)\s*\('
    )
    for i, line in enumerate(lines):
        m = func_pat.search(line)
        if not m:
            continue
        fname = m.group(1)
        if len(fname) < 3 or fname in ('if', 'for', 'while', 'return', 'class', 'import',
                                        'from', 'with', 'elif', 'else', 'try', 'except'):
            continue
        body = "\n".join(lines[i:i+12]).strip()
        if len(body) < 40:
            continue
        anchor = fname.replace("_", " ").strip()
        if anchor:
            pairs.append((anchor, body))
    return pairs

def load_pairs():
    """Load pairs from local codebases by scanning source files."""
    pairs = []
    extensions = {".py", ".gd", ".js", ".ts", ".cs", ".go", ".rs", ".java", ".cpp", ".c", ".lua"}
    for scan_dir in LOCAL_SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for ext in extensions:
            for fp in scan_dir.rglob(f"*{ext}"):
                try:
                    code = fp.read_text(encoding="utf-8", errors="ignore")
                    pairs.extend(extract_pairs_from_code(code))
                except Exception:
                    pass
    print(f"  Extracted {len(pairs):,} pairs from local codebases")
    random.shuffle(pairs)
    if len(pairs) > MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]
    return pairs

def mean_pool(outputs, attention_mask):
    token_embs = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

# ── Training ──────────────────────────────────────────────────────────────────

def train_qat(device):
    print("Loading pairs...")
    pairs = load_pairs()
    print(f"  {len(pairs):,} pairs loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # Replace all Linear layers with TernaryLinear
    replace_linears(base)
    n_ternary = count_ternary_layers(base)
    print(f"  Replaced {n_ternary} Linear layers with TernaryLinear")
    set_quantize(base, False)  # float32 warmup first

    optimizer = torch.optim.AdamW(base.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = (len(pairs) // BATCH_SIZE + 1) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    quant_step   = int(total_steps * QUANT_WARMUP_FRAC)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    dataset = PairDataset(pairs)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: b)

    print(f"\nTraining {EPOCHS} epochs, {total_steps} steps total")
    print(f"  Float32 warmup: first {quant_step} steps, then ternary QAT\n")

    step = 0
    quant_active = False
    t0 = time.time()

    for epoch in range(EPOCHS):
        base.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if step == quant_step:
                set_quantize(base, True)
                quant_active = True
                print(f"  [step {step}] switching to TERNARY weights")

            anchors   = [p[0] for p in batch]
            positives = [p[1] for p in batch]

            # tokenize
            def tok(texts):
                return tokenizer(texts, padding=True, truncation=True,
                                 max_length=MAX_LEN, return_tensors="pt").to(device)

            anc_enc = tok(anchors)
            pos_enc = tok(positives)

            anc_out = base(**anc_enc)
            pos_out = base(**pos_enc)

            anc_emb = mean_pool(anc_out, anc_enc["attention_mask"])
            pos_emb = mean_pool(pos_out, pos_enc["attention_mask"])

            anc_emb = F.normalize(anc_emb, dim=-1)
            pos_emb = F.normalize(pos_emb, dim=-1)

            # InfoNCE / NT-Xent contrastive loss (in-batch negatives)
            sim = torch.matmul(anc_emb, pos_emb.T) / 0.05
            labels = torch.arange(len(anchors), device=device)
            loss = F.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1
            step       += 1

        mode_tag = "TERNARY" if quant_active else "float32"
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/n_batches:.4f}  "
              f"[{mode_tag}]  ({time.time()-t0:.0f}s)")

    return base, tokenizer

# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, device, triplets, quantize=True):
    model.eval()
    set_quantize(model, quantize)

    def encode(texts):
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(device)
        out = model(**enc)
        emb = mean_pool(out, enc["attention_mask"])
        return F.normalize(emb, dim=-1).cpu().numpy()

    queries   = [t[0] for t in triplets]
    positives = [t[1] for t in triplets]
    negatives = [t[2] for t in triplets]

    q_emb = encode(queries)
    p_emb = encode(positives)
    n_emb = encode(negatives)

    pos_scores = (q_emb * p_emb).sum(axis=1)
    neg_scores = (q_emb * n_emb).sum(axis=1)
    return float((pos_scores > neg_scores).mean())

# ── Sparsity ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def ternary_sparsity(model):
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            w = ternary_ste(m.weight.data)
            zeros += (w == 0).sum().item()
            total += w.numel()
    return zeros / total if total > 0 else 0.0

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Float32 baseline (existing code-minilm)
    print("Loading float32 baseline (code-minilm)...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(FLOAT32_PATH, device=device)
    st_model.eval()
    all_texts = [t for tri in TRIPLETS for t in tri]
    embs = st_model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
    n = len(TRIPLETS)
    pos_s = (embs[:n] * embs[n:2*n]).sum(axis=1)
    neg_s = (embs[:n] * embs[2*n:]).sum(axis=1)
    base_acc = float((pos_s > neg_s).mean())
    print(f"  Float32 baseline: {base_acc*100:.1f}%\n")

    # QAT training
    print("=" * 60)
    print("  Training with Ternary QAT (STE)...")
    print("=" * 60)
    qat_model, tokenizer = train_qat(device)

    # Evaluate QAT with ternary weights active
    print("\nEvaluating QAT model (ternary weights)...")
    qat_acc = evaluate(qat_model, tokenizer, device, TRIPLETS, quantize=True)

    # Also evaluate QAT model with float32 weights (upper bound — shows what QAT learned)
    print("Evaluating QAT model (float32 weights, no quantization)...")
    qat_f32_acc = evaluate(qat_model, tokenizer, device, TRIPLETS, quantize=False)

    sp = ternary_sparsity(qat_model)

    bits_per_weight = (1 - sp) * 1.585 + 0.2
    compression = 32.0 / bits_per_weight

    print("\n" + "=" * 70)
    print("  OPTION B — Quantization-Aware Training (QAT) Results")
    print("=" * 70)
    print(f"  {'Mode':<30} {'Accuracy':>10}  {'vs Float32':>12}")
    print(f"  {'-'*55}")
    print(f"  {'Float32 (code-minilm)':<30} {base_acc*100:>9.1f}%  {'—':>12}")
    print(f"  {'Ternary PTQ (Option A)':<30} {'50.0':>9}%  {'-45.0pp':>12}")
    print(f"  {'Ternary QAT (this) — ternary':<30} {qat_acc*100:>9.1f}%  {(qat_acc-base_acc)*100:>+11.1f}pp")
    print(f"  {'Ternary QAT (this) — float32':<30} {qat_f32_acc*100:>9.1f}%  {(qat_f32_acc-base_acc)*100:>+11.1f}pp")
    print()
    print(f"  QAT ternary sparsity:    {sp*100:.1f}%")
    print(f"  Theoretical compression: {compression:.1f}x vs float32")
    print()
    recovery = qat_acc - 0.50
    ptq_gap  = base_acc - 0.50
    if recovery > 0:
        print(f"  QAT recovered {recovery/ptq_gap*100:.0f}% of the accuracy lost by PTQ")
    print("=" * 70)

    # Save if it recovered meaningfully
    if qat_acc > 0.70:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(SAVE_PATH))
        torch.save(qat_model.state_dict(), str(SAVE_PATH / "pytorch_model.bin"))
        print(f"\n  Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
