"""
Option C — Mixed Precision PTQ: INT8 attention, Ternary FFN

The key insight from Option A: PTQ ternary collapses accuracy by 45pp.
The question: is all of that loss spread evenly, or concentrated in specific
layer types? Transformers have two kinds of Linear layers:
  - Attention (Q, K, V, output): compute pairwise token similarities
  - FFN (intermediate + output): per-token nonlinear projection (2x larger)

Hypothesis: attention weights are precision-sensitive (similarity geometry
breaks under ternary), FFN weights are more redundant. Mixed precision:
  - Attention layers → INT8 (4x compression, near-zero accuracy cost)
  - FFN layers → Ternary (29.7x compression, but applied to ~2/3 of params)
  - Net: bigger compression than pure INT8, smaller accuracy loss than pure ternary

Also tests ablation configs to precisely identify which layer type is responsible
for the accuracy collapse.

Usage:
  python trit_mixed_precision_test.py
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_PATH = "models/code-minilm"
BATCH_SIZE = 64

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

# ── Quantization functions ─────────────────────────────────────────────────────

def ternary_ptq(w: torch.Tensor) -> torch.Tensor:
    t = 0.7 * w.abs().mean()
    return torch.where(w > t,  torch.ones_like(w),
           torch.where(w < -t, -torch.ones_like(w), torch.zeros_like(w)))

def int8_ptq(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().max() / 127.0 + 1e-8
    return torch.round(w / scale).clamp(-127, 127) * scale

# ── Layer classification ───────────────────────────────────────────────────────

def classify_layer(name: str) -> str:
    """Classify a Linear layer by its role in the transformer."""
    n = name.lower()
    if any(k in n for k in ("query", "key", "value", "attention.output", "self.output")):
        return "attention"
    if any(k in n for k in ("intermediate", "output.dense", "pooler")):
        return "ffn"
    return "other"

# ── Apply mixed PTQ ───────────────────────────────────────────────────────────

def apply_mixed_ptq(model, attn_mode: str, ffn_mode: str):
    """
    Apply per-layer-type quantization.
    mode: 'float32' | 'int8' | 'ternary'
    """
    stats = {"attention": {"n": 0, "params": 0, "zeros": 0},
             "ffn":       {"n": 0, "params": 0, "zeros": 0},
             "other":     {"n": 0, "params": 0, "zeros": 0}}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        role = classify_layer(name)
        mode = attn_mode if role == "attention" else (ffn_mode if role == "ffn" else "float32")

        with torch.no_grad():
            w = module.weight.data
            if mode == "ternary":
                q = ternary_ptq(w)
                stats[role]["zeros"] += (q == 0).sum().item()
            elif mode == "int8":
                q = int8_ptq(w)
            else:
                q = w.clone()
            module.weight.data.copy_(q)

        stats[role]["n"]      += 1
        stats[role]["params"] += w.numel()

    return stats

def effective_compression(stats, attn_mode, ffn_mode):
    """Compute overall effective compression ratio."""
    total_params = sum(s["params"] for s in stats.values())
    total_bits = 0
    for role, s in stats.items():
        mode = attn_mode if role == "attention" else (ffn_mode if role == "ffn" else "float32")
        if mode == "ternary":
            sp = s["zeros"] / s["params"] if s["params"] else 0
            bpw = (1 - sp) * 1.585 + 0.2
        elif mode == "int8":
            bpw = 8.0
        else:
            bpw = 32.0
        total_bits += s["params"] * bpw
    return (total_params * 32.0) / total_bits

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, triplets):
    queries   = [t[0] for t in triplets]
    positives = [t[1] for t in triplets]
    negatives = [t[2] for t in triplets]
    all_texts = queries + positives + negatives
    embs = model.encode(all_texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                        show_progress_bar=False)
    n = len(triplets)
    pos_s = (embs[:n] * embs[n:2*n]).sum(axis=1)
    neg_s = (embs[:n] * embs[2*n:]).sum(axis=1)
    return float((pos_s > neg_s).mean())

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Benchmark: {len(TRIPLETS)} triplets\n")

    print("Loading code-minilm (float32 baseline)...")
    base_model = SentenceTransformer(MODEL_PATH, device=device)
    base_model.eval()
    base_acc = evaluate(base_model, TRIPLETS)
    print(f"  Float32 baseline: {base_acc*100:.1f}%\n")

    # Configurations to test:
    # (label, attn_mode, ffn_mode)
    configs = [
        ("Float32 (baseline)",          "float32", "float32"),
        ("INT8 all",                    "int8",    "int8"),
        ("Ternary all (PTQ, Option A)", "ternary", "ternary"),
        ("Ternary attn only",           "ternary", "float32"),
        ("Ternary FFN only",            "float32", "ternary"),
        ("Mixed: INT8 attn + Ternary FFN", "int8", "ternary"),
    ]

    results = {}
    for label, attn_mode, ffn_mode in configs:
        if label == "Float32 (baseline)":
            results[label] = (base_acc, 1.0)
            continue

        print(f"Testing: {label}...")
        m = copy.deepcopy(base_model)
        m.eval()
        stats = apply_mixed_ptq(m, attn_mode, ffn_mode)
        ratio = effective_compression(stats, attn_mode, ffn_mode)
        acc = evaluate(m, TRIPLETS)
        results[label] = (acc, ratio)

        attn_s = stats["attention"]
        ffn_s  = stats["ffn"]
        print(f"  Accuracy: {acc*100:.1f}%  drop: {(base_acc-acc)*100:+.1f}pp  compression: {ratio:.1f}x")
        if attn_mode == "ternary":
            sp = attn_s["zeros"] / attn_s["params"] * 100
            print(f"  Attn sparsity: {sp:.1f}%  ({attn_s['n']} layers, {attn_s['params']:,} params)")
        if ffn_mode == "ternary":
            sp = ffn_s["zeros"] / ffn_s["params"] * 100
            print(f"  FFN  sparsity: {sp:.1f}%  ({ffn_s['n']} layers, {ffn_s['params']:,} params)")
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 72)
    print("  OPTION C — Mixed Precision PTQ Results")
    print("=" * 72)
    print(f"  {'Config':<40} {'Accuracy':>10}  {'Drop':>8}  {'Compression':>12}")
    print(f"  {'-'*72}")
    for label, attn_mode, ffn_mode in configs:
        acc, ratio = results[label]
        drop = base_acc - acc
        print(f"  {label:<40} {acc*100:>9.1f}%  {drop*100:>+7.1f}pp  {ratio:>10.1f}x")

    print()
    # Key findings
    _, attn_ratio = results["Ternary attn only"]
    attn_acc, _   = results["Ternary attn only"]
    ffn_acc, _    = results["Ternary FFN only"]
    mix_acc, mix_ratio = results["Mixed: INT8 attn + Ternary FFN"]

    print("  Key findings:")
    print(f"  Ternary attn only:    {(base_acc-attn_acc)*100:+.1f}pp drop — isolates attention sensitivity")
    print(f"  Ternary FFN only:     {(base_acc-ffn_acc)*100:+.1f}pp drop — isolates FFN sensitivity")
    if abs(base_acc - attn_acc) > abs(base_acc - ffn_acc) + 0.05:
        print(f"  -> Attention is the bottleneck: more precision-sensitive than FFN")
    elif abs(base_acc - ffn_acc) > abs(base_acc - attn_acc) + 0.05:
        print(f"  -> FFN is the bottleneck: more precision-sensitive than attention")
    else:
        print(f"  -> Both layer types contribute roughly equally to accuracy loss")
    print(f"  Best mixed config ({mix_ratio:.1f}x compression): {mix_acc*100:.1f}% accuracy "
          f"({(base_acc-mix_acc)*100:+.1f}pp vs float32)")
    print("=" * 72)

if __name__ == "__main__":
    main()
