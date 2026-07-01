"""
Option A — Post-Training Ternary Weight Quantization on code-minilm

Takes the already-fine-tuned code-minilm transformer, applies ternary PTQ
({-1, 0, +1}) to all nn.Linear weight matrices (no retraining), and measures
how much embedding quality degrades vs the float32 baseline and INT8 PTQ.

Benchmark: same triplet structure as trit_benchmark.py — (query, positive, negative).
Score = fraction of pairs where model ranks positive above negative.

Usage:
  python trit_ptq_ternary_test.py
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_PATH = "models/code-minilm"
BATCH_SIZE = 64

# ── benchmark triplets (same hard pairs as trit_benchmark.py) ─────────────────

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


# ── quantization helpers ───────────────────────────────────────────────────────

def ternary_ptq(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric threshold-based ternary quantization, no gradient needed."""
    t = 0.7 * weight.abs().mean()
    return torch.where(weight > t,  torch.ones_like(weight),
           torch.where(weight < -t, -torch.ones_like(weight),
                       torch.zeros_like(weight)))

def int8_ptq(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor INT8 (fake-quant: store as float, simulate 8-bit range)."""
    scale = weight.abs().max() / 127.0 + 1e-8
    q = torch.round(weight / scale).clamp(-127, 127)
    return q * scale

def apply_ptq(model, mode: str):
    """Replace all nn.Linear weight data in-place. mode: 'ternary' | 'int8'"""
    n_layers = 0
    total_params = 0
    zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                w = module.weight.data
                if mode == "ternary":
                    q = ternary_ptq(w)
                    zero_params += (q == 0).sum().item()
                else:
                    q = int8_ptq(w)
                module.weight.data.copy_(q)
            n_layers += 1
            total_params += w.numel()
    sparsity = zero_params / total_params if mode == "ternary" else 0.0
    return n_layers, total_params, sparsity

def compression_ratio(mode: str, sparsity: float = 0.0) -> float:
    """Theoretical bits/param vs float32 baseline (32 bits/param)."""
    if mode == "ternary":
        # packed: log2(3) ≈ 1.585 bits/trit, but sparsity means more zeros
        # zero trits can be sparse-coded; rough estimate: (1-sparsity)*log2(3) + small overhead
        bits = (1 - sparsity) * 1.585 + 0.2
        return 32.0 / bits
    elif mode == "int8":
        return 32.0 / 8.0
    return 1.0


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, triplets):
    queries   = [t[0] for t in triplets]
    positives = [t[1] for t in triplets]
    negatives = [t[2] for t in triplets]
    all_texts = queries + positives + negatives
    t0 = time.time()
    embs = model.encode(all_texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.time() - t0
    n = len(triplets)
    q_embs = embs[:n]
    p_embs = embs[n:2*n]
    g_embs = embs[2*n:]
    pos_scores = (q_embs * p_embs).sum(axis=1)
    neg_scores = (q_embs * g_embs).sum(axis=1)
    acc = (pos_scores > neg_scores).mean()
    return float(acc), elapsed


# ── layer-level sparsity report ───────────────────────────────────────────────

def layer_sparsity(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            zeros = (w == 0).sum().item()
            total = w.numel()
            rows.append((name, total, zeros / total))
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Benchmark: {len(TRIPLETS)} triplets\n")

    print("Loading code-minilm (float32 baseline)...")
    base_model = SentenceTransformer(MODEL_PATH, device=device)
    base_model.eval()

    print("Evaluating float32 baseline...")
    base_acc, base_time = evaluate(base_model, TRIPLETS)
    print(f"  Accuracy: {base_acc*100:.1f}%  ({base_time:.2f}s)\n")

    results = {"float32": (base_acc, base_time, 1.0, 0.0)}

    for mode, label in [("int8", "INT8 PTQ"), ("ternary", "Ternary PTQ")]:
        print(f"Applying {label}...")
        quant_model = copy.deepcopy(base_model)
        quant_model.eval()
        n_layers, n_params, sparsity = apply_ptq(quant_model, mode)
        ratio = compression_ratio(mode, sparsity)
        print(f"  Quantized {n_layers} Linear layers, {n_params:,} weights")
        if mode == "ternary":
            print(f"  Sparsity (fraction zeroed): {sparsity*100:.1f}%")
        print(f"  Theoretical compression vs float32: {ratio:.1f}x")

        print(f"  Evaluating...")
        acc, elapsed = evaluate(quant_model, TRIPLETS)
        results[mode] = (acc, elapsed, ratio, sparsity)
        print(f"  Accuracy: {acc*100:.1f}%  (dropped {(base_acc-acc)*100:.1f}pp)  ({elapsed:.2f}s)\n")

        if mode == "ternary":
            print("  Per-layer sparsity (ternary):")
            for name, total, sp in layer_sparsity(quant_model):
                bar = "#" * int(sp * 20)
                print(f"    {name:<50} {total:>8,}w  {sp*100:5.1f}%  |{bar:<20}|")
            print()

    # ── summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  OPTION A — PTQ Ternary Weight Quantization on code-minilm")
    print("=" * 70)
    print(f"  {'Mode':<15} {'Accuracy':>10} {'Drop':>8} {'Compression':>13} {'Sparsity':>10}")
    print(f"  {'-'*60}")
    modes = [("float32", "Float32"), ("int8", "INT8 PTQ"), ("ternary", "Ternary PTQ")]
    for mode, label in modes:
        acc, elapsed, ratio, sparsity = results[mode]
        drop = base_acc - acc
        sp_str = f"{sparsity*100:.0f}%" if mode == "ternary" else "—"
        print(f"  {label:<15} {acc*100:>9.1f}%  {drop*100:>+7.1f}pp  {ratio:>10.1f}x  {sp_str:>10}")
    print()
    t_acc, _, t_ratio, t_sp = results["ternary"]
    i_acc, _, i_ratio, _ = results["int8"]
    if t_acc >= i_acc - 0.03:
        verdict = f"Ternary PTQ holds within 3pp of INT8 at {t_ratio:.0f}x compression vs INT8's {i_ratio:.0f}x — viable tradeoff."
    else:
        verdict = f"Ternary PTQ loses {(i_acc-t_acc)*100:.1f}pp more than INT8 ({t_ratio:.0f}x vs {i_ratio:.0f}x compression) — high cost for high gain."
    print(f"  Verdict: {verdict}")
    print()
    print("  Note: PTQ applies no retraining — this is the worst case for ternary.")
    print("  Option B (QAT) will retrain with ternary weights and should recover accuracy.")
    print("=" * 70)

if __name__ == "__main__":
    main()
