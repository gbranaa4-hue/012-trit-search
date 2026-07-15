"""
Option A' — Scaled-ternary control + BCQ comparison, on the REAL code-minilm checkpoint.

Run this yourself (needs your models/code-minilm checkpoint + GPU for BCQ's
bcq_quant/bcq.py which calls .cuda() internally -- CPU-adapted version is at
option_d/bcq_cpu.py if you don't have a GPU handy, same algorithm either way).

Pre-registered before running:
  CONFIRM  (boring explanation holds): scaled-ternary accuracy recovers most
           of the way from 50% toward INT8's 95%, on the real trained model.
  DISCONFIRM: scaled-ternary stays near 50% even with the scale factor added
           -- meaning the original 45pp collapse really is about geometry,
           not the missing scale, and the earlier finding stands as written.

This is a straightforward extension of trit_ptq_ternary_test.py -- same
TRIPLETS, same MODEL_PATH, same evaluate() -- add one method, don't change
the existing ones (so the paper's existing numbers stay reproducible).
"""
import copy, time, sys
sys.path.insert(0, "/home/claude/option_d")  # for bcq_cpu.py if no GPU
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

MODEL_PATH = "models/code-minilm"
BATCH_SIZE = 64

# vvv paste TRIPLETS from trit_ptq_ternary_test.py unchanged vvv
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

def ternary_ptq_raw(weight: torch.Tensor) -> torch.Tensor:
    t = 0.7 * weight.abs().mean()
    return torch.where(weight > t,  torch.ones_like(weight),
           torch.where(weight < -t, -torch.ones_like(weight),
                       torch.zeros_like(weight)))

def ternary_ptq_scaled(weight: torch.Tensor) -> torch.Tensor:
    """Same sign/threshold rule, but rescaled by mean(|w|) over kept (nonzero) entries per row."""
    sign = ternary_ptq_raw(weight)
    alpha = torch.zeros(weight.shape[0], 1, device=weight.device)
    for i in range(weight.shape[0]):
        nz = sign[i] != 0
        alpha[i] = weight[i][nz].abs().mean() if nz.any() else 0.0
    return sign * alpha

def apply_ptq(model, fn):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.data.copy_(fn(module.weight.data))

def evaluate(model, triplets):
    queries, positives, negatives = zip(*triplets)
    all_texts = list(queries) + list(positives) + list(negatives)
    embs = model.encode(all_texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
    n = len(triplets)
    q, p, g = embs[:n], embs[n:2*n], embs[2*n:]
    pos = (q * p).sum(axis=1)
    neg = (q * g).sum(axis=1)
    return float((pos > neg).mean())

def main():
    assert TRIPLETS, "Paste the 20 TRIPLETS from trit_ptq_ternary_test.py before running."
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = SentenceTransformer(MODEL_PATH, device=device); base.eval()
    base_acc = evaluate(base, TRIPLETS)
    print(f"Float32 baseline: {base_acc*100:.1f}%")

    for name, fn in [("Ternary PTQ (raw, original)", ternary_ptq_raw),
                     ("Ternary PTQ (scaled, new control)", ternary_ptq_scaled)]:
        m = copy.deepcopy(base); m.eval()
        apply_ptq(m, fn)
        acc = evaluate(m, TRIPLETS)
        print(f"{name}: {acc*100:.1f}%  (drop {(base_acc-acc)*100:+.1f}pp)")

if __name__ == "__main__":
    main()
