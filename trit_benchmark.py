"""
012 Embedding Benchmark
Tests semantic search quality across hundreds of code concepts
and multiple languages. Compares baseline MiniLM vs your fine-tuned model.

Usage:
  python trit_benchmark.py                    Run full benchmark
  python trit_benchmark.py --quick            Run 20-query quick test
  python trit_benchmark.py --compare          Side-by-side baseline vs fine-tuned
  python trit_benchmark.py --stream           Pull 100k lines from GitHub and test
"""

import argparse, time, random
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK DATASET
# (query, correct_code, wrong_code) triples
# Score = did the model rank correct_code above wrong_code?
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK = [
    # ── HARD: wrong answer is semantically similar but subtly different ──

    # Health vs damage — both involve health values
    ("take damage from enemy",
     "func take_damage(amount: float):\n\thealth -= amount\n\thealth = clamp(health, 0, max_health)\n\tif health <= 0: _die()",
     "func heal(amount: float):\n\thealth += amount\n\thealth = clamp(health, 0, max_health)"),

    # Wave spawn vs enemy spawn — both spawn things
    ("spawn wave of enemies",
     "func start_wave():\n\tfor i in wave_size:\n\t\tvar e = enemy_scene.instantiate()\n\t\tspawn_points[i % spawn_points.size()].add_child(e)",
     "func spawn_single_enemy(pos: Vector3):\n\tvar e = enemy_scene.instantiate()\n\te.global_position = pos\n\tadd_child(e)"),

    # Chase vs patrol — both are enemy AI movement
    ("enemy chases player",
     "func _chase(delta):\n\tvar next = nav_agent.get_next_path_position()\n\tvelocity = (next - global_position).normalized() * chase_speed\n\tmove_and_slide()",
     "func _patrol(delta):\n\tif position.distance_to(patrol_target) < 1.0:\n\t\tpatrol_index = (patrol_index + 1) % patrol_points.size()\n\tvelocity = (patrol_target - position).normalized() * patrol_speed"),

    # Training loss vs eval loss — both compute loss
    ("compute training loss",
     "model.train()\noptimizer.zero_grad()\noutput = model(input_ids, labels=labels)\nloss = output.loss\nloss.backward()\noptimizer.step()",
     "model.eval()\nwith torch.no_grad():\n\toutput = model(input_ids, labels=labels)\n\tval_loss = output.loss.item()"),

    # Contrastive vs cross-entropy — both are loss functions
    ("contrastive learning loss",
     "logits = torch.matmul(q, k.T) / temperature\nlabels = torch.arange(q.size(0)).to(device)\nloss = F.cross_entropy(logits, labels)",
     "logits = model(input_ids).logits\nshift_logits = logits[..., :-1, :]\nshift_labels = labels[..., 1:]\nloss = F.cross_entropy(shift_logits.reshape(-1, vocab), shift_labels.reshape(-1))"),

    # Merge weights vs quantize weights — both modify model weights
    ("merge lora into base model",
     "model = PeftModel.from_pretrained(base_model, lora_path)\nmodel = model.merge_and_unload()\nmodel.save_pretrained(output_path)",
     "bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')\nmodel = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config)"),

    # Save vs load — both touch same file path
    ("save game state to disk",
     "func save_game():\n\tvar data = {wave=current_wave, health=player.health, gold=gold}\n\tvar f = FileAccess.open(SAVE_PATH, FileAccess.WRITE)\n\tf.store_string(JSON.stringify(data))",
     "func load_game():\n\tvar f = FileAccess.open(SAVE_PATH, FileAccess.READ)\n\tvar data = JSON.parse_string(f.get_as_text())\n\tcurrent_wave = data.wave\n\tgold = data.gold"),

    # Card draw vs card play — both involve cards
    ("draw card from deck",
     "func draw_card() -> Card:\n\tif deck.is_empty(): reshuffle_discard()\n\tvar card = deck.pop_back()\n\thand.append(card)\n\treturn card",
     "func play_card(card: Card):\n\thand.erase(card)\n\tdiscard.append(card)\n\tcard.activate(player)"),

    # Turret shoot vs turret target — both turret behavior
    ("turret fires at target",
     "func _shoot():\n\tvar bullet = bullet_scene.instantiate()\n\tbullet.direction = (target.global_position - barrel.global_position).normalized()\n\tadd_child(bullet)",
     "func _find_target():\n\tvar enemies = get_tree().get_nodes_in_group('enemies')\n\ttarget = enemies.reduce(func(a,b): return a if position.distance_to(a.position) < position.distance_to(b.position) else b)"),

    # Fetch POST vs GET — both are HTTP calls
    ("send post request with json body",
     "const res = await fetch('/api/submit', {\n\tmethod: 'POST',\n\theaders: {'Content-Type': 'application/json'},\n\tbody: JSON.stringify(payload)\n});\nreturn res.json();",
     "const res = await fetch(`/api/items/${id}`, {\n\tmethod: 'GET',\n\theaders: {'Authorization': `Bearer ${token}`}\n});\nreturn res.json();"),

    # useState vs useEffect — both are React hooks
    ("react state management hook",
     "const [user, setUser] = useState(null);\nconst [loading, setLoading] = useState(true);\nconst [error, setError] = useState(null);",
     "useEffect(() => {\n\tfetchUser(id).then(setUser).catch(setError).finally(() => setLoading(false));\n\treturn () => controller.abort();\n}, [id]);"),

    # Mutex vs channel — both are Go concurrency
    ("go mutex lock shared data",
     "mu.Lock()\ndefer mu.Unlock()\nsharedMap[key] = value",
     "ch := make(chan int, 1)\ngo func() { ch <- compute() }()\nresult := <-ch"),

    # Auth middleware vs rate limit — both are middleware
    ("jwt authentication middleware",
     "func AuthMiddleware(next http.Handler) http.Handler {\n\treturn http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {\n\t\ttoken := r.Header.Get('Authorization')\n\t\tif !validateJWT(token) { http.Error(w, 'Unauthorized', 401); return }\n\t\tnext.ServeHTTP(w, r)\n\t})\n}",
     "func RateLimitMiddleware(next http.Handler) http.Handler {\n\tlimiter := rate.NewLimiter(10, 30)\n\treturn http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {\n\t\tif !limiter.Allow() { http.Error(w, 'Too Many Requests', 429); return }\n\t\tnext.ServeHTTP(w, r)\n\t})\n}"),

    # SQL insert vs select — both touch same table
    ("insert new user into database",
     "cursor.execute('INSERT INTO users (email, password_hash, created_at) VALUES (%s, %s, NOW())', (email, hash_password(password)))\nconn.commit()",
     "cursor.execute('SELECT id, email, role FROM users WHERE email = %s AND is_active = true', (email,))\nuser = cursor.fetchone()"),

    # Replay buffer add vs sample — both are replay buffer methods
    ("add experience to replay buffer",
     "def add(self, state, action, reward, next_state, done):\n\tif len(self.buffer) >= self.capacity:\n\t\tself.buffer.popleft()\n\tself.buffer.append((state, action, reward, next_state, done))",
     "def sample(self, batch_size):\n\tbatch = random.sample(self.buffer, batch_size)\n\tstates, actions, rewards, next_states, dones = zip(*batch)\n\treturn np.array(states), np.array(actions), np.array(rewards)"),

    # LoRA config vs LoRA training — both are LoRA
    ("configure lora adapter ranks",
     "config = LoraConfig(\n\tr=16,\n\tlora_alpha=32,\n\ttarget_modules=['q_proj', 'v_proj', 'k_proj'],\n\tlora_dropout=0.05,\n\tbias='none'\n)",
     "model = get_peft_model(base_model, lora_config)\nmodel.enable_input_require_grads()\nmodel.gradient_checkpointing_enable()\ntrainable = sum(p.numel() for p in model.parameters() if p.requires_grad)"),

    # Ternary snap vs ternary threshold — both quantize
    ("snap weights to ternary values",
     "threshold = 0.7 * weight.abs().mean()\nreturn torch.where(weight > threshold, 1.0,\n\ttorch.where(weight < -threshold, -1.0, 0.0))",
     "scale = weight.abs().mean()\nnorm_weight = weight / (scale + 1e-8)\nreturn norm_weight, scale"),

    # FAISS index vs FAISS search — both FAISS operations
    ("add vectors to faiss index",
     "index = faiss.IndexFlatIP(dim)\nindex.add(vectors.astype('float32'))\nfaiss.write_index(index, index_path)",
     "index = faiss.read_index(index_path)\nscores, indices = index.search(query_vec.reshape(1,-1), k=10)\nreturn [(metadata[i], float(scores[0][j])) for j, i in enumerate(indices[0])]"),

    # Death vs respawn — both happen after player dies
    ("player death handler",
     "func _die():\n\tis_dead = true\n\t$CollisionShape3D.disabled = true\n\t$AnimationPlayer.play('death')\n\tdied.emit()\n\tget_tree().create_timer(3.0).timeout.connect(_on_respawn_timer)",
     "func _respawn():\n\tis_dead = false\n\thealth = max_health\n\t$CollisionShape3D.disabled = false\n\tglobal_position = spawn_point.global_position"),

    # Binary search vs linear search — both search arrays
    ("binary search sorted array",
     "def binary_search(arr, target):\n\tleft, right = 0, len(arr)-1\n\twhile left <= right:\n\t\tmid = (left+right)//2\n\t\tif arr[mid] == target: return mid\n\t\telif arr[mid] < target: left = mid+1\n\t\telse: right = mid-1\n\treturn -1",
     "def linear_search(arr, target):\n\tfor i, val in enumerate(arr):\n\t\tif val == target: return i\n\treturn -1"),

    # Gradient checkpointing vs gradient clipping — both gradient related
    ("enable gradient checkpointing to save memory",
     "model.gradient_checkpointing_enable()\nmodel.enable_input_require_grads()\nprint('Gradient checkpointing enabled — trades compute for memory')",
     "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\noptimizer.step()\nscheduler.step()"),

    # WebSocket send vs receive — both WebSocket operations
    ("send message over websocket",
     "async def send_to_player(websocket, event: str, data: dict):\n\tpayload = json.dumps({'event': event, 'data': data})\n\tawait websocket.send(payload)",
     "async def receive_from_player(websocket):\n\traw = await websocket.recv()\n\tmessage = json.loads(raw)\n\treturn message.get('event'), message.get('data')"),

    # Docker build vs Docker run — both Docker commands
    ("build docker image from dockerfile",
     "docker build --no-cache -t myapp:$VERSION .\ndocker tag myapp:$VERSION registry.io/org/myapp:$VERSION\ndocker push registry.io/org/myapp:$VERSION",
     "docker run -d --name myapp -p 8080:8080 \\\n\t-e DATABASE_URL=$DB_URL \\\n\t-v /data:/app/data \\\n\tregistry.io/org/myapp:latest"),

    # LRU get vs LRU put — both LRU cache operations
    ("get item from lru cache",
     "def get(self, key: int) -> int:\n\tif key not in self.cache: return -1\n\tself.cache.move_to_end(key)\n\treturn self.cache[key]",
     "def put(self, key: int, value: int) -> None:\n\tif key in self.cache: self.cache.move_to_end(key)\n\tself.cache[key] = value\n\tif len(self.cache) > self.capacity: self.cache.popitem(last=False)"),

    # Rate limit check vs rate limit reset — both rate limiting
    ("check if request exceeds rate limit",
     "def is_rate_limited(user_id: str) -> bool:\n\tkey = f'rate:{user_id}'\n\tcount = redis.incr(key)\n\tif count == 1: redis.expire(key, WINDOW_SECONDS)\n\treturn count > MAX_REQUESTS",
     "def reset_rate_limit(user_id: str) -> None:\n\tkey = f'rate:{user_id}'\n\tredis.delete(key)"),
]


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(model, name: str, pairs: list) -> dict:
    """
    For each (query, correct, wrong) triple:
    - Model passes if sim(query, correct) > sim(query, wrong)
    - Score = % of triples passed
    """
    queries  = [p[0] for p in pairs]
    corrects = [p[1] for p in pairs]
    wrongs   = [p[2] for p in pairs]

    all_texts = queries + corrects + wrongs
    t0   = time.time()
    vecs = model.encode(all_texts, normalize_embeddings=True)
    elapsed = time.time() - t0

    n = len(pairs)
    q_vecs = vecs[:n]
    c_vecs = vecs[n:2*n]
    w_vecs = vecs[2*n:]

    correct_scores = (q_vecs * c_vecs).sum(axis=1)
    wrong_scores   = (q_vecs * w_vecs).sum(axis=1)
    passed         = correct_scores > wrong_scores

    results = []
    for i, (query, correct, wrong) in enumerate(pairs):
        results.append({
            "query":   query,
            "correct": correct_scores[i],
            "wrong":   wrong_scores[i],
            "passed":  bool(passed[i]),
            "margin":  float(correct_scores[i] - wrong_scores[i]),
        })

    accuracy = passed.mean() * 100
    avg_margin = (correct_scores - wrong_scores).mean()

    return {
        "name":       name,
        "accuracy":   accuracy,
        "avg_margin": float(avg_margin),
        "elapsed":    elapsed,
        "results":    results,
        "n":          n,
    }


def print_report(report: dict, verbose=False):
    print(f"\n{'='*60}")
    print(f"  Model    : {report['name']}")
    print(f"  Accuracy : {report['accuracy']:.1f}%  ({int(report['accuracy']*report['n']/100)}/{report['n']} correct)")
    print(f"  Margin   : {report['avg_margin']:+.3f}  (correct score - wrong score)")
    print(f"  Time     : {report['elapsed']*1000:.0f}ms for {report['n']} queries")
    print(f"{'='*60}")

    if verbose:
        print("\nPer-query breakdown:")
        for r in sorted(report['results'], key=lambda x: x['margin']):
            icon = "✓" if r['passed'] else "✗"
            print(f"  {icon} [{r['margin']:+.3f}]  {r['query'][:50]}")


def stream_github_test(model, n_files=200):
    """
    Pull real GitHub code and test search quality on it.
    Generates pairs automatically from function docstrings.
    """
    try:
        from datasets import load_dataset
        from trit_embed_train import extract_pairs_from_code
    except ImportError:
        print("Install: pip install datasets")
        return

    print(f"\nStreaming {n_files} files from GitHub for live test...")
    pairs = []

    try:
        ds = load_dataset("codeparrot/github-code", streaming=True, split="train")
        for i, item in enumerate(ds):
            if i >= n_files * 5:
                break
            code = item.get("content", "")
            if len(code) < 300:
                continue
            extracted = extract_pairs_from_code(code, "unknown")
            if extracted:
                # Take first pair as (anchor=function_name, positive=body)
                # Shuffle body lines as negative
                anchor, positive = extracted[0]
                lines = positive.splitlines()
                random.shuffle(lines)
                wrong = "\n".join(lines[:5])
                if len(wrong) > 20:
                    pairs.append((anchor, positive, wrong))
            if len(pairs) >= 100:
                break
    except Exception as e:
        print(f"Stream failed: {e}")
        return

    if not pairs:
        print("No pairs extracted from stream.")
        return

    print(f"  Extracted {len(pairs)} live pairs from GitHub")
    report = run_benchmark(model, "live-github", pairs)
    print_report(report, verbose=False)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_model(path):
    from sentence_transformers import SentenceTransformer
    print(f"Loading: {path}")
    return SentenceTransformer(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",   action="store_true", help="Run 20-query subset")
    parser.add_argument("--compare", action="store_true", help="Baseline vs fine-tuned")
    parser.add_argument("--stream",  action="store_true", help="Live GitHub test")
    parser.add_argument("--verbose", action="store_true", help="Show per-query results")
    args = parser.parse_args()

    FINE_TUNED = r"C:\Users\gbran\OneDrive\Documents\012-ternary\models\code-minilm"
    BASELINE   = "all-MiniLM-L6-v2"

    pairs = BENCHMARK[:20] if args.quick else BENCHMARK

    if args.compare:
        print("\nRunning comparison: baseline vs fine-tuned...\n")
        base_model  = load_model(BASELINE)
        base_report = run_benchmark(base_model, "baseline-MiniLM", pairs)
        print_report(base_report, args.verbose)

        del base_model

        ft_model   = load_model(FINE_TUNED)
        ft_report  = run_benchmark(ft_model, "fine-tuned-012", pairs)
        print_report(ft_report, args.verbose)

        diff = ft_report['accuracy'] - base_report['accuracy']
        print(f"\n  Improvement: {diff:+.1f}% accuracy")
        print(f"  Margin gain: {ft_report['avg_margin'] - base_report['avg_margin']:+.3f}")

        if args.stream:
            print("\n--- Live GitHub test (fine-tuned) ---")
            stream_github_test(ft_model)

    else:
        model  = load_model(FINE_TUNED)
        report = run_benchmark(model, "fine-tuned-012", pairs)
        print_report(report, args.verbose)

        if args.stream:
            stream_github_test(model)
