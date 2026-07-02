"""
Test the saved ternary-quantized Qwen2.5-7B-Instruct weights for real.
Loads the real base model (embeddings, norms, lm_head — not ternary),
splices in the saved ternary Linear-layer weights (q/k/v/o_proj,
mlp gate/up/down_proj, all 28 layers), and runs one real generation
to see honestly whether it produces coherent output or the same
collapse-to-garbage failure measured earlier on the smaller MiniLM model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TERNARY_WEIGHTS = "models/qwen2.5-7b-instruct-ternary/trit_weights.pt"

print("Loading real base model (bfloat16, CPU)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"  Base model loaded in {time.time()-t0:.0f}s")

print("Loading ternary weights file...")
trit_sd = torch.load(TERNARY_WEIGHTS, map_location="cpu", weights_only=False)
print(f"  {len(trit_sd)} ternary tensors loaded")

# ── Splice: replace each targeted Linear layer's weight with the ternary version ──

model_sd = model.state_dict()
replaced = 0
skipped = []
for key, trit_tensor in trit_sd.items():
    full_key = key + ".weight"
    if full_key in model_sd:
        if model_sd[full_key].shape == trit_tensor.shape:
            model_sd[full_key] = trit_tensor.to(torch.bfloat16)
            replaced += 1
        else:
            skipped.append((full_key, "shape mismatch", tuple(model_sd[full_key].shape), tuple(trit_tensor.shape)))
    else:
        skipped.append((full_key, "not found in model", None, None))

print(f"Replaced {replaced} / {len(trit_sd)} weight tensors with ternary values")
if skipped:
    print(f"Skipped {len(skipped)}:")
    for s in skipped[:5]:
        print(f"  {s}")

model.load_state_dict(model_sd)
model.eval()

# ── Real generation test ──────────────────────────────────────────────────────

prompt = "Write a one-sentence description of what a Python for loop does."
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

print(f"\nPrompt: {prompt}")
print("Generating (this may take a while on CPU)...")
t0 = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=60, do_sample=False)
elapsed = time.time() - t0

response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

with open("qwen_response.txt", "w", encoding="utf-8") as f:
    f.write(f"Generated in {elapsed:.0f}s:\n")
    f.write(response + "\n")
    words = response.split()
    f.write(f"\nWord count: {len(words)}\n")
    verdict = "LIKELY COHERENT" if len(words) > 3 and any(c.isalpha() for c in response) else "LIKELY GARBAGE/COLLAPSED"
    f.write(f"Verdict: {verdict}\n")

print(f"\nGenerated in {elapsed:.0f}s — response written to qwen_response.txt (avoiding console encoding issue)")
