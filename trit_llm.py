"""
trit_llm -- train the existing TritGPT (triadic attention, ternary {-1,0,+1}
weights, predictive-coding loss) into a working char-level language model and
generate from it.

Honest scope, stated plainly: this is a ~1M-param CHARACTER-level model trained
on Shakespeare on CPU. It learns English-ish structure -- character names, line
breaks, archaic cadence, plausible-looking words -- but it is semantically
nonsense. It is a REAL, working ternary language model; it is NOT a useful one,
and it is nothing like a modern LLM. The point is a genuine end-to-end ternary
LM that generates text, honestly bounded.

Fixes the reusability bug in the old index/trit_lm.pt: saves the tokenizer
(stoi/itos) and config INSIDE the checkpoint, so it can actually be reloaded
and generated from later.

Reuses trit_transformer.py (TritGPT, get_data, train, generate).
"""
import sys
import torch
import trit_transformer as tt

# trit_transformer prints box-drawing chars Windows' cp1252 console can't encode
# (the same crash class hit earlier tonight); replace unencodable chars instead.
sys.stdout.reconfigure(errors="replace")

tt.device = torch.device("cpu")   # trit_transformer hardcodes cuda; this box is CPU-only
tt.CFG.update(dict(
    n_embd=160, n_head=4, n_layer=4, block_size=80,
    batch_size=32, epochs=2500, quant_warmup=400, eval_every=250,
))


def main():
    train_data, val_data, vocab_size, stoi, itos = tt.get_data()
    _, trit = tt.build_models(vocab_size)
    nparams = sum(p.numel() for p in trit.parameters())
    print(f"TritGPT: vocab={vocab_size}, params={nparams:,}, ternary weights {{-1,0,+1}}")

    tt.train(trit, train_data, val_data, "TritGPT", use_pred=True, is_trit=True)

    tt.set_quant(trit, True)   # generate in the deployed ternary state
    try:
        tt.trit_dist(trit)      # print how ternary/sparse the weights are
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("  GENERATED SAMPLES (char-level ternary LM)")
    print("=" * 60)
    for temp in (0.5, 0.8, 1.0):
        print(f"\n--- temperature {temp} ---")
        print(tt.generate(trit, itos, stoi, prompt="ROMEO:", max_new=400, temperature=temp))

    torch.save({"state": trit.state_dict(), "stoi": stoi, "itos": itos, "cfg": dict(tt.CFG)},
               "trit_llm.pt")
    print("\nSaved trit_llm.pt (model + tokenizer + config bundled -- reloadable)")
    print("\nHonest read: recognizable Shakespeare-ish STRUCTURE, semantic nonsense.")
    print("A real working ternary LM, not a useful one -- exactly what a ~1M-param")
    print("char model on CPU produces.")


if __name__ == "__main__":
    main()
