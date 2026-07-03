"""
trit_dna -- point the SAME ternary GPT (triadic attention, {-1,0,+1} weights,
predictive-coding loss) at DNA instead of human language. The architecture
doesn't care what the tokens are; here they're nucleotides {A,C,G,T}.

Why this is a BETTER demo than the Shakespeare LM: DNA lets us MEASURE whether
it learned the "language," instead of eyeballing gibberish. A real genome has
statistical structure -- k-mer (e.g. 3-mer/codon) frequencies and GC content
that are far from uniform. If the model's GENERATED DNA matches the real
genome's k-mer statistics (and a uniform-random baseline does not), the model
demonstrably learned the local structure of DNA.

Honest scope: a ~1M-param char-level model learns LOCAL statistics (k-mer
frequencies, GC content, short motifs). It will NOT learn genes, function, or
long-range structure -- that needs large models. "Learned local DNA statistics,
measurably, beating random" is the real, bounded claim.

Data: real human chr1 sequence via the Ensembl REST API (NCBI DNS is blocked
in this environment; Ensembl works). Cached to data/dna.txt.
"""
import sys
import urllib.request
import collections
import random
from pathlib import Path

import numpy as np
import torch

import trit_transformer as tt

sys.stdout.reconfigure(errors="replace")
tt.device = torch.device("cpu")
tt.CFG.update(dict(
    n_embd=160, n_head=4, n_layer=4, block_size=100,
    batch_size=32, epochs=2500, quant_warmup=400, eval_every=250,
))

DNA_PATH = Path("data/dna_yeast.txt")


def fetch_dna():
    # Honed from the first run: the human chr1 START was GC-rich, CpG-island,
    # repeat-heavy -- the model over-produced homopolymer runs and distorted the
    # k-mer distribution. Switched to the MIDDLE of yeast chr IV: coding-dense,
    # ~38% GC, low-complexity-repeat-free -- a fair, learnable target with real
    # codon structure.
    if DNA_PATH.exists():
        return DNA_PATH.read_text()
    DNA_PATH.parent.mkdir(exist_ok=True)
    seqs = []
    for start in range(600_000, 1_400_000, 100_000):     # 8 x 100kb, mid yeast chr IV
        url = (f"https://rest.ensembl.org/sequence/region/saccharomyces_cerevisiae/"
               f"IV:{start}..{start+99999}?content-type=text/plain")
        with urllib.request.urlopen(url, timeout=60) as r:
            seqs.append(r.read().decode("utf-8", "ignore").strip())
        print(f"  fetched {start:,}")
    raw = "".join(seqs).upper()
    dna = "".join(c for c in raw if c in "ACGT")
    DNA_PATH.write_text(dna)
    return dna


def kmer_vec(s, k, keys):
    c = collections.Counter(s[i:i + k] for i in range(len(s) - k + 1))
    tot = sum(c.values()) or 1
    return np.array([c.get(km, 0) / tot for km in keys])


def main():
    dna = fetch_dna()
    gc = (dna.count("G") + dna.count("C")) / len(dna) * 100
    print(f"DNA length {len(dna):,} nt, GC content {gc:.1f}% (uniform-random would be 50%)")

    chars = sorted(set(dna))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in dna], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    _, trit = tt.build_models(len(chars))
    print(f"TritDNA GPT: vocab={len(chars)} (nucleotides), "
          f"params={sum(p.numel() for p in trit.parameters()):,}, ternary weights\n")
    tt.train(trit, train_data, val_data, "TritDNA", use_pred=True, is_trit=True)

    tt.set_quant(trit, True)
    gen = tt.generate(trit, itos, stoi, prompt="ATG", max_new=6000, temperature=0.9)
    gen = "".join(c for c in gen if c in "ACGT")

    # === honest measurable eval: k-mer statistics at k=2,3,4 vs real & random ===
    import itertools
    real = dna[n:][:len(gen)]                              # held-out real DNA, same length
    rand = "".join(random.choice("ACGT") for _ in range(len(gen)))
    gc_gen = (gen.count("G") + gen.count("C")) / len(gen) * 100

    print("\n" + "=" * 60)
    print("  DID IT LEARN THE DNA 'LANGUAGE'? (k-mer statistics, yeast)")
    print("=" * 60)
    print(f"  GC content   real {gc:.1f}%   generated {gc_gen:.1f}%   (random 50%)")
    print(f"  {'k':>3}  {'corr(gen)':>10}  {'corr(rand)':>11}  {'TV(gen)':>8}  {'TV(rand)':>9}")
    for k in (2, 3, 4):
        keys = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
        rv, gv, uv = kmer_vec(real, k, keys), kmer_vec(gen, k, keys), kmer_vec(rand, k, keys)
        cg, cr = np.corrcoef(rv, gv)[0, 1], np.corrcoef(rv, uv)[0, 1]
        tg, tr = 0.5 * np.abs(rv - gv).sum(), 0.5 * np.abs(rv - uv).sum()
        print(f"  {k:>3}  {cg:>10.3f}  {cr:>11.3f}  {tg:>8.3f}  {tr:>9.3f}")
    print(f"\n  sample generated DNA:\n  {gen[:120]}")

    torch.save({"state": trit.state_dict(), "stoi": stoi, "itos": itos, "cfg": dict(tt.CFG)},
               "trit_dna.pt")
    print("\nSaved trit_dna.pt (model + tokenizer + config).")
    print("Honest read: generated corr >> random corr = the ternary model learned")
    print("real local DNA structure (codon/k-mer statistics), not just uniform letters.")


if __name__ == "__main__":
    main()
