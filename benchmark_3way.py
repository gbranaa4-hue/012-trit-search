"""
3-way benchmark: Microsoft baseline vs your fine-tuned vs Jina code-specific model.
Uses the same BENCHMARK triples as trit_benchmark.py.
"""
from pathlib import Path
from sentence_transformers import SentenceTransformer
from trit_benchmark import BENCHMARK, run_benchmark, print_report

FINE_TUNED = str(Path(__file__).resolve().parent / "models" / "code-minilm")
BASELINE   = "all-MiniLM-L6-v2"
JINA_CODE  = "flax-sentence-embeddings/st-codesearch-distilroberta-base"

reports = []

print("\n[1/3] Microsoft baseline (all-MiniLM-L6-v2)...")
m = SentenceTransformer(BASELINE)
r = run_benchmark(m, "Microsoft-baseline", BENCHMARK)
print_report(r)
reports.append(r)
del m

print("\n[2/3] Your fine-tuned model...")
m = SentenceTransformer(FINE_TUNED)
r = run_benchmark(m, "Your-fine-tuned", BENCHMARK)
print_report(r)
reports.append(r)
del m

print("\n[3/3] CodeSearch-DistilRoBERTa (purpose-built for code search)...")
m = SentenceTransformer(JINA_CODE)
r = run_benchmark(m, "CodeSearch-specific", BENCHMARK)
print_report(r)
reports.append(r)
del m

print("\n" + "="*65)
print("  FINAL RANKING")
print("="*65)
for r in sorted(reports, key=lambda x: -x["accuracy"]):
    print(f"  {r['name']:<22} {r['accuracy']:>5.1f}%  margin={r['avg_margin']:+.3f}  {r['elapsed']*1000:.0f}ms")
