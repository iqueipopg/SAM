"""
run_all.py — Ejecuta los 6 toy experiments en secuencia.
Uso: python experiments/run_all.py

Nota: E4 (model comparison) solo ejecuta Qwen-7B en local.
      En Rudiger, descomentar los modelos adicionales en exp4_model_comparison.py.

Tiempo estimado local (RTX 4060):
  E1 ~10 min  | E2 ~15 min  | E3 ~45 min  | E4 ~15 min  | E5 ~30 min  | E6 ~20 min
  Total: ~2.5h
"""
import json, os, time
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    ("E1 — Framework Selection (A1)", "experiments/exp1_framework_selection.py"),
    ("E2 — Screening Gold Standard (A3)", "experiments/exp2_screening_gold_standard.py"),
    ("E3 — Ensemble Ablation (A3)", "experiments/exp3_ensemble_ablation.py"),
    ("E4 — Model Comparison (A3)", "experiments/exp4_model_comparison.py"),
    ("E5 — End-to-End Benchmark (A1-A4)", "experiments/exp5_end_to_end_benchmark.py"),
    ("E6 — Full-text Impact (A4)", "experiments/exp6_fulltext_impact.py"),
]


def run_experiment(name, module_path):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    t0 = time.time()
    exit_code = os.system(f"python {module_path}")
    elapsed = round(time.time() - t0, 1)
    status = "OK" if exit_code == 0 else "ERROR"
    print(f"  -> {status}  ({elapsed}s)")
    return {"name": name, "status": status, "elapsed_s": elapsed}


if __name__ == "__main__":
    print("\nRunning all 6 experiments...")
    print(f"Results will be saved to: {RESULTS_DIR}/\n")

    results = []
    total_t0 = time.time()

    for name, path in EXPERIMENTS:
        results.append(run_experiment(name, path))

    total_elapsed = round(time.time() - total_t0, 1)

    # Summary
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({"experiments": results, "total_elapsed_s": total_elapsed}, f, indent=2)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        icon = "OK" if r["status"] == "OK" else "XX"
        print(f"  [{icon}] {r['name']:<45} {r['elapsed_s']:>7}s")
    print(f"\n  Total: {total_elapsed}s")
    print(f"  Results: {RESULTS_DIR}/")
