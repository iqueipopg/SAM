# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
run_all.py — Orquestador de experimentos SAM.

Ejecuta todos los experimentos en experiments/ en secuencia, captura fallos
individuales sin abortar el runner y genera un resumen JSON + tabla LaTeX.

Uso:
    python experiments/run_all.py              # ejecucion completa
    python experiments/run_all.py --dry-run    # pasa --dry-run a cada experimento

Nota: E4 (model comparison) solo ejecuta Qwen-7B en local.
      En Rudiger, descomentar los modelos adicionales en exp4_model_comparison.py.
      E8 requiere data/gold_standard.json (ver project_t5_gold_standard.md).

Tiempo estimado local (RTX 4060):
  E1 ~10 min | E2 ~15 min | E3 ~45 min | E4 ~15 min |
  E5 ~30 min | E6 ~20 min | E7 ~25 min | E8 ~20 min
  Total: ~3h 20m
"""
import json, os, time, argparse, traceback
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    ("E1 -- Framework Selection (A1)", "experiments/exp1_framework_selection.py"),
    ("E2 -- Screening Gold Standard (A3)", "experiments/exp2_screening_gold_standard.py"),
    ("E3 -- Ensemble Ablation (A3)", "experiments/exp3_ensemble_ablation.py"),
    ("E4 -- Model Comparison (A3)", "experiments/exp4_model_comparison.py"),
    ("E5 -- End-to-End Benchmark (A1-A4)", "experiments/exp5_end_to_end_benchmark.py"),
    ("E6 -- Full-text Impact (A4)", "experiments/exp6_fulltext_impact.py"),
    ("E7 -- VRAM Profiling (A1-A8)", "experiments/exp7_vram_profiling.py"),
    ("E8 -- Gold Standard Eval (A3/A3+A4)", "experiments/exp8_gold_standard_eval.py"),
]


def run_experiment(name, module_path, dry_run=False):
    """Ejecuta un experimento aislando excepciones para no matar el runner."""
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    t0 = time.time()
    status = "ERROR"
    error_msg = None
    try:
        if not Path(module_path).exists():
            raise FileNotFoundError(f"No existe {module_path}")
        flag = " --dry-run" if dry_run else ""
        exit_code = os.system(f"python {module_path}{flag}")
        status = "OK" if exit_code == 0 else "ERROR"
        if exit_code != 0:
            error_msg = f"exit_code={exit_code}"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    elapsed = round(time.time() - t0, 1)
    print(f"  -> {status}  ({elapsed}s)" + (f"  [{error_msg}]" if error_msg else ""))
    result = {"name": name, "status": status, "elapsed_s": elapsed}
    if error_msg:
        result["error"] = error_msg
    return result


def generate_latex_table(results):
    """Generate a summary LaTeX table for all experiments."""
    lines = []
    lines.append(r"\begin{table}[h]\centering")
    lines.append(r"\begin{tabular}{llr}\hline")
    lines.append(r"Experiment & Status & Time (s) \\")
    lines.append(r"\hline")
    for r in results:
        name = r["name"].replace("&", r"\&").replace("--", "---")
        lines.append(f"{name} & {r['status']} & {r['elapsed_s']} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Resumen de todos los experimentos}\label{tab:all_experiments}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to all experiments")
    args = parser.parse_args()

    n_exp = len(EXPERIMENTS)
    print(f"\nRunning all {n_exp} experiments{'  [DRY RUN]' if args.dry_run else ''}...")
    print(f"Results will be saved to: {RESULTS_DIR}/\n")

    results = []
    total_t0 = time.time()

    for name, path in EXPERIMENTS:
        results.append(run_experiment(name, path, dry_run=args.dry_run))

    total_elapsed = round(time.time() - total_t0, 1)

    latex = generate_latex_table(results)

    # Summary
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({
            "experiments": results,
            "total_elapsed_s": total_elapsed,
            "latex_table": latex,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        icon = "OK" if r["status"] == "OK" else "XX"
        print(f"  [{icon}] {r['name']:<45} {r['elapsed_s']:>7}s")
    print(f"\n  Total: {total_elapsed}s")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"\n  LaTeX table:\n{latex}")
