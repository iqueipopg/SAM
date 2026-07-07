# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Exp 8 — Gold Standard Evaluation: A3 and A3+A4 vs Human Annotations

Evaluates pipeline screening accuracy against a human-annotated gold standard.
Computes: confusion matrix, sensitivity, specificity, precision, F1, Cohen's kappa.
Two evaluation modes:
  (a) A3 alone (title/abstract screening)
  (b) A3 + A4 combined (title/abstract + full-text screening)

Input:  data/gold_standard.json
Format:
  {
    "query": "...",
    "protocol": { ... },
    "papers": [
      {
        "pmid": "12345678",
        "title": "...",
        "abstract": "...",
        "gold_decision": "INCLUDE" | "EXCLUDE",
        "gold_reason": "optional reason for exclusion"
      },
      ...
    ]
  }

Output: experiments/results/exp8_results.json + LaTeX table + console report

BLOCKER: This experiment requires data/gold_standard.json which must be
created manually by a domain expert. The file does NOT exist yet.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLD_STANDARD_PATH = Path("data/gold_standard.json")
RESULTS_DIR = Path("experiments/results")


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(tp, fp, fn, tn):
    """Compute classification metrics from confusion matrix counts."""
    n = tp + fp + fn + tn
    if n == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0,
            "precision": 0.0, "f1": 0.0, "cohen_kappa": 0.0,
        }

    accuracy = round((tp + tn) / n, 4)
    sensitivity = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    f1 = (
        round(2 * precision * sensitivity / (precision + sensitivity), 4)
        if (precision + sensitivity) > 0 else 0.0
    )

    # Cohen's kappa
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    kappa = round((po - pe) / (1 - pe), 4) if (1 - pe) != 0 else 1.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "cohen_kappa": kappa,
    }


def evaluate_decisions(gold_labels, predicted_labels):
    """Compare gold vs predicted INCLUDE/EXCLUDE decisions.

    Args:
        gold_labels: dict mapping pmid -> "INCLUDE" | "EXCLUDE"
        predicted_labels: dict mapping pmid -> "INCLUDE" | "EXCLUDE"

    Returns:
        (metrics_dict, rows_list)
    """
    tp = fp = fn = tn = 0
    rows = []

    for pmid, gold in gold_labels.items():
        pred = predicted_labels.get(pmid, "EXCLUDE")  # missing = EXCLUDE (conservative)

        if gold == "INCLUDE" and pred == "INCLUDE":
            tp += 1
        elif gold == "EXCLUDE" and pred == "INCLUDE":
            fp += 1
        elif gold == "INCLUDE" and pred == "EXCLUDE":
            fn += 1
        elif gold == "EXCLUDE" and pred == "EXCLUDE":
            tn += 1

        rows.append({
            "pmid": pmid,
            "gold": gold,
            "predicted": pred,
            "correct": gold == pred,
        })

    metrics = compute_metrics(tp, fp, fn, tn)
    return metrics, rows


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table
# ══════════════════════════════════════════════════════════════════════════════

def generate_latex_table(a3_metrics, a3a4_metrics):
    """Generate a LaTeX comparison table for A3 vs A3+A4."""
    lines = []
    lines.append(r"\begin{table}[h]\centering")
    lines.append(r"\begin{tabular}{lrr}\hline")
    lines.append(r"Metric & A3 Only & A3 + A4 \\")
    lines.append(r"\hline")
    for metric in ["accuracy", "sensitivity", "specificity", "precision", "f1", "cohen_kappa"]:
        label = metric.replace("_", " ").title()
        v_a3 = a3_metrics.get(metric, 0)
        v_a3a4 = a3a4_metrics.get(metric, 0)
        lines.append(f"{label} & {v_a3:.4f} & {v_a3a4:.4f} \\\\")
    lines.append(r"\hline")
    for cm in ["tp", "fp", "fn", "tn"]:
        label = cm.upper()
        lines.append(f"{label} & {a3_metrics.get(cm, 0)} & {a3a4_metrics.get(cm, 0)} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Exp 8 --- Gold Standard Evaluation: A3 vs A3+A4}\label{tab:exp8}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline evaluation
# ══════════════════════════════════════════════════════════════════════════════

def load_gold_standard():
    """Load and validate gold_standard.json."""
    if not GOLD_STANDARD_PATH.exists():
        return None

    with open(GOLD_STANDARD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data.get("papers", [])
    if not papers:
        print("  ERROR: gold_standard.json has no papers.")
        return None

    # Validate structure
    for p in papers:
        if "pmid" not in p:
            print(f"  ERROR: Paper missing 'pmid': {p}")
            return None
        if p.get("gold_decision") not in ("INCLUDE", "EXCLUDE"):
            print(f"  ERROR: Invalid gold_decision for PMID {p['pmid']}: {p.get('gold_decision')}")
            return None

    return data


def evaluate_a3(papers, protocol):
    """Run A3 screening and compare to gold labels."""
    import gc
    import torch
    from agents.agent3_screening import Agent3Screening

    gold_labels = {
        str(p["pmid"]): p["gold_decision"] for p in papers
    }

    # Prepare papers in the format A3 expects
    a3_input = []
    for p in papers:
        a3_input.append({
            "pmid": str(p["pmid"]),
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "authors": p.get("authors", []),
            "journal": p.get("journal", ""),
            "year": p.get("year", ""),
        })

    print(f"  Running A3 screening on {len(a3_input)} papers...")
    a3 = Agent3Screening()
    results = a3.screen_batch(a3_input, protocol)

    # Free VRAM
    if hasattr(a3, "model") and a3.model is not None:
        del a3.model
        a3.model = None
    if hasattr(a3, "tokenizer") and a3.tokenizer is not None:
        del a3.tokenizer
        a3.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

    # Collect predictions: UNCERTAIN -> EXCLUDE (conservative, Cochrane standard)
    predicted = {}
    all_screened = results.get("include", []) + results.get("exclude", []) + results.get("uncertain", [])
    for paper in all_screened:
        pmid = str(paper.get("pmid", ""))
        decision = paper.get("decision", "EXCLUDE").upper()
        predicted[pmid] = "INCLUDE" if decision == "INCLUDE" else "EXCLUDE"

    metrics, rows = evaluate_decisions(gold_labels, predicted)
    return metrics, rows, results


def evaluate_a3a4(papers, protocol, a3_results):
    """Run A4 full-text screening on A3 included papers and compare to gold."""
    import gc
    import torch
    from agents.agent4_fulltext import Agent4FullText

    gold_labels = {
        str(p["pmid"]): p["gold_decision"] for p in papers
    }

    # A3 excluded papers stay excluded
    a3_excluded_pmids = {
        str(p.get("pmid", "")) for p in a3_results.get("exclude", [])
    }
    a3_uncertain_pmids = {
        str(p.get("pmid", "")) for p in a3_results.get("uncertain", [])
    }

    # Papers that passed A3
    a3_included = a3_results.get("include", [])

    if not a3_included:
        print("  No papers passed A3 -- A4 evaluation skipped.")
        # All predictions are EXCLUDE
        predicted = {str(p["pmid"]): "EXCLUDE" for p in papers}
        metrics, rows = evaluate_decisions(gold_labels, predicted)
        return metrics, rows

    print(f"  Running A4 on {len(a3_included)} A3-included papers...")
    a4 = Agent4FullText()
    a4_results = a4.process_papers(a3_included, protocol)

    # Free VRAM
    if hasattr(a4, "model") and a4.model is not None:
        del a4.model
        a4.model = None
    if hasattr(a4, "tokenizer") and a4.tokenizer is not None:
        del a4.tokenizer
        a4.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

    # Combine A3 + A4 decisions
    predicted = {}

    # A3 excluded -> EXCLUDE
    for pmid in a3_excluded_pmids | a3_uncertain_pmids:
        predicted[pmid] = "EXCLUDE"

    # A4 decisions
    for p in a4_results.get("include", []):
        predicted[str(p.get("pmid", ""))] = "INCLUDE"
    for p in a4_results.get("exclude", []):
        predicted[str(p.get("pmid", ""))] = "EXCLUDE"

    # Any paper not in predicted -> EXCLUDE
    for p in papers:
        pmid = str(p["pmid"])
        if pmid not in predicted:
            predicted[pmid] = "EXCLUDE"

    metrics, rows = evaluate_decisions(gold_labels, predicted)
    return metrics, rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run(dry_run=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Exp 8] Gold Standard Evaluation: A3 and A3+A4")
    print("=" * 55)

    # Check for gold standard file
    if not GOLD_STANDARD_PATH.exists():
        print(f"\n  BLOCKER: {GOLD_STANDARD_PATH} not found.")
        print("  This experiment requires a human-annotated gold standard file.")
        print("  Expected format:")
        print('  {')
        print('    "query": "research question",')
        print('    "protocol": { "framework": "PICO", ... },')
        print('    "papers": [')
        print('      { "pmid": "12345678", "title": "...", "abstract": "...",')
        print('        "gold_decision": "INCLUDE", "gold_reason": "" },')
        print('      ...')
        print('    ]')
        print('  }')
        print("\n  Create data/gold_standard.json and re-run this experiment.")

        # Save blocker result
        blocker = {
            "experiment": "exp8_gold_standard_eval",
            "status": "BLOCKED",
            "reason": f"{GOLD_STANDARD_PATH} not found",
            "instructions": (
                "Create data/gold_standard.json with human-annotated papers. "
                "Each paper needs: pmid, title, abstract, gold_decision (INCLUDE/EXCLUDE)."
            ),
        }
        out_path = RESULTS_DIR / "exp8_results.json"
        with open(out_path, "w") as f:
            json.dump(blocker, f, indent=2, ensure_ascii=False)
        print(f"\n  Blocker saved -> {out_path}")
        return blocker

    # Load gold standard
    gold_data = load_gold_standard()
    if gold_data is None:
        print("  ERROR: Failed to load gold standard.")
        return None

    papers = gold_data["papers"]
    protocol = gold_data.get("protocol", {})
    n_include = sum(1 for p in papers if p["gold_decision"] == "INCLUDE")
    n_exclude = sum(1 for p in papers if p["gold_decision"] == "EXCLUDE")
    print(f"  Gold standard: {n_include} INCLUDE + {n_exclude} EXCLUDE = {len(papers)} papers")
    print(f"  Query: {gold_data.get('query', 'N/A')}")

    if dry_run:
        print("  [DRY RUN] Generating mock results...")
        mock_a3 = compute_metrics(n_include, 0, 0, n_exclude)
        mock_a3a4 = compute_metrics(n_include, 0, 0, n_exclude)
        output = {
            "experiment": "exp8_gold_standard_eval",
            "status": "DRY_RUN",
            "n_papers": len(papers),
            "n_include": n_include,
            "n_exclude": n_exclude,
            "a3_only": mock_a3,
            "a3_plus_a4": mock_a3a4,
            "latex_table": generate_latex_table(mock_a3, mock_a3a4),
        }
        out_path = RESULTS_DIR / "exp8_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  Results -> {out_path}")
        return output

    # Evaluate A3
    print("\n  --- A3 Only ---")
    a3_metrics, a3_rows, a3_results = evaluate_a3(papers, protocol)
    _print_metrics("A3", a3_metrics, a3_rows)

    # Evaluate A3 + A4
    print("\n  --- A3 + A4 ---")
    a3a4_metrics, a3a4_rows = evaluate_a3a4(papers, protocol, a3_results)
    _print_metrics("A3+A4", a3a4_metrics, a3a4_rows)

    # Comparison
    latex = generate_latex_table(a3_metrics, a3a4_metrics)

    output = {
        "experiment": "exp8_gold_standard_eval",
        "status": "COMPLETED",
        "query": gold_data.get("query", ""),
        "protocol": protocol,
        "n_papers": len(papers),
        "n_include": n_include,
        "n_exclude": n_exclude,
        "a3_only": {**a3_metrics, "rows": a3_rows},
        "a3_plus_a4": {**a3a4_metrics, "rows": a3a4_rows},
        "latex_table": latex,
    }

    out_path = RESULTS_DIR / "exp8_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  LaTeX table:\n{latex}")
    print(f"\n  Results -> {out_path}")
    return output


def _print_metrics(label, metrics, rows):
    """Print confusion matrix and metrics to console."""
    n_correct = sum(1 for r in rows if r["correct"])
    n_wrong = sum(1 for r in rows if not r["correct"])

    for r in rows:
        icon = "OK" if r["correct"] else "XX"
        print(f"    [{icon}] PMID {r['pmid']}: gold={r['gold']} pred={r['predicted']}")

    print(f"\n  [{label}] Confusion: TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TN={metrics['tn']}")
    print(f"  [{label}] Accuracy   : {metrics['accuracy']:.1%}")
    print(f"  [{label}] Sensitivity: {metrics['sensitivity']:.1%}  (goal >75%)")
    print(f"  [{label}] Specificity: {metrics['specificity']:.1%}  (goal >80%)")
    print(f"  [{label}] Precision  : {metrics['precision']:.1%}")
    print(f"  [{label}] F1-score   : {metrics['f1']:.1%}")
    print(f"  [{label}] Cohen Kappa: {metrics['cohen_kappa']:.3f}  (goal >0.60)")
    print(f"  [{label}] Correct: {n_correct}/{len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 8 - Gold Standard Evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Mock run without GPU")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
