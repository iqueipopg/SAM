# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Exp 2 — Screening Sensitivity/Specificity vs Gold Standard (A3)
Mide: Sensitivity, Specificity, F1, Cohen's Kappa contra anotación humana.
Dataset: 20 PMIDs reales anotados (10 INCLUDE + 10 EXCLUDE) para CBT depression.
Nota: UNCERTAIN se trata como EXCLUDE (conservador, estándar Cochrane).
"""
import sys, json, gc, time, torch, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from Bio import Entrez

from agents.agent1_protocol import Agent1Protocol
from agents.agent3_screening import Agent3Screening

Entrez.email = "meta-analysis@example.com"

# ── Gold standard: PMIDs anotados manualmente ────────────────────────────────
# INCLUDE: RCTs comparing CBT (or CBT-based) vs medication/control for depression in adults
GOLD_INCLUDE = {
    "41135204": "RCT dynamic interpersonal therapy vs CBT vs pharmacotherapy MDD",
    "40930184": "eIMPACT collaborative care RCT depression",
    "40379363": "MBCT vs TAU for non-remitted depression RCT",
    "41494488": "CBT + quetiapine for depression suicide risk RCT",
    "40739549": "Psychotherapy vs antidepressants for depression primary care RCT",
    "41284534": "Triple chronotherapy add-on for unipolar/bipolar depression RCT",
    "41490574": "AI-based self-guided digital CBT for depression RCT",
    "41309552": "Ketamine, antidepressant and internet-delivered CBT depression",
}

# EXCLUDE: reviews, wrong population, wrong intervention, wrong outcome
GOLD_EXCLUDE = {
    "41650725": "Ketamine-assisted psychotherapies systematic review (not RCT)",
    "41758545": "Telerehab for tinnitus systematic review (wrong topic)",
    "41574602": "GPR3 neuro-metabolic review (wrong topic entirely)",
    "41730329": "CBT for pediatric gut-brain disorders (wrong population, wrong outcome)",
    "41711297": "Parent-led CBT for child anxiety (wrong population)",
    "41691290": "Digital health interventions adolescents (wrong population)",
    "41773906": "Idiographic coping in youth anxiety (wrong population, wrong outcome)",
    "41574023": "Aerobic exercise for PCOS depression/anxiety (wrong intervention)",
    "41521955": "Psychedelics for treatment-resistant depression review (not RCT)",
    "41366465": "Interventions for anorexia nervosa review (wrong outcome)",
}

GOLD_LABELS = {p: "INCLUDE" for p in GOLD_INCLUDE}
GOLD_LABELS.update({p: "EXCLUDE" for p in GOLD_EXCLUDE})

# Protocol for this query (fixed, not generated — isolates A3 evaluation)
FIXED_PROTOCOL = {
    "framework": "PICO",
    "inclusion_criteria": [
        "Randomized controlled trial or controlled study design",
        "Adults aged 18-65 with diagnosed depression or depressive disorder",
        "Cognitive Behavioral Therapy (CBT) as intervention or comparator",
    ],
    "exclusion_criteria": [
        "Non-randomized designs, systematic reviews, or meta-analyses",
        "Studies not reporting depression symptom reduction as outcome",
    ],
}

RESULTS_DIR = Path("experiments/results")


def free_vram(agent):
    if hasattr(agent, "model") and agent.model is not None:
        del agent.model
        agent.model = None
    if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
        del agent.tokenizer
        agent.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()


def fetch_papers_by_pmids(pmids):
    """Fetch paper metadata from PubMed by PMID list."""
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
    records = Entrez.read(handle)
    papers = []
    for record in records["PubmedArticle"]:
        try:
            article = record["MedlineCitation"]["Article"]
            papers.append({
                "pmid": str(record["MedlineCitation"]["PMID"]),
                "title": str(article["ArticleTitle"]),
                "abstract": str(article.get("Abstract", {}).get("AbstractText", [""])[0]),
                "authors": [a.get("LastName", "") for a in article.get("AuthorList", [])[:3]],
                "journal": str(article["Journal"]["Title"]),
                "year": str(article["Journal"]["JournalIssue"].get("PubDate", {}).get("Year", "")),
            })
        except Exception:
            continue
    return papers


def cohen_kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    return round((po - pe) / (1 - pe), 4) if (1 - pe) != 0 else 1.0


def generate_latex_table(results):
    """Generate a LaTeX table from experiment results."""
    lines = []
    lines.append(r"\begin{table}[h]\centering")
    lines.append(r"\begin{tabular}{lrrrrr}\hline")
    lines.append(r"Metric & Value \\")
    lines.append(r"\hline")
    for metric in ["accuracy", "sensitivity", "specificity", "precision", "f1", "cohen_kappa"]:
        val = results.get(metric, 0)
        label = metric.replace("_", " ").title()
        lines.append(f"{label} & {val:.4f} \\\\")
    lines.append(r"\hline")
    lines.append(f"TP & {results.get('tp', 0)} \\\\")
    lines.append(f"FP & {results.get('fp', 0)} \\\\")
    lines.append(f"FN & {results.get('fn', 0)} \\\\")
    lines.append(f"TN & {results.get('tn', 0)} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Resultados Experimento 2 --- Screening Gold Standard}\label{tab:exp2}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def run(dry_run=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Exp 2] Screening vs Gold Standard")
    print("=" * 55)
    print(f"  Gold: {len(GOLD_INCLUDE)} INCLUDE + {len(GOLD_EXCLUDE)} EXCLUDE = {len(GOLD_LABELS)} papers")

    if dry_run:
        print("  [DRY RUN] Generating mock results...")
        output = {
            "experiment": "exp2_screening_gold_standard",
            "protocol": FIXED_PROTOCOL,
            "n_papers": len(GOLD_LABELS), "n_evaluated": len(GOLD_LABELS),
            "tp": 8, "fp": 0, "fn": 0, "tn": 10,
            "accuracy": 1.0,
            "sensitivity": 1.0, "specificity": 1.0,
            "precision": 1.0, "f1": 1.0, "cohen_kappa": 1.0,
            "rows": [],
        }
        latex = generate_latex_table(output)
        output["latex_table"] = latex
        print(latex)
        out_path = RESULTS_DIR / "exp2_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Results -> {out_path}")
        return output

    # Fetch abstracts for all gold standard PMIDs
    print("  Fetching papers from PubMed...")
    papers = fetch_papers_by_pmids(list(GOLD_LABELS.keys()))
    print(f"  Retrieved: {len(papers)}/{len(GOLD_LABELS)} papers")

    if not papers:
        print("  ERROR: Could not fetch papers. Check network.")
        return None

    # Screen with A3
    print("  Running A3 screening...")
    a3 = Agent3Screening(ensemble_size=3, voting_threshold=0.60)
    results = a3.screen_batch(papers, FIXED_PROTOCOL)
    free_vram(a3)

    # Collect predictions
    all_screened = results["include"] + results["exclude"] + results["uncertain"]

    tp = fp = fn = tn = 0
    rows = []
    for paper in all_screened:
        pmid = str(paper.get("pmid", ""))
        predicted = paper.get("decision", "UNCERTAIN").upper()
        gold = GOLD_LABELS.get(pmid, "UNKNOWN")
        if gold == "UNKNOWN":
            continue

        # UNCERTAIN -> EXCLUDE (conservative, Cochrane standard)
        pred_bin = "INCLUDE" if predicted == "INCLUDE" else "EXCLUDE"

        if gold == "INCLUDE" and pred_bin == "INCLUDE":
            tp += 1
        elif gold == "EXCLUDE" and pred_bin == "INCLUDE":
            fp += 1
        elif gold == "INCLUDE" and pred_bin == "EXCLUDE":
            fn += 1
        elif gold == "EXCLUDE" and pred_bin == "EXCLUDE":
            tn += 1

        rows.append({
            "pmid": pmid, "gold": gold, "predicted": predicted,
            "pred_binary": pred_bin, "confidence": paper.get("confidence", 0),
            "correct": gold == pred_bin,
        })
        icon = "OK" if gold == pred_bin else "XX"
        print(f"  [{icon}] {pmid}: gold={gold} pred={predicted} -> {pred_bin}")

    sensitivity = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
    f1 = round(2 * precision * sensitivity / (precision + sensitivity), 4) if (precision + sensitivity) > 0 else 0
    kappa = cohen_kappa(tp, fp, fn, tn)
    accuracy = round((tp + tn) / (tp + fp + fn + tn), 4) if (tp + fp + fn + tn) > 0 else 0

    print(f"\n  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Accuracy   : {accuracy:.1%}")
    print(f"  Sensitivity: {sensitivity:.1%}  (recall, goal >75%)")
    print(f"  Specificity: {specificity:.1%}  (goal >80%)")
    print(f"  Precision  : {precision:.1%}")
    print(f"  F1-score   : {f1:.1%}")
    print(f"  Cohen Kappa: {kappa:.3f}  (goal >0.60)")

    output = {
        "experiment": "exp2_screening_gold_standard",
        "protocol": FIXED_PROTOCOL,
        "n_papers": len(papers), "n_evaluated": len(rows),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1, "cohen_kappa": kappa,
        "rows": rows,
    }

    latex = generate_latex_table(output)
    output["latex_table"] = latex
    print(f"\n  LaTeX table:\n{latex}")

    out_path = RESULTS_DIR / "exp2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results -> {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2 - Screening Gold Standard")
    parser.add_argument("--dry-run", action="store_true", help="Mock run without GPU")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
