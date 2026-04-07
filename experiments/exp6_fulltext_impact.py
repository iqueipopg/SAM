"""
Exp 6 — Full-text Screening Impact (A4 value-add)
Mide: Cuántos falsos positivos elimina A4 que A3 dejó pasar.
Compara pipeline con A4 vs sin A4 (solo T/A screening).
Usa el gold standard de Exp 2 + papers que A3 incluye incorrectamente.
"""
import sys, json, gc, time, torch, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from Bio import Entrez

from agents.agent3_screening import Agent3Screening
from agents.agent4_fulltext import Agent4FullText

Entrez.email = "meta-analysis@example.com"

# ── Gold standard: same as Exp 2 ─────────────────────────────────────────────
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


def compute_metrics(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    sensitivity = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
    f1 = round(2 * precision * sensitivity / (precision + sensitivity), 4) if (precision + sensitivity) > 0 else 0
    accuracy = round((tp + tn) / n, 4) if n > 0 else 0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy, "sensitivity": sensitivity,
        "specificity": specificity, "precision": precision, "f1": f1,
    }


def run(dry_run=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Exp 6] Full-text Screening Impact")
    print("=" * 55)
    print(f"  Gold: {len(GOLD_INCLUDE)} INCLUDE + {len(GOLD_EXCLUDE)} EXCLUDE")

    if dry_run:
        print("  [DRY RUN] Generating mock results...")
        metrics_a3 = {"tp": 7, "fp": 2, "fn": 1, "tn": 8, "accuracy": 0.8333,
                      "sensitivity": 0.875, "specificity": 0.8, "precision": 0.7778, "f1": 0.8235}
        metrics_a3a4 = {"tp": 7, "fp": 0, "fn": 1, "tn": 10, "accuracy": 0.9444,
                        "sensitivity": 0.875, "specificity": 1.0, "precision": 1.0, "f1": 0.9333}
        output = {
            "experiment": "exp6_fulltext_impact",
            "n_papers": len(GOLD_LABELS),
            "a3_only": metrics_a3,
            "a3_plus_a4": metrics_a3a4,
            "impact": {"fp_reduced": 2, "fn_added": 0, "specificity_delta": 0.2, "f1_delta": 0.1098},
            "a3_rows": [], "a4_rows": [],
        }
        latex = generate_latex_table(output)
        output["latex_table"] = latex
        print(latex)
        out_path = RESULTS_DIR / "exp6_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Results -> {out_path}")
        return output

    # Fetch papers
    print("  Fetching papers...")
    papers = fetch_papers_by_pmids(list(GOLD_LABELS.keys()))
    print(f"  Retrieved: {len(papers)}/{len(GOLD_LABELS)}")

    # ── Phase 1: A3 only (title/abstract screening) ──────────────────────────
    print("\n  --- Pipeline A: A3 only (title/abstract) ---")
    a3 = Agent3Screening(ensemble_size=3, voting_threshold=0.60)
    a3_results = a3.screen_batch(papers, FIXED_PROTOCOL)
    free_vram(a3)

    # Evaluate A3-only
    all_a3 = a3_results["include"] + a3_results["exclude"] + a3_results["uncertain"]
    tp_a3 = fp_a3 = fn_a3 = tn_a3 = 0
    a3_rows = []

    for paper in all_a3:
        pmid = str(paper.get("pmid", ""))
        gold = GOLD_LABELS.get(pmid, "UNKNOWN")
        if gold == "UNKNOWN":
            continue
        pred = "INCLUDE" if paper.get("decision") == "INCLUDE" else "EXCLUDE"
        if gold == "INCLUDE" and pred == "INCLUDE": tp_a3 += 1
        elif gold == "EXCLUDE" and pred == "INCLUDE": fp_a3 += 1
        elif gold == "INCLUDE" and pred == "EXCLUDE": fn_a3 += 1
        elif gold == "EXCLUDE" and pred == "EXCLUDE": tn_a3 += 1
        a3_rows.append({"pmid": pmid, "gold": gold, "a3_decision": paper.get("decision"), "a3_final": pred})

    metrics_a3 = compute_metrics(tp_a3, fp_a3, fn_a3, tn_a3)
    print(f"    A3-only: Sens={metrics_a3['sensitivity']:.1%} Spec={metrics_a3['specificity']:.1%} "
          f"F1={metrics_a3['f1']:.1%} (FP={fp_a3})")

    # ── Phase 2: A3 + A4 (add full-text screening) ───────────────────────────
    print("\n  --- Pipeline B: A3 + A4 (full-text) ---")
    a3_included = a3_results["include"]

    if not a3_included:
        print("    A3 included 0 papers, A4 has nothing to screen.")
        metrics_a3a4 = metrics_a3.copy()
        a4_rows = []
    else:
        a4 = Agent4FullText(ensemble_size=1, voting_threshold=0.60)
        a4_result = a4.run(a3_included, papers, FIXED_PROTOCOL)
        free_vram(a4)

        # Re-evaluate: papers that A3 included but A4 excluded are now EXCLUDE
        a4_decisions = {str(p["pmid"]): p.get("fulltext_decision", "INCLUDE") for p in a4_result["papers"]}

        tp_a4 = fp_a4 = fn_a4 = tn_a4 = 0
        a4_rows = []

        for paper in all_a3:
            pmid = str(paper.get("pmid", ""))
            gold = GOLD_LABELS.get(pmid, "UNKNOWN")
            if gold == "UNKNOWN":
                continue

            a3_dec = paper.get("decision", "EXCLUDE")
            if a3_dec == "INCLUDE":
                # A4 may have overridden
                final = a4_decisions.get(pmid, "INCLUDE")
            else:
                final = "EXCLUDE"

            if gold == "INCLUDE" and final == "INCLUDE": tp_a4 += 1
            elif gold == "EXCLUDE" and final == "INCLUDE": fp_a4 += 1
            elif gold == "INCLUDE" and final == "EXCLUDE": fn_a4 += 1
            elif gold == "EXCLUDE" and final == "EXCLUDE": tn_a4 += 1
            a4_rows.append({"pmid": pmid, "gold": gold, "a3_decision": a3_dec, "a4_decision": final})

        metrics_a3a4 = compute_metrics(tp_a4, fp_a4, fn_a4, tn_a4)
        print(f"    A3+A4:   Sens={metrics_a3a4['sensitivity']:.1%} Spec={metrics_a3a4['specificity']:.1%} "
              f"F1={metrics_a3a4['f1']:.1%} (FP={fp_a4})")

    # ── Comparison ────────────────────────────────────────────────────────────
    fp_reduced = fp_a3 - metrics_a3a4.get("fp", fp_a3)
    fn_added = metrics_a3a4.get("fn", fn_a3) - fn_a3

    print("\n  Impact of A4:")
    print(f"    False positives reduced : {fp_a3} -> {metrics_a3a4.get('fp', fp_a3)} ({fp_reduced} eliminated)")
    print(f"    False negatives added   : {fn_a3} -> {metrics_a3a4.get('fn', fn_a3)} ({fn_added} new)")
    print(f"    Specificity improvement : {metrics_a3['specificity']:.1%} -> {metrics_a3a4['specificity']:.1%}")
    print(f"    F1 change               : {metrics_a3['f1']:.1%} -> {metrics_a3a4['f1']:.1%}")

    output = {
        "experiment": "exp6_fulltext_impact",
        "n_papers": len(papers),
        "a3_only": metrics_a3,
        "a3_plus_a4": metrics_a3a4,
        "impact": {
            "fp_reduced": fp_reduced,
            "fn_added": fn_added,
            "specificity_delta": round(metrics_a3a4["specificity"] - metrics_a3["specificity"], 4),
            "f1_delta": round(metrics_a3a4["f1"] - metrics_a3["f1"], 4),
        },
        "a3_rows": a3_rows,
        "a4_rows": a4_rows,
    }

    latex = generate_latex_table(output)
    output["latex_table"] = latex
    print(f"\n  LaTeX table:\n{latex}")

    out_path = RESULTS_DIR / "exp6_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results -> {out_path}")
    return output


def generate_latex_table(results):
    """Generate a LaTeX table from experiment results."""
    lines = []
    lines.append(r"\begin{table}[h]\centering")
    lines.append(r"\begin{tabular}{lrrrrrr}\hline")
    lines.append(r"Pipeline & Sensitivity & Specificity & Precision & F1 & FP & FN \\")
    lines.append(r"\hline")
    for label, key in [("A3 only", "a3_only"), ("A3 + A4", "a3_plus_a4")]:
        m = results.get(key, {})
        lines.append(
            f"{label} & {m.get('sensitivity', 0):.4f} & {m.get('specificity', 0):.4f} & "
            f"{m.get('precision', 0):.4f} & {m.get('f1', 0):.4f} & "
            f"{m.get('fp', 0)} & {m.get('fn', 0)} \\\\"
        )
    lines.append(r"\hline")
    impact = results.get("impact", {})
    lines.append(f"$\\Delta$ & -- & {impact.get('specificity_delta', 0):+.4f} & "
                 f"-- & {impact.get('f1_delta', 0):+.4f} & "
                 f"{impact.get('fp_reduced', 0):+d} & {impact.get('fn_added', 0):+d} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Resultados Experimento 6 --- Full-text Impact}\label{tab:exp6}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 6 - Full-text Impact")
    parser.add_argument("--dry-run", action="store_true", help="Mock run without GPU")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
