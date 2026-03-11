"""
Exp 4 — Model Comparison for Screening (A3)
Mide: Sensitivity, Specificity, F1, Kappa para DIFERENTES modelos LLM.
Usa el mismo gold standard de Exp 2 para comparar modelos side-by-side.

Diseñado para Rudiger: probar Qwen-7B, Qwen-14B, Qwen-72B, etc.
En local: solo ejecuta Qwen-7B (el único que cabe en 8GB VRAM).

Para añadir modelos, edita MODEL_CONFIGS abajo.
"""
import sys, json, gc, time, torch, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from Bio import Entrez
from agents.agent3_screening import Agent3Screening

Entrez.email = "meta-analysis@example.com"

# ── Model configs to compare ─────────────────────────────────────────────────
# Each entry overrides config.yaml's agent3_screening model path.
# Add more models as available (especially on Rudiger).
MODEL_CONFIGS = [
    {
        "name": "Qwen2.5-7B-4bit",
        "model_path": "./models/qwen2.5-7b",
        "quantization": "4bit",
    },
    # Uncomment when running on Rudiger with more VRAM:
    # {
    #     "name": "Qwen2.5-14B-4bit",
    #     "model_path": "./models/qwen2.5-14b",
    #     "quantization": "4bit",
    # },
    # {
    #     "name": "Qwen2.5-72B-4bit",
    #     "model_path": "./models/qwen2.5-72b",
    #     "quantization": "4bit",
    # },
    # {
    #     "name": "Qwen2.5-7B-fp16",
    #     "model_path": "./models/qwen2.5-7b",
    #     "quantization": None,  # No quantization, needs ~14GB VRAM
    # },
]

# ── Same gold standard as Exp 2 ──────────────────────────────────────────────
GOLD_INCLUDE = [
    "41135204", "40930184", "40379363", "41494488",
    "40739549", "41284534", "41490574", "41309552",
]
GOLD_EXCLUDE = [
    "41650725", "41758545", "41574602", "41730329",
    "41711297", "41691290", "41773906", "41574023",
    "41521955", "41366465",
]
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


def cohen_kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    return round((po - pe) / (1 - pe), 4) if (1 - pe) != 0 else 1.0


def evaluate_model(model_cfg, papers):
    """Run screening with a specific model and return metrics."""
    name = model_cfg["name"]
    print(f"\n  -- {name} {'='*40}")

    # Create agent and override model path
    a3 = Agent3Screening(ensemble_size=3, voting_threshold=0.60)
    a3.model_path = model_cfg["model_path"]
    if model_cfg.get("quantization") is None:
        a3.quantization = None

    results = a3.screen_batch(papers, FIXED_PROTOCOL)
    free_vram(a3)

    all_screened = results["include"] + results["exclude"] + results["uncertain"]

    tp = fp = fn = tn = 0
    rows = []
    for paper in all_screened:
        pmid = str(paper.get("pmid", ""))
        predicted = paper.get("decision", "UNCERTAIN").upper()
        gold = GOLD_LABELS.get(pmid, "UNKNOWN")
        if gold == "UNKNOWN":
            continue
        pred_bin = "INCLUDE" if predicted == "INCLUDE" else "EXCLUDE"

        if gold == "INCLUDE" and pred_bin == "INCLUDE": tp += 1
        elif gold == "EXCLUDE" and pred_bin == "INCLUDE": fp += 1
        elif gold == "INCLUDE" and pred_bin == "EXCLUDE": fn += 1
        elif gold == "EXCLUDE" and pred_bin == "EXCLUDE": tn += 1
        rows.append({"pmid": pmid, "gold": gold, "predicted": predicted, "pred_binary": pred_bin})

    sensitivity = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
    f1 = round(2 * precision * sensitivity / (precision + sensitivity), 4) if (precision + sensitivity) > 0 else 0
    kappa = cohen_kappa(tp, fp, fn, tn)

    elapsed = results["stats"].get("elapsed_s", 0)
    s_per_paper = results["stats"].get("s_per_paper", 0)

    print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"    Sens={sensitivity:.1%} Spec={specificity:.1%} F1={f1:.1%} Kappa={kappa:.3f}")
    print(f"    Time: {elapsed}s ({s_per_paper}s/paper)")

    return {
        "model": name,
        "model_path": model_cfg["model_path"],
        "quantization": model_cfg.get("quantization", "4bit"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1, "cohen_kappa": kappa,
        "elapsed_s": elapsed, "s_per_paper": s_per_paper,
        "rows": rows,
    }


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Exp 4] Model Comparison for Screening")
    print("=" * 55)
    print(f"  Models to test: {len(MODEL_CONFIGS)}")
    print(f"  Gold standard : {len(GOLD_LABELS)} papers")

    # Fetch papers once
    print("  Fetching papers...")
    papers = fetch_papers_by_pmids(list(GOLD_LABELS.keys()))
    print(f"  Retrieved: {len(papers)}/{len(GOLD_LABELS)}")

    # Evaluate each model
    model_results = []
    for cfg in MODEL_CONFIGS:
        try:
            result = evaluate_model(cfg, papers)
            model_results.append(result)
        except Exception as e:
            print(f"  ERROR with {cfg['name']}: {e}")
            model_results.append({"model": cfg["name"], "error": str(e)})

    # Summary table
    print(f"\n  {'Model':<25} {'Sens':>6} {'Spec':>6} {'F1':>6} {'Kappa':>7} {'s/paper':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*8}")
    for r in model_results:
        if "error" in r:
            print(f"  {r['model']:<25} ERROR: {r['error'][:40]}")
        else:
            print(f"  {r['model']:<25} {r['sensitivity']:>5.1%} {r['specificity']:>5.1%} "
                  f"{r['f1']:>5.1%} {r['cohen_kappa']:>6.3f} {r['s_per_paper']:>7.1f}s")

    output = {
        "experiment": "exp4_model_comparison",
        "n_papers": len(papers),
        "gold_include": len(GOLD_INCLUDE),
        "gold_exclude": len(GOLD_EXCLUDE),
        "protocol": FIXED_PROTOCOL,
        "models": model_results,
    }
    out_path = RESULTS_DIR / "exp4_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results -> {out_path}")
    return output


if __name__ == "__main__":
    run()
