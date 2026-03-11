"""
Exp 3 — Ensemble Size Ablation (A3)
Mide: %INCLUDE, %UNCERTAIN, s/paper, estabilidad (std entre runs) vs ensemble_size.
Justifica el hiperparámetro ensemble_size=3.
Requiere: outputs/pipeline_a1_a2_a3.json o pipeline_a1_a2_a3_a4.json
"""
import sys, json, gc, statistics, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.agent2_search import Agent2Search
from agents.agent3_screening import Agent3Screening

ENSEMBLE_SIZES = [1, 3, 5]
N_RUNS = 2  # repetitions per size for stability measurement
MAX_RESULTS = 30
DEDUP_THR = 0.90

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


def find_pipeline_output():
    """Find the most recent pipeline output."""
    candidates = [
        Path("outputs/pipeline_a1_a2_a3_a4.json"),
        Path("outputs/pipeline_a1_a2_a3.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def stats_from_results(results: dict) -> dict:
    s = results.get("stats", {})
    total = max(s.get("total", 1), 1)
    return {
        "n_include": s.get("included", 0),
        "n_exclude": s.get("excluded", 0),
        "n_uncertain": s.get("uncertain", 0),
        "include_pct": round(s.get("included", 0) / total * 100, 1),
        "exclude_pct": round(s.get("excluded", 0) / total * 100, 1),
        "uncertain_pct": round(s.get("uncertain", 0) / total * 100, 1),
        "elapsed_s": s.get("elapsed_s", 0),
        "s_per_paper": s.get("s_per_paper", 0),
    }


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Exp 3] Ensemble Size Ablation")
    print("=" * 55)

    pipeline_path = find_pipeline_output()
    if not pipeline_path:
        print("  ERROR: No pipeline output found. Run test_pipeline_complete.py first.")
        sys.exit(1)

    with open(pipeline_path) as f:
        prev = json.load(f)

    pubmed_query = prev.get("pubmed_query", "")
    protocol = prev.get("protocol", {})
    print(f"  Query    : {pubmed_query}")
    print(f"  Framework: {protocol.get('framework', '?')}")
    print(f"  Sizes    : {ENSEMBLE_SIZES}")
    print(f"  Runs/size: {N_RUNS}")

    # Re-fetch papers
    print("\n  Fetching papers...")
    a2 = Agent2Search()
    papers = a2.search_pubmed(pubmed_query, max_results=MAX_RESULTS)
    papers = a2.deduplicate(papers, threshold=DEDUP_THR)
    print(f"  Papers: {len(papers)}")

    if not papers:
        print("  ERROR: 0 papers.")
        sys.exit(1)

    # Ablation
    experiment_results = []

    for size in ENSEMBLE_SIZES:
        print(f"\n  -- ensemble_size={size} {'='*35}")
        run_stats = []

        for i in range(N_RUNS):
            a3 = Agent3Screening(ensemble_size=size, voting_threshold=0.60)
            results = a3.screen_batch(papers, protocol)
            free_vram(a3)

            s = stats_from_results(results)
            run_stats.append(s)
            print(f"    Run {i+1}: INC={s['n_include']} UNC={s['n_uncertain']} "
                  f"EXC={s['n_exclude']} ({s['s_per_paper']}s/paper)")

        inc_vals = [s["include_pct"] for s in run_stats]
        unc_vals = [s["uncertain_pct"] for s in run_stats]
        std_inc = round(statistics.stdev(inc_vals), 2) if len(inc_vals) > 1 else 0.0
        std_unc = round(statistics.stdev(unc_vals), 2) if len(unc_vals) > 1 else 0.0

        experiment_results.append({
            "ensemble_size": size,
            "mean_include_pct": round(statistics.mean(inc_vals), 1),
            "mean_uncertain_pct": round(statistics.mean(unc_vals), 1),
            "mean_s_per_paper": round(statistics.mean([s["s_per_paper"] for s in run_stats]), 2),
            "stability_std_include": std_inc,
            "stability_std_uncertain": std_unc,
            "runs": run_stats,
        })

    # Summary table
    print(f"\n  {'Size':>6}  {'%INC':>6}  {'%UNC':>6}  {'s/paper':>8}  {'std(%INC)':>10}")
    for r in experiment_results:
        print(f"  {r['ensemble_size']:>6}  {r['mean_include_pct']:>5}%  "
              f"{r['mean_uncertain_pct']:>5}%  {r['mean_s_per_paper']:>7}s  "
              f"  +/-{r['stability_std_include']:>5}%")

    output = {
        "experiment": "exp3_ensemble_ablation",
        "pubmed_query": pubmed_query,
        "framework": protocol.get("framework"),
        "n_papers": len(papers),
        "ensemble_sizes": ENSEMBLE_SIZES,
        "n_runs": N_RUNS,
        "results": experiment_results,
    }
    out_path = RESULTS_DIR / "exp3_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results -> {out_path}")
    return output


if __name__ == "__main__":
    run()
