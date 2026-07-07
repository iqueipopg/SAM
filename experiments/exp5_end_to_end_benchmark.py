# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Exp 5 — End-to-End Pipeline Benchmark (A1->A2->A3->A4)
Mide: PRISMA flow, retrieval rates, timing breakdown para MULTIPLES queries.
Demuestra que el sistema generaliza a cualquier pregunta de investigacion.
Includes VRAM profiling per agent (torch.cuda.memory_allocated).
"""
import sys, json, gc, time, torch, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.agent1_protocol import Agent1Protocol
from agents.agent2_search import Agent2Search
from agents.agent3_screening import Agent3Screening
from agents.agent4_fulltext import Agent4FullText

# ── Benchmark queries (diverse domains + frameworks) ──────────────────────────
BENCHMARK_QUERIES = [
    {
        "query": "¿TCC versus medicación para depresión en adultos?",
        "expected_framework": "PICO",
        "domain": "psychiatry",
    },
    {
        "query": "Does exercise improve cognitive function in elderly with dementia?",
        "expected_framework": "PICO",
        "domain": "neurology",
    },
    {
        "query": "What is the prevalence of burnout among medical residents worldwide?",
        "expected_framework": "PCC",
        "domain": "occupational_health",
    },
]

MAX_RESULTS = 20  # smaller to keep experiment tractable
DEDUP_THR = 0.90
ENSEMBLE_A3 = 3
ENSEMBLE_A4 = 1
VOTE_THR = 0.60

RESULTS_DIR = Path("experiments/results")


def get_vram_mb():
    """Return current GPU memory allocated in MB, or 0 if no GPU."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
    return 0


def free_vram(agent):
    if hasattr(agent, "model") and agent.model is not None:
        del agent.model
        agent.model = None
    if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
        del agent.tokenizer
        agent.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()


def run_single_query(query_cfg):
    """Run the full pipeline for one query, return metrics with VRAM profiling."""
    user_query = query_cfg["query"]
    print(f"\n  Query: {user_query}")
    timings = {}
    vram_profile = {}

    # A1
    vram_before = get_vram_mb()
    t0 = time.time()
    a1 = Agent1Protocol()
    protocol = a1.generate_protocol(user_query)
    pubmed_query = a1.generate_pubmed_query(user_query, protocol)
    vram_peak = get_vram_mb()
    free_vram(a1)
    vram_after = get_vram_mb()
    timings["a1_s"] = round(time.time() - t0, 1)
    vram_profile["a1"] = {"vram_before_mb": vram_before, "vram_peak_mb": vram_peak, "vram_after_free_mb": vram_after}
    print(f"    A1: framework={protocol.get('framework', '?')}, query='{pubmed_query[:60]}' "
          f"[VRAM peak={vram_peak}MB]")

    # A2 (CPU only)
    vram_before = get_vram_mb()
    t0 = time.time()
    a2 = Agent2Search()
    papers = a2.search_pubmed(pubmed_query, max_results=MAX_RESULTS)
    papers = a2.deduplicate(papers, threshold=DEDUP_THR)
    timings["a2_s"] = round(time.time() - t0, 1)
    vram_profile["a2"] = {"vram_before_mb": vram_before, "vram_peak_mb": get_vram_mb(), "vram_after_free_mb": get_vram_mb()}
    print(f"    A2: {len(papers)} papers")

    if not papers:
        return {"query": user_query, "error": "0 papers from PubMed", "timings": timings, "vram_profile": vram_profile}

    # A3
    vram_before = get_vram_mb()
    t0 = time.time()
    a3 = Agent3Screening(ensemble_size=ENSEMBLE_A3, voting_threshold=VOTE_THR)
    a3_results = a3.screen_batch(papers, protocol)
    vram_peak = get_vram_mb()
    free_vram(a3)
    vram_after = get_vram_mb()
    timings["a3_s"] = round(time.time() - t0, 1)
    vram_profile["a3"] = {"vram_before_mb": vram_before, "vram_peak_mb": vram_peak, "vram_after_free_mb": vram_after}
    s3 = a3_results["stats"]
    print(f"    A3: INC={s3['included']} EXC={s3['excluded']} UNC={s3['uncertain']} ({s3['elapsed_s']}s) "
          f"[VRAM peak={vram_peak}MB]")

    # A4
    included = a3_results["include"]
    a4_stats = None
    if included:
        vram_before = get_vram_mb()
        t0 = time.time()
        a4 = Agent4FullText(ensemble_size=ENSEMBLE_A4, voting_threshold=VOTE_THR)
        a4_result = a4.run(included, papers, protocol)
        vram_peak = get_vram_mb()
        free_vram(a4)
        vram_after = get_vram_mb()
        timings["a4_s"] = round(time.time() - t0, 1)
        vram_profile["a4"] = {"vram_before_mb": vram_before, "vram_peak_mb": vram_peak, "vram_after_free_mb": vram_after}
        a4_stats = a4_result["stats"]
        print(f"    A4: INC={a4_stats['fulltext_included']} EXC={a4_stats['fulltext_excluded']} "
              f"(PMC={a4_stats['pmc_fulltext']}, UPW={a4_stats['unpaywall_pdf']}, ABS={a4_stats['abstract_only']}) "
              f"[VRAM peak={vram_peak}MB]")
    else:
        timings["a4_s"] = 0
        vram_profile["a4"] = {"vram_before_mb": 0, "vram_peak_mb": 0, "vram_after_free_mb": 0}
        print("    A4: skipped (0 papers from A3)")

    total_time = sum(timings.values())

    return {
        "query": user_query,
        "domain": query_cfg["domain"],
        "expected_framework": query_cfg["expected_framework"],
        "detected_framework": protocol.get("framework", "?"),
        "framework_correct": protocol.get("framework", "?").upper() == query_cfg["expected_framework"].upper(),
        "pubmed_query": pubmed_query,
        "prisma": {
            "identified": s3["total"],
            "excluded_ta": s3["excluded"],
            "uncertain": s3["uncertain"],
            "assessed_fulltext": s3["included"],
            "excluded_fulltext": a4_stats["fulltext_excluded"] if a4_stats else 0,
            "included_final": a4_stats["fulltext_included"] if a4_stats else 0,
        },
        "retrieval": {
            "pmc_fulltext": a4_stats["pmc_fulltext"] if a4_stats else 0,
            "unpaywall_pdf": a4_stats["unpaywall_pdf"] if a4_stats else 0,
            "abstract_only": a4_stats["abstract_only"] if a4_stats else 0,
        },
        "timings": timings,
        "total_time_s": round(total_time, 1),
        "vram_profile": vram_profile,
    }


def generate_latex_table(results):
    """Generate a LaTeX table from experiment results."""
    lines = []
    lines.append(r"\begin{table}[h]\centering")
    lines.append(r"\begin{tabular}{lllrrrr}\hline")
    lines.append(r"Query & Domain & FW & Identified & Included & Time (s) & VRAM Peak (MB) \\")
    lines.append(r"\hline")
    for r in results.get("results", []):
        if "error" in r:
            q = r["query"][:30].replace("&", r"\&").replace("%", r"\%")
            lines.append(f"{q} & -- & -- & \\multicolumn{{4}}{{c}}{{ERROR}} \\\\")
            continue
        q = r["query"][:30].replace("&", r"\&").replace("%", r"\%")
        domain = r.get("domain", "")
        fw = r.get("detected_framework", "?")
        p = r["prisma"]
        vp = r.get("vram_profile", {})
        max_vram = max((v.get("vram_peak_mb", 0) for v in vp.values()), default=0)
        lines.append(
            f"{q} & {domain} & {fw} & {p['identified']} & "
            f"{p['included_final']} & {r['total_time_s']} & {max_vram} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Resultados Experimento 5 --- End-to-End Benchmark}\label{tab:exp5}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def run(dry_run=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Exp 5] End-to-End Pipeline Benchmark")
    print("=" * 55)
    print(f"  Queries: {len(BENCHMARK_QUERIES)}")
    print(f"  Config : max_results={MAX_RESULTS}, ensemble_a3={ENSEMBLE_A3}, ensemble_a4={ENSEMBLE_A4}")

    if dry_run:
        print("  [DRY RUN] Generating mock results...")
        all_results = []
        for cfg in BENCHMARK_QUERIES:
            all_results.append({
                "query": cfg["query"],
                "domain": cfg["domain"],
                "expected_framework": cfg["expected_framework"],
                "detected_framework": cfg["expected_framework"],
                "framework_correct": True,
                "pubmed_query": "mock query",
                "prisma": {
                    "identified": 20, "excluded_ta": 10, "uncertain": 2,
                    "assessed_fulltext": 8, "excluded_fulltext": 3, "included_final": 5,
                },
                "retrieval": {"pmc_fulltext": 3, "unpaywall_pdf": 1, "abstract_only": 4},
                "timings": {"a1_s": 0, "a2_s": 0, "a3_s": 0, "a4_s": 0},
                "total_time_s": 0,
                "vram_profile": {
                    "a1": {"vram_before_mb": 0, "vram_peak_mb": 0, "vram_after_free_mb": 0},
                    "a2": {"vram_before_mb": 0, "vram_peak_mb": 0, "vram_after_free_mb": 0},
                    "a3": {"vram_before_mb": 0, "vram_peak_mb": 0, "vram_after_free_mb": 0},
                    "a4": {"vram_before_mb": 0, "vram_peak_mb": 0, "vram_after_free_mb": 0},
                },
            })
        output = {
            "experiment": "exp5_end_to_end_benchmark",
            "config": {
                "max_results": MAX_RESULTS, "dedup_threshold": DEDUP_THR,
                "ensemble_a3": ENSEMBLE_A3, "ensemble_a4": ENSEMBLE_A4,
            },
            "n_queries": len(BENCHMARK_QUERIES),
            "results": all_results,
        }
        latex = generate_latex_table(output)
        output["latex_table"] = latex
        print(latex)
        out_path = RESULTS_DIR / "exp5_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Results -> {out_path}")
        return output

    all_results = []
    for cfg in BENCHMARK_QUERIES:
        try:
            result = run_single_query(cfg)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR with '{cfg['query'][:40]}': {e}")
            all_results.append({"query": cfg["query"], "error": str(e)})

    # Summary
    print(f"\n  {'Query':<45} {'FW':>5} {'Found':>6} {'INC':>4} {'EXC':>4} {'Final':>6} {'Time':>6}")
    print(f"  {'-'*45} {'-'*5} {'-'*6} {'-'*4} {'-'*4} {'-'*6} {'-'*6}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['query'][:45]:<45} ERROR")
            continue
        p = r["prisma"]
        fw_ok = "OK" if r["framework_correct"] else "XX"
        print(f"  {r['query'][:45]:<45} {fw_ok:>5} {p['identified']:>6} "
              f"{p['assessed_fulltext']:>4} {p['excluded_fulltext']:>4} "
              f"{p['included_final']:>6} {r['total_time_s']:>5}s")

    output = {
        "experiment": "exp5_end_to_end_benchmark",
        "config": {
            "max_results": MAX_RESULTS, "dedup_threshold": DEDUP_THR,
            "ensemble_a3": ENSEMBLE_A3, "ensemble_a4": ENSEMBLE_A4,
        },
        "n_queries": len(BENCHMARK_QUERIES),
        "results": all_results,
    }

    latex = generate_latex_table(output)
    output["latex_table"] = latex
    print(f"\n  LaTeX table:\n{latex}")

    out_path = RESULTS_DIR / "exp5_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results -> {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 5 - End-to-End Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Mock run without GPU")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
