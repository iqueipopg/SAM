"""
Exp 7 — VRAM Profiling per Agent (A1-A8)
Mide: time_seconds y VRAM (peak, after_free) por cada agente en un pipeline completo.
Query: "Does mindfulness reduce anxiety in adolescents?"
Output: Markdown table a stdout + JSON con per-agent metrics.
"""
import sys, json, gc, time, torch, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

QUERY = "Does mindfulness reduce anxiety in adolescents?"
MAX_RESULTS = 15
DEDUP_THR = 0.90
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


def profile_agent(name, fn):
    """Run fn(), measuring time and VRAM. Returns (result, profile_dict).

    Uses torch.cuda.max_memory_allocated() to capture the true peak during
    fn() execution. Reset peak stats before fn() so the measurement is local
    to this agent's run, even if fn() calls free_vram() internally.
    """
    vram_before = get_vram_mb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = fn()
    elapsed = round(time.time() - t0, 2)
    if torch.cuda.is_available():
        vram_peak = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    else:
        vram_peak = 0
    return result, {
        "agent": name,
        "time_seconds": elapsed,
        "vram_peak_mb": vram_peak,
        "vram_before_mb": vram_before,
    }


def run(dry_run=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Exp 7] VRAM Profiling per Agent")
    print("=" * 55)
    print(f"  Query: {QUERY}")
    print(f"  GPU  : {'available' if torch.cuda.is_available() else 'NOT available (values will be 0)'}")

    profiles = []

    if dry_run:
        print("  [DRY RUN] Generating mock VRAM profiles...")
        agent_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
        for i, name in enumerate(agent_names):
            profiles.append({
                "agent": name,
                "time_seconds": round((i + 1) * 1.5, 2),
                "vram_peak_mb": 0,
                "vram_before_mb": 0,
                "vram_after_free_mb": 0,
            })
    else:
        # A1 - Protocol
        from agents.agent1_protocol import Agent1Protocol
        def run_a1():
            a1 = Agent1Protocol()
            protocol = a1.generate_protocol(QUERY)
            pubmed_query = a1.generate_pubmed_query(QUERY, protocol)
            free_vram(a1)
            return {"protocol": protocol, "pubmed_query": pubmed_query}

        a1_result, a1_profile = profile_agent("A1", run_a1)
        a1_profile["vram_after_free_mb"] = get_vram_mb()
        profiles.append(a1_profile)
        protocol = a1_result["protocol"]
        pubmed_query = a1_result["pubmed_query"]
        print(f"  A1: {a1_profile['time_seconds']}s, VRAM peak={a1_profile['vram_peak_mb']}MB")

        # A2 - Search (CPU)
        from agents.agent2_search import Agent2Search
        def run_a2():
            a2 = Agent2Search()
            papers = a2.search_pubmed(pubmed_query, max_results=MAX_RESULTS)
            papers = a2.deduplicate(papers, threshold=DEDUP_THR)
            return papers

        papers, a2_profile = profile_agent("A2", run_a2)
        a2_profile["vram_after_free_mb"] = get_vram_mb()
        profiles.append(a2_profile)
        print(f"  A2: {a2_profile['time_seconds']}s, {len(papers)} papers")

        if not papers:
            print("  ERROR: 0 papers. Stopping.")
            _save_and_print(profiles)
            return

        # A3 - Screening
        from agents.agent3_screening import Agent3Screening
        def run_a3():
            a3 = Agent3Screening(ensemble_size=3, voting_threshold=0.60)
            result = a3.screen_batch(papers, protocol)
            free_vram(a3)
            return result

        a3_result, a3_profile = profile_agent("A3", run_a3)
        a3_profile["vram_after_free_mb"] = get_vram_mb()
        profiles.append(a3_profile)
        included_a3 = a3_result["include"]
        print(f"  A3: {a3_profile['time_seconds']}s, VRAM peak={a3_profile['vram_peak_mb']}MB, "
              f"included={len(included_a3)}")

        # A4 - Full-text
        from agents.agent4_fulltext import Agent4FullText
        def run_a4():
            if not included_a3:
                return {"papers": [], "stats": {}}
            a4 = Agent4FullText(ensemble_size=1, voting_threshold=0.60)
            result = a4.run(included_a3, papers, protocol)
            free_vram(a4)
            return result

        a4_result, a4_profile = profile_agent("A4", run_a4)
        a4_profile["vram_after_free_mb"] = get_vram_mb()
        profiles.append(a4_profile)
        included_papers = a4_result.get("papers", [])
        print(f"  A4: {a4_profile['time_seconds']}s, VRAM peak={a4_profile['vram_peak_mb']}MB")

        # A5 - Extraction
        from agents.agent5_extraction import Agent5Extraction
        def run_a5():
            if not included_papers:
                return {"papers": [], "stats": {}}
            a5 = Agent5Extraction()
            result = a5.extract_all(included_papers, protocol)
            free_vram(a5)
            return result

        extractions, a5_profile = profile_agent("A5", run_a5)
        a5_profile["vram_after_free_mb"] = get_vram_mb()
        profiles.append(a5_profile)
        print(f"  A5: {a5_profile['time_seconds']}s, VRAM peak={a5_profile['vram_peak_mb']}MB")

        # A6 - Quality
        from agents.agent6_quality import Agent6Quality
        def run_a6():
            if not extractions or not extractions.get("papers"):
                return {"papers": [], "stats": {}}
            a6 = Agent6Quality()
            result = a6.assess_all(extractions, protocol)
            free_vram(a6)
            return result

        quality, a6_profile = profile_agent("A6", run_a6)
        a6_profile["vram_after_free_mb"] = get_vram_mb()
        profiles.append(a6_profile)
        print(f"  A6: {a6_profile['time_seconds']}s, VRAM peak={a6_profile['vram_peak_mb']}MB")

        # A7 - Synthesis (R, no GPU)
        from agents.agent7_synthesis import Agent7Synthesis
        def run_a7():
            if not quality or not quality.get("papers"):
                return {}
            a7 = Agent7Synthesis()
            result = a7.synthesize(quality, protocol)
            return result

        synthesis, a7_profile = profile_agent("A7", run_a7)
        a7_profile["vram_after_free_mb"] = get_vram_mb()
        profiles.append(a7_profile)
        print(f"  A7: {a7_profile['time_seconds']}s (R/metafor, no GPU)")

        # A8 - Manuscript
        from agents.agent8_manuscript import Agent8Manuscript
        def run_a8():
            a8 = Agent8Manuscript()
            prisma = {
                "identified": len(papers),
                "screened": len(papers),
                "excluded_ta": len(papers) - len(included_a3),
                "fulltext_assessed": len(included_a3),
                "excluded_ft": len(included_a3) - len(included_papers),
                "included": len(included_papers),
                "quality_assessed": len(quality.get("papers", [])) if quality else 0,
            }
            result = a8.generate_manuscript(
                protocol=protocol,
                included_papers=included_papers,
                extraction=extractions if extractions else None,
                quality=quality if quality else None,
                synthesis=synthesis if synthesis else None,
                prisma=prisma,
            )
            free_vram(a8)
            return result

        _, a8_profile = profile_agent("A8", run_a8)
        a8_profile["vram_after_free_mb"] = get_vram_mb()
        profiles.append(a8_profile)
        print(f"  A8: {a8_profile['time_seconds']}s, VRAM peak={a8_profile['vram_peak_mb']}MB")

    _save_and_print(profiles)


def _save_and_print(profiles):
    """Print markdown table and save JSON."""
    # Markdown table to stdout
    print(f"\n  {'Agent':<8} {'Time (s)':>10} {'VRAM Peak (MB)':>16} {'VRAM After Free (MB)':>22}")
    print(f"  {'-'*8} {'-'*10} {'-'*16} {'-'*22}")
    total_time = 0
    for p in profiles:
        total_time += p["time_seconds"]
        print(f"  {p['agent']:<8} {p['time_seconds']:>10.2f} {p['vram_peak_mb']:>16.1f} "
              f"{p.get('vram_after_free_mb', 0):>22.1f}")
    print(f"  {'TOTAL':<8} {total_time:>10.2f}")

    output = {
        "experiment": "exp7_vram_profiling",
        "query": QUERY,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "profiles": profiles,
        "total_time_s": round(total_time, 2),
    }

    latex = generate_latex_table(output)
    output["latex_table"] = latex
    print(f"\n  LaTeX table:\n{latex}")

    out_path = RESULTS_DIR / "exp7_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results -> {out_path}")
    return output


def generate_latex_table(results):
    """Generate a LaTeX table from VRAM profiling results."""
    lines = []
    lines.append(r"\begin{table}[h]\centering")
    lines.append(r"\begin{tabular}{lrrr}\hline")
    lines.append(r"Agent & Time (s) & VRAM Peak (MB) & VRAM After Free (MB) \\")
    lines.append(r"\hline")
    for p in results.get("profiles", []):
        lines.append(
            f"{p['agent']} & {p['time_seconds']:.2f} & "
            f"{p['vram_peak_mb']:.1f} & {p.get('vram_after_free_mb', 0):.1f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(f"Total & {results.get('total_time_s', 0):.2f} & -- & -- \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    gpu = results.get("gpu_name", "N/A")
    lines.append(r"\caption{Resultados Experimento 7 --- VRAM Profiling (" + gpu + r")}\label{tab:exp7}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 7 - VRAM Profiling")
    parser.add_argument("--dry-run", action="store_true", help="Mock run without GPU")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
