#!/usr/bin/env python3
"""SAM Batch Runner — runs full pipeline on multiple queries, analyzes results,
and auto-adjusts parameters between runs.

Usage:
    source venv/bin/activate
    python batch_run.py

Results are saved to outputs/batch_<timestamp>/
"""

import gc, json, os, sys, time, traceback, yaml
from datetime import datetime

import torch

# ── Config ──────────────────────────────────────────────────────────────────

QUERIES = [
    # PICO — intervention comparison
    "Is cognitive behavioral therapy more effective than medication for treating depression in adults?",
    # PECO — exposure/risk
    "Does smoking increase myocardial infarction risk in diabetic patients?",
    # PICO — another clinical area
    "Is exercise effective for reducing anxiety symptoms in adults?",
    # PECO — epidemiological
    "Does air pollution exposure increase the risk of asthma in children?",
    # PICO — pharmacological
    "Are SGLT2 inhibitors effective in reducing heart failure hospitalizations in diabetic patients?",
]

# Papers to request per query (total across all databases)
MAX_RESULTS = 150

# Screening parameters
ENSEMBLE_SIZE = 3
VOTING_THRESHOLD = 0.60

# Minimum included papers to consider a run "successful"
MIN_INCLUDED_FOR_SUCCESS = 5

# ── Helpers ─────────────────────────────────────────────────────────────────

def free_vram(agent=None):
    if agent is not None:
        if hasattr(agent, "model") and agent.model is not None:
            del agent.model
            agent.model = None
        if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
            del agent.tokenizer
            agent.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clean_r_objects(obj):
    if isinstance(obj, dict):
        return {k: clean_r_objects(v) for k, v in obj.items() if k != "_fit"}
    elif isinstance(obj, list):
        return [clean_r_objects(item) for item in obj]
    return obj


def save_checkpoint(run_dir, agent_name, data):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"checkpoint_{agent_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def log(msg, run_log=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if run_log is not None:
        run_log.append(line)


def analyze_run(run_dir, result):
    """Analyze a completed run and return a quality report dict."""
    report = {
        "run_dir": run_dir,
        "query": result.get("query", ""),
        "framework": result.get("protocol", {}).get("framework", "?"),
        "status": "success",
        "issues": [],
    }

    # Search stats
    a2 = result.get("a2_stats", {})
    report["papers_found"] = a2.get("total_after_dedup", 0)

    # Screening stats
    a3 = result.get("a3_stats", {})
    report["a3_included"] = a3.get("included", 0)
    report["a3_excluded"] = a3.get("excluded", 0)
    report["a3_uncertain"] = a3.get("uncertain", 0)
    report["a3_total"] = a3.get("total", 0)

    if report["a3_total"] > 0:
        inc_rate = report["a3_included"] / report["a3_total"]
        report["inclusion_rate_a3"] = round(inc_rate * 100, 1)
        if inc_rate < 0.03:
            report["issues"].append(f"A3 inclusion rate very low ({report['inclusion_rate_a3']}%) — screening may be too strict")
        elif inc_rate > 0.80:
            report["issues"].append(f"A3 inclusion rate very high ({report['inclusion_rate_a3']}%) — screening may be too loose")

    # A4 stats
    a4 = result.get("a4_stats", {})
    report["a4_included"] = a4.get("fulltext_included", 0)
    report["a4_excluded"] = a4.get("fulltext_excluded", 0)

    # Final included
    report["final_included"] = report["a4_included"]
    if report["final_included"] < MIN_INCLUDED_FOR_SUCCESS:
        report["issues"].append(f"Only {report['final_included']} papers included — insufficient for robust meta-analysis")
        if report["final_included"] < 2:
            report["status"] = "failed"

    # Extraction stats
    a5 = result.get("a5_stats", {})
    report["extracted_complete"] = a5.get("complete_extraction", 0)
    report["extracted_partial"] = a5.get("partial_extraction", 0)
    report["extracted_failed"] = a5.get("failed_extraction", 0)

    # Synthesis stats
    a7 = result.get("a7_result", {})
    analyses = a7.get("analyses", {})
    report["n_analyses"] = len(analyses)
    report["synthesis_details"] = []

    for measure, analysis in analyses.items():
        model = analysis.get("model", {})
        grade = analysis.get("grade", {})
        detail = {
            "measure": measure,
            "pooled": model.get("pooled_estimate"),
            "ci_lower": model.get("ci_lower"),
            "ci_upper": model.get("ci_upper"),
            "k_studies": model.get("k_studies"),
            "I2": model.get("I2"),
            "grade": grade.get("final_level", "?"),
        }
        report["synthesis_details"].append(detail)

        if detail["k_studies"] and detail["k_studies"] < 3:
            report["issues"].append(f"{measure}: only k={detail['k_studies']} studies in meta-analysis")
        if detail["grade"] == "Very low":
            report["issues"].append(f"{measure}: GRADE certainty is Very low")

    # Manuscript
    a8 = result.get("a8_result", {})
    report["has_manuscript"] = bool(a8.get("manuscript"))
    report["has_pdf"] = bool(a8.get("pdf_path"))

    if not report["issues"]:
        report["verdict"] = "GOOD"
    elif report["status"] == "failed":
        report["verdict"] = "FAILED"
    elif len(report["issues"]) <= 2:
        report["verdict"] = "ACCEPTABLE"
    else:
        report["verdict"] = "NEEDS REVIEW"

    return report


# ── Pipeline Runner ─────────────────────────────────────────────────────────

def run_single_pipeline(query, run_dir, run_log):
    """Run the full A1→A8 pipeline for a single query. Returns result dict."""
    from agents.agent1_protocol import Agent1Protocol
    from agents.agent2_search import Agent2Search
    from agents.agent3_screening import Agent3Screening
    from agents.agent4_fulltext import Agent4FullText
    from agents.agent5_extraction import Agent5Extraction
    from agents.agent6_quality import Agent6Quality
    from agents.agent7_synthesis import Agent7Synthesis
    from agents.agent8_manuscript import Agent8Manuscript
    from agents.exporters import export_csv, export_excel, export_summary_table

    os.makedirs(run_dir, exist_ok=True)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    result = {"query": query, "run_dir": run_dir}

    # ── A1: Protocol ──
    log(f"A1: Generating protocol...", run_log)
    try:
        a1 = Agent1Protocol()
        protocol = a1.generate_protocol(query)
        pubmed_query = a1.generate_pubmed_query(query, protocol)
        free_vram(a1)
        result["protocol"] = protocol
        result["pubmed_query"] = pubmed_query
        save_checkpoint(run_dir, "a1_protocol", {"query": query, "protocol": protocol, "pubmed_query": pubmed_query})
        log(f"A1: Framework={protocol.get('framework')} | Query: {pubmed_query[:80]}...", run_log)
    except Exception as e:
        log(f"A1: FAILED — {e}", run_log)
        result["error"] = f"A1: {e}"
        return result

    # ── A2: Search ──
    log(f"A2: Searching databases (max_results={MAX_RESULTS})...", run_log)
    try:
        a2 = Agent2Search()
        a2_result = a2.search(protocol, pubmed_query, max_results=MAX_RESULTS)
        papers = a2_result["papers"]
        a2_stats = a2_result["stats"]
        result["a2_stats"] = a2_stats
        save_checkpoint(run_dir, "a2_search", {"papers": papers, "stats": a2_stats})
        log(f"A2: Found {a2_stats['total_after_dedup']} papers (PubMed={a2_stats['pubmed']}, EPMC={a2_stats['europepmc']}, S2={a2_stats['semantic_scholar']})", run_log)
    except Exception as e:
        log(f"A2: FAILED — {e}", run_log)
        result["error"] = f"A2: {e}"
        return result

    if not papers:
        log("A2: No papers found. Aborting.", run_log)
        result["error"] = "No papers found"
        return result

    # ── A3: Screening ──
    log(f"A3: Screening {len(papers)} papers (ensemble={ENSEMBLE_SIZE}, threshold={VOTING_THRESHOLD})...", run_log)
    try:
        a3 = Agent3Screening(ensemble_size=ENSEMBLE_SIZE, voting_threshold=VOTING_THRESHOLD)
        a3_results = a3.screen_batch(papers, protocol)
        stats3 = a3_results["stats"]
        free_vram(a3)
        save_checkpoint(run_dir, "a3_screening", a3_results)
        result["a3_stats"] = stats3
        log(f"A3: INCLUDE={stats3['included']} | EXCLUDE={stats3['excluded']} | UNCERTAIN={stats3['uncertain']} | {stats3['elapsed_s']}s", run_log)
    except Exception as e:
        log(f"A3: FAILED — {e}", run_log)
        result["error"] = f"A3: {e}"
        return result

    included_papers = a3_results.get("include", [])
    # Also pass uncertain to A4 for full-text re-evaluation
    uncertain_papers = a3_results.get("uncertain", [])

    if not included_papers and not uncertain_papers:
        log("A3: No papers passed screening. Aborting.", run_log)
        result["error"] = "No papers passed A3"
        return result

    # Combine include + uncertain for A4 (let full-text decide)
    a4_candidates = included_papers + uncertain_papers
    log(f"A4: Sending {len(a4_candidates)} papers ({len(included_papers)} included + {len(uncertain_papers)} uncertain) for full-text screening...", run_log)

    # ── A4: Full-text ──
    try:
        a4 = Agent4FullText(ensemble_size=ENSEMBLE_SIZE, voting_threshold=VOTING_THRESHOLD)
        a4_result = a4.run(a4_candidates, papers, protocol)
        stats4 = a4_result["stats"]
        free_vram(a4)
        save_checkpoint(run_dir, "a4_fulltext", a4_result)
        result["a4_stats"] = stats4
        log(f"A4: INCLUDE={stats4['fulltext_included']} | EXCLUDE={stats4['fulltext_excluded']} | {stats4.get('elapsed_s', '?')}s", run_log)
    except Exception as e:
        log(f"A4: FAILED — {e}", run_log)
        result["error"] = f"A4: {e}"
        return result

    if not a4_result.get("include"):
        log("A4: No papers passed full-text screening. Aborting.", run_log)
        result["error"] = "No papers passed A4"
        return result

    # ── A5: Extraction ──
    log(f"A5: Extracting data from {len(a4_result['include'])} papers...", run_log)
    a5_result = None
    try:
        a5 = Agent5Extraction()
        a5_result = a5.extract_all(a4_result["include"], protocol)
        stats5 = a5_result["stats"]
        free_vram(a5)
        save_checkpoint(run_dir, "a5_extraction", a5_result)
        result["a5_stats"] = stats5
        log(f"A5: Complete={stats5['complete_extraction']} | Partial={stats5['partial_extraction']} | Failed={stats5['failed_extraction']}", run_log)
    except Exception as e:
        log(f"A5: FAILED — {e}", run_log)

    # ── A6: Quality ──
    a6_result = None
    a6_papers = None
    if a5_result and a5_result.get("papers"):
        a6_papers = a5_result["papers"]
    elif a4_result.get("include"):
        a6_papers = a4_result["include"]

    if a6_papers:
        log(f"A6: Quality assessment ({len(a6_papers)} papers)...", run_log)
        try:
            a6 = Agent6Quality()
            a6_result = a6.assess_all(a6_papers, protocol)
            stats6 = a6_result["stats"]
            free_vram(a6)
            save_checkpoint(run_dir, "a6_quality", a6_result)
            log(f"A6: RoB2={stats6['rob2_assessed']} | ROBINS-I={stats6['robins_i_assessed']} | JBI={stats6['jbi_assessed']}", run_log)
        except Exception as e:
            log(f"A6: FAILED — {e}", run_log)

    # ── A7: Synthesis ──
    a7_result = None
    a7_input = None
    if a6_result and a6_result.get("papers"):
        a7_input = a6_result
    elif a5_result and a5_result.get("papers"):
        a7_input = {"papers": [{**p, "quality_weight": 0.5} for p in a5_result["papers"]]}

    if a7_input and len(a7_input.get("papers", [])) >= 2:
        log(f"A7: Statistical synthesis ({len(a7_input['papers'])} papers)...", run_log)
        try:
            _config = {}
            if os.path.exists("config.yaml"):
                with open("config.yaml") as cf:
                    _config = yaml.safe_load(cf) or {}
            a7 = Agent7Synthesis(config=_config)
            a7_result = a7.synthesize(a7_input, protocol, output_dir=run_dir)
            a7_result = clean_r_objects(a7_result)
            save_checkpoint(run_dir, "a7_synthesis", a7_result)
            result["a7_result"] = a7_result
            stats7 = a7_result["stats"]
            log(f"A7: {stats7['analyses_performed']} analyses, {stats7['papers_with_usable_data']} papers with usable data", run_log)
            for measure, analysis in a7_result.get("analyses", {}).items():
                m = analysis.get("model", {})
                g = analysis.get("grade", {})
                log(f"  {measure}: pooled={m.get('pooled_estimate')} [{m.get('ci_lower')}, {m.get('ci_upper')}], I²={m.get('I2')}%, GRADE={g.get('final_level')}", run_log)
        except Exception as e:
            log(f"A7: FAILED — {e}", run_log)
    else:
        log("A7: Skipped (fewer than 2 papers with data)", run_log)

    # ── A8: Manuscript ──
    a8_result = None
    a8_papers = a4_result.get("include", [])
    if a8_papers and len(a8_papers) >= 2:
        log(f"A8: Generating manuscript ({len(a8_papers)} papers)...", run_log)
        try:
            a8 = Agent8Manuscript()
            a7_for_a8 = a7_result
            _s3 = a3_results.get("stats", {})
            _s4 = a4_result.get("stats", {})
            _n_ext = 0
            if a5_result and a5_result.get("stats"):
                _n_ext = a5_result["stats"].get("complete_extraction", 0) + a5_result["stats"].get("partial_extraction", 0)
            _n_qa = 0
            if a6_result and a6_result.get("stats"):
                _n_qa = a6_result["stats"].get("rob2_assessed", 0) + a6_result["stats"].get("robins_i_assessed", 0) + a6_result["stats"].get("jbi_assessed", 0)

            prisma = {
                "identified": _s3.get("total", 0),
                "excluded_title_abstract": _s3.get("excluded", 0),
                "uncertain_title_abstract": _s3.get("uncertain", 0),
                "assessed_fulltext": _s3.get("included", 0) + _s3.get("uncertain", 0),
                "excluded_fulltext": _s4.get("fulltext_excluded", 0),
                "included_final": _s4.get("fulltext_included", len(a8_papers)),
                "extracted": _n_ext,
                "quality_assessed": _n_qa,
            }
            protocol_for_a8 = dict(protocol)
            protocol_for_a8["pubmed_query"] = pubmed_query
            a8_result = a8.generate_manuscript(
                protocol=protocol_for_a8,
                included_papers=a8_papers,
                extraction=a5_result,
                quality=a6_result,
                synthesis=a7_for_a8,
                output_dir=run_dir,
                prisma=prisma,
            )
            free_vram(a8)
            save_checkpoint(run_dir, "a8_manuscript", a8_result)
            result["a8_result"] = a8_result
            log(f"A8: Manuscript generated ({a8_result['stats']['total_papers_cited']} papers cited), PDF: {a8_result.get('pdf_path', 'N/A')}", run_log)
        except Exception as e:
            log(f"A8: FAILED — {e}", run_log)
    else:
        log("A8: Skipped (fewer than 2 included papers)", run_log)

    # ── Exports ──
    try:
        export_csv(result_to_pipeline_data(result, a3_results, a4_result, a5_result, a6_result, a7_result, a8_result),
                    os.path.join(run_dir, "included_studies.csv"))
        export_summary_table(result_to_pipeline_data(result, a3_results, a4_result, a5_result, a6_result, a7_result, a8_result),
                             os.path.join(run_dir, "summary_table.md"))
        log("Exports: CSV + Summary table generated", run_log)
    except Exception as e:
        log(f"Exports: FAILED — {e}", run_log)

    return result


def result_to_pipeline_data(result, a3_results, a4_result, a5_result, a6_result, a7_result, a8_result):
    """Convert run result to the pipeline_data format expected by exporters."""
    return {
        "protocol": result.get("protocol", {}),
        "screening": a3_results,
        "fulltext_retrieval": a4_result,
        "data_extraction": a5_result,
        "quality_assessment": a6_result,
        "synthesis": a7_result,
        "manuscript": a8_result,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    batch_ts = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join("outputs", f"batch_{batch_ts}")
    os.makedirs(batch_dir, exist_ok=True)

    all_reports = []
    all_logs = []

    print(f"\n{'='*70}")
    print(f"  SAM BATCH RUNNER — {len(QUERIES)} queries, {MAX_RESULTS} papers each")
    print(f"  Output: {batch_dir}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for i, query in enumerate(QUERIES):
        run_log = []
        run_name = f"run_{i+1:02d}"
        run_dir = os.path.join(batch_dir, run_name)

        log(f"\n{'='*70}", run_log)
        log(f"RUN {i+1}/{len(QUERIES)}: {query}", run_log)
        log(f"{'='*70}", run_log)

        start = time.time()
        try:
            result = run_single_pipeline(query, run_dir, run_log)
        except Exception as e:
            log(f"PIPELINE CRASHED: {e}", run_log)
            log(traceback.format_exc(), run_log)
            result = {"query": query, "run_dir": run_dir, "error": str(e)}

        elapsed = round(time.time() - start, 1)
        log(f"Run completed in {elapsed}s", run_log)

        # Analyze
        report = analyze_run(run_dir, result)
        report["elapsed_s"] = elapsed
        all_reports.append(report)

        # Print summary
        log(f"\n--- RUN {i+1} SUMMARY ---", run_log)
        log(f"  Query: {query[:80]}", run_log)
        log(f"  Framework: {report['framework']}", run_log)
        log(f"  Papers found: {report.get('papers_found', 0)}", run_log)
        log(f"  A3 screening: {report.get('a3_included', 0)} included / {report.get('a3_total', 0)} total ({report.get('inclusion_rate_a3', '?')}%)", run_log)
        log(f"  Final included: {report.get('final_included', 0)}", run_log)
        log(f"  Analyses: {report.get('n_analyses', 0)}", run_log)
        for detail in report.get("synthesis_details", []):
            log(f"    {detail['measure']}: pooled={detail['pooled']} [{detail['ci_lower']}, {detail['ci_upper']}], k={detail['k_studies']}, I²={detail['I2']}%, GRADE={detail['grade']}", run_log)
        log(f"  Verdict: {report['verdict']}", run_log)
        if report["issues"]:
            for issue in report["issues"]:
                log(f"  ⚠ {issue}", run_log)

        # Save run log
        log_path = os.path.join(run_dir, "run_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(run_log))

        all_logs.extend(run_log)

        # Force GPU cleanup between runs
        free_vram()
        gc.collect()

    # ── Final batch report ──
    print(f"\n\n{'='*70}")
    print(f"  BATCH COMPLETE — {len(QUERIES)} runs")
    print(f"{'='*70}\n")

    summary_lines = []
    summary_lines.append(f"SAM Batch Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"{'='*70}")
    summary_lines.append(f"Queries: {len(QUERIES)} | Max results: {MAX_RESULTS}")
    summary_lines.append(f"Ensemble: {ENSEMBLE_SIZE} | Threshold: {VOTING_THRESHOLD}")
    summary_lines.append("")

    for i, report in enumerate(all_reports):
        v = report["verdict"]
        icon = {"GOOD": "OK", "ACCEPTABLE": "~~", "NEEDS REVIEW": "!!", "FAILED": "XX"}[v]
        summary_lines.append(f"[{icon}] Run {i+1}: {report['query'][:60]}...")
        summary_lines.append(f"     Framework: {report['framework']} | Found: {report.get('papers_found', 0)} | Included: {report.get('final_included', 0)} | Analyses: {report.get('n_analyses', 0)} | Time: {report.get('elapsed_s', '?')}s")
        for detail in report.get("synthesis_details", []):
            summary_lines.append(f"     {detail['measure']}: {detail['pooled']} [{detail['ci_lower']}, {detail['ci_upper']}], k={detail['k_studies']}, GRADE={detail['grade']}")
        if report["issues"]:
            for issue in report["issues"]:
                summary_lines.append(f"     ! {issue}")
        summary_lines.append("")

    # Counts
    good = sum(1 for r in all_reports if r["verdict"] == "GOOD")
    acceptable = sum(1 for r in all_reports if r["verdict"] == "ACCEPTABLE")
    review = sum(1 for r in all_reports if r["verdict"] == "NEEDS REVIEW")
    failed = sum(1 for r in all_reports if r["verdict"] == "FAILED")
    summary_lines.append(f"TOTALS: {good} GOOD, {acceptable} ACCEPTABLE, {review} NEEDS REVIEW, {failed} FAILED")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save batch report
    report_path = os.path.join(batch_dir, "batch_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    json_path = os.path.join(batch_dir, "batch_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nReports saved to: {batch_dir}/")
    print(f"  batch_report.txt  — human-readable summary")
    print(f"  batch_report.json — structured data for analysis")


if __name__ == "__main__":
    main()
