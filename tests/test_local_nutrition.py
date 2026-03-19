#!/usr/bin/env python3
"""Local nutrition meta-analysis test — runs A5→A8 on Paula's pre-selected PDFs.

Skips A1-A4 (papers are already selected and downloaded).
Reads PDFs from docs/meta-nutricion-paula/, extracts text, groups supplements
with main papers, then runs extraction → quality → synthesis → manuscript.

Usage:
    source venv/bin/activate
    python tests/test_local_nutrition.py                # full pipeline
    python tests/test_local_nutrition.py --a5-only      # only A5 extraction
    python tests/test_local_nutrition.py --max-papers 10 # limit for quick test
"""

import gc, json, os, re, sys, time, argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import yaml

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "docs" / "meta-nutricion-paula"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Hardcoded protocol (PICO) matching Paula's nutrition meta-analysis
PROTOCOL = {
    "framework": "PICO",
    "research_question": (
        "Are behavioral and dietary interventions effective for improving "
        "weight, metabolic markers, and dietary outcomes in adults with "
        "chronic conditions such as obesity, type 2 diabetes, or "
        "cardiovascular disease?"
    ),
    "population": (
        "Adults (18+) with chronic metabolic conditions including obesity, "
        "type 2 diabetes, metabolic syndrome, or cardiovascular disease"
    ),
    "intervention": (
        "Behavioral or dietary lifestyle interventions including nutritional "
        "counseling, motivational interviewing, computer-tailored programs, "
        "dietary self-management, weight management programs, or combined "
        "diet and exercise interventions"
    ),
    "comparator": "Usual care, standard treatment, education-only control, or waitlist",
    "outcome": (
        "Primary: weight change (kg or BMI). Secondary: HbA1c, fasting glucose, "
        "lipid profiles, dietary intake (fruit/vegetable servings, fat intake, "
        "caloric intake), physical activity, treatment adherence"
    ),
    "original_query": (
        "Efectividad de intervenciones nutricionales y conductuales sobre "
        "peso y marcadores metabolicos en adultos con enfermedades cronicas"
    ),
    "inclusion_criteria": [
        "Randomized controlled trial (RCT) or cluster-RCT",
        "Adults (18+) with obesity, diabetes, metabolic syndrome, or cardiovascular disease",
        "Behavioral or dietary intervention targeting nutrition or weight management",
        "Comparison to control, usual care, or standard treatment",
        "Reports at least one quantitative outcome (weight, BMI, HbA1c, dietary intake)",
    ],
    "exclusion_criteria": [
        "Non-randomized studies, case reports, editorials, reviews",
        "Pediatric populations (<18 years)",
        "Pharmacological-only interventions without behavioral/dietary component",
    ],
}


# ── PDF loading ───────────────────────────────────────────────────────────────

def find_pdf_dir():
    """Find the actual PDF directory (handles unicode encoding issues)."""
    for d in PDF_DIR.iterdir():
        if d.is_dir() and d.name.startswith("Incluidos"):
            return d
    # Fallback: check if PDFs are directly in PDF_DIR
    if any(PDF_DIR.glob("*.pdf")):
        return PDF_DIR
    raise FileNotFoundError(f"No PDF directory found in {PDF_DIR}")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF (primary) + pdfplumber (fallback)."""
    text = ""

    # Primary: PyMuPDF
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(pages)
    except Exception:
        pass

    # Fallback: pdfplumber (better with tables/complex layouts)
    if len(text.strip()) < 500:
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            fallback = "\n\n".join(pages)
            if len(fallback) > len(text):
                text = fallback
        except Exception:
            pass

    return text.strip()


def extract_docx_text(docx_path: Path) -> str:
    """Extract text from DOCX files (some supplements are .docx)."""
    try:
        import docx
        doc = docx.Document(str(docx_path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def extract_title_from_text(text: str) -> str:
    """Best-effort title extraction from first lines of PDF text."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Skip common header noise (journal names, DOIs, page numbers)
    skip_patterns = [
        r"^(doi|DOI|http|www\.|Vol\.|Downloaded|See discussion)",
        r"^\d+$",  # page numbers
        r"^(ORIGINAL|RESEARCH|ARTICLE|INVESTIGATION)",
    ]
    title_lines = []
    for line in lines[:20]:
        if any(re.search(p, line) for p in skip_patterns):
            continue
        if len(line) > 15:  # skip very short lines
            title_lines.append(line)
            # Title is usually 1-3 lines
            if len(" ".join(title_lines)) > 40:
                break

    title = " ".join(title_lines)[:200]
    return title if title else f"Unknown paper"


def group_papers_from_pdfs(pdf_dir: Path) -> list:
    """Load PDFs, group main papers with supplements, return paper dicts."""
    files = sorted(
        [f for f in pdf_dir.iterdir() if f.suffix.lower() in (".pdf", ".docx")
         and not f.name.startswith(".")]
    )

    # Group by numeric ID: #5298.pdf, #5298sup1.pdf, #5298sup2.pdf → "5298"
    groups = {}
    for f in files:
        match = re.match(r"#?(\d+)", f.name)
        if not match:
            continue
        paper_id = match.group(1)
        is_supplement = "sup" in f.name.lower() or "primeros" in f.name.lower()

        if paper_id not in groups:
            groups[paper_id] = {"main": [], "supplements": []}
        if is_supplement:
            groups[paper_id]["supplements"].append(f)
        else:
            groups[paper_id]["main"].append(f)

    papers = []
    for paper_id, group in sorted(groups.items(), key=lambda x: int(x[0])):
        # Extract main text
        main_texts = []
        for f in group["main"]:
            if f.suffix.lower() == ".pdf":
                t = extract_pdf_text(f)
            else:
                t = extract_docx_text(f)
            if t:
                main_texts.append(t)

        if not main_texts:
            print(f"  [SKIP] #{paper_id}: no readable main PDF")
            continue

        main_text = "\n\n".join(main_texts)

        # Extract supplement text
        sup_texts = []
        for f in group["supplements"]:
            if f.suffix.lower() == ".pdf":
                t = extract_pdf_text(f)
            elif f.suffix.lower() == ".docx":
                t = extract_docx_text(f)
            else:
                continue
            if t:
                sup_texts.append(t)

        # Combine main + supplements
        full_text = main_text
        if sup_texts:
            full_text += "\n\n--- SUPPLEMENTARY MATERIAL ---\n\n" + "\n\n".join(sup_texts)

        title = extract_title_from_text(main_text)

        paper = {
            "pmid": f"LOCAL-{paper_id}",
            "title": title,
            "abstract": main_text[:1500],  # first ~1500 chars as abstract proxy
            "full_text": full_text,
            "text_source": "local_pdf",
            "full_text_length": len(full_text),
            "authors": [],
            "journal": "",
            "year": "",
            "doi": "",
            "_local_id": paper_id,
            "_n_supplements": len(sup_texts),
            "_main_files": [f.name for f in group["main"]],
            "_supplement_files": [f.name for f in group["supplements"]],
        }
        papers.append(paper)

    return papers


# ── VRAM management ───────────────────────────────────────────────────────────

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


# ── Pipeline stages ───────────────────────────────────────────────────────────

def run_a5(papers, protocol, run_dir):
    """A5: Data extraction."""
    from agents.agent5_extraction import Agent5Extraction

    print(f"\n{'='*60}")
    print(f"A5: EXTRACTING DATA FROM {len(papers)} PAPERS")
    print(f"{'='*60}")

    a5 = Agent5Extraction()
    a5_result = a5.extract_all(papers, protocol)
    free_vram(a5)

    stats = a5_result["stats"]
    print(f"\nA5 Results:")
    print(f"  Complete: {stats['complete_extraction']}")
    print(f"  Partial:  {stats['partial_extraction']}")
    print(f"  Failed:   {stats['failed_extraction']}")
    print(f"  Numeric:  {stats.get('numeric_complete', 0)}")

    # Save checkpoint
    ckpt_path = os.path.join(run_dir, "checkpoint_a5_extraction.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(a5_result, f, indent=2, ensure_ascii=False, default=str)

    # Detailed per-paper extraction report
    report_lines = ["# A5 Extraction Report\n"]
    for p in a5_result.get("papers", []):
        pid = p.get("pmid", "?")
        title = p.get("title", "?")[:80]
        quality = p.get("extraction_quality", "?")
        confidence = p.get("extraction_confidence", "?")

        report_lines.append(f"\n## {pid}: {title}")
        report_lines.append(f"- Quality: **{quality}** | Confidence: **{confidence}**")

        # Study characteristics
        sc = p.get("study_characteristics")
        if sc:
            report_lines.append(f"- Design: {sc.get('study_design', '?')}")
            report_lines.append(f"- N: {sc.get('sample_size', '?')}")
            report_lines.append(f"- Population: {sc.get('population', '?')}")

        # Outcomes
        outcomes = p.get("outcomes", [])
        if outcomes:
            report_lines.append(f"- Outcomes extracted: {len(outcomes)}")
            for o in outcomes:
                name = o.get("outcome_name", o.get("name", "?"))
                measure = o.get("measure_type", o.get("effect_measure", "?"))
                effect = o.get("effect_size", o.get("value", "?"))
                ci = f"[{o.get('ci_lower', '?')}, {o.get('ci_upper', '?')}]"
                report_lines.append(f"  - {name}: {measure}={effect} {ci}")
        else:
            report_lines.append("- **NO OUTCOMES EXTRACTED**")

    report_path = os.path.join(run_dir, "a5_extraction_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nDetailed report: {report_path}")

    return a5_result


def run_a6(a5_result, protocol, run_dir):
    """A6: Quality assessment."""
    from agents.agent6_quality import Agent6Quality

    papers = a5_result.get("papers", [])
    print(f"\n{'='*60}")
    print(f"A6: QUALITY ASSESSMENT ({len(papers)} papers)")
    print(f"{'='*60}")

    a6 = Agent6Quality()
    a6_result = a6.assess_all(papers, protocol)
    free_vram(a6)

    stats = a6_result["stats"]
    print(f"\nA6 Results:")
    print(f"  RoB 2:     {stats['rob2_assessed']}")
    print(f"  ROBINS-I:  {stats['robins_i_assessed']}")
    print(f"  JBI:       {stats['jbi_assessed']}")
    print(f"  Failed:    {stats['failed']}")

    ckpt_path = os.path.join(run_dir, "checkpoint_a6_quality.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(a6_result, f, indent=2, ensure_ascii=False, default=str)

    return a6_result


def run_a7(a6_result, protocol, run_dir):
    """A7: Statistical synthesis."""
    from agents.agent7_synthesis import Agent7Synthesis

    papers = a6_result.get("papers", [])
    print(f"\n{'='*60}")
    print(f"A7: STATISTICAL SYNTHESIS ({len(papers)} papers)")
    print(f"{'='*60}")

    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}

    a7 = Agent7Synthesis(config=cfg)
    a7_result = a7.synthesize(a6_result, protocol, output_dir=str(run_dir))
    a7_result = clean_r_objects(a7_result)

    stats = a7_result["stats"]
    print(f"\nA7 Results:")
    print(f"  Papers with usable data: {stats['papers_with_usable_data']}")
    print(f"  Analyses performed:      {stats['analyses_performed']}")

    for measure, analysis in a7_result.get("analyses", {}).items():
        m = analysis.get("model", {})
        g = analysis.get("grade", {})
        print(f"  {measure}: pooled={m.get('pooled_estimate')} "
              f"[{m.get('ci_lower')}, {m.get('ci_upper')}], "
              f"I²={m.get('I2')}%, GRADE={g.get('final_level')}")

    ckpt_path = os.path.join(run_dir, "checkpoint_a7_synthesis.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(a7_result, f, indent=2, ensure_ascii=False, default=str)

    return a7_result


def run_a8(papers, a5_result, a6_result, a7_result, protocol, run_dir):
    """A8: Manuscript generation."""
    from agents.agent8_manuscript import Agent8Manuscript

    print(f"\n{'='*60}")
    print(f"A8: GENERATING MANUSCRIPT ({len(papers)} papers)")
    print(f"{'='*60}")

    a8 = Agent8Manuscript()

    n_ext = 0
    if a5_result and a5_result.get("stats"):
        n_ext = (a5_result["stats"].get("complete_extraction", 0) +
                 a5_result["stats"].get("partial_extraction", 0))
    n_qa = 0
    if a6_result and a6_result.get("stats"):
        n_qa = (a6_result["stats"].get("rob2_assessed", 0) +
                a6_result["stats"].get("robins_i_assessed", 0) +
                a6_result["stats"].get("jbi_assessed", 0))

    prisma = {
        "identified": len(papers),
        "excluded_title_abstract": 0,
        "assessed_fulltext": len(papers),
        "excluded_fulltext": 0,
        "included_final": len(papers),
        "extracted": n_ext,
        "quality_assessed": n_qa,
    }

    protocol_for_a8 = dict(protocol)
    a8_result = a8.generate_manuscript(
        protocol=protocol_for_a8,
        included_papers=papers,
        extraction=a5_result,
        quality=a6_result,
        synthesis=a7_result,
        output_dir=str(run_dir),
        prisma=prisma,
    )
    free_vram(a8)

    stats = a8_result["stats"]
    print(f"\nA8 Results:")
    print(f"  Papers cited: {stats['total_papers_cited']}")
    print(f"  RAG chunks:   {stats['total_chunks']}")
    print(f"  PDF: {a8_result.get('pdf_path', 'N/A')}")

    ckpt_path = os.path.join(run_dir, "checkpoint_a8_manuscript.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(a8_result, f, indent=2, ensure_ascii=False, default=str)

    return a8_result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local nutrition meta-analysis test (A5→A8)")
    parser.add_argument("--a5-only", action="store_true", help="Only run A5 extraction")
    parser.add_argument("--max-papers", type=int, default=0, help="Limit number of papers (0=all)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing run dir")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SAM LOCAL TEST — Nutrition Meta-Analysis (Paula's PDFs)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── Setup run directory ──
    if args.resume and os.path.isdir(args.resume):
        run_dir = args.resume
        print(f"\nResuming from: {run_dir}")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = str(PROJECT_ROOT / "outputs" / f"nutrition_test_{ts}")

    os.makedirs(run_dir, exist_ok=True)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Load PDFs ──
    print(f"\nLoading PDFs from: {PDF_DIR}")
    pdf_dir = find_pdf_dir()
    papers = group_papers_from_pdfs(pdf_dir)

    if args.max_papers > 0:
        papers = papers[:args.max_papers]

    print(f"\nLoaded {len(papers)} papers:")
    for p in papers:
        sups = p["_n_supplements"]
        sup_label = f" (+{sups} sup)" if sups else ""
        print(f"  #{p['_local_id']}: {p['title'][:70]}...{sup_label} [{len(p['full_text'])} chars]")

    # Save paper manifest
    manifest = [{
        "id": p["_local_id"],
        "pmid": p["pmid"],
        "title": p["title"][:100],
        "text_length": len(p["full_text"]),
        "n_supplements": p["_n_supplements"],
        "main_files": p["_main_files"],
        "supplement_files": p["_supplement_files"],
    } for p in papers]
    manifest_path = os.path.join(run_dir, "paper_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Save protocol
    protocol_path = os.path.join(run_dir, "protocol.json")
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(PROTOCOL, f, indent=2, ensure_ascii=False)

    start = time.time()

    # ── A5: Extraction ──
    a5_result = run_a5(papers, PROTOCOL, run_dir)

    if args.a5_only:
        elapsed = round(time.time() - start, 1)
        print(f"\n{'='*60}")
        print(f"A5-only test complete in {elapsed}s")
        print(f"Results: {run_dir}/")
        print(f"{'='*60}")
        return

    # ── A6: Quality ──
    a6_result = run_a6(a5_result, PROTOCOL, run_dir)

    # ── A7: Synthesis ──
    a7_result = None
    if len(a6_result.get("papers", [])) >= 2:
        a7_result = run_a7(a6_result, PROTOCOL, run_dir)
    else:
        print("\nA7: Skipped (fewer than 2 papers)")

    # ── A8: Manuscript ──
    a8_result = None
    if len(papers) >= 2:
        a8_result = run_a8(papers, a5_result, a6_result, a7_result, PROTOCOL, run_dir)
    else:
        print("\nA8: Skipped (fewer than 2 papers)")

    # ── Save unified pipeline JSON ──
    _sys_info = {"timestamp": datetime.now().isoformat()}
    try:
        _sys_info["python"] = sys.version.split()[0]
        _sys_info["torch"] = torch.__version__
        if torch.cuda.is_available():
            _sys_info["gpu"] = torch.cuda.get_device_name(0)
            _sys_info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
    except Exception:
        pass

    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}

    pipeline_data = {
        "query": PROTOCOL["research_question"],
        "protocol": PROTOCOL,
        "config_snapshot": cfg,
        "system_info": _sys_info,
        "paper_manifest": manifest,
        "data_extraction": a5_result,
        "quality_assessment": a6_result,
        "synthesis": a7_result,
        "manuscript": a8_result,
    }
    pipeline_path = os.path.join(run_dir, "pipeline.json")
    with open(pipeline_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_data, f, indent=2, ensure_ascii=False, default=str)

    elapsed = round(time.time() - start, 1)

    print(f"\n{'='*60}")
    print(f"  LOCAL NUTRITION TEST COMPLETE")
    print(f"{'='*60}")
    print(f"  Papers: {len(papers)}")
    a5s = a5_result.get("stats", {})
    print(f"  A5: {a5s.get('complete_extraction', 0)} complete, "
          f"{a5s.get('partial_extraction', 0)} partial, "
          f"{a5s.get('numeric_complete', 0)} numeric")
    if a6_result:
        a6s = a6_result.get("stats", {})
        print(f"  A6: {a6s.get('rob2_assessed', 0)} RoB2, "
              f"{a6s.get('robins_i_assessed', 0)} ROBINS-I")
    if a7_result:
        a7s = a7_result.get("stats", {})
        print(f"  A7: {a7s.get('analyses_performed', 0)} analyses, "
              f"{a7s.get('papers_with_usable_data', 0)} usable")
    if a8_result:
        a8s = a8_result.get("stats", {})
        print(f"  A8: {a8s.get('total_papers_cited', 0)} cited, "
              f"PDF: {a8_result.get('pdf_path', 'N/A')}")
    print(f"  Time: {elapsed}s")
    print(f"  Output: {run_dir}/")


if __name__ == "__main__":
    main()
