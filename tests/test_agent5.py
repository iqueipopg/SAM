# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Test standalone Agent 5: Structured Data Extraction
Uses the 2 real included papers from the last A1→A2→A3→A4 pipeline run.

Ejecutar: python test_agent5.py
"""

import json, os, sys, gc, torch, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent5_extraction import Agent5Extraction

# ─────────────────────────────────────────────────────────────
#  LOAD REAL DATA from last pipeline output
# ─────────────────────────────────────────────────────────────

PIPELINE_FILE = "./outputs/pipeline_20260304_160956.json"


def load_test_data():
    """Load included papers and protocol from the last pipeline output."""
    if not os.path.exists(PIPELINE_FILE):
        print(f"Pipeline output not found: {PIPELINE_FILE}")
        print("Using fallback test data.")
        return _fallback_data()

    with open(PIPELINE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    included = data.get("fulltext_retrieval", {}).get("include", [])
    protocol = data.get("protocol", {})

    if not included:
        print("No included papers found in pipeline output. Using fallback.")
        return _fallback_data()

    print(f"Loaded {len(included)} papers from {PIPELINE_FILE}")
    return included, protocol


def _fallback_data():
    """Minimal fallback if pipeline output is not available."""
    protocol = {
        "framework": "PECO",
        "research_question": "Does smoking increase the risk of myocardial infarction in diabetic patients?",
        "population": "Diabetic patients",
        "exposure": "Smoking",
        "comparator": "Non-smoking",
        "outcome": "Risk of myocardial infarction",
        "inclusion_criteria": [
            "Cohort study, case-control study, or cross-sectional study",
            "Adults with relevant clinical diagnosis",
            "Includes Smoking as intervention or exposure",
        ],
        "exclusion_criteria": [
            "Case reports, editorials, or animal studies",
            "Studies not reporting relevant clinical outcomes",
        ],
    }

    papers = [
        {
            "pmid": "41519686",
            "title": "Prevalence and clinical correlates of peripheral artery disease in patients undergoing hemodialysis: a cross-sectional study from Palestine.",
            "abstract": "Peripheral artery disease (PAD) is a common but underrecognized complication among patients with end-stage renal disease (ESRD) receiving hemodialysis.",
            "authors": ["Shakhshir", "Zabin", "Nawahda"],
            "journal": "BMC nephrology",
            "year": "2026",
            "doi": "10.1093/ndt/gfr760",
            "text_source": "abstract_only",
            "full_text": "Peripheral artery disease (PAD) is a common but underrecognized complication among patients with end-stage renal disease (ESRD) receiving hemodialysis.",
            "full_text_length": 150,
            "sections": {},
        },
    ]
    return papers, protocol


def free_vram(agent):
    if hasattr(agent, "model") and agent.model is not None:
        del agent.model
        agent.model = None
    if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
        del agent.tokenizer
        agent.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    print(f"   VRAM freed -> {torch.cuda.memory_allocated()/1e9:.1f} GB in use")


# ─────────────────────────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  TEST AGENT 5 - STRUCTURED DATA EXTRACTION")
    print("=" * 65)

    included, protocol = load_test_data()
    print(f"  Papers: {len(included)}")
    print(f"  Framework: {protocol.get('framework', '?')}")
    print()

    start = time.time()

    a5 = Agent5Extraction()
    result = a5.extract_all(included, protocol)

    free_vram(a5)

    # ── Show results ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EXTRACTION RESULTS")
    print("=" * 65)

    for p in result["papers"]:
        q_icon = {"complete": "✅", "partial": "⚠", "failed": "❌"}.get(
            p["extraction_quality"], "?"
        )
        print(f"\n  {q_icon} [{p['pmid']}] {p['title'][:70]}...")
        print(f"     Quality  : {p['extraction_quality']}")
        print(f"     Source   : {p['text_source']}")
        print(f"     Sections : {p['sections_used']}")

        if p.get("study_characteristics"):
            sc = p["study_characteristics"]
            print(f"     Design   : {sc.get('study_design', '?')}")
            print(f"     N        : {sc.get('sample_size', '?')}")
            print(f"     Country  : {sc.get('country', '?')}")

        if p.get("intervention_exposure"):
            ie = p["intervention_exposure"]
            print(f"     Exposure : {ie.get('name', '?')}")
            print(f"     Comparator: {ie.get('comparator', '?')}")

        if p.get("outcomes"):
            print(f"     Outcomes : {len(p['outcomes'])}")
            for o in p["outcomes"][:3]:
                print(f"       - {o.get('name', '?')}: {o.get('measure_type', '?')} = {o.get('value', '?')}")

        if p.get("extraction_notes"):
            print(f"     Notes    : {p['extraction_notes']}")

    # ── Save output ───────────────────────────────────────────
    os.makedirs("./outputs", exist_ok=True)
    out_path = "./outputs/agent5_extraction_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed = round(time.time() - start, 1)
    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {elapsed}s")
    print("  Test Agent 5 completed.")
