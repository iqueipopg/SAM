# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Test A6 + A7 secuencial usando pipeline_20260304_171352.json"""

import json, sys, os, gc, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PIPELINE_PATH = "outputs/pipeline_20260304_171352.json"


def main():
    # Load pipeline
    print(f"Loading: {PIPELINE_PATH}")
    with open(PIPELINE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    a5 = data["data_extraction"]
    protocol = data["protocol"]
    papers = a5["papers"]
    print(f"Papers: {len(papers)}")
    print(f"Protocol: {protocol.get('framework')} — {protocol.get('intervention')}")

    # ── A6: Quality Assessment ──────────────────────────────
    print(f"\n{'='*60}")
    print("RUNNING AGENT 6")
    print(f"{'='*60}")

    import torch
    from agents.agent6_quality import Agent6Quality

    a6 = Agent6Quality()
    a6_result = a6.assess_all(papers, protocol)

    # Free VRAM
    del a6.model
    del a6.tokenizer
    a6.model = None
    a6.tokenizer = None
    del a6
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nVRAM freed. Current: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Print A6 results table
    print(f"\n{'='*60}")
    print(f"{'PMID':<12} {'Design':<15} {'Tool':<10} {'Overall':<20} {'Weight'}")
    print(f"{'-'*70}")
    for p in a6_result["papers"]:
        qa = p["quality_assessment"]
        extra = ""
        if qa["tool"] == "JBI":
            extra = f" ({qa.get('yes_count', '?')}/10 Yes)"
        print(f"{p['pmid']:<12} {p['study_design_detected']:<15} {qa['tool']:<10} {qa['overall']:<20} {p['quality_weight']}{extra}")

    # Save A6 result
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/test_a6_from_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(a6_result, f, indent=2, ensure_ascii=False)
    print(f"\nA6 saved to outputs/test_a6_from_pipeline.json")

    # ── A7: Statistical Synthesis ───────────────────────────
    print(f"\n{'='*60}")
    print("RUNNING AGENT 7")
    print(f"{'='*60}")

    from agents.agent7_synthesis import Agent7Synthesis

    a7 = Agent7Synthesis()
    a7_result = a7.synthesize(a6_result, protocol)

    # Clean and save
    def clean_r(obj):
        if isinstance(obj, dict):
            return {k: clean_r(v) for k, v in obj.items() if k != "_fit"}
        elif isinstance(obj, list):
            return [clean_r(i) for i in obj]
        return obj

    a7_clean = clean_r(a7_result)
    with open("outputs/test_a7_from_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(a7_clean, f, indent=2, ensure_ascii=False)
    print(f"\nA7 saved to outputs/test_a7_from_pipeline.json")

    # Summary
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    for measure, analysis in a7_result.get("analyses", {}).items():
        m = analysis.get("model", {})
        g = analysis.get("grade", {})
        h = analysis.get("heterogeneity", {})
        print(f"\n  {measure} ({m.get('k_studies', '?')} studies):")
        print(f"    Pooled = {m.get('pooled_estimate')} [{m.get('ci_lower')}, {m.get('ci_upper')}]")
        print(f"    p = {m.get('p_value')}, I² = {h.get('I2')}% ({h.get('interpretation')})")
        print(f"    GRADE: {g.get('starting_level')} → {g.get('final_level')} (-{g.get('total_downgrades')})")
        plots = analysis.get("plots", {})
        if plots.get("forest"):
            print(f"    Forest: {plots['forest']}")
        if plots.get("funnel"):
            print(f"    Funnel: {plots['funnel']}")


if __name__ == "__main__":
    main()
