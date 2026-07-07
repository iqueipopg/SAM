# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Test Agent 6: Quality / Risk of Bias Assessment

Loads papers from the latest pipeline JSON (A5 extraction output),
runs Agent6Quality.assess_all(), and verifies output structure.
"""

import json, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent6_quality import Agent6Quality


def load_latest_pipeline():
    """Load the most recent pipeline output JSON."""
    new_pattern = os.path.join("outputs", "run_*", "pipeline.json")
    old_pattern = os.path.join("outputs", "pipeline_*.json")
    files = sorted(glob.glob(new_pattern) + glob.glob(old_pattern), reverse=True)
    if not files:
        print("No pipeline JSON found in outputs/. Run the pipeline first.")
        sys.exit(1)

    path = files[0]
    print(f"Loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    data = load_latest_pipeline()

    # Get A5 extraction papers
    a5 = data.get("data_extraction")
    if not a5 or not a5.get("papers"):
        print("No A5 extraction data found. Using A4 included papers as fallback.")
        a4 = data.get("fulltext_retrieval", {})
        papers = a4.get("include", [])
        if not papers:
            print("No included papers found at all.")
            sys.exit(1)
    else:
        papers = a5["papers"]

    protocol = data.get("protocol", {})
    print(f"\nPapers to assess: {len(papers)}")
    print(f"Protocol framework: {protocol.get('framework', '?')}")

    # Run Agent 6
    a6 = Agent6Quality()
    result = a6.assess_all(papers, protocol)

    # Verify output structure
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    errors = []

    # Check stats
    stats = result.get("stats", {})
    assert stats["total"] == len(papers), "Stats total mismatch"
    assert stats["rob2_assessed"] + stats["robins_i_assessed"] + stats["jbi_assessed"] + stats["failed"] == stats["total"], "Stats don't add up"

    # Check each paper
    for p in result["papers"]:
        pmid = p.get("pmid", "?")

        if "quality_assessment" not in p:
            errors.append(f"  {pmid}: missing quality_assessment")
            continue

        qa = p["quality_assessment"]

        # Check tool
        if qa["tool"] not in ("RoB2", "ROBINS-I", "JBI"):
            errors.append(f"  {pmid}: unknown tool '{qa['tool']}'")

        # Check overall
        if qa["tool"] == "RoB2":
            if qa["overall"] not in ("Low", "Some concerns", "High"):
                errors.append(f"  {pmid}: invalid RoB2 overall '{qa['overall']}'")
            # Check all 5 domains present
            expected_domains = {"randomization", "deviations", "missing_data", "measurement", "selection"}
            actual_domains = set(qa.get("domains", {}).keys())
            if expected_domains != actual_domains:
                errors.append(f"  {pmid}: RoB2 domains mismatch {actual_domains}")

        elif qa["tool"] == "ROBINS-I":
            if qa["overall"] not in ("Low", "Moderate", "Serious", "Critical", "No information"):
                errors.append(f"  {pmid}: invalid ROBINS-I overall '{qa['overall']}'")
            # Check all 7 domains present
            expected_domains = {
                "confounding", "selection_of_participants", "classification_of_interventions",
                "deviations_from_interventions", "missing_data", "measurement_of_outcomes",
                "selection_of_results",
            }
            actual_domains = set(qa.get("domains", {}).keys())
            if expected_domains != actual_domains:
                errors.append(f"  {pmid}: ROBINS-I domains mismatch {actual_domains}")

        elif qa["tool"] == "JBI":
            if qa["overall"] not in ("Include", "Include with caution", "Exclude"):
                errors.append(f"  {pmid}: invalid JBI overall '{qa['overall']}'")
            yes_count = qa.get("yes_count", -1)
            if not (0 <= yes_count <= 10):
                errors.append(f"  {pmid}: JBI yes_count out of range: {yes_count}")
            # Check all 10 items present
            expected_items = {
                "philosophical_perspective", "methodology_question", "methodology_methods",
                "methodology_analysis", "methodology_interpretation", "researcher_positioning",
                "researcher_influence", "participant_voices", "ethics", "conclusions",
            }
            actual_items = set(qa.get("items", {}).keys())
            if expected_items != actual_items:
                errors.append(f"  {pmid}: JBI items mismatch {actual_items}")

        # Check weight range
        w = p.get("quality_weight", -1)
        if not (0.0 <= w <= 1.0):
            errors.append(f"  {pmid}: quality_weight out of [0,1]: {w}")

        # Check study_design_detected
        if p.get("study_design_detected") not in ("rct", "observational", "qualitative", "unknown"):
            errors.append(f"  {pmid}: unexpected study_design_detected '{p.get('study_design_detected')}'")

    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for e in errors:
            print(e)
    else:
        print("\nALL CHECKS PASSED")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'PMID':<12} {'Design':<15} {'Tool':<10} {'Overall':<22} {'Weight'}")
    print(f"{'-'*70}")
    for p in result["papers"]:
        qa = p["quality_assessment"]
        print(f"{p['pmid']:<12} {p['study_design_detected']:<15} {qa['tool']:<10} {qa['overall']:<22} {p['quality_weight']}")

    # Save result
    out_path = "outputs/agent6_quality_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Free VRAM
    del a6
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
