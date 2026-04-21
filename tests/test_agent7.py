"""Test Agent 7: Statistical Meta-Analysis

Two modes:
  A) Synthetic data with known SMD values to verify pooled estimate
  B) Real pipeline data (if available)
"""

import json, glob, os, sys, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent7_synthesis import Agent7Synthesis


def run_synthetic():
    """Test with synthetic data: 5 studies with known SMD values."""
    print(f"\n{'='*60}")
    print("TEST A: SYNTHETIC DATA")
    print(f"{'='*60}")

    # 5 studies with known effect sizes and CIs
    papers = [
        {
            "pmid": "SYNTH001",
            "title": "Synthetic Study 1 (Low risk)",
            "quality_weight": 1.0,
            "study_characteristics": {"study_design": "RCT"},
            "text_source": "pmc_fulltext",
            "outcomes": [{
                "name": "Primary outcome",
                "measure_type": "SMD",
                "value": -0.50,
                "ci_lower": -0.80,
                "ci_upper": -0.20,
                "p_value": 0.001,
            }],
        },
        {
            "pmid": "SYNTH002",
            "title": "Synthetic Study 2 (Low risk)",
            "quality_weight": 1.0,
            "study_characteristics": {"study_design": "RCT"},
            "text_source": "pmc_fulltext",
            "outcomes": [{
                "name": "Primary outcome",
                "measure_type": "SMD",
                "value": -0.35,
                "ci_lower": -0.70,
                "ci_upper": 0.00,
                "p_value": 0.05,
            }],
        },
        {
            "pmid": "SYNTH003",
            "title": "Synthetic Study 3 (Medium risk)",
            "quality_weight": 0.6,
            "study_characteristics": {"study_design": "RCT"},
            "text_source": "pmc_fulltext",
            "outcomes": [{
                "name": "Primary outcome",
                "measure_type": "SMD",
                "value": -0.60,
                "ci_lower": -1.00,
                "ci_upper": -0.20,
                "p_value": 0.003,
            }],
        },
        {
            "pmid": "SYNTH004",
            "title": "Synthetic Study 4 (High risk)",
            "quality_weight": 0.3,
            "study_characteristics": {"study_design": "RCT"},
            "text_source": "abstract_only",
            "outcomes": [{
                "name": "Primary outcome",
                "measure_type": "SMD",
                "value": -0.80,
                "ci_lower": -1.40,
                "ci_upper": -0.20,
                "p_value": 0.009,
            }],
        },
        {
            "pmid": "SYNTH005",
            "title": "Synthetic Study 5 (Low risk)",
            "quality_weight": 1.0,
            "study_characteristics": {"study_design": "RCT"},
            "text_source": "pmc_fulltext",
            "outcomes": [{
                "name": "Primary outcome",
                "measure_type": "SMD",
                "value": -0.40,
                "ci_lower": -0.65,
                "ci_upper": -0.15,
                "p_value": 0.002,
            }],
        },
    ]

    protocol = {
        "framework": "PICO",
        "population": "Adults with depression",
        "intervention": "CBT",
        "outcome": "Depression severity",
    }

    a7 = Agent7Synthesis()
    result = a7.synthesize({"papers": papers}, protocol)

    # Verify
    errors = []

    # Check stats
    stats = result.get("stats", {})
    if stats["total_papers_input"] != 5:
        errors.append(f"Expected 5 papers input, got {stats['total_papers_input']}")
    if stats["analyses_performed"] < 1:
        errors.append("No analyses performed")

    # Check SMD analysis exists
    analyses = result.get("analyses", {})
    if "SMD" not in analyses:
        errors.append("SMD analysis not found")
    else:
        smd = analyses["SMD"]

        # Model
        model = smd.get("model", {})
        est = model.get("pooled_estimate", 0)
        if not (-1.0 < est < 0.0):
            errors.append(f"Pooled SMD estimate out of expected range: {est}")
        print(f"\n  Pooled SMD: {est} [{model.get('ci_lower')}, {model.get('ci_upper')}]")
        print(f"  p = {model.get('p_value')}, k = {model.get('k_studies')}")

        # Heterogeneity
        het = smd.get("heterogeneity", {})
        print(f"  I^2 = {het.get('I2')}%, Q = {het.get('Q')}, Q_p = {het.get('Q_p')}")

        # Plots
        plots = smd.get("plots", {})
        forest = plots.get("forest")
        funnel = plots.get("funnel")
        if forest and os.path.exists(forest):
            print(f"  Forest plot: {forest} (exists)")
        else:
            errors.append(f"Forest plot not created: {forest}")
        if funnel and os.path.exists(funnel):
            print(f"  Funnel plot: {funnel} (exists)")
        else:
            errors.append(f"Funnel plot not created: {funnel}")

        # GRADE
        grade = smd.get("grade", {})
        print(f"  GRADE: {grade.get('starting_level')} -> {grade.get('final_level')} (-{grade.get('total_downgrades')})")

        # Sensitivity
        sens = smd.get("sensitivity", {})
        l1o = sens.get("leave_one_out", [])
        print(f"  Leave-one-out: {len(l1o)} iterations")
        excl = sens.get("exclude_high_risk", {})
        print(f"  Excl. high-risk: k_excluded={excl.get('k_excluded', 0)}, est={excl.get('pooled_estimate', '?')}")

    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n  ALL SYNTHETIC CHECKS PASSED")

    return result


def run_real_data():
    """Test with real pipeline data if available."""
    print(f"\n{'='*60}")
    print("TEST B: REAL PIPELINE DATA")
    print(f"{'='*60}")

    # Try loading A6 output first, then latest pipeline
    a6_path = "outputs/agent6_quality_test.json"
    if os.path.exists(a6_path):
        print(f"  Loading A6 output: {a6_path}")
        with open(a6_path, "r", encoding="utf-8") as f:
            a6_result = json.load(f)
    else:
        # Try latest pipeline JSON
        new_pattern = os.path.join("outputs", "run_*", "pipeline.json")
        old_pattern = os.path.join("outputs", "pipeline_*.json")
        files = sorted(glob.glob(new_pattern) + glob.glob(old_pattern), reverse=True)
        if not files:
            print("  No pipeline data available. Skipping real data test.")
            return None

        path = files[0]
        print(f"  Loading pipeline: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        a5 = data.get("data_extraction")
        if not a5 or not a5.get("papers"):
            print("  No A5 extraction data. Skipping.")
            return None

        # Mock A6 result with default weights
        a6_result = {
            "papers": [
                {**p, "quality_weight": p.get("quality_weight", 0.5)}
                for p in a5["papers"]
            ]
        }

    protocol = {"framework": "PICO", "population": "?", "intervention": "?", "outcome": "?"}

    a7 = Agent7Synthesis()
    result = a7.synthesize(a6_result, protocol)

    # Save result
    out_path = "outputs/agent7_synthesis_test.json"
    # Remove non-serializable _fit objects
    clean = _remove_fit(result)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    return result


def _remove_fit(obj):
    """Recursively remove non-serializable R objects from result dict."""
    if isinstance(obj, dict):
        return {k: _remove_fit(v) for k, v in obj.items() if k != "_fit"}
    elif isinstance(obj, list):
        return [_remove_fit(item) for item in obj]
    return obj


def main():
    # Test A: Synthetic
    synth_result = run_synthetic()

    # Save synthetic result
    if synth_result:
        os.makedirs("outputs", exist_ok=True)
        clean = _remove_fit(synth_result)
        with open("outputs/agent7_synthetic_test.json", "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)

    print()

    # Test B: Real data
    run_real_data()


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# Pytest tests for BUG2: forest plot study-label fallback cascade.
# These are pure-function tests — no R, no plotting, CPU-only.
# ──────────────────────────────────────────────────────────────────────────────

import pytest  # noqa: E402


def test_build_study_label_prefers_pmid():
    label = Agent7Synthesis._build_study_label(
        {"pmid": "12345678", "authors": ["Smith J"], "year": "2020"}, 1
    )
    assert label == "12345678"


def test_build_study_label_pmid_plus_outcome_name():
    label = Agent7Synthesis._build_study_label(
        {"pmid": "42", "authors": ["X"]}, 1, outcome_name="Anxiety (BAI)"
    )
    assert label == "42: Anxiety (BAI)"


def test_build_study_label_fallback_to_first_author_and_year():
    label = Agent7Synthesis._build_study_label(
        {"pmid": "", "first_author": "Smith J", "year": "2021"}, 1
    )
    assert label == "Smith J (2021)"


def test_build_study_label_fallback_to_first_author_alone():
    label = Agent7Synthesis._build_study_label(
        {"pmid": "", "first_author": "Smith J"}, 2
    )
    assert label == "Smith J"


def test_build_study_label_fallback_to_authors_list_first():
    label = Agent7Synthesis._build_study_label(
        {"pmid": "", "authors": ["García-López R", "Müller K"]}, 3
    )
    assert label == "García-López R"


def test_build_study_label_fallback_to_title():
    paper = {"pmid": "", "authors": [], "title": "Effects of CBT on anxiety in adults"}
    label = Agent7Synthesis._build_study_label(paper, 4)
    assert label.startswith("Effects of CBT")
    # Must NOT begin with a colon (the original BUG2 symptom).
    assert not label.startswith(":")


def test_build_study_label_last_resort_study_n():
    label = Agent7Synthesis._build_study_label({}, 7)
    assert label == "Study 7"


def test_build_study_label_never_starts_with_colon_even_with_outcome_name():
    # This is the exact BUG2 regression scenario: empty pmid, outcome present.
    label = Agent7Synthesis._build_study_label(
        {"pmid": "", "authors": [], "title": ""},
        5,
        outcome_name="Reduction in anxiety",
    )
    assert not label.startswith(":"), f"label started with colon: {label!r}"
    assert "Study 5" in label or "Reduction in anxiety" in label


def test_build_study_label_handles_missing_everything():
    # Dict with None values must not raise.
    label = Agent7Synthesis._build_study_label(
        {"pmid": None, "authors": None, "title": None, "year": None}, 9
    )
    assert label == "Study 9"
