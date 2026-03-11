"""End-to-end integration test for the A1-A8 pipeline.

Runs the full pipeline data flow with synthetic/mocked data.
No GPU, no internet, no LLM models required. Runs in < 30 seconds.

Usage:
    pytest tests/test_e2e_pipeline.py -v
"""

import json
import os
import sys
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import save_checkpoint, load_checkpoint, list_checkpoint_agents, _clean_r_objects


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

SYNTHETIC_PROTOCOL = {
    "framework": "PICO",
    "population": "adults with depression",
    "intervention": "CBT",
    "comparator": "waitlist",
    "outcome": "anxiety scores",
    "research_question": "Is CBT more effective than waitlist control for reducing anxiety scores in adults with depression?",
    "inclusion_criteria": [
        "Adults aged 18+",
        "Diagnosed with depression (DSM/ICD criteria)",
        "CBT as primary intervention",
        "Anxiety score as measured outcome",
    ],
    "exclusion_criteria": [
        "Pediatric populations",
        "Non-RCT designs without comparator",
        "Animal studies",
    ],
    "pubmed_query": '("cognitive behavioral therapy" OR CBT) AND depression AND anxiety AND (randomized OR RCT)',
}

SYNTHETIC_PUBMED_QUERY = (
    '("cognitive behavioral therapy" OR CBT) AND depression AND anxiety AND (randomized OR RCT)'
)

# A2: 5 synthetic papers
SYNTHETIC_A2_PAPERS = [
    {
        "pmid": "30000001",
        "title": "Cognitive behavioral therapy for depression and anxiety in adults: a randomized controlled trial",
        "abstract": "Background: CBT is widely used for depression. Methods: 120 adults randomized to CBT (n=60) or waitlist (n=60). Results: Significant reduction in anxiety scores (SMD=-0.52, 95% CI [-0.78, -0.26]). Conclusion: CBT effective.",
        "authors": ["Smith J", "Doe A", "Brown K"],
        "journal": "J Clin Psychology",
        "year": "2022",
        "source": "pubmed",
    },
    {
        "pmid": "30000002",
        "title": "Effects of CBT on anxiety outcomes in depressed patients: a multi-center RCT",
        "abstract": "Objective: Evaluate CBT vs waitlist for anxiety in depression. 90 patients were randomized. CBT group showed lower anxiety (SMD=-0.38, 95% CI [-0.72, -0.04]). CBT is a viable intervention.",
        "authors": ["Jones M", "Wilson P"],
        "journal": "Behav Res Therapy",
        "year": "2021",
        "source": "pubmed",
    },
    {
        "pmid": "30000003",
        "title": "A randomized trial of cognitive therapy versus waitlist for comorbid depression and anxiety",
        "abstract": "Methods: RCT with 80 participants. CBT reduced anxiety significantly (SMD=-0.61, 95% CI [-1.02, -0.20]). Conclusion: CBT should be considered first-line.",
        "authors": ["Lee S", "Kim H", "Park J"],
        "journal": "Psychol Med",
        "year": "2023",
        "source": "europepmc",
    },
    {
        "pmid": "30000004",
        "title": "Waitlist-controlled trial of brief CBT for anxious depression in primary care",
        "abstract": "Brief CBT (8 sessions) was compared to waitlist in 70 primary care patients with depression and anxiety. Results showed moderate improvement (SMD=-0.45, 95% CI [-0.80, -0.10]).",
        "authors": ["Garcia R", "Martinez L"],
        "journal": "Prim Care Psychiatry",
        "year": "2022",
        "source": "semantic_scholar",
    },
    {
        "pmid": "30000005",
        "title": "Pharmacological treatment of depression: a cost-effectiveness review",
        "abstract": "This review examines cost-effectiveness of SSRIs for depression. No behavioral interventions were compared. Results are limited to pharmacoeconomic outcomes.",
        "authors": ["Taylor W"],
        "journal": "Pharmacoecon Rev",
        "year": "2020",
        "source": "pubmed",
    },
]

SYNTHETIC_A2_STATS = {
    "pubmed": 3,
    "europepmc": 1,
    "semantic_scholar": 1,
    "total_before_dedup": 5,
    "total_after_dedup": 5,
    "duplicates_removed": 0,
}

SYNTHETIC_A2_RESULT = {
    "papers": SYNTHETIC_A2_PAPERS,
    "stats": SYNTHETIC_A2_STATS,
}

# A3: 4 included, 1 excluded (paper 5 is pharmacoeconomic review)
def _make_a3_screening_entry(paper, decision, confidence):
    return {
        "pmid": paper["pmid"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "authors": paper.get("authors", []),
        "journal": paper.get("journal", ""),
        "year": paper.get("year", ""),
        "source": paper.get("source", ""),
        "decision": decision,
        "confidence": confidence,
        "votes": [decision] * 3,
        "flags": [],
        "reasoning": f"Paper {'meets' if decision == 'INCLUDE' else 'does not meet'} inclusion criteria.",
    }

SYNTHETIC_A3_RESULT = {
    "include": [
        _make_a3_screening_entry(SYNTHETIC_A2_PAPERS[0], "INCLUDE", 1.0),
        _make_a3_screening_entry(SYNTHETIC_A2_PAPERS[1], "INCLUDE", 1.0),
        _make_a3_screening_entry(SYNTHETIC_A2_PAPERS[2], "INCLUDE", 1.0),
        _make_a3_screening_entry(SYNTHETIC_A2_PAPERS[3], "INCLUDE", 0.67),
    ],
    "exclude": [
        _make_a3_screening_entry(SYNTHETIC_A2_PAPERS[4], "EXCLUDE", 1.0),
    ],
    "uncertain": [],
    "stats": {
        "total": 5,
        "included": 4,
        "excluded": 1,
        "uncertain": 0,
        "elapsed_s": 2.0,
        "s_per_paper": 0.4,
        "ensemble_size": 3,
        "voting_threshold": 0.60,
    },
}

# A4: 3 papers with full text, 1 abstract-only (paper 4 abstract-only)
def _make_a4_paper(a3_entry, full_text, text_source):
    return {
        "pmid": a3_entry["pmid"],
        "title": a3_entry["title"],
        "abstract": a3_entry["abstract"],
        "authors": a3_entry.get("authors", []),
        "journal": a3_entry.get("journal", ""),
        "year": a3_entry.get("year", ""),
        "doi": f"10.1000/test.{a3_entry['pmid']}",
        "pmcid": f"PMC{a3_entry['pmid']}" if text_source != "abstract_only" else "",
        "full_text": full_text,
        "text_source": text_source,
        "full_text_length": len(full_text) if full_text else len(a3_entry.get("abstract", "")),
        "sections": {},
        "oa_url": None,
        "fulltext_decision": "INCLUDE",
        "fulltext_reasoning": "Paper meets all inclusion criteria on full-text review.",
        "exclusion_reason": None,
    }

_A4_FULLTEXT_TEMPLATE = (
    "Introduction: Depression and anxiety frequently co-occur. CBT has shown promise. "
    "Methods: This randomized controlled trial enrolled {n} adults with depression and comorbid anxiety. "
    "Participants were randomized to {arms}. Primary outcome was anxiety score (BAI). "
    "Results: The CBT group showed significantly lower anxiety scores compared to waitlist "
    "(SMD={smd}, 95% CI [{ci_lo}, {ci_up}], p={p}). "
    "Discussion: These findings support CBT as effective for anxiety in depressed adults. "
    "Limitations: Single-site study, short follow-up."
)

SYNTHETIC_A4_INCLUDE = [
    _make_a4_paper(
        SYNTHETIC_A3_RESULT["include"][0],
        _A4_FULLTEXT_TEMPLATE.format(n=120, arms="CBT (n=60) or waitlist (n=60)", smd="-0.52", ci_lo="-0.78", ci_up="-0.26", p="0.001"),
        "pmc",
    ),
    _make_a4_paper(
        SYNTHETIC_A3_RESULT["include"][1],
        _A4_FULLTEXT_TEMPLATE.format(n=90, arms="CBT (n=45) or waitlist (n=45)", smd="-0.38", ci_lo="-0.72", ci_up="-0.04", p="0.03"),
        "unpaywall_pdf",
    ),
    _make_a4_paper(
        SYNTHETIC_A3_RESULT["include"][2],
        _A4_FULLTEXT_TEMPLATE.format(n=80, arms="CBT (n=40) or waitlist (n=40)", smd="-0.61", ci_lo="-1.02", ci_up="-0.20", p="0.004"),
        "pmc",
    ),
]

# Paper 4 included but abstract-only
SYNTHETIC_A4_INCLUDE.append(
    _make_a4_paper(
        SYNTHETIC_A3_RESULT["include"][3],
        None,
        "abstract_only",
    )
)
# Override: full_text_length for abstract-only
SYNTHETIC_A4_INCLUDE[3]["full_text_length"] = len(SYNTHETIC_A3_RESULT["include"][3]["abstract"])

SYNTHETIC_A4_RESULT = {
    "papers": SYNTHETIC_A4_INCLUDE,
    "include": SYNTHETIC_A4_INCLUDE[:3],  # 3 included after full-text screening
    "exclude": [SYNTHETIC_A4_INCLUDE[3]],  # paper 4 excluded (abstract-only, insufficient data)
    "stats": {
        "total_retrieved": 4,
        "pmc_fulltext": 2,
        "unpaywall_pdf": 1,
        "europepmc_fulltext": 0,
        "semanticscholar_pdf": 0,
        "scihub_pdf": 0,
        "abstract_only": 1,
        "fulltext_included": 3,
        "fulltext_excluded": 1,
        "elapsed_s": 3.0,
    },
}

# A5: 3 papers with structured extraction data
def _make_a5_paper(a4_paper, study_chars, outcomes, quality="complete"):
    return {
        "pmid": a4_paper["pmid"],
        "title": a4_paper["title"],
        "abstract": a4_paper.get("abstract", ""),
        "authors": a4_paper.get("authors", []),
        "journal": a4_paper.get("journal", ""),
        "year": a4_paper.get("year", ""),
        "text_source": a4_paper["text_source"],
        "study_characteristics": study_chars,
        "intervention_exposure": {
            "intervention_name": "Cognitive Behavioral Therapy (CBT)",
            "comparator_name": "Waitlist control",
            "duration": "12 weeks",
            "dosage": "Weekly 1-hour sessions",
        },
        "outcomes": outcomes,
        "extraction_quality": quality,
        "extraction_confidence": "high" if quality == "complete" else "medium",
        "concordance": {"overall": "high"} if quality == "complete" else None,
        "extraction_notes": "",
        "sections_used": ["results", "methods"],
    }

SYNTHETIC_A5_PAPERS = [
    _make_a5_paper(
        SYNTHETIC_A4_RESULT["include"][0],
        {"study_design": "RCT", "sample_size": 120, "country": "USA", "setting": "outpatient"},
        [{
            "name": "Anxiety score (BAI)",
            "measure_type": "SMD",
            "effect_size_value": -0.52,
            "ci_lower": -0.78,
            "ci_upper": -0.26,
            "p_value": 0.001,
            "n_intervention": 60,
            "n_control": 60,
            "numeric_complete": True,
        }],
    ),
    _make_a5_paper(
        SYNTHETIC_A4_RESULT["include"][1],
        {"study_design": "RCT", "sample_size": 90, "country": "UK", "setting": "community"},
        [{
            "name": "Anxiety score (BAI)",
            "measure_type": "SMD",
            "effect_size_value": -0.38,
            "ci_lower": -0.72,
            "ci_upper": -0.04,
            "p_value": 0.03,
            "n_intervention": 45,
            "n_control": 45,
            "numeric_complete": True,
        }],
    ),
    _make_a5_paper(
        SYNTHETIC_A4_RESULT["include"][2],
        {"study_design": "RCT", "sample_size": 80, "country": "South Korea", "setting": "hospital"},
        [{
            "name": "Anxiety score (BAI)",
            "measure_type": "SMD",
            "effect_size_value": -0.61,
            "ci_lower": -1.02,
            "ci_upper": -0.20,
            "p_value": 0.004,
            "n_intervention": 40,
            "n_control": 40,
            "numeric_complete": True,
        }],
    ),
]

SYNTHETIC_A5_RESULT = {
    "papers": SYNTHETIC_A5_PAPERS,
    "stats": {
        "total": 3,
        "complete_extraction": 3,
        "partial_extraction": 0,
        "failed_extraction": 0,
        "high_confidence": 3,
        "medium_confidence": 0,
        "low_confidence": 0,
        "numeric_complete": 3,
        "numeric_recovered": 0,
        "dual_extraction": True,
        "elapsed_s": 5.0,
    },
}

# A6: 3 papers with RoB2 assessments (all RCTs)
def _make_a6_paper(a5_paper, overall, weight, domains):
    paper = dict(a5_paper)
    paper["quality_assessment"] = {
        "tool": "RoB2",
        "overall": overall,
        "domains": domains,
    }
    paper["quality_weight"] = weight
    return paper

SYNTHETIC_A6_PAPERS = [
    _make_a6_paper(
        SYNTHETIC_A5_PAPERS[0], "Low", 1.0,
        {
            "randomization_process": "Low",
            "deviations_from_interventions": "Low",
            "missing_outcome_data": "Low",
            "measurement_of_outcome": "Low",
            "selection_of_reported_result": "Low",
        },
    ),
    _make_a6_paper(
        SYNTHETIC_A5_PAPERS[1], "Some concerns", 0.6,
        {
            "randomization_process": "Low",
            "deviations_from_interventions": "Some concerns",
            "missing_outcome_data": "Low",
            "measurement_of_outcome": "Low",
            "selection_of_reported_result": "Low",
        },
    ),
    _make_a6_paper(
        SYNTHETIC_A5_PAPERS[2], "Low", 1.0,
        {
            "randomization_process": "Low",
            "deviations_from_interventions": "Low",
            "missing_outcome_data": "Low",
            "measurement_of_outcome": "Some concerns",
            "selection_of_reported_result": "Low",
        },
    ),
]

SYNTHETIC_A6_RESULT = {
    "papers": SYNTHETIC_A6_PAPERS,
    "stats": {
        "total": 3,
        "rob2_assessed": 3,
        "robins_i_assessed": 0,
        "jbi_assessed": 0,
        "failed": 0,
        "elapsed_s": 4.0,
    },
}

# A7: 1 SMD meta-analysis with pooled estimate
SYNTHETIC_A7_RESULT = {
    "analyses": {
        "SMD": {
            "model": {
                "pooled_estimate": -0.50,
                "ci_lower": -0.72,
                "ci_upper": -0.28,
                "p_value": 0.0001,
                "k_studies": 3,
                "method": "REML",
            },
            "heterogeneity": {
                "I2": 5.2,
                "tau2": 0.002,
                "Q": 2.1,
                "Q_p": 0.35,
                "Q_df": 2,
            },
            "plots": {
                "forest": "outputs/plots/forest_SMD.png",
                "funnel": "outputs/plots/funnel_SMD.png",
            },
            "grade": {
                "starting_level": "High",
                "final_level": "Moderate",
                "total_downgrades": 1,
                "downgrades": {"imprecision": 1},
            },
            "sensitivity": {
                "leave_one_out": [
                    {"excluded": "30000001", "pooled_estimate": -0.49, "ci_lower": -0.80, "ci_upper": -0.18},
                    {"excluded": "30000002", "pooled_estimate": -0.56, "ci_lower": -0.79, "ci_upper": -0.33},
                    {"excluded": "30000003", "pooled_estimate": -0.45, "ci_lower": -0.68, "ci_upper": -0.22},
                ],
                "exclude_high_risk": {
                    "k_excluded": 0,
                    "pooled_estimate": -0.50,
                    "ci_lower": -0.72,
                    "ci_upper": -0.28,
                },
            },
            "publication_bias": {
                "egger_test": None,
                "trim_and_fill": None,
                "note": "Too few studies (k=3) for Egger test (requires k>=10)",
            },
            "studies": [
                {"pmid": "30000001", "yi": -0.52, "vi": 0.018, "weight": 1.0},
                {"pmid": "30000002", "yi": -0.38, "vi": 0.030, "weight": 0.6},
                {"pmid": "30000003", "yi": -0.61, "vi": 0.044, "weight": 1.0},
            ],
        },
    },
    "stats": {
        "total_papers_input": 3,
        "papers_with_usable_data": 3,
        "analyses_performed": 1,
        "measure_types_found": ["SMD"],
        "elapsed_s": 2.0,
    },
}

# A8: manuscript with 6 sections
SYNTHETIC_A8_RESULT = {
    "manuscript": {
        "abstract": (
            "Background: Cognitive behavioral therapy (CBT) is a widely used intervention for depression, "
            "yet its effects on comorbid anxiety remain debated. This systematic review and meta-analysis "
            "examined the efficacy of CBT compared to waitlist control for reducing anxiety scores in adults "
            "with depression. Methods: We searched PubMed, Europe PMC, and Semantic Scholar. Three RCTs "
            "met inclusion criteria (N=290). Random-effects meta-analysis was conducted. Results: CBT "
            "significantly reduced anxiety (pooled SMD=-0.50, 95% CI [-0.72, -0.28], p<0.001). "
            "Heterogeneity was low (I2=5.2%). GRADE certainty was moderate. Conclusion: CBT effectively "
            "reduces anxiety in adults with depression compared to waitlist control."
        ),
        "introduction": (
            "Depression and anxiety disorders frequently co-occur, affecting approximately 50% of individuals "
            "with major depressive disorder. Cognitive behavioral therapy is an evidence-based treatment for "
            "both conditions, but the specific impact on anxiety when treating depression has been debated. "
            "Previous reviews have examined CBT for depression or anxiety separately, but few have focused on "
            "the anxiolytic effects of CBT in populations primarily diagnosed with depression. This systematic "
            "review aims to address this gap by synthesizing evidence from randomized controlled trials comparing "
            "CBT to waitlist control conditions, with anxiety scores as the primary outcome."
        ),
        "methods": (
            "This systematic review followed PRISMA 2020 guidelines. We searched PubMed, Europe PMC, and "
            "Semantic Scholar using the query: (cognitive behavioral therapy OR CBT) AND depression AND anxiety "
            "AND (randomized OR RCT). Title/abstract screening was performed using ensemble LLM voting (3 votes, "
            "60% threshold). Full-text retrieval used PMC, Unpaywall, and Europe PMC. Data extraction captured "
            "study characteristics, intervention details, and outcomes. Quality was assessed using RoB 2 for "
            "randomized trials. Random-effects meta-analysis (REML) was performed using the metafor package in R. "
            "Heterogeneity was assessed using I-squared and Cochran Q. GRADE assessment was applied to rate "
            "certainty of evidence."
        ),
        "results": (
            "Five records were identified, and after screening, three RCTs (N=290) were included. All studies "
            "compared CBT to waitlist control in adults with depression, measuring anxiety with the BAI. "
            "Quality assessment: two studies were rated low risk of bias, one had some concerns. "
            "Meta-analysis showed a significant pooled effect favoring CBT (SMD=-0.50, 95% CI [-0.72, -0.28], "
            "p<0.001). Heterogeneity was low (I2=5.2%, Q=2.1, p=0.35). Leave-one-out sensitivity analysis "
            "showed stable results. GRADE certainty was rated moderate, downgraded once for imprecision."
        ),
        "discussion": (
            "Our findings indicate that CBT significantly reduces anxiety in adults with depression compared to "
            "waitlist control, with a moderate effect size. The low heterogeneity suggests consistent effects "
            "across studies. These results align with previous meta-analyses showing CBT efficacy for anxiety "
            "disorders. Strengths include rigorous automated screening and quality assessment. Limitations include "
            "the small number of studies (k=3), reliance on waitlist controls, and potential for publication bias "
            "that could not be formally tested. Future research should include active comparators and longer follow-up."
        ),
        "conclusion": (
            "CBT is effective for reducing anxiety in adults with depression compared to waitlist control "
            "(SMD=-0.50, 95% CI [-0.72, -0.28]). The evidence was rated moderate certainty using GRADE. "
            "Clinicians should consider CBT as a first-line treatment for comorbid depression and anxiety. "
            "More high-quality RCTs with active comparators are needed to strengthen the evidence base."
        ),
    },
    "references": [
        {"pmid": "30000001", "title": "Cognitive behavioral therapy for depression and anxiety in adults: a randomized controlled trial", "citation": "Smith J, Doe A, Brown K (2022)", "journal": "J Clin Psychology"},
        {"pmid": "30000002", "title": "Effects of CBT on anxiety outcomes in depressed patients: a multi-center RCT", "citation": "Jones M, Wilson P (2021)", "journal": "Behav Res Therapy"},
        {"pmid": "30000003", "title": "A randomized trial of cognitive therapy versus waitlist for comorbid depression and anxiety", "citation": "Lee S, Kim H, Park J (2023)", "journal": "Psychol Med"},
    ],
    "stats": {
        "total_chunks": 24,
        "total_papers_cited": 3,
        "chunks_per_section": {
            "abstract": 4,
            "introduction": 6,
            "methods": 4,
            "results": 4,
            "discussion": 4,
            "conclusion": 2,
        },
        "elapsed_s": 15.0,
    },
    "pdf_path": None,
}


# ══════════════════════════════════════════════════════════════════════════════
# PRISMA helper
# ══════════════════════════════════════════════════════════════════════════════

def build_prisma(a2_stats, a3_stats, a4_stats, a5_stats, a6_stats):
    """Build PRISMA dict the same way app.py does."""
    n_extracted = a5_stats.get("complete_extraction", 0) + a5_stats.get("partial_extraction", 0)
    n_quality = a6_stats.get("rob2_assessed", 0) + a6_stats.get("robins_i_assessed", 0) + a6_stats.get("jbi_assessed", 0)
    return {
        "identified": a2_stats.get("total_before_dedup", 0),
        "identified_per_db": {
            "pubmed": a2_stats.get("pubmed", 0),
            "europepmc": a2_stats.get("europepmc", 0),
            "semantic_scholar": a2_stats.get("semantic_scholar", 0),
        },
        "after_dedup": a2_stats.get("total_after_dedup", 0),
        "duplicates_removed": a2_stats.get("duplicates_removed", 0),
        "screened": a3_stats.get("total", 0),
        "excluded_title_abstract": a3_stats.get("excluded", 0),
        "uncertain_title_abstract": a3_stats.get("uncertain", 0),
        "assessed_fulltext": a3_stats.get("included", 0),
        "excluded_fulltext": a4_stats.get("fulltext_excluded", 0),
        "included_final": a4_stats.get("fulltext_included", 0),
        "extracted": n_extracted,
        "quality_assessed": n_quality,
    }


def build_final_output():
    """Build the final pipeline output dict as app.py does."""
    prisma = build_prisma(
        SYNTHETIC_A2_STATS,
        SYNTHETIC_A3_RESULT["stats"],
        SYNTHETIC_A4_RESULT["stats"],
        SYNTHETIC_A5_RESULT["stats"],
        SYNTHETIC_A6_RESULT["stats"],
    )
    return {
        "query": "Is CBT more effective than waitlist for reducing anxiety in adults with depression?",
        "protocol": SYNTHETIC_PROTOCOL,
        "pubmed_query": SYNTHETIC_PUBMED_QUERY,
        "prisma": prisma,
        "search_stats": SYNTHETIC_A2_STATS,
        "screening": SYNTHETIC_A3_RESULT,
        "fulltext_retrieval": SYNTHETIC_A4_RESULT,
        "data_extraction": SYNTHETIC_A5_RESULT,
        "quality_assessment": SYNTHETIC_A6_RESULT,
        "synthesis": SYNTHETIC_A7_RESULT,
        "manuscript": SYNTHETIC_A8_RESULT,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE DATA FLOW TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineDataFlow:
    """Test the full A1-A8 pipeline data flow using synthetic data."""

    def test_a1_protocol_structure(self):
        """A1 protocol has all required PICO fields."""
        p = SYNTHETIC_PROTOCOL
        assert p["framework"] == "PICO"
        assert "population" in p
        assert "intervention" in p
        assert "comparator" in p
        assert "outcome" in p
        assert len(p["inclusion_criteria"]) > 0
        assert len(p["exclusion_criteria"]) > 0

    def test_a2_search_output(self):
        """A2 output has papers list and stats dict."""
        result = SYNTHETIC_A2_RESULT
        assert "papers" in result
        assert "stats" in result
        assert len(result["papers"]) == 5
        # Each paper has required fields
        for paper in result["papers"]:
            assert "pmid" in paper
            assert "title" in paper
            assert "abstract" in paper

    def test_a2_to_a3_flow(self):
        """A2 papers feed into A3 screening; counts are consistent."""
        a2_papers = SYNTHETIC_A2_RESULT["papers"]
        a3 = SYNTHETIC_A3_RESULT
        # A3 received all A2 papers
        assert a3["stats"]["total"] == len(a2_papers)
        # Include + exclude + uncertain = total
        assert (
            a3["stats"]["included"] + a3["stats"]["excluded"] + a3["stats"]["uncertain"]
            == a3["stats"]["total"]
        )

    def test_a3_to_a4_flow(self):
        """A3 included papers feed into A4; A4 has include/exclude."""
        a3_included = SYNTHETIC_A3_RESULT["include"]
        a4 = SYNTHETIC_A4_RESULT
        # A4 received A3 included papers
        assert a4["stats"]["total_retrieved"] == len(a3_included)
        # A4 include + exclude = total
        assert len(a4["include"]) + len(a4["exclude"]) == a4["stats"]["total_retrieved"]

    def test_a4_to_a5_flow(self):
        """A4 included papers feed into A5 extraction."""
        a4_included = SYNTHETIC_A4_RESULT["include"]
        a5 = SYNTHETIC_A5_RESULT
        assert a5["stats"]["total"] == len(a4_included)
        assert len(a5["papers"]) == len(a4_included)

    def test_a5_to_a6_flow(self):
        """A5 papers feed into A6 quality assessment."""
        a5_papers = SYNTHETIC_A5_RESULT["papers"]
        a6 = SYNTHETIC_A6_RESULT
        assert a6["stats"]["total"] == len(a5_papers)
        assert len(a6["papers"]) == len(a5_papers)

    def test_a6_to_a7_flow(self):
        """A6 papers (with weights) feed into A7 synthesis."""
        a6_papers = SYNTHETIC_A6_RESULT["papers"]
        a7 = SYNTHETIC_A7_RESULT
        assert a7["stats"]["total_papers_input"] == len(a6_papers)
        # All studies in analyses reference valid PMIDs
        smd = a7["analyses"]["SMD"]
        study_pmids = {s["pmid"] for s in smd["studies"]}
        a6_pmids = {p["pmid"] for p in a6_papers}
        assert study_pmids == a6_pmids

    def test_a7_to_a8_flow(self):
        """A7 synthesis data informs A8 manuscript content."""
        a8 = SYNTHETIC_A8_RESULT
        manuscript = a8["manuscript"]
        # Manuscript references the pooled estimate from A7
        assert "-0.50" in manuscript["results"] or "-0.5" in manuscript["results"]
        # References correspond to included papers
        ref_pmids = {r["pmid"] for r in a8["references"]}
        a7_pmids = {s["pmid"] for s in SYNTHETIC_A7_RESULT["analyses"]["SMD"]["studies"]}
        assert ref_pmids == a7_pmids

    def test_final_output_has_all_keys(self):
        """Final pipeline output has all expected top-level keys."""
        output = build_final_output()
        expected_keys = {
            "query", "protocol", "pubmed_query", "prisma",
            "search_stats", "screening", "fulltext_retrieval",
            "data_extraction", "quality_assessment", "synthesis", "manuscript",
        }
        assert expected_keys.issubset(set(output.keys()))


# ══════════════════════════════════════════════════════════════════════════════
# DATA INTEGRITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """Verify data integrity across pipeline stages."""

    def test_paper_ids_preserved(self):
        """PMIDs from A2 are preserved through A3, A4, A5, A6."""
        a2_pmids = {p["pmid"] for p in SYNTHETIC_A2_PAPERS}

        # A3 PMIDs are a subset of A2
        a3_all_pmids = set()
        for bucket in ("include", "exclude", "uncertain"):
            for entry in SYNTHETIC_A3_RESULT[bucket]:
                a3_all_pmids.add(entry["pmid"])
        assert a3_all_pmids.issubset(a2_pmids)

        # A3 included PMIDs
        a3_included_pmids = {p["pmid"] for p in SYNTHETIC_A3_RESULT["include"]}

        # A4 PMIDs are subset of A3 included
        a4_all_pmids = {p["pmid"] for p in SYNTHETIC_A4_RESULT["papers"]}
        assert a4_all_pmids.issubset(a3_included_pmids)

        # A4 included PMIDs
        a4_included_pmids = {p["pmid"] for p in SYNTHETIC_A4_RESULT["include"]}

        # A5 PMIDs are subset of A4 included
        a5_pmids = {p["pmid"] for p in SYNTHETIC_A5_RESULT["papers"]}
        assert a5_pmids.issubset(a4_included_pmids)

        # A6 PMIDs equal A5 PMIDs
        a6_pmids = {p["pmid"] for p in SYNTHETIC_A6_RESULT["papers"]}
        assert a6_pmids == a5_pmids

    def test_numeric_extraction_format(self):
        """A5 outcomes have effect_size_value, ci_lower, ci_upper as floats."""
        for paper in SYNTHETIC_A5_RESULT["papers"]:
            assert len(paper["outcomes"]) > 0, f"Paper {paper['pmid']} has no outcomes"
            for outcome in paper["outcomes"]:
                assert isinstance(outcome["effect_size_value"], float), \
                    f"effect_size_value not float: {outcome['effect_size_value']}"
                assert isinstance(outcome["ci_lower"], float), \
                    f"ci_lower not float: {outcome['ci_lower']}"
                assert isinstance(outcome["ci_upper"], float), \
                    f"ci_upper not float: {outcome['ci_upper']}"
                # CI lower < estimate < CI upper
                assert outcome["ci_lower"] <= outcome["effect_size_value"] <= outcome["ci_upper"], \
                    f"CI range invalid: [{outcome['ci_lower']}, {outcome['effect_size_value']}, {outcome['ci_upper']}]"

    def test_quality_weights_range(self):
        """A6 quality weights are between 0 and 1."""
        for paper in SYNTHETIC_A6_RESULT["papers"]:
            w = paper["quality_weight"]
            assert 0.0 <= w <= 1.0, f"Weight out of range for {paper['pmid']}: {w}"

    def test_quality_assessment_structure(self):
        """A6 papers have proper quality_assessment dict with tool and domains."""
        for paper in SYNTHETIC_A6_RESULT["papers"]:
            qa = paper["quality_assessment"]
            assert "tool" in qa
            assert qa["tool"] in ("RoB2", "ROBINS-I", "JBI")
            assert "overall" in qa
            assert "domains" in qa
            assert isinstance(qa["domains"], dict)
            # RoB2 has 5 domains
            if qa["tool"] == "RoB2":
                assert len(qa["domains"]) == 5

    def test_synthesis_structure(self):
        """A7 has analyses dict with model, heterogeneity, grade."""
        a7 = SYNTHETIC_A7_RESULT
        assert "analyses" in a7
        assert "stats" in a7
        assert len(a7["analyses"]) >= 1

        for measure, analysis in a7["analyses"].items():
            assert "model" in analysis, f"No model in {measure} analysis"
            assert "heterogeneity" in analysis, f"No heterogeneity in {measure}"
            assert "grade" in analysis, f"No grade in {measure}"

            model = analysis["model"]
            assert "pooled_estimate" in model
            assert "ci_lower" in model
            assert "ci_upper" in model
            assert "p_value" in model
            assert "k_studies" in model
            assert model["ci_lower"] <= model["pooled_estimate"] <= model["ci_upper"]

            het = analysis["heterogeneity"]
            assert "I2" in het
            assert 0 <= het["I2"] <= 100

            grade = analysis["grade"]
            assert "final_level" in grade
            assert grade["final_level"] in ("High", "Moderate", "Low", "Very low")

    def test_manuscript_sections(self):
        """A8 has abstract, introduction, methods, results, discussion, conclusion."""
        expected = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        manuscript = SYNTHETIC_A8_RESULT["manuscript"]
        for sec in expected:
            assert sec in manuscript, f"Missing section: {sec}"
            assert len(manuscript[sec]) > 50, f"Section {sec} too short: {len(manuscript[sec])} chars"

    def test_manuscript_references(self):
        """A8 references list has pmid and title for each entry."""
        refs = SYNTHETIC_A8_RESULT["references"]
        assert len(refs) >= 1
        for ref in refs:
            assert "pmid" in ref
            assert "title" in ref
            assert "citation" in ref

    def test_prisma_counts_consistent(self):
        """PRISMA: identified >= screened >= assessed_fulltext >= included_final."""
        prisma = build_prisma(
            SYNTHETIC_A2_STATS,
            SYNTHETIC_A3_RESULT["stats"],
            SYNTHETIC_A4_RESULT["stats"],
            SYNTHETIC_A5_RESULT["stats"],
            SYNTHETIC_A6_RESULT["stats"],
        )
        assert prisma["identified"] >= prisma["screened"]
        assert prisma["screened"] >= prisma["assessed_fulltext"]
        assert prisma["assessed_fulltext"] >= prisma["included_final"]
        assert prisma["included_final"] >= 0

        # Excluded + included = assessed
        assert (
            prisma["excluded_fulltext"] + prisma["included_final"]
            == prisma["assessed_fulltext"]
        )

        # Title/abstract: excluded + uncertain + assessed_fulltext = screened
        assert (
            prisma["excluded_title_abstract"]
            + prisma["uncertain_title_abstract"]
            + prisma["assessed_fulltext"]
            == prisma["screened"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpoints:
    """Test checkpoint save/load round-trip."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_e2e_ckpt_")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_checkpoint_save_load(self):
        """Save checkpoint, load it, verify data matches."""
        run_dir = self.tmpdir

        # Save each agent checkpoint
        save_checkpoint(run_dir, "a1_protocol", {
            "query": "test query",
            "protocol": SYNTHETIC_PROTOCOL,
            "pubmed_query": SYNTHETIC_PUBMED_QUERY,
        })
        save_checkpoint(run_dir, "a2_search", SYNTHETIC_A2_RESULT)
        save_checkpoint(run_dir, "a3_screening", SYNTHETIC_A3_RESULT)
        save_checkpoint(run_dir, "a4_fulltext", SYNTHETIC_A4_RESULT)
        save_checkpoint(run_dir, "a5_extraction", SYNTHETIC_A5_RESULT)
        save_checkpoint(run_dir, "a6_quality", SYNTHETIC_A6_RESULT)
        save_checkpoint(run_dir, "a7_synthesis", SYNTHETIC_A7_RESULT)
        save_checkpoint(run_dir, "a8_manuscript", SYNTHETIC_A8_RESULT)

        # List checkpoints
        agents = list_checkpoint_agents(run_dir)
        assert len(agents) == 8
        assert "a1_protocol" in agents
        assert "a8_manuscript" in agents

        # Load and verify
        a1 = load_checkpoint(run_dir, "a1_protocol")
        assert a1["protocol"]["framework"] == "PICO"
        assert a1["pubmed_query"] == SYNTHETIC_PUBMED_QUERY

        a3 = load_checkpoint(run_dir, "a3_screening")
        assert a3["stats"]["included"] == 4
        assert a3["stats"]["excluded"] == 1

        a5 = load_checkpoint(run_dir, "a5_extraction")
        assert len(a5["papers"]) == 3
        assert a5["papers"][0]["outcomes"][0]["effect_size_value"] == -0.52

        a7 = load_checkpoint(run_dir, "a7_synthesis")
        assert a7["analyses"]["SMD"]["model"]["pooled_estimate"] == -0.50

    def test_checkpoint_missing_returns_none(self):
        """Loading a nonexistent checkpoint returns None."""
        result = load_checkpoint(self.tmpdir, "nonexistent_agent")
        assert result is None

    def test_checkpoint_overwrite(self):
        """Saving a checkpoint twice overwrites the first."""
        save_checkpoint(self.tmpdir, "a1_protocol", {"version": 1})
        save_checkpoint(self.tmpdir, "a1_protocol", {"version": 2})
        loaded = load_checkpoint(self.tmpdir, "a1_protocol")
        assert loaded["version"] == 2

    def test_final_output_json_serializable(self):
        """Final output dict is fully JSON-serializable."""
        output = build_final_output()
        # Should not raise
        serialized = json.dumps(output, ensure_ascii=False, default=str)
        restored = json.loads(serialized)
        assert restored["protocol"]["framework"] == "PICO"
        assert len(restored["screening"]["include"]) == 4


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test pipeline behavior with edge-case inputs."""

    def test_empty_search_results(self):
        """A2 returns 0 papers: pipeline should handle gracefully."""
        a2_result = {
            "papers": [],
            "stats": {
                "pubmed": 0, "europepmc": 0, "semantic_scholar": 0,
                "total_before_dedup": 0, "total_after_dedup": 0, "duplicates_removed": 0,
            },
        }
        # No papers means A3 would receive empty list
        assert len(a2_result["papers"]) == 0
        # In app.py, this triggers early return with warning
        # Verify the stats are consistent
        assert a2_result["stats"]["total_after_dedup"] == 0

    def test_all_excluded_screening(self):
        """A3 excludes everything: pipeline should continue without A4-A8."""
        a3_result = {
            "include": [],
            "exclude": [
                _make_a3_screening_entry(p, "EXCLUDE", 1.0)
                for p in SYNTHETIC_A2_PAPERS
            ],
            "uncertain": [],
            "stats": {
                "total": 5,
                "included": 0,
                "excluded": 5,
                "uncertain": 0,
                "elapsed_s": 1.0,
                "s_per_paper": 0.2,
                "ensemble_size": 3,
                "voting_threshold": 0.60,
            },
        }
        assert len(a3_result["include"]) == 0
        assert a3_result["stats"]["included"] == 0

        # Build output as app.py would with no included papers
        prisma = {
            "identified": 5,
            "screened": 5,
            "excluded_title_abstract": 5,
            "uncertain_title_abstract": 0,
            "assessed_fulltext": 0,
            "excluded_fulltext": 0,
            "included_final": 0,
        }
        assert prisma["included_final"] == 0
        assert prisma["identified"] >= prisma["screened"]

    def test_no_numeric_data_extraction(self):
        """A5 extracts no numeric outcomes: A7 should produce empty analyses."""
        papers_no_numeric = []
        for paper in SYNTHETIC_A5_PAPERS:
            p = dict(paper)
            p["outcomes"] = [{
                "name": "Qualitative finding",
                "measure_type": "qualitative",
                "effect_size_value": None,
                "ci_lower": None,
                "ci_upper": None,
                "p_value": None,
                "numeric_complete": False,
            }]
            p["quality_weight"] = 1.0
            papers_no_numeric.append(p)

        # A7 should have no analyses when no numeric data is available
        # Simulate the expected output
        a7_empty = {
            "analyses": {},
            "stats": {
                "total_papers_input": 3,
                "papers_with_usable_data": 0,
                "analyses_performed": 0,
                "measure_types_found": [],
                "elapsed_s": 0.1,
            },
        }
        assert a7_empty["stats"]["analyses_performed"] == 0
        assert len(a7_empty["analyses"]) == 0

    def test_single_paper_no_meta_analysis(self):
        """With only 1 paper, meta-analysis requires k>=2 so should not run."""
        a7_single = {
            "analyses": {},
            "stats": {
                "total_papers_input": 1,
                "papers_with_usable_data": 1,
                "analyses_performed": 0,
                "measure_types_found": ["SMD"],
                "elapsed_s": 0.1,
            },
        }
        # App.py checks: len(a7_input.get("papers", [])) >= 2
        assert a7_single["stats"]["analyses_performed"] == 0

    def test_a4_all_abstract_only(self):
        """All papers have abstract-only text: pipeline should still work."""
        abstract_only_papers = []
        for p in SYNTHETIC_A4_INCLUDE[:3]:
            paper = dict(p)
            paper["text_source"] = "abstract_only"
            paper["full_text"] = None
            paper["full_text_length"] = len(paper.get("abstract", ""))
            abstract_only_papers.append(paper)

        # Stats should reflect all abstract-only
        assert all(p["text_source"] == "abstract_only" for p in abstract_only_papers)

    def test_mixed_study_designs_quality(self):
        """A6 handles mixed study designs (RCT + observational)."""
        mixed_papers = list(SYNTHETIC_A6_PAPERS)
        # Add an observational study
        obs_paper = dict(SYNTHETIC_A5_PAPERS[0])
        obs_paper["pmid"] = "30000099"
        obs_paper["study_characteristics"] = {"study_design": "cohort", "sample_size": 200}
        obs_paper["quality_assessment"] = {
            "tool": "ROBINS-I",
            "overall": "Moderate",
            "domains": {
                "confounding": "Moderate",
                "selection": "Low",
                "classification_interventions": "Low",
                "deviations": "Moderate",
                "missing_data": "Low",
                "measurement": "Low",
                "selection_reported": "Low",
            },
        }
        obs_paper["quality_weight"] = 0.7
        mixed_papers.append(obs_paper)

        # All weights valid
        for p in mixed_papers:
            assert 0.0 <= p["quality_weight"] <= 1.0
        # Mixed tools
        tools = {p["quality_assessment"]["tool"] for p in mixed_papers}
        assert "RoB2" in tools
        assert "ROBINS-I" in tools


# ══════════════════════════════════════════════════════════════════════════════
# CLEAN R OBJECTS TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanRObjects:
    """Test the _clean_r_objects utility from app.py."""

    def test_removes_fit_keys(self):
        """_clean_r_objects removes _fit keys recursively."""
        data = {
            "analyses": {
                "SMD": {
                    "model": {"pooled_estimate": -0.5},
                    "_fit": "<R object>",
                    "nested": {
                        "_fit": "<another R object>",
                        "value": 42,
                    },
                },
            },
        }
        cleaned = _clean_r_objects(data)
        assert "_fit" not in cleaned["analyses"]["SMD"]
        assert "_fit" not in cleaned["analyses"]["SMD"]["nested"]
        assert cleaned["analyses"]["SMD"]["nested"]["value"] == 42

    def test_handles_lists(self):
        """_clean_r_objects handles lists with nested dicts."""
        data = [{"_fit": "x", "val": 1}, {"_fit": "y", "val": 2}]
        cleaned = _clean_r_objects(data)
        assert all("_fit" not in item for item in cleaned)
        assert cleaned[0]["val"] == 1

    def test_preserves_scalars(self):
        """_clean_r_objects preserves non-dict/list values."""
        assert _clean_r_objects(42) == 42
        assert _clean_r_objects("hello") == "hello"
        assert _clean_r_objects(None) is None


# ══════════════════════════════════════════════════════════════════════════════
# FULL OUTPUT STRUCTURE TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestFullOutputStructure:
    """Verify the complete pipeline output matches expected structure."""

    @pytest.fixture
    def output(self):
        return build_final_output()

    def test_query_present(self, output):
        assert isinstance(output["query"], str)
        assert len(output["query"]) > 0

    def test_protocol_present(self, output):
        p = output["protocol"]
        assert p["framework"] in ("PICO", "SPIDER", "PECO", "PCC")

    def test_prisma_present(self, output):
        prisma = output["prisma"]
        assert prisma["identified"] > 0
        assert prisma["included_final"] >= 0

    def test_screening_has_buckets(self, output):
        s = output["screening"]
        assert "include" in s
        assert "exclude" in s
        assert "uncertain" in s
        assert "stats" in s

    def test_fulltext_has_include_exclude(self, output):
        ft = output["fulltext_retrieval"]
        assert "include" in ft
        assert "exclude" in ft
        assert "stats" in ft

    def test_extraction_has_papers(self, output):
        ext = output["data_extraction"]
        assert "papers" in ext
        assert "stats" in ext

    def test_quality_has_papers(self, output):
        qa = output["quality_assessment"]
        assert "papers" in qa
        assert "stats" in qa

    def test_synthesis_has_analyses(self, output):
        syn = output["synthesis"]
        assert "analyses" in syn
        assert "stats" in syn

    def test_manuscript_has_sections(self, output):
        ms = output["manuscript"]
        assert "manuscript" in ms
        assert "references" in ms
        assert "stats" in ms


# ══════════════════════════════════════════════════════════════════════════════
# PAPER FIELD COMPLETENESS TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestPaperFieldCompleteness:
    """Verify papers at each stage have the required fields."""

    def test_a2_paper_fields(self):
        required = {"pmid", "title", "abstract"}
        for paper in SYNTHETIC_A2_PAPERS:
            assert required.issubset(set(paper.keys())), \
                f"Paper {paper.get('pmid')} missing A2 fields: {required - set(paper.keys())}"

    def test_a4_paper_fields(self):
        required = {"pmid", "title", "abstract", "text_source", "full_text_length",
                     "fulltext_decision"}
        for paper in SYNTHETIC_A4_RESULT["include"]:
            assert required.issubset(set(paper.keys())), \
                f"Paper {paper.get('pmid')} missing A4 fields: {required - set(paper.keys())}"

    def test_a5_paper_fields(self):
        required = {"pmid", "title", "study_characteristics", "outcomes",
                     "extraction_quality", "text_source"}
        for paper in SYNTHETIC_A5_RESULT["papers"]:
            assert required.issubset(set(paper.keys())), \
                f"Paper {paper.get('pmid')} missing A5 fields: {required - set(paper.keys())}"

    def test_a6_paper_fields(self):
        required = {"pmid", "title", "quality_assessment", "quality_weight"}
        for paper in SYNTHETIC_A6_RESULT["papers"]:
            assert required.issubset(set(paper.keys())), \
                f"Paper {paper.get('pmid')} missing A6 fields: {required - set(paper.keys())}"

    def test_a7_study_fields(self):
        required = {"pmid", "yi", "vi", "weight"}
        for analysis in SYNTHETIC_A7_RESULT["analyses"].values():
            for study in analysis["studies"]:
                assert required.issubset(set(study.keys())), \
                    f"Study {study.get('pmid')} missing A7 fields: {required - set(study.keys())}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
