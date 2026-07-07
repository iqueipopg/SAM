# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for SAM pipeline tests.

Provides reusable mock data and helpers for testing new features (T1-T8).
All fixtures are CPU-only, no GPU/LLM/internet required.
"""

import json
import os
import sys
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ══════════════════════════════════════════════════════════════════════════════
# T1 FIXTURES: Author extraction (problematic author names)
# ══════════════════════════════════════════════════════════════════════════════

# Papers with author names that historically break extraction:
# - Accented characters (Garcia -> García, Muller -> Müller)
# - Hyphens in surnames (O'Brien-Smith)
# - Multiple given names (José María)
# - Non-Latin diacritics (Günther-Straße)
# - CJK-origin romanized names (Zhang Wei, Kim Hyun-Soo)
# - Single-name authors (mononym, common in some cultures)

FIXTURE_AUTHORS_SPECIAL = [
    "García-López R",
    "Müller K",
    "O'Brien-Smith T",
    "José María F",
    "Günther-Straße H",
]

FIXTURE_AUTHORS_CJK_ROMANIZED = [
    "Zhang W",
    "Kim H-S",
    "Tanaka Y",
]

FIXTURE_AUTHORS_MONONYM = [
    "Sukarno",
]

FIXTURE_A5_PAPERS_WITH_PROBLEMATIC_AUTHORS = [
    # Paper 1: Accented/hyphenated authors
    {
        "pmid": "90000001",
        "title": "Intervenciones cognitivas en pacientes con depresión: un ensayo controlado",
        "abstract": (
            "Methods: RCT with 100 adults diagnosed with depression. "
            "Participants were randomized to CBT (n=50) or waitlist (n=50). "
            "Results: Significant reduction in anxiety (SMD=-0.48, 95% CI [-0.76, -0.20])."
        ),
        "authors": FIXTURE_AUTHORS_SPECIAL,
        "journal": "Revista de Psiquiatría Clínica",
        "year": "2024",
        "text_source": "pmc",
        "full_text": (
            "Introduction: Depression is a leading cause of disability. "
            "Methods: This was a single-blind RCT conducted across 3 hospitals in Spain. "
            "100 adults (age 18-65) with MDD were randomized to CBT (n=50) or waitlist (n=50). "
            "Results: CBT group showed significant improvement (SMD=-0.48, 95% CI [-0.76, -0.20], p=0.001). "
            "Discussion: Results support CBT for depression in Spanish-speaking populations."
        ),
        "full_text_length": 350,
        "sections": {},
        "doi": "10.1000/test.90000001",
        "pmcid": "PMC90000001",
        "oa_url": None,
        "fulltext_decision": "INCLUDE",
        "fulltext_reasoning": "Paper meets all inclusion criteria.",
        "exclusion_reason": None,
        "study_characteristics": {
            "study_design": "RCT",
            "sample_size": 100,
            "country": "Spain",
            "setting": "hospital",
        },
        "intervention_exposure": {
            "intervention_name": "CBT",
            "comparator_name": "Waitlist",
            "duration": "12 weeks",
            "dosage": "Weekly sessions",
        },
        "outcomes": [{
            "name": "Depression score (BDI)",
            "measure_type": "SMD",
            "effect_size_value": -0.48,
            "ci_lower": -0.76,
            "ci_upper": -0.20,
            "p_value": 0.001,
            "n_intervention": 50,
            "n_control": 50,
            "numeric_complete": True,
        }],
        "extraction_quality": "complete",
        "extraction_confidence": "high",
    },
    # Paper 2: CJK-romanized authors
    {
        "pmid": "90000002",
        "title": "Mindfulness-based stress reduction in East Asian populations: RCT",
        "abstract": (
            "Objective: Evaluate MBSR for anxiety in East Asian adults. "
            "Methods: 80 participants randomized. "
            "Results: SMD=-0.35, 95% CI [-0.65, -0.05]."
        ),
        "authors": FIXTURE_AUTHORS_CJK_ROMANIZED,
        "journal": "Asian J Psychiatry",
        "year": "2023",
        "text_source": "unpaywall_pdf",
        "full_text": None,
        "full_text_length": 0,
        "sections": {},
        "doi": "10.1000/test.90000002",
        "pmcid": "",
        "oa_url": None,
        "fulltext_decision": "INCLUDE",
        "fulltext_reasoning": "Meets criteria.",
        "exclusion_reason": None,
        "study_characteristics": {
            "study_design": "RCT",
            "sample_size": 80,
            "country": "South Korea",
            "setting": "community",
        },
        "intervention_exposure": {
            "intervention_name": "MBSR",
            "comparator_name": "Waitlist",
            "duration": "8 weeks",
            "dosage": "Weekly 2-hour sessions",
        },
        "outcomes": [{
            "name": "Anxiety score (GAD-7)",
            "measure_type": "SMD",
            "effect_size_value": -0.35,
            "ci_lower": -0.65,
            "ci_upper": -0.05,
            "p_value": 0.02,
            "n_intervention": 40,
            "n_control": 40,
            "numeric_complete": True,
        }],
        "extraction_quality": "complete",
        "extraction_confidence": "high",
    },
    # Paper 3: Mononym author + missing fields
    {
        "pmid": "90000003",
        "title": "Community mental health interventions in Southeast Asia",
        "abstract": (
            "Methods: Cross-sectional survey of 200 participants. "
            "Results: OR=2.1, 95% CI [1.3, 3.4]."
        ),
        "authors": FIXTURE_AUTHORS_MONONYM,
        "journal": "Lancet Psychiatry",
        "year": "2022",
        "text_source": "abstract_only",
        "full_text": None,
        "full_text_length": 0,
        "sections": {},
        "doi": "10.1000/test.90000003",
        "pmcid": "",
        "oa_url": None,
        "fulltext_decision": "INCLUDE",
        "fulltext_reasoning": "Meets criteria.",
        "exclusion_reason": None,
        "study_characteristics": {
            "study_design": "Cross-sectional",
            "sample_size": 200,
            "country": "Indonesia",
            "setting": "community",
        },
        "intervention_exposure": {
            "intervention_name": "Community intervention",
            "comparator_name": "No intervention",
            "duration": "6 months",
            "dosage": "N/A",
        },
        "outcomes": [{
            "name": "Mental health utilization",
            "measure_type": "OR",
            "effect_size_value": 2.1,
            "ci_lower": 1.3,
            "ci_upper": 3.4,
            "p_value": 0.003,
            "n_intervention": 100,
            "n_control": 100,
            "numeric_complete": True,
        }],
        "extraction_quality": "partial",
        "extraction_confidence": "medium",
    },
]


@pytest.fixture
def a5_papers_special_authors():
    """A5-style papers with problematic author names for T1 regression testing."""
    return FIXTURE_A5_PAPERS_WITH_PROBLEMATIC_AUTHORS


# ══════════════════════════════════════════════════════════════════════════════
# T2 FIXTURES: config/agents.json per-agent LLM params
# ══════════════════════════════════════════════════════════════════════════════

# Canonical agents.json that overrides defaults from config.yaml
FIXTURE_AGENTS_JSON = {
    "agents": {
        "agent1_protocol": {
            "temperature": 0.3,
            "repetition_penalty": 1.3,
            "max_tokens": 400,
            "max_input_chars": 4000,
            "model_name": "qwen2.5-7b",
        },
        "agent3_screening": {
            "temperature": 0.2,
            "repetition_penalty": 1.3,
            "max_tokens": 150,
            "max_input_chars": 800,
            "model_name": "qwen2.5-7b",
        },
        "agent4_validation": {
            "temperature": 0.2,
            "repetition_penalty": 1.3,
            "max_tokens": 200,
            "max_input_chars": 6000,
            "model_name": "qwen2.5-7b",
        },
        "agent5_extraction": {
            "temperature": 0.1,
            "repetition_penalty": 1.3,
            "max_tokens": 500,
            "max_input_chars": 6000,
            "model_name": "qwen2.5-7b",
        },
        "agent6_quality": {
            "temperature": 0.2,
            "repetition_penalty": 1.3,
            "max_tokens": 600,
            "max_input_chars": 5000,
            "model_name": "qwen2.5-7b",
        },
        "agent8_manuscript": {
            "temperature": {
                "abstract": 0.3,
                "introduction": 0.6,
                "methods": 0.3,
                "results": 0.3,
                "discussion": 0.6,
                "conclusion": 0.4,
            },
            "repetition_penalty": 1.3,
            "max_tokens": {
                "abstract": 400,
                "introduction": 800,
                "methods": 1000,
                "results": 1200,
                "discussion": 1000,
                "conclusion": 500,
            },
            "max_input_chars": 8000,
            "model_name": "qwen2.5-7b",
        },
    },
    "defaults": {
        "temperature": 0.3,
        "repetition_penalty": 1.3,
        "max_tokens": 300,
        "max_input_chars": 4000,
    },
}

# agents.json with a custom temperature override for A5
FIXTURE_AGENTS_JSON_CUSTOM = {
    "agents": {
        "agent5_extraction": {
            "temperature": 0.05,
            "repetition_penalty": 1.5,
            "max_tokens": 800,
            "max_input_chars": 8000,
            "model_name": "qwen2.5-7b",
        },
    },
    "defaults": {
        "temperature": 0.3,
        "repetition_penalty": 1.3,
        "max_tokens": 300,
        "max_input_chars": 4000,
    },
}


@pytest.fixture
def agents_json_path(tmp_path):
    """Write FIXTURE_AGENTS_JSON to a temp config/agents.json and return the path."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "agents.json"
    path.write_text(json.dumps(FIXTURE_AGENTS_JSON, indent=2))
    return str(path)


@pytest.fixture
def agents_json_custom_path(tmp_path):
    """Write FIXTURE_AGENTS_JSON_CUSTOM to a temp config/agents.json."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "agents.json"
    path.write_text(json.dumps(FIXTURE_AGENTS_JSON_CUSTOM, indent=2))
    return str(path)


@pytest.fixture
def no_agents_json_path(tmp_path):
    """Return a config dir path where agents.json does NOT exist (backward compat)."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return str(config_dir / "agents.json")


# ══════════════════════════════════════════════════════════════════════════════
# T4 FIXTURES: A9 Explainer agent + Excel export
# ══════════════════════════════════════════════════════════════════════════════

# Full pipeline output dict as A9 would receive it
# Reuses the synthetic data from test_e2e_pipeline.py but builds a self-contained
# version here so conftest doesn't import from the test file.

def _build_full_pipeline_for_a9():
    """Build a complete pipeline output dict for A9 explainability testing."""
    protocol = {
        "framework": "PICO",
        "population": "adults with depression",
        "intervention": "CBT",
        "comparator": "waitlist",
        "outcome": "anxiety scores",
        "research_question": "Is CBT effective for anxiety in adults with depression?",
        "inclusion_criteria": [
            "Adults aged 18+",
            "Diagnosed with depression",
            "CBT as intervention",
        ],
        "exclusion_criteria": [
            "Pediatric populations",
            "Animal studies",
        ],
        "pubmed_query": '("CBT" OR "cognitive behavioral therapy") AND depression AND anxiety',
    }

    papers_a2 = [
        {
            "pmid": "40000001",
            "title": "CBT for anxiety in depressed adults: an RCT",
            "abstract": "Methods: 120 adults randomized. Results: SMD=-0.52.",
            "authors": ["Smith J", "García-López R"],
            "journal": "J Clin Psychology",
            "year": "2022",
            "source": "pubmed",
        },
        {
            "pmid": "40000002",
            "title": "Multi-site CBT trial for comorbid depression and anxiety",
            "abstract": "90 patients randomized. CBT group showed lower anxiety (SMD=-0.38).",
            "authors": ["Müller K", "O'Brien-Smith T"],
            "journal": "Behav Res Therapy",
            "year": "2021",
            "source": "europepmc",
        },
        {
            "pmid": "40000003",
            "title": "Cognitive therapy vs waitlist for depression and anxiety: RCT",
            "abstract": "80 participants. CBT reduced anxiety (SMD=-0.61).",
            "authors": ["Zhang W", "Kim H-S", "Tanaka Y"],
            "journal": "Psychol Med",
            "year": "2023",
            "source": "semantic_scholar",
        },
        {
            "pmid": "40000004",
            "title": "Pharmacotherapy vs CBT for depression: cost analysis",
            "abstract": "This review examines costs of SSRIs vs CBT. No anxiety outcomes measured.",
            "authors": ["Taylor W"],
            "journal": "Pharmacoecon Rev",
            "year": "2020",
            "source": "pubmed",
        },
    ]

    screening = {
        "include": [
            {"pmid": "40000001", "title": papers_a2[0]["title"],
             "abstract": papers_a2[0]["abstract"], "authors": papers_a2[0]["authors"],
             "decision": "INCLUDE", "confidence": 1.0, "votes": ["INCLUDE"] * 3,
             "reasoning": "Meets all inclusion criteria."},
            {"pmid": "40000002", "title": papers_a2[1]["title"],
             "abstract": papers_a2[1]["abstract"], "authors": papers_a2[1]["authors"],
             "decision": "INCLUDE", "confidence": 1.0, "votes": ["INCLUDE"] * 3,
             "reasoning": "Meets all inclusion criteria."},
            {"pmid": "40000003", "title": papers_a2[2]["title"],
             "abstract": papers_a2[2]["abstract"], "authors": papers_a2[2]["authors"],
             "decision": "INCLUDE", "confidence": 0.67, "votes": ["INCLUDE", "INCLUDE", "EXCLUDE"],
             "reasoning": "Borderline but majority included."},
        ],
        "exclude": [
            {"pmid": "40000004", "title": papers_a2[3]["title"],
             "abstract": papers_a2[3]["abstract"], "authors": papers_a2[3]["authors"],
             "decision": "EXCLUDE", "confidence": 1.0, "votes": ["EXCLUDE"] * 3,
             "reasoning": "No anxiety outcome; cost analysis only."},
        ],
        "uncertain": [],
        "stats": {
            "total": 4, "included": 3, "excluded": 1, "uncertain": 0,
            "elapsed_s": 2.0, "s_per_paper": 0.5, "ensemble_size": 3, "voting_threshold": 0.60,
        },
    }

    fulltext_retrieval = {
        "papers": [
            {"pmid": "40000001", "title": papers_a2[0]["title"],
             "authors": papers_a2[0]["authors"], "journal": papers_a2[0]["journal"],
             "year": papers_a2[0]["year"], "doi": "10.1000/test.40000001",
             "text_source": "pmc", "full_text_length": 5000,
             "fulltext_decision": "INCLUDE", "fulltext_reasoning": "Meets criteria.",
             "exclusion_reason": None, "abstract": papers_a2[0]["abstract"]},
            {"pmid": "40000002", "title": papers_a2[1]["title"],
             "authors": papers_a2[1]["authors"], "journal": papers_a2[1]["journal"],
             "year": papers_a2[1]["year"], "doi": "10.1000/test.40000002",
             "text_source": "unpaywall_pdf", "full_text_length": 4200,
             "fulltext_decision": "INCLUDE", "fulltext_reasoning": "Meets criteria.",
             "exclusion_reason": None, "abstract": papers_a2[1]["abstract"]},
            {"pmid": "40000003", "title": papers_a2[2]["title"],
             "authors": papers_a2[2]["authors"], "journal": papers_a2[2]["journal"],
             "year": papers_a2[2]["year"], "doi": "10.1000/test.40000003",
             "text_source": "abstract_only", "full_text_length": 80,
             "fulltext_decision": "EXCLUDE", "fulltext_reasoning": "Abstract-only, insufficient data.",
             "exclusion_reason": "abstract_only", "abstract": papers_a2[2]["abstract"]},
        ],
        "include": [],  # filled below
        "exclude": [],  # filled below
        "stats": {
            "total_retrieved": 3, "pmc_fulltext": 1, "unpaywall_pdf": 1,
            "europepmc_fulltext": 0, "semanticscholar_pdf": 0, "scihub_pdf": 0,
            "abstract_only": 1, "fulltext_included": 2, "fulltext_excluded": 1,
            "elapsed_s": 3.0,
        },
    }
    fulltext_retrieval["include"] = [fulltext_retrieval["papers"][0], fulltext_retrieval["papers"][1]]
    fulltext_retrieval["exclude"] = [fulltext_retrieval["papers"][2]]

    data_extraction = {
        "papers": [
            {
                "pmid": "40000001", "title": papers_a2[0]["title"],
                "abstract": papers_a2[0]["abstract"],
                "authors": papers_a2[0]["authors"],
                "journal": papers_a2[0]["journal"], "year": papers_a2[0]["year"],
                "text_source": "pmc",
                "study_characteristics": {
                    "study_design": "RCT", "sample_size": 120,
                    "country": "USA", "setting": "outpatient",
                },
                "intervention_exposure": {
                    "intervention_name": "CBT", "comparator_name": "Waitlist",
                    "duration": "12 weeks", "dosage": "Weekly sessions",
                },
                "outcomes": [{
                    "name": "Anxiety score (BAI)", "measure_type": "SMD",
                    "effect_size_value": -0.52, "ci_lower": -0.78, "ci_upper": -0.26,
                    "p_value": 0.001, "n_intervention": 60, "n_control": 60,
                    "numeric_complete": True,
                }],
                "extraction_quality": "complete", "extraction_confidence": "high",
            },
            {
                "pmid": "40000002", "title": papers_a2[1]["title"],
                "abstract": papers_a2[1]["abstract"],
                "authors": papers_a2[1]["authors"],
                "journal": papers_a2[1]["journal"], "year": papers_a2[1]["year"],
                "text_source": "unpaywall_pdf",
                "study_characteristics": {
                    "study_design": "RCT", "sample_size": 90,
                    "country": "UK", "setting": "community",
                },
                "intervention_exposure": {
                    "intervention_name": "CBT", "comparator_name": "Waitlist",
                    "duration": "10 weeks", "dosage": "Biweekly sessions",
                },
                "outcomes": [{
                    "name": "Anxiety score (BAI)", "measure_type": "SMD",
                    "effect_size_value": -0.38, "ci_lower": -0.72, "ci_upper": -0.04,
                    "p_value": 0.03, "n_intervention": 45, "n_control": 45,
                    "numeric_complete": True,
                }],
                "extraction_quality": "complete", "extraction_confidence": "high",
            },
        ],
        "stats": {
            "total": 2, "complete_extraction": 2, "partial_extraction": 0,
            "failed_extraction": 0, "high_confidence": 2, "medium_confidence": 0,
            "low_confidence": 0, "numeric_complete": 2, "numeric_recovered": 0,
            "dual_extraction": True, "elapsed_s": 5.0,
        },
    }

    quality_assessment = {
        "papers": [
            {
                **data_extraction["papers"][0],
                "quality_assessment": {
                    "tool": "RoB2", "overall": "Low",
                    "domains": {
                        "randomization_process": "Low",
                        "deviations_from_interventions": "Low",
                        "missing_outcome_data": "Low",
                        "measurement_of_outcome": "Low",
                        "selection_of_reported_result": "Low",
                    },
                },
                "quality_weight": 1.0,
            },
            {
                **data_extraction["papers"][1],
                "quality_assessment": {
                    "tool": "RoB2", "overall": "Some concerns",
                    "domains": {
                        "randomization_process": "Low",
                        "deviations_from_interventions": "Some concerns",
                        "missing_outcome_data": "Low",
                        "measurement_of_outcome": "Low",
                        "selection_of_reported_result": "Low",
                    },
                },
                "quality_weight": 0.6,
            },
        ],
        "stats": {
            "total": 2, "rob2_assessed": 2, "robins_i_assessed": 0,
            "jbi_assessed": 0, "failed": 0, "elapsed_s": 4.0,
        },
    }

    synthesis = {
        "analyses": {
            "SMD": {
                "model": {
                    "pooled_estimate": -0.47, "ci_lower": -0.70, "ci_upper": -0.24,
                    "p_value": 0.0002, "k_studies": 2, "method": "REML",
                },
                "heterogeneity": {
                    "I2": 3.1, "tau2": 0.001, "Q": 1.03, "Q_p": 0.31, "Q_df": 1,
                },
                "plots": {
                    "forest": "outputs/plots/forest_SMD.png",
                    "funnel": "outputs/plots/funnel_SMD.png",
                },
                "grade": {
                    "starting_level": "High", "final_level": "Moderate",
                    "total_downgrades": 1, "downgrades": {"imprecision": 1},
                },
                "sensitivity": {
                    "leave_one_out": [
                        {"excluded": "40000001", "pooled_estimate": -0.38,
                         "ci_lower": -0.72, "ci_upper": -0.04},
                        {"excluded": "40000002", "pooled_estimate": -0.52,
                         "ci_lower": -0.78, "ci_upper": -0.26},
                    ],
                    "exclude_high_risk": {
                        "k_excluded": 0, "pooled_estimate": -0.47,
                        "ci_lower": -0.70, "ci_upper": -0.24,
                    },
                },
                "publication_bias": {
                    "egger_test": None, "trim_and_fill": None,
                    "note": "Too few studies (k=2) for Egger test (requires k>=10)",
                },
                "studies": [
                    {"pmid": "40000001", "yi": -0.52, "vi": 0.018, "weight": 1.0},
                    {"pmid": "40000002", "yi": -0.38, "vi": 0.030, "weight": 0.6},
                ],
            },
        },
        "stats": {
            "total_papers_input": 2, "papers_with_usable_data": 2,
            "analyses_performed": 1, "measure_types_found": ["SMD"], "elapsed_s": 2.0,
        },
    }

    manuscript = {
        "manuscript": {
            "abstract": "This systematic review examined CBT for anxiety in depression. Two RCTs met criteria (N=210). Pooled SMD=-0.47.",
            "introduction": "Depression and anxiety frequently co-occur. CBT is widely used.",
            "methods": "We searched PubMed, Europe PMC, and Semantic Scholar. Two RCTs met inclusion criteria.",
            "results": "Two RCTs (N=210) were included. Pooled SMD=-0.47, 95% CI [-0.70, -0.24].",
            "discussion": "CBT significantly reduces anxiety in depressed adults.",
            "conclusion": "CBT is effective for anxiety in depression (SMD=-0.47).",
        },
        "references": [
            {"pmid": "40000001", "title": papers_a2[0]["title"],
             "citation": "Smith J, García-López R (2022)", "journal": "J Clin Psychology"},
            {"pmid": "40000002", "title": papers_a2[1]["title"],
             "citation": "Müller K, O'Brien-Smith T (2021)", "journal": "Behav Res Therapy"},
        ],
        "stats": {
            "total_chunks": 16, "total_papers_cited": 2,
            "chunks_per_section": {
                "abstract": 2, "introduction": 3, "methods": 3,
                "results": 3, "discussion": 3, "conclusion": 2,
            },
            "elapsed_s": 12.0,
        },
        "pdf_path": None,
    }

    return {
        "query": "Is CBT effective for anxiety in adults with depression?",
        "protocol": protocol,
        "pubmed_query": protocol["pubmed_query"],
        "prisma": {
            "identified": 4, "identified_per_db": {"pubmed": 2, "europepmc": 1, "semantic_scholar": 1},
            "after_dedup": 4, "duplicates_removed": 0, "screened": 4,
            "excluded_title_abstract": 1, "uncertain_title_abstract": 0,
            "assessed_fulltext": 3, "excluded_fulltext": 1, "included_final": 2,
            "extracted": 2, "quality_assessed": 2,
        },
        "search_stats": {"pubmed": 2, "europepmc": 1, "semantic_scholar": 1,
                         "total_before_dedup": 4, "total_after_dedup": 4, "duplicates_removed": 0},
        "screening": screening,
        "fulltext_retrieval": fulltext_retrieval,
        "data_extraction": data_extraction,
        "quality_assessment": quality_assessment,
        "synthesis": synthesis,
        "manuscript": manuscript,
    }


FIXTURE_FULL_PIPELINE_FOR_A9 = _build_full_pipeline_for_a9()


@pytest.fixture
def full_pipeline_data():
    """Complete pipeline output (A1-A8) for A9 explainability testing."""
    import copy
    return copy.deepcopy(FIXTURE_FULL_PIPELINE_FOR_A9)


# Expected columns in the A9 Excel export (papers_raw.xlsx)
# Must match the headers list in agent9_explainer.py._generate_excel()
FIXTURE_A9_EXCEL_EXPECTED_COLUMNS = [
    "PMID", "Title", "Authors", "Year", "Journal", "DOI",
    "Abstract",
    "N (sample)", "Population", "Intervention", "Control",
    "Outcome", "Effect Size Type", "Effect Size Value",
    "CI Lower", "CI Upper",
    "A3 Decision", "A3 Confidence",
    "A4 Decision", "A4 Text Source",
    "A5 Extraction Quality", "A5 Confidence",
    "A6 Tool", "A6 Overall", "A6 Weight",
    "Reason Excluded", "Exclusion Stage",
]


@pytest.fixture
def a9_excel_columns():
    """Expected column names for the A9 Excel export."""
    return FIXTURE_A9_EXCEL_EXPECTED_COLUMNS


# ══════════════════════════════════════════════════════════════════════════════
# T6 FIXTURES: Fail-Safe N + publication bias stats
# ══════════════════════════════════════════════════════════════════════════════

# Synthetic dataset with 12 studies for Egger test (requires k >= 10)
# All SMD outcomes, varying effect sizes and sample sizes to create
# some publication bias signal for trim-and-fill.
FIXTURE_A7_STUDIES_FOR_PUB_BIAS = [
    {"pmid": f"60000{i:03d}", "yi": yi, "vi": vi, "weight": w}
    for i, (yi, vi, w) in enumerate([
        (-0.65, 0.015, 1.0),   # large effect, large sample
        (-0.58, 0.018, 1.0),   # large effect
        (-0.52, 0.020, 0.6),   # moderate effect
        (-0.48, 0.022, 1.0),   # moderate effect
        (-0.45, 0.025, 1.0),   # moderate effect
        (-0.40, 0.028, 0.6),   # moderate effect
        (-0.35, 0.032, 1.0),   # small-moderate effect
        (-0.30, 0.035, 0.6),   # small effect
        (-0.22, 0.040, 1.0),   # small effect
        (-0.15, 0.045, 0.6),   # very small effect
        (-0.10, 0.050, 1.0),   # negligible effect
        (-0.05, 0.060, 0.6),   # near-zero effect
    ], start=1)
]

# Expected A7 output structure WITH Fail-Safe N added
FIXTURE_A7_WITH_FAILSAFE_N = {
    "analyses": {
        "SMD": {
            "model": {
                "pooled_estimate": -0.35,
                "ci_lower": -0.45,
                "ci_upper": -0.25,
                "p_value": 0.00001,
                "k_studies": 12,
                "method": "REML",
            },
            "heterogeneity": {
                "I2": 28.5,
                "tau2": 0.008,
                "Q": 15.4,
                "Q_p": 0.17,
                "Q_df": 11,
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
                    {"excluded": f"60000{i:03d}", "pooled_estimate": -0.35 + (i * 0.005),
                     "ci_lower": -0.46 + (i * 0.003), "ci_upper": -0.24 + (i * 0.003)}
                    for i in range(1, 13)
                ],
                "exclude_high_risk": {
                    "k_excluded": 2,
                    "pooled_estimate": -0.38,
                    "ci_lower": -0.50,
                    "ci_upper": -0.26,
                },
            },
            "publication_bias": {
                "egger_test": {
                    "intercept": 1.42,
                    "se": 0.65,
                    "p_value": 0.052,
                    "significant": False,
                },
                "trim_and_fill": {
                    "k_missing": 2,
                    "adjusted_estimate": -0.30,
                    "adjusted_ci_lower": -0.42,
                    "adjusted_ci_upper": -0.18,
                    "side": "right",
                },
                "failsafe_n": {
                    "n": 187,
                    "alpha": 0.05,
                    "method": "Rosenthal",
                    "interpretation": "187 null studies needed to make result non-significant",
                },
            },
            "studies": FIXTURE_A7_STUDIES_FOR_PUB_BIAS,
        },
    },
    "stats": {
        "total_papers_input": 12,
        "papers_with_usable_data": 12,
        "analyses_performed": 1,
        "measure_types_found": ["SMD"],
        "elapsed_s": 3.5,
    },
}


@pytest.fixture
def a7_studies_for_pub_bias():
    """12 synthetic studies for publication bias / Fail-Safe N testing."""
    return list(FIXTURE_A7_STUDIES_FOR_PUB_BIAS)


@pytest.fixture
def a7_result_with_failsafe_n():
    """Expected A7 output structure with Fail-Safe N in publication_bias."""
    import copy
    return copy.deepcopy(FIXTURE_A7_WITH_FAILSAFE_N)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test artifacts."""
    out = tmp_path / "outputs"
    out.mkdir()
    plots = out / "plots"
    plots.mkdir()
    return str(out)
