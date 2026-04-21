"""Unit tests for Agent5._needs_second_pass numeric-recovery heuristic (BUG5).

These tests exercise the heuristic in isolation without loading an LLM:
- `_count_statistical_markers` regex-based counter
- `_needs_second_pass` decision boundary for numeric-poor pass-1 results

Run: python -m pytest tests/test_agent5_heuristic.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agents.agent5_extraction import Agent5Extraction  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def a5_instance():
    """Return an Agent5Extraction instance without loading any LLM."""
    return Agent5Extraction()


def _long_text(inner: str) -> str:
    """Return a >=2000-char body (so the length gate doesn't mask the tests)."""
    filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 60
    return filler + "\n\n" + inner + "\n\n" + filler


def _paper(full_text: str, text_source: str = "pmc") -> dict:
    return {
        "pmid": "90009001",
        "title": "Heuristic test paper",
        "abstract": "abstract text",
        "authors": ["Smith J"],
        "journal": "Test J",
        "year": "2024",
        "text_source": text_source,
        "full_text": full_text,
        "sections": {},
    }


def _complete_outcomes(effect: float = -0.45):
    """Return outcomes list that would mark pass 1 as confident."""
    return [{
        "name": "Outcome A",
        "measure_type": "SMD",
        "value": effect,
        "effect_size_value": effect,
        "ci_lower": effect - 0.2,
        "ci_upper": effect + 0.2,
        "se": 0.1,
        "numeric_complete": True,
    }]


def _empty_outcomes_no_effect():
    """Return outcomes where the name is captured but no effect size is populated."""
    return [{
        "name": "Outcome A",
        "measure_type": "SMD",
        "value": None,
        "effect_size_value": None,
        "ci_lower": None,
        "ci_upper": None,
        "se": None,
        "numeric_complete": False,
    }]


# ---------------------------------------------------------------------------
# _count_statistical_markers
# ---------------------------------------------------------------------------


class TestCountStatisticalMarkers:

    def test_no_markers_returns_zero(self, a5_instance):
        text = "This is a narrative paragraph about a study with no numbers at all."
        assert a5_instance._count_statistical_markers(text) == 0

    def test_cohens_d_equals(self, a5_instance):
        text = "The effect was large (d = 0.82)."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_cohens_d_label(self, a5_instance):
        text = "We computed Cohen's d as the main effect size."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_eta_squared(self, a5_instance):
        text = "The ANOVA showed η² = 0.14 indicating a large effect."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_eta2_ascii(self, a5_instance):
        text = "Partial eta2 = 0.09 for the main effect."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_hedges_g(self, a5_instance):
        text = "Pooled Hedges' g was 0.55, favoring the intervention."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_mean_sd_format(self, a5_instance):
        text = "Baseline weight was 82.3 ± 6.1 kg in the intervention group."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_correlation_r(self, a5_instance):
        text = "We observed a moderate correlation (r = 0.42)."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_odds_ratio(self, a5_instance):
        text = "Adjusted OR = 2.10 (95% CI 1.20-3.80)."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_risk_ratio(self, a5_instance):
        text = "The RR = 0.75 was statistically significant."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_hazard_ratio(self, a5_instance):
        text = "Mortality risk was reduced (HR = 0.68)."
        assert a5_instance._count_statistical_markers(text) >= 1

    def test_multiple_markers_accumulate(self, a5_instance):
        text = (
            "Mean BDI change was 12.1 ± 4.2 in the CBT group. "
            "Cohen's d = 0.48 favoring CBT. "
            "OR = 2.1 for response at 12 weeks."
        )
        # d, Cohen's d, SD, ±, OR   → clearly >= 2
        assert a5_instance._count_statistical_markers(text) >= 2

    def test_empty_text_is_zero(self, a5_instance):
        assert a5_instance._count_statistical_markers("") == 0
        assert a5_instance._count_statistical_markers(None) == 0


# ---------------------------------------------------------------------------
# _needs_second_pass
# ---------------------------------------------------------------------------


class TestNeedsSecondPass:

    def test_abstract_only_always_triggers(self, a5_instance):
        paper = _paper(_long_text("body"), text_source="abstract_only")
        result = {"outcomes": _complete_outcomes(), "study_characteristics": {"sample_size": 100}}
        assert a5_instance._needs_second_pass(result, paper) is True

    def test_short_text_triggers(self, a5_instance):
        paper = _paper("short body", text_source="pmc")
        result = {"outcomes": _complete_outcomes(), "study_characteristics": {"sample_size": 100}}
        assert a5_instance._needs_second_pass(result, paper) is True

    def test_no_outcomes_triggers(self, a5_instance):
        paper = _paper(_long_text("body"))
        result = {"outcomes": [], "study_characteristics": {"sample_size": 100}}
        assert a5_instance._needs_second_pass(result, paper) is True

    def test_complete_numeric_no_trigger(self, a5_instance):
        paper = _paper(_long_text("body with no stats"))
        result = {
            "outcomes": _complete_outcomes(),
            "study_characteristics": {"sample_size": 100},
        }
        # Confident: numeric_complete=True, study chars present, no stat markers
        assert a5_instance._needs_second_pass(result, paper) is False

    def test_missing_study_chars_triggers(self, a5_instance):
        paper = _paper(_long_text("body"))
        result = {
            "outcomes": _complete_outcomes(),
            "study_characteristics": None,
        }
        assert a5_instance._needs_second_pass(result, paper) is True

    def test_extended_heuristic_triggers_on_multiple_markers(self, a5_instance):
        """BUG5: full_text contains rich stats but pass 1 missed effect sizes."""
        stat_body = (
            "Cohen's d = 0.48 favored the intervention. "
            "Mean difference was 5.2 ± 1.8 points. "
            "OR = 2.1 (95% CI 1.2-3.8)."
        )
        paper = _paper(_long_text(stat_body))
        # Outcome captured by name but NO numeric value populated
        result = {
            "outcomes": _empty_outcomes_no_effect(),
            "study_characteristics": {"sample_size": 100},
        }
        assert a5_instance._needs_second_pass(result, paper) is True

    def test_extended_heuristic_skipped_when_effect_present(self, a5_instance):
        """If the effect_size is already populated, extra stat markers shouldn't force a pass."""
        stat_body = (
            "Cohen's d = 0.48 favored the intervention. "
            "Mean difference was 5.2 ± 1.8 points."
        )
        paper = _paper(_long_text(stat_body))
        result = {
            "outcomes": _complete_outcomes(),
            "study_characteristics": {"sample_size": 100},
        }
        # All gates passed and effect size present, so the heuristic is a no-op
        assert a5_instance._needs_second_pass(result, paper) is False

    def test_extended_heuristic_no_trigger_with_single_marker(self, a5_instance):
        """A single stat marker is not enough to trigger the extended heuristic."""
        stat_body = "The analysis used Cohen's d as the main effect size."  # 1 marker only
        paper = _paper(_long_text(stat_body))
        result = {
            "outcomes": [{
                "name": "Outcome A",
                "measure_type": "SMD",
                "value": -0.3,
                "effect_size_value": -0.3,
                "ci_lower": -0.5,
                "ci_upper": -0.1,
                "se": 0.1,
                "numeric_complete": True,
            }],
            "study_characteristics": {"sample_size": 100},
        }
        # Outcome is numeric-complete, so _needs_second_pass short-circuits to False
        assert a5_instance._needs_second_pass(result, paper) is False

    def test_half_outcomes_numeric_incomplete_triggers(self, a5_instance):
        paper = _paper(_long_text("body"))
        result = {
            "outcomes": [
                {"name": "A", "numeric_complete": True, "value": 0.3},
                {"name": "B", "numeric_complete": False, "value": None},
                {"name": "C", "numeric_complete": False, "value": None},
            ],
            "study_characteristics": {"sample_size": 100},
        }
        assert a5_instance._needs_second_pass(result, paper) is True
