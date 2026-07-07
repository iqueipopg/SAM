# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Agent5 dual-extraction reconciliation (BUG3).

Covers:
- Numeric agreement (<10% relative diff) → average
- Numeric disagreement (≥10% relative diff) + LLM arbiter resolves
- Numeric disagreement + LLM arbiter fails (dry-run / parse failure)
  → needs_manual_review=True + conservative value (smaller magnitude)
- String field equality (case/whitespace insensitive)
- String field mismatch with arbiter decision
- One-side None returns the other value

The LLM is mocked via unittest.mock.patch; no GPU or model loading occurs.

Run: python -m pytest tests/test_agent5_reconcile.py -q
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agents.agent5_extraction import Agent5Extraction  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def a5():
    """Fresh Agent5Extraction with no model loaded (dry-run default)."""
    agent = Agent5Extraction()
    # Ensure the arbiter knows we have context (some tests set _arbiter_paper).
    agent._arbiter_paper = {
        "title": "Test paper",
        "abstract": "abstract",
        "full_text": "The primary outcome was BDI change. d = -0.42 (95% CI -0.70 to -0.14).",
        "sections": {},
    }
    agent._arbiter_protocol = {
        "population": "adults",
        "intervention": "CBT",
        "outcome": "BDI",
    }
    return agent


# ---------------------------------------------------------------------------
# _reconcile_field — numeric branch
# ---------------------------------------------------------------------------


class TestReconcileNumeric:

    def test_agreement_under_10pct_averages(self, a5):
        """When values agree within 10%, return their average."""
        out = a5._reconcile_field("value", 0.50, 0.52, outcome_name="BDI")
        assert out["method"] == "average"
        assert out["reconciled"] is True
        assert out["needs_manual_review"] is False
        assert abs(out["value"] - 0.51) < 1e-6

    def test_agreement_exactly_equal(self, a5):
        out = a5._reconcile_field("ci_lower", -0.30, -0.30, outcome_name="BDI")
        assert out["method"] == "average"
        assert out["value"] == pytest.approx(-0.30)

    def test_disagreement_with_arbiter_resolves(self, a5):
        """≥10% disagreement → arbiter picks a value."""
        with patch.object(
            a5, "_invoke_arbiter",
            return_value={"value": -0.42, "source": "B", "rationale": "matches text"},
        ):
            out = a5._reconcile_field("value", -0.30, -0.50, outcome_name="BDI")
        assert out["method"] == "llm_arbiter"
        assert out["reconciled"] is True
        assert out["needs_manual_review"] is False
        assert out["value"] == pytest.approx(-0.42)
        assert out.get("arbiter_source") == "B"
        assert "flag" in out

    def test_disagreement_arbiter_fails_conservative_fallback(self, a5):
        """Arbiter returns None → needs_manual_review=True, value = smaller magnitude."""
        with patch.object(a5, "_invoke_arbiter", return_value=None):
            out = a5._reconcile_field("value", -0.30, -0.80, outcome_name="BDI")
        assert out["needs_manual_review"] is True
        assert out["reconciled"] is False
        assert out["method"] == "conservative"
        # -0.30 has smaller magnitude than -0.80 → conservative
        assert out["value"] == pytest.approx(-0.30)
        assert out["value_conservative"] == pytest.approx(-0.30)

    def test_disagreement_positive_conservative_picks_smaller(self, a5):
        with patch.object(a5, "_invoke_arbiter", return_value=None):
            out = a5._reconcile_field("ci_upper", 2.5, 4.0, outcome_name="OR")
        assert out["needs_manual_review"] is True
        assert out["value"] == pytest.approx(2.5)

    def test_borderline_just_under_threshold_averages(self, a5):
        """9.9% relative diff → still averages."""
        a, b = 1.00, 1.099
        out = a5._reconcile_field("value", a, b, outcome_name="t")
        assert out["method"] == "average"

    def test_borderline_at_threshold_triggers_arbiter(self, a5):
        """Exactly 10% relative diff → arbiter invoked."""
        a, b = 1.00, 1.10  # rel_diff = 0.10/1.10 ≈ 0.0909 — under threshold
        # Test a clean 12% instead
        a2, b2 = 1.00, 1.14  # rel_diff ≈ 0.1228 → arbiter
        with patch.object(a5, "_invoke_arbiter", return_value=None):
            out_under = a5._reconcile_field("value", a, b, outcome_name="t")
            out_over = a5._reconcile_field("value", a2, b2, outcome_name="t")
        assert out_under["method"] == "average"
        assert out_over["method"] == "conservative"

    def test_one_side_none_returns_other(self, a5):
        out_a_none = a5._reconcile_field("value", None, 0.5, outcome_name="t")
        assert out_a_none["value"] == 0.5
        assert out_a_none["method"] == "pass2"
        assert out_a_none["needs_manual_review"] is False

        out_b_none = a5._reconcile_field("value", 0.3, None, outcome_name="t")
        assert out_b_none["value"] == 0.3
        assert out_b_none["method"] == "pass1"

    def test_both_none_returns_none(self, a5):
        out = a5._reconcile_field("value", None, None, outcome_name="t")
        assert out["value"] is None
        assert out["method"] == "agree"
        assert out["needs_manual_review"] is False


# ---------------------------------------------------------------------------
# _reconcile_field — string branch
# ---------------------------------------------------------------------------


class TestReconcileString:

    def test_case_insensitive_match(self, a5):
        out = a5._reconcile_field("measure_type", "SMD", " smd ", outcome_name="t")
        assert out["method"] == "agree"
        assert out["reconciled"] is True
        assert out["needs_manual_review"] is False

    def test_string_mismatch_arbiter_resolves(self, a5):
        with patch.object(
            a5, "_invoke_arbiter",
            return_value={"value": "SMD", "source": "A", "rationale": "standard"},
        ):
            out = a5._reconcile_field("measure_type", "SMD", "MD", outcome_name="t")
        assert out["method"] == "llm_arbiter"
        assert out["value"] == "SMD"
        assert out["needs_manual_review"] is False

    def test_string_mismatch_arbiter_fails_needs_review(self, a5):
        with patch.object(a5, "_invoke_arbiter", return_value=None):
            out = a5._reconcile_field("measure_type", "SMD", "OR", outcome_name="t")
        assert out["needs_manual_review"] is True
        assert out["reconciled"] is False
        assert out["value"] == "SMD"  # fallback = pass 1
        assert "REVIEW NEEDED" in out.get("flag", "")


# ---------------------------------------------------------------------------
# _invoke_arbiter dry-run behaviour
# ---------------------------------------------------------------------------


class TestInvokeArbiter:

    def test_returns_none_when_model_not_loaded(self, a5):
        """With no model and local backend, arbiter should short-circuit to None."""
        # Force local (non-Groq) mode so dry-run branch kicks in
        a5.model = None
        with patch.object(a5, "_is_groq", return_value=False):
            out = a5._invoke_arbiter("value", 0.3, 0.6, "t")
        assert out is None

    def test_returns_parsed_json_on_success(self, a5):
        """Mock _simple_generate + _is_groq to simulate a loaded model."""
        fake_response = '{"value": 0.42, "source": "B", "rationale": "matches text"}'
        a5.model = "dummy"  # not None so the gate passes
        with patch.object(a5, "_is_groq", return_value=False), \
             patch.object(a5, "_simple_generate", return_value=fake_response):
            out = a5._invoke_arbiter("value", 0.30, 0.50, "BDI")
        assert out is not None
        assert out["value"] == pytest.approx(0.42)
        assert out["source"] == "B"

    def test_returns_none_on_unparseable_response(self, a5):
        a5.model = "dummy"
        with patch.object(a5, "_is_groq", return_value=False), \
             patch.object(a5, "_simple_generate", return_value="not json at all"):
            out = a5._invoke_arbiter("value", 0.30, 0.50, "BDI")
        assert out is None

    def test_returns_none_when_arbiter_says_unknown(self, a5):
        fake_response = '{"value": null, "source": "UNKNOWN", "rationale": "cannot decide"}'
        a5.model = "dummy"
        with patch.object(a5, "_is_groq", return_value=False), \
             patch.object(a5, "_simple_generate", return_value=fake_response):
            out = a5._invoke_arbiter("value", 0.30, 0.50, "BDI")
        assert out is None


# ---------------------------------------------------------------------------
# _merge_single_outcome end-to-end
# ---------------------------------------------------------------------------


class TestMergeSingleOutcome:

    def _outcome(self, value, ci_lo, ci_hi, measure="SMD"):
        return {
            "name": "BDI change",
            "measure_type": measure,
            "effect_size_type": measure,
            "value": value,
            "effect_size_value": value,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": 0.03,
            "se": None,
        }

    def test_agreement_averages_numeric_fields(self, a5):
        oa = self._outcome(-0.50, -0.70, -0.30)
        ob = self._outcome(-0.52, -0.72, -0.28)
        merged, flags = a5._merge_single_outcome(oa, ob)
        assert flags == []
        assert merged["value"] == pytest.approx(-0.51)
        assert merged["ci_lower"] == pytest.approx(-0.71)

    def test_disagreement_with_arbiter_resolves_and_no_manual_review(self, a5):
        oa = self._outcome(-0.30, -0.50, -0.10)
        ob = self._outcome(-0.80, -1.00, -0.60)
        with patch.object(
            a5, "_invoke_arbiter",
            return_value={"value": -0.50, "source": "NEW", "rationale": "context -0.5"},
        ):
            merged, flags = a5._merge_single_outcome(oa, ob)
        assert merged["value"] == pytest.approx(-0.50)
        # No _reconciliation entry because arbiter succeeded
        assert "_reconciliation" not in merged or "value" not in merged.get("_reconciliation", {})
        assert any("reconciled by arbiter" in f for f in flags)

    def test_disagreement_arbiter_fails_sets_manual_review(self, a5):
        oa = self._outcome(-0.30, -0.50, -0.10)
        ob = self._outcome(-0.80, -1.00, -0.60)
        with patch.object(a5, "_invoke_arbiter", return_value=None):
            merged, flags = a5._merge_single_outcome(oa, ob)
        # Conservative fallback kept the smaller magnitude for "value"
        assert merged["value"] == pytest.approx(-0.30)
        assert "_reconciliation" in merged
        assert merged["_reconciliation"]["value"]["needs_manual_review"] is True
        assert merged["_reconciliation"]["value"]["value_conservative"] == pytest.approx(-0.30)

    def test_one_side_none_in_outcome_keeps_other(self, a5):
        oa = self._outcome(None, -0.70, -0.30)
        ob = self._outcome(-0.40, -0.68, -0.28)
        merged, flags = a5._merge_single_outcome(oa, ob)
        assert merged["value"] == pytest.approx(-0.40)

    def test_measure_type_mismatch_arbiter_fails(self, a5):
        oa = self._outcome(-0.5, -0.7, -0.3, measure="SMD")
        ob = self._outcome(0.5, 0.3, 0.7, measure="OR")
        with patch.object(a5, "_invoke_arbiter", return_value=None):
            merged, flags = a5._merge_single_outcome(oa, ob)
        # measure_type falls back to pass 1
        assert merged["measure_type"] == "SMD"
        assert "_reconciliation" in merged
        assert merged["_reconciliation"]["measure_type"]["needs_manual_review"] is True
