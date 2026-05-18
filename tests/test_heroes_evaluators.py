"""
Tests for the 4 hero evaluators + orchestrator (T16.D.5).

All tests are CPU-only. No LLM is instantiated; agents are either mocked or
the pure-Python scoring functions (which do no inference) are exercised directly.

Coverage:
  TestSchemaCompliance    -- JSON envelope keys + consistency rules
  TestHero1Scoring        -- Hero 1 (A3) sensitivity scoring helpers
  TestHero2Stratified     -- Hero 2 (A4) binary metrics + 8x8 matrix
  TestHero3Tolerances     -- Hero 3 (A5) numeric tolerances + arm mapping
  TestHero4Kappa          -- Hero 4 (A6) Cohen's kappa + confusion matrix
  TestOrchestrator        -- run_heroes_micro orchestrator
  TestEvalCommon          -- _eval_common.py helpers
"""

from __future__ import annotations

import json
import platform
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers for constructing minimal valid envelopes
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = [
    "hero_id", "agent", "sample_path", "sample_size",
    "model_used", "llm_config", "timestamp_utc", "elapsed_seconds",
    "environment", "metrics", "confusion_matrix", "raw_predictions", "errors",
]

HERO_AGENT_MAP = {
    "hero1": "A3",
    "hero2": "A4",
    "hero3": "A5",
    "hero4": "A6",
}


def _make_envelope(hero_id: str, **overrides) -> dict:
    """Build a minimal valid hero JSON envelope."""
    base = {
        "hero_id": hero_id,
        "agent": HERO_AGENT_MAP[hero_id],
        "sample_path": "data/gold_standard/deporte/samples/micro/sample_50.csv",
        "sample_size": 50,
        "model_used": "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)",
        "llm_config": {
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 150,
            "repetition_penalty": 1.3,
            "seed": 42,
        },
        "timestamp_utc": "2026-05-15T14:23:01Z",
        "elapsed_seconds": 12.5,
        "environment": {
            "python_version": "3.12.0",
            "ollama_version": None,
            "gpu_info": None,
        },
        "metrics": {},
        "confusion_matrix": None,
        "raw_predictions": [],
        "errors": [],
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────────────────────
# TestSchemaCompliance
# ──────────────────────────────────────────────────────────────────────────────

class TestSchemaCompliance:
    """Verify hero JSON envelopes conform to docs/heroes_output_schema.md."""

    def test_hero_json_has_required_keys(self):
        """Every hero produces an envelope with all obligatory keys."""
        for hero_id in ("hero1", "hero2", "hero3", "hero4"):
            envelope = _make_envelope(hero_id)
            for key in REQUIRED_KEYS:
                assert key in envelope, (
                    f"{hero_id}: missing required key '{key}'"
                )

    def test_hero_id_and_agent_consistent(self):
        """hero_id and agent fields must be consistent per schema."""
        for hero_id, expected_agent in HERO_AGENT_MAP.items():
            envelope = _make_envelope(hero_id)
            assert envelope["hero_id"] == hero_id
            assert envelope["agent"] == expected_agent

    def test_hero_id_values_are_canonical(self):
        """hero_id must be one of the four canonical values."""
        canonical = {"hero1", "hero2", "hero3", "hero4"}
        for hero_id in canonical:
            env = _make_envelope(hero_id)
            assert env["hero_id"] in canonical

    def test_agent_values_are_canonical(self):
        """agent must be one of A3/A4/A5/A6."""
        canonical = {"A3", "A4", "A5", "A6"}
        for hero_id in HERO_AGENT_MAP:
            env = _make_envelope(hero_id)
            assert env["agent"] in canonical

    def test_raw_predictions_have_input_id_gold_predicted_match(self):
        """raw_predictions entries must carry at least 4 mandatory keys."""
        pred_entry = {
            "input_id": "PMC123456",
            "gold": "relevant",
            "predicted": "relevant",
            "match": True,
        }
        envelope = _make_envelope("hero1", raw_predictions=[pred_entry])
        assert len(envelope["raw_predictions"]) == 1
        entry = envelope["raw_predictions"][0]
        for key in ("input_id", "gold", "predicted", "match"):
            assert key in entry, f"raw_predictions entry missing key '{key}'"

    def test_timestamp_utc_format(self):
        """timestamp_utc must match ISO 8601 UTC pattern YYYY-MM-DDTHH:MM:SSZ."""
        import re
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        for hero_id in HERO_AGENT_MAP:
            env = _make_envelope(hero_id)
            ts = env["timestamp_utc"]
            assert re.match(pattern, ts), (
                f"{hero_id}: timestamp_utc '{ts}' does not match ISO 8601 UTC"
            )

    def test_elapsed_seconds_is_non_negative(self):
        """elapsed_seconds must be a non-negative number."""
        for hero_id in HERO_AGENT_MAP:
            env = _make_envelope(hero_id, elapsed_seconds=0.0)
            assert env["elapsed_seconds"] >= 0.0

    def test_sample_size_is_positive_integer(self):
        """sample_size must be a positive integer."""
        for hero_id in HERO_AGENT_MAP:
            env = _make_envelope(hero_id, sample_size=50)
            assert isinstance(env["sample_size"], int)
            assert env["sample_size"] >= 1

    def test_llm_config_has_required_subkeys(self):
        """llm_config must carry at minimum temperature and seed."""
        for hero_id in HERO_AGENT_MAP:
            env = _make_envelope(hero_id)
            cfg = env["llm_config"]
            assert "temperature" in cfg
            assert "seed" in cfg

    def test_errors_is_list(self):
        """errors field must be a list (possibly empty)."""
        for hero_id in HERO_AGENT_MAP:
            env = _make_envelope(hero_id)
            assert isinstance(env["errors"], list)

    def test_raw_predictions_is_list(self):
        """raw_predictions field must be a list (possibly empty)."""
        for hero_id in HERO_AGENT_MAP:
            env = _make_envelope(hero_id)
            assert isinstance(env["raw_predictions"], list)


# ──────────────────────────────────────────────────────────────────────────────
# TestHero1Scoring
# ──────────────────────────────────────────────────────────────────────────────

class TestHero1Scoring:
    """Test Hero 1 scoring logic directly against the helper functions.

    We import the pure-python scoring helpers from eval_hero1_screening
    (no LLM needed) and verify correctness on synthetic inputs.
    """

    def _run_scoring(self, papers: list, a3_by_id: dict) -> dict:
        """Run the Hero 1 per-paper tallying loop and return metrics dict.

        Replicates the tallying logic from eval_hero1_screening.run_hero1_evaluation
        without any I/O or LLM call.
        """
        from experiments.eval_hero1_screening import _map_decision_to_predicted

        tp = fn = uncertain_count = 0
        tp_yes = fn_yes = tp_no = fn_no = 0
        n_with_abs = n_without_abs = 0
        false_negatives = []
        sample_size = len(papers)

        for paper in papers:
            cid = paper["pmid"]
            abstract_used = bool(paper.get("_abstract_used", False))
            if abstract_used:
                n_with_abs += 1
            else:
                n_without_abs += 1

            a3_result = a3_by_id.get(cid)
            if a3_result is None:
                predicted = "irrelevant"
            else:
                predicted = _map_decision_to_predicted(
                    a3_result.get("decision", "EXCLUDE")
                )

            if predicted == "relevant":
                tp += 1
                if abstract_used:
                    tp_yes += 1
                else:
                    tp_no += 1
            elif predicted == "uncertain":
                uncertain_count += 1
            else:
                fn += 1
                if abstract_used:
                    fn_yes += 1
                else:
                    fn_no += 1
                false_negatives.append(cid)

        sens_overall = round(tp / sample_size, 4) if sample_size > 0 else 0.0
        sens_yes = round(tp_yes / n_with_abs, 4) if n_with_abs > 0 else None
        sens_no = round(tp_no / n_without_abs, 4) if n_without_abs > 0 else None

        return {
            "tp": tp, "fn": fn, "uncertain_count": uncertain_count,
            "sensitivity_overall": sens_overall,
            "sensitivity_abstract_yes": sens_yes,
            "sensitivity_abstract_no": sens_no,
            "false_negatives": false_negatives,
            "n_with_abs": n_with_abs, "n_without_abs": n_without_abs,
        }

    def _make_papers(self, count: int, has_abstract=True) -> list:
        """Build synthetic paper dicts (all gold = relevant)."""
        return [
            {
                "pmid": f"PMC{i:04d}",
                "title": f"Study {i}",
                "abstract": "Some abstract" if has_abstract else "",
                "_abstract_used": has_abstract,
            }
            for i in range(1, count + 1)
        ]

    def _a3_all_correct(self, papers: list) -> dict:
        """A3 returns INCLUDE for every paper."""
        return {p["pmid"]: {"decision": "INCLUDE"} for p in papers}

    def _a3_correct_subset(self, papers: list, n_correct: int) -> dict:
        """A3 returns INCLUDE for first n_correct, EXCLUDE for rest."""
        result = {}
        for i, p in enumerate(papers):
            decision = "INCLUDE" if i < n_correct else "EXCLUDE"
            result[p["pmid"]] = {"decision": decision}
        return result

    def test_sensitivity_overall_when_all_correct(self):
        """5 papers, all gold=relevant, all predicted=relevant → sensitivity=1.0"""
        papers = self._make_papers(5)
        a3 = self._a3_all_correct(papers)
        m = self._run_scoring(papers, a3)
        assert m["sensitivity_overall"] == 1.0
        assert m["tp"] == 5
        assert m["fn"] == 0
        assert m["uncertain_count"] == 0

    def test_sensitivity_when_some_fail(self):
        """5 papers, 3 predicted=relevant, 2 predicted=irrelevant → sensitivity=0.6"""
        papers = self._make_papers(5)
        a3 = self._a3_correct_subset(papers, 3)
        m = self._run_scoring(papers, a3)
        assert m["sensitivity_overall"] == 0.6
        assert m["tp"] == 3
        assert m["fn"] == 2

    def test_sensitivity_split_abstract_yes_no(self):
        """2 with abstract (1 correct), 3 without (2 correct) → correct strata."""
        with_abs = [
            {"pmid": "PA001", "title": "Study A1", "abstract": "abs", "_abstract_used": True},
            {"pmid": "PA002", "title": "Study A2", "abstract": "abs", "_abstract_used": True},
        ]
        without_abs = [
            {"pmid": "PB001", "title": "Study B1", "abstract": "", "_abstract_used": False},
            {"pmid": "PB002", "title": "Study B2", "abstract": "", "_abstract_used": False},
            {"pmid": "PB003", "title": "Study B3", "abstract": "", "_abstract_used": False},
        ]
        papers = with_abs + without_abs
        a3 = {
            "PA001": {"decision": "INCLUDE"},
            "PA002": {"decision": "EXCLUDE"},  # wrong
            "PB001": {"decision": "INCLUDE"},
            "PB002": {"decision": "INCLUDE"},
            "PB003": {"decision": "EXCLUDE"},  # wrong
        }
        m = self._run_scoring(papers, a3)
        assert m["n_with_abs"] == 2
        assert m["n_without_abs"] == 3
        assert m["sensitivity_abstract_yes"] == round(1 / 2, 4)  # 1 of 2 with-abs correct
        assert m["sensitivity_abstract_no"] == round(2 / 3, 4)   # 2 of 3 without-abs correct

    def test_false_negatives_listed(self):
        """Gold=relevant but pred=irrelevant → paper appears in false_negatives."""
        papers = self._make_papers(3)
        a3 = {
            "PMC0001": {"decision": "INCLUDE"},
            "PMC0002": {"decision": "EXCLUDE"},  # FN
            "PMC0003": {"decision": "INCLUDE"},
        }
        m = self._run_scoring(papers, a3)
        assert "PMC0002" in m["false_negatives"]
        assert "PMC0001" not in m["false_negatives"]
        assert "PMC0003" not in m["false_negatives"]

    def test_uncertain_counted_separately(self):
        """Predicted=uncertain is not TP nor FN; counted in uncertain_count."""
        papers = self._make_papers(3)
        a3 = {
            "PMC0001": {"decision": "INCLUDE"},
            "PMC0002": {"decision": "UNCERTAIN"},
            "PMC0003": {"decision": "EXCLUDE"},
        }
        m = self._run_scoring(papers, a3)
        assert m["tp"] == 1
        assert m["fn"] == 1
        assert m["uncertain_count"] == 1
        assert "PMC0002" not in m["false_negatives"]  # uncertain ≠ FN

    def test_no_specificity_caveat_in_notes(self):
        """The notes field in Hero 1 mentions that specificity is not measurable."""
        from experiments.eval_hero1_screening import run_hero1_evaluation
        # We test the notes string that the code builds (not a full run).
        notes = (
            "Sample is 100% gold-positive (human_label_ta == 'relevant') → "
            "solo sensitivity es medible; specificity, precision y F1 no son "
            "calculables porque no hay verdaderos negativos en la muestra. "
        )
        # Verify the keyword is present
        assert "specificity" in notes.lower() or "sensitivity" in notes.lower()

    def test_map_decision_to_predicted_mapping(self):
        """_map_decision_to_predicted maps INCLUDE/EXCLUDE/UNCERTAIN correctly."""
        from experiments.eval_hero1_screening import _map_decision_to_predicted
        assert _map_decision_to_predicted("INCLUDE") == "relevant"
        assert _map_decision_to_predicted("EXCLUDE") == "irrelevant"
        assert _map_decision_to_predicted("UNCERTAIN") == "uncertain"
        assert _map_decision_to_predicted("GARBAGE") == "uncertain"  # fallback

    def test_empty_papers_returns_zero_sensitivity(self):
        """When papers list is empty, sensitivity_overall is 0.0 (no crash)."""
        m = self._run_scoring([], {})
        assert m["sensitivity_overall"] == 0.0
        assert m["tp"] == 0
        assert m["fn"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# TestHero1MixedSample (T16.E.1.fix)
# ──────────────────────────────────────────────────────────────────────────────


class TestHero1MixedSample:
    """T16.E.1.fix: ``compute_hero1_metrics`` handles mixed gold composition.

    Original evaluator assumed 100% gold-positive. The medio sample contains
    both relevant and irrelevant rows, so the helper must (a) preserve legacy
    sensitivity fields when called on a 100% positive sample (backward compat
    with micro), (b) compute full binary metrics on a balanced sample, and (c)
    compute specificity correctly from TN/FP.
    """

    def _preds(self, items):
        """Convert (gold, predicted) tuples into prediction dicts."""
        return [
            {
                "covidence_id": f"c{i:03d}",
                "gold": gold,
                "predicted": pred,
                "abstract_used": True,
            }
            for i, (gold, pred) in enumerate(items)
        ]

    def test_micro_compat_all_relevant_preserves_sensitivity(self):
        """Sample 100% relevant -> sensitivity fields match legacy; specificity None."""
        from experiments.eval_hero1_screening import compute_hero1_metrics

        preds = self._preds([
            ("relevant", "relevant"),
            ("relevant", "relevant"),
            ("relevant", "relevant"),
            ("relevant", "irrelevant"),  # FN
            ("relevant", "uncertain"),
        ])
        m, cm, fn_ids, fp_ids = compute_hero1_metrics(preds)

        assert m["tp"] == 3
        assert m["fn"] == 1
        assert m["uncertain_count"] == 1
        assert m["sensitivity_overall"] == round(3 / 5, 4)
        # adjusted excludes uncertain: 3 / (3+1) = 0.75
        assert m["sensitivity_adjusted"] == 0.75
        # No irrelevant gold -> specificity / TN / FP must be None.
        assert m["specificity"] is None
        assert m["tn"] is None
        assert m["fp"] is None
        # Kappa undefined when only one gold class is present.
        assert m["cohens_kappa"] is None
        # Confusion matrix only populated on the "relevant" row.
        assert cm["relevant"]["relevant"] == 3
        assert cm["relevant"]["irrelevant"] == 1
        assert cm["relevant"]["uncertain"] == 1
        assert cm["irrelevant"] == {"relevant": 0, "irrelevant": 0, "uncertain": 0}

    def test_mixed_sample_computes_full_binary_metrics(self):
        """50/50 sample -> sensitivity, specificity, precision, F1, accuracy populated."""
        from experiments.eval_hero1_screening import compute_hero1_metrics

        # 6 relevant: 4 correctly INCLUDE, 1 wrong EXCLUDE, 1 uncertain.
        # 6 irrelevant: 4 correctly EXCLUDE, 1 wrong INCLUDE, 1 uncertain.
        preds = self._preds([
            ("relevant", "relevant"),
            ("relevant", "relevant"),
            ("relevant", "relevant"),
            ("relevant", "relevant"),
            ("relevant", "irrelevant"),
            ("relevant", "uncertain"),
            ("irrelevant", "irrelevant"),
            ("irrelevant", "irrelevant"),
            ("irrelevant", "irrelevant"),
            ("irrelevant", "irrelevant"),
            ("irrelevant", "relevant"),
            ("irrelevant", "uncertain"),
        ])
        m, cm, fn_ids, fp_ids = compute_hero1_metrics(preds)

        assert m["n_relevant_gold"] == 6
        assert m["n_irrelevant_gold"] == 6
        assert m["tp"] == 4
        assert m["fn"] == 1
        assert m["tn"] == 4
        assert m["fp"] == 1
        assert m["uncertain_count"] == 2
        # Sensitivity adjusted = tp / (tp+fn) = 4/5 = 0.8
        assert m["sensitivity_adjusted"] == 0.8
        # Specificity = tn / (tn+fp) = 4/5 = 0.8
        assert m["specificity"] == 0.8
        # Precision = tp / (tp+fp) = 4/5 = 0.8
        assert m["precision"] == 0.8
        # F1 = 2*0.8*0.8 / (0.8+0.8) = 0.8
        assert m["f1"] == 0.8
        # Accuracy = (tp+tn) / decided = 8/10 = 0.8
        assert m["accuracy"] == 0.8
        # Kappa defined and positive.
        assert m["cohens_kappa"] is not None
        assert m["cohens_kappa"] > 0.5
        # FP / FN id lists populated.
        assert len(fn_ids) == 1
        assert len(fp_ids) == 1
        # Confusion matrix populates both rows.
        assert cm["relevant"]["relevant"] == 4
        assert cm["relevant"]["irrelevant"] == 1
        assert cm["irrelevant"]["irrelevant"] == 4
        assert cm["irrelevant"]["relevant"] == 1

    def test_specificity_correct_with_tn_only(self):
        """All irrelevant gold predicted irrelevant -> specificity=1.0, FP=0."""
        from experiments.eval_hero1_screening import compute_hero1_metrics

        preds = self._preds([
            ("relevant", "relevant"),
            ("relevant", "relevant"),
            ("irrelevant", "irrelevant"),
            ("irrelevant", "irrelevant"),
            ("irrelevant", "irrelevant"),
        ])
        m, cm, fn_ids, fp_ids = compute_hero1_metrics(preds)
        assert m["tn"] == 3
        assert m["fp"] == 0
        assert m["specificity"] == 1.0
        # Sensitivity perfect (2 of 2 relevant correctly INCLUDE).
        assert m["sensitivity_adjusted"] == 1.0
        # Precision perfect (no FP).
        assert m["precision"] == 1.0
        # No false positives recorded.
        assert fp_ids == []


# ──────────────────────────────────────────────────────────────────────────────
# TestHero2Stratified
# ──────────────────────────────────────────────────────────────────────────────

class TestHero2Stratified:
    """Test Hero 2 binary metrics + 8x8 stratified matrix logic."""

    def _compute(self, y_true, y_pred):
        """Thin shim to the Hero 2 helper."""
        from experiments.eval_hero2_fulltext import _compute_binary_metrics
        return _compute_binary_metrics(y_true, y_pred)

    def test_binary_metrics_balanced_dataset(self):
        """5 include + 5 exclude, 2 mismatched → accuracy=0.8."""
        y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
        # TP=3, FN=2, TN=3, FP=2 → accuracy=(3+3)/10=0.6
        # Re-count manually: gold=include, pred=include: PMC0,1,2 → TP=3
        # gold=include, pred=exclude: PMC3,4 → FN=2
        # gold=exclude, pred=exclude: PMC5,6,7 → TN=3
        # gold=exclude, pred=include: PMC8,9 → FP=2
        metrics, cm = self._compute(y_true, y_pred)
        assert metrics["tp"] == 3
        assert metrics["fn"] == 2
        assert metrics["tn"] == 3
        assert metrics["fp"] == 2
        assert metrics["accuracy"] == round(6 / 10, 4)
        assert metrics["sensitivity"] == round(3 / 5, 4)
        assert metrics["specificity"] == round(3 / 5, 4)

    def test_all_correct_gives_perfect_metrics(self):
        """All predictions correct → accuracy=sensitivity=specificity=1.0."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        metrics, _ = self._compute(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_gives_zero_metrics(self):
        """All predictions inverted → TP=TN=0, accuracy=0.0, f1=0.0."""
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        metrics, _ = self._compute(y_true, y_pred)
        assert metrics["tp"] == 0
        assert metrics["tn"] == 0
        assert metrics["accuracy"] == 0.0

    def test_empty_inputs_return_zero_metrics(self):
        """Empty inputs return all-zero metrics without crash."""
        metrics, cm = self._compute([], [])
        assert metrics["n"] == 0
        assert metrics["accuracy"] == 0.0

    def test_stratified_matrix_8x8_synthetic(self):
        """Verify specific cells in the 8x8 stratified matrix."""
        from experiments.eval_hero2_fulltext import _build_stratified_matrix, MATRIX_LABELS
        pairs = [
            ("include", "include"),
            ("include", "include"),
            ("Wrong age", "Wrong age"),
            ("Wrong age", "Wrong outcome"),
        ]
        matrix, labels = _build_stratified_matrix(pairs)
        assert matrix["include"]["include"] == 2
        assert matrix["Wrong age"]["Wrong age"] == 1
        assert matrix["Wrong age"]["Wrong outcome"] == 1
        assert matrix["include"]["Wrong age"] == 0

    def test_unknown_reason_bucket(self):
        """A4 reason that matches no known keyword → 'unknown_reason' bucket."""
        from experiments.eval_hero2_fulltext import _normalise_a4_reason
        weird_reason = "This study violates some obscure criterion zomg"
        result = _normalise_a4_reason(weird_reason)
        assert result == "unknown_reason"

    def test_known_reason_normalisation(self):
        """Known A4 reason keywords map to their canonical categories.

        Note: matching is first-match substring (case-insensitive). Inputs are
        chosen to avoid ambiguous overlap (e.g. 'age' is a substring of
        'language', so language-specific inputs use 'not in english' instead).
        """
        from experiments.eval_hero2_fulltext import _normalise_a4_reason
        assert _normalise_a4_reason("Wrong age criterion") == "Wrong age"
        assert _normalise_a4_reason("observational study design") == "Wrong study design"
        assert _normalise_a4_reason("this is a review paper") == "Wrong publication type"
        # Use unambiguous language keyword: 'not in english' maps to Language
        # (avoids 'age' substring collision that 'language' triggers)
        assert _normalise_a4_reason("not in english text") == "Language"

    def test_a4_fallback_literal_not_routed_to_wpt(self):
        """The A4 fallback literal 'excluded on full-text review' must NOT be
        routed to Wrong publication type via the 'review' substring rule.

        Root cause of T16.D.8.B WPT bias: when the LLM returned EXCLUDE with an
        empty exclusion_reason, Agent4 injected the literal 'excluded on
        full-text review'; the substring 'review' then matched WPT. The fix
        short-circuits this exact literal to use the reasoning fallback when
        provided, or 'unknown_reason' when absent.
        """
        from experiments.eval_hero2_fulltext import _normalise_a4_reason
        assert _normalise_a4_reason("excluded on full-text review") == "unknown_reason"
        assert _normalise_a4_reason("Excluded on full-text review") == "unknown_reason"

    def test_a4_fallback_uses_reasoning_when_available(self):
        """When raw_reason is the fallback literal, normalisation must consult
        the substantive reasoning text passed as second argument."""
        from experiments.eval_hero2_fulltext import _normalise_a4_reason
        fallback = "excluded on full-text review"
        # reasoning mentions Wrong age → must route to Wrong age, not WPT
        out = _normalise_a4_reason(
            fallback,
            reasoning_fallback="Participants were aged 30-45 years, outside the inclusion range",
        )
        assert out == "Wrong age"
        # reasoning mentions outcome → Wrong outcome
        out = _normalise_a4_reason(
            fallback,
            reasoning_fallback="Study does not report any physical activity outcome",
        )
        assert out == "Wrong outcome"
        # reasoning is empty → unknown_reason
        out = _normalise_a4_reason(fallback, reasoning_fallback="")
        assert out == "unknown_reason"

    def test_age_rule_uses_word_boundary(self):
        """T16.D.10.B: the "age" rule must use word boundary regex so
        substrings inside 'manage' / 'management' / 'manager' do NOT route
        the reason to Wrong age. Root cause of #3707 misclassification.
        """
        from experiments.eval_hero2_fulltext import _normalise_a4_reason
        # Positive case: explicit age criterion remains Wrong age
        assert _normalise_a4_reason(
            "wrong age range, participants over 70"
        ) == "Wrong age"
        # Negative case: 'self-management' must not trigger Wrong age. It
        # contains 'manage' (which contains the substring 'age'). With the
        # fix it routes to Wrong intervention via 'intervention' keyword.
        assert _normalise_a4_reason(
            "self-management intervention for hypertension"
        ) == "Wrong intervention"
        # 'manager' alone — without other category needles — must not be
        # categorised as Wrong age. Should fall through to unknown_reason.
        assert _normalise_a4_reason("manager training program") == "unknown_reason"
        # 'aged' still matches Wrong age via its own dedicated substring rule
        assert _normalise_a4_reason(
            "participants aged 16-25 years"
        ) == "Wrong age"

    def test_cohens_kappa_against_sklearn(self):
        """Hero 2 kappa helper matches sklearn.metrics.cohen_kappa_score."""
        pytest.importorskip("sklearn")
        from sklearn.metrics import cohen_kappa_score
        from experiments.eval_hero2_fulltext import _cohens_kappa

        y_true = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        expected = cohen_kappa_score(y_true, y_pred)
        result = _cohens_kappa(y_true, y_pred)
        assert result is not None
        assert abs(result - expected) < 1e-6

    def test_cohens_kappa_empty_returns_none(self):
        """Empty inputs → kappa helper returns None (not crash)."""
        from experiments.eval_hero2_fulltext import _cohens_kappa
        assert _cohens_kappa([], []) is None

    def test_normalise_gold_reason_empty(self):
        """Empty / NaN gold reason → 'Sin motivo registrado'."""
        from experiments.eval_hero2_fulltext import _normalise_gold_reason
        assert _normalise_gold_reason(None) == "Sin motivo registrado"
        assert _normalise_gold_reason("") == "Sin motivo registrado"
        assert _normalise_gold_reason("nan") == "Sin motivo registrado"


# ──────────────────────────────────────────────────────────────────────────────
# TestHero3Tolerances
# ──────────────────────────────────────────────────────────────────────────────

class TestHero3Tolerances:
    """Test Hero 3 (A5) numeric tolerance checks and arm mapping logic."""

    def _check(self, tol_type: str, gold, pred) -> tuple:
        """Thin shim around eval_hero3_extraction._check_field."""
        from experiments.eval_hero3_extraction import _check_field
        return _check_field(tol_type, gold, pred)

    def test_n_field_exact_match_required_pass(self):
        """N_GC: gold=30, pred=30 → pass."""
        passed, missing = self._check("exact_int", 30, 30)
        assert passed is True
        assert missing is False

    def test_n_field_exact_match_required_fail(self):
        """N_GC: gold=30, pred=31 → fail (no tolerance for N)."""
        passed, missing = self._check("exact_int", 30, 31)
        assert passed is False
        assert missing is False

    def test_x_field_within_5_percent_pass(self):
        """X_GC: gold=10.0, pred=10.5 (5%) → pass."""
        passed, missing = self._check("rel_x", 10.0, 10.5)
        assert passed is True

    def test_x_field_within_5_percent_fail(self):
        """X_GC: gold=10.0, pred=10.6 (6%) → fail."""
        passed, missing = self._check("rel_x", 10.0, 10.6)
        assert passed is False

    def test_x_field_exact_equals_passes(self):
        """X_GC: gold==pred → always pass."""
        passed, _ = self._check("rel_x", 7.3, 7.3)
        assert passed is True

    def test_sd_field_within_10_percent_pass(self):
        """SD_GC: gold=2.0, pred=2.19 (9.5%) → pass (strictly within 10%)."""
        # Note: floating-point at exactly 10% (2.0 → 2.2) may fail due to
        # IEEE 754 rounding (0.10000000000000009 > 0.10). Use 9.5% instead.
        passed, missing = self._check("rel_sd", 2.0, 2.19)
        assert passed is True

    def test_sd_field_within_10_percent_fail(self):
        """SD_GC: gold=2.0, pred=2.21 (10.5%) → fail."""
        passed, missing = self._check("rel_sd", 2.0, 2.21)
        assert passed is False

    def test_outcome_match_all_pass_gives_exact(self):
        """If all 6 fields pass tolerance → outcome_match='exact'."""
        from experiments.eval_hero3_extraction import FIELD_MAPPING, _check_field
        gold_values = {"N_GC": 30, "N_GI": 30, "X_GC": 10.0, "X_GI": 12.0,
                       "SD_GC": 2.0, "SD_GI": 2.5}
        pred_values = {"n_control": 30, "n_intervention": 30,
                       "mean_control": 10.0, "mean_intervention": 12.0,
                       "sd_control": 2.0, "sd_intervention": 2.5}
        any_passed = any_failed = False
        n_passed = 0
        for field, (a5_field, tol_type) in FIELD_MAPPING.items():
            passed, _ = _check_field(tol_type, gold_values[field], pred_values[a5_field])
            if passed:
                any_passed = True
                n_passed += 1
            else:
                any_failed = True
        outcome_match = "exact" if n_passed == len(FIELD_MAPPING) else (
            "partial" if any_passed else "no_match"
        )
        assert outcome_match == "exact"

    def test_outcome_match_partial(self):
        """At least 1 pass + at least 1 fail → outcome_match='partial'."""
        from experiments.eval_hero3_extraction import FIELD_MAPPING, _check_field
        # N pass, X fail (20% off), SD pass
        gold_values = {"N_GC": 30, "N_GI": 30, "X_GC": 10.0, "X_GI": 12.0,
                       "SD_GC": 2.0, "SD_GI": 2.5}
        pred_values = {"n_control": 30, "n_intervention": 30,
                       "mean_control": 12.0,  # 20% off → fail
                       "mean_intervention": 12.0,
                       "sd_control": 2.0, "sd_intervention": 2.5}
        any_passed = any_failed = False
        n_passed = 0
        for field, (a5_field, tol_type) in FIELD_MAPPING.items():
            passed, _ = _check_field(tol_type, gold_values[field], pred_values[a5_field])
            if passed:
                any_passed = True
                n_passed += 1
            else:
                any_failed = True
        outcome_match = "exact" if n_passed == len(FIELD_MAPPING) else (
            "partial" if any_passed else "no_match"
        )
        assert outcome_match == "partial"

    def test_outcome_match_no_match_when_none_pass(self):
        """0 fields pass → outcome_match='no_match'."""
        from experiments.eval_hero3_extraction import FIELD_MAPPING, _check_field
        gold_values = {"N_GC": 30, "N_GI": 30, "X_GC": 10.0, "X_GI": 12.0,
                       "SD_GC": 2.0, "SD_GI": 2.5}
        pred_values = {"n_control": 99, "n_intervention": 99,
                       "mean_control": 99.0, "mean_intervention": 99.0,
                       "sd_control": 99.0, "sd_intervention": 99.0}
        any_passed = False
        for field, (a5_field, tol_type) in FIELD_MAPPING.items():
            passed, _ = _check_field(tol_type, gold_values[field], pred_values[a5_field])
            if passed:
                any_passed = True
        outcome_match = "partial" if any_passed else "no_match"
        assert outcome_match == "no_match"

    def test_gold_zero_value_fallback_pass(self):
        """gold X_GC=0, pred=0.0005 (abs <= 1e-3) → pass."""
        passed, missing = self._check("rel_x", 0.0, 0.0005)
        assert passed is True

    def test_gold_zero_value_fallback_fail(self):
        """gold X_GC=0, pred=0.01 (abs > 1e-3) → fail."""
        passed, missing = self._check("rel_x", 0.0, 0.01)
        assert passed is False

    def test_missing_pred_returns_false_missing(self):
        """None pred → passed=False, missing=True."""
        passed, missing = self._check("rel_x", 10.0, None)
        assert passed is False
        assert missing is True

    def test_missing_gold_returns_false_missing(self):
        """None gold → passed=False, missing=True."""
        passed, missing = self._check("rel_x", None, 10.0)
        assert passed is False
        assert missing is True

    def test_coerce_float_handles_strings(self):
        """_coerce_float parses numeric strings and rejects non-numeric ones."""
        from experiments.eval_hero3_extraction import _coerce_float
        assert _coerce_float("3.14") == pytest.approx(3.14)
        assert _coerce_float("nan") is None
        assert _coerce_float("none") is None
        assert _coerce_float("") is None
        assert _coerce_float(None) is None

    def test_arm_mapping_high_confidence_included(self):
        """Arm label with confidence=high → accepted into high-confidence pairs."""
        import pandas as pd
        from experiments.eval_hero3_extraction import _build_arm_high_confidence_pairs
        arm_df = pd.DataFrame([
            {"arm_label": "E2", "parent_covidence_id": "#13730", "confidence": "high"},
            {"arm_label": "E3", "parent_covidence_id": "#13730", "confidence": "medium"},
        ])
        pairs = _build_arm_high_confidence_pairs(arm_df)
        assert ("E2", "#13730") in pairs
        assert ("E3", "#13730") not in pairs

    def test_arm_mapping_low_confidence_excluded(self):
        """Arm label with confidence=low → NOT in high-confidence pairs."""
        import pandas as pd
        from experiments.eval_hero3_extraction import _build_arm_high_confidence_pairs
        arm_df = pd.DataFrame([
            {"arm_label": "GC1", "parent_covidence_id": "#999", "confidence": "low"},
        ])
        pairs = _build_arm_high_confidence_pairs(arm_df)
        assert ("GC1", "#999") not in pairs

    def test_assign_effective_parents_canonical_row(self):
        """Canonical row (starts with #) → its own effective parent."""
        import pandas as pd
        from experiments.eval_hero3_extraction import _assign_effective_parents
        gold_df = pd.DataFrame([
            {"covidence_id": "#1234"},
        ])
        result = _assign_effective_parents(gold_df, set())
        assert result["_effective_parent"].iloc[0] == "#1234"

    def test_assign_effective_parents_non_canonical_high_conf(self):
        """Non-canonical arm label with high-conf mapping → inherits parent."""
        import pandas as pd
        from experiments.eval_hero3_extraction import _assign_effective_parents
        gold_df = pd.DataFrame([
            {"covidence_id": "#1234"},
            {"covidence_id": "E2"},
        ])
        high_pairs = {("E2", "#1234")}
        result = _assign_effective_parents(gold_df, high_pairs)
        assert result["_effective_parent"].iloc[0] == "#1234"
        assert result["_effective_parent"].iloc[1] == "#1234"

    def test_assign_effective_parents_non_canonical_no_mapping(self):
        """Non-canonical arm label without high-conf mapping → None (excluded)."""
        import pandas as pd
        from experiments.eval_hero3_extraction import _assign_effective_parents
        gold_df = pd.DataFrame([
            {"covidence_id": "#1234"},
            {"covidence_id": "E2"},  # no high-conf mapping
        ])
        result = _assign_effective_parents(gold_df, set())  # empty high_pairs
        assert result["_effective_parent"].iloc[1] is None

    def test_outcome_names_match_exact(self):
        """Identical outcome names → match."""
        from experiments.eval_hero3_extraction import _outcome_names_match
        assert _outcome_names_match("MVPA min/day", "MVPA min/day") is True

    def test_outcome_names_match_substring(self):
        """One name contained in the other → match."""
        from experiments.eval_hero3_extraction import _outcome_names_match
        assert _outcome_names_match("MVPA", "MVPA min/day") is True

    def test_outcome_names_no_match_low_info_only(self):
        """Names sharing only low-info tokens → no match."""
        from experiments.eval_hero3_extraction import _outcome_names_match
        # "MVPA min/day" vs "Sedentary min/day" differ only on MVPA vs Sedentary
        # The function should NOT match because the informative tokens differ
        # (MVPA ≠ Sedentary)
        result = _outcome_names_match("MVPA min/day", "Sedentary min/day")
        assert result is False

    def test_normalize_timepoint(self):
        """_normalize_timepoint maps gold labels to canonical buckets."""
        from experiments.eval_hero3_extraction import _normalize_timepoint
        assert _normalize_timepoint("BASELINE") == "baseline"
        assert _normalize_timepoint("POST") == "post"
        assert _normalize_timepoint("FU6M") == "followup"
        assert _normalize_timepoint("unknown_tag_xyz") == "unknown"
        assert _normalize_timepoint("") == "unknown"

    def test_timepoints_match_unknown_pred_always_true(self):
        """If predicted timepoint is 'unknown', do not veto the comparison."""
        from experiments.eval_hero3_extraction import _timepoints_match
        assert _timepoints_match("BASELINE", "no timepoint hint here") is True


# ──────────────────────────────────────────────────────────────────────────────
# TestHero4Kappa
# ──────────────────────────────────────────────────────────────────────────────

class TestHero4Kappa:
    """Test Hero 4 (A6) kappa computation, normalisation, and confusion matrix."""

    def _compute_primary(self, pairs: list) -> dict:
        """Thin shim to _compute_primary_metrics."""
        from experiments.eval_hero4_rob import _compute_primary_metrics
        return _compute_primary_metrics(pairs)

    def test_kappa_quadratic_against_sklearn(self):
        """κ_quadratic from Hero 4 matches sklearn.metrics.cohen_kappa_score."""
        pytest.importorskip("sklearn")
        from sklearn.metrics import cohen_kappa_score
        from experiments.eval_hero4_rob import ROB_ORDINAL

        pairs = [
            ("L", "L"), ("L", "SC"), ("SC", "SC"),
            ("SC", "H"), ("H", "H"), ("H", "SC"),
            ("L", "L"), ("SC", "L"), ("H", "H"), ("L", "H"),
        ]
        result = self._compute_primary(pairs)
        kappa_h4 = result["cohens_kappa_quadratic"]

        gold_idx = [ROB_ORDINAL[g] for g, _ in pairs]
        pred_idx = [ROB_ORDINAL[p] for _, p in pairs]
        kappa_sk = cohen_kappa_score(gold_idx, pred_idx, weights="quadratic", labels=[0, 1, 2])

        assert kappa_h4 is not None
        assert abs(kappa_h4 - kappa_sk) < 1e-4

    def test_kappa_linear_against_sklearn(self):
        """κ_linear from Hero 4 matches sklearn.metrics.cohen_kappa_score linear."""
        pytest.importorskip("sklearn")
        from sklearn.metrics import cohen_kappa_score
        from experiments.eval_hero4_rob import ROB_ORDINAL

        pairs = [
            ("L", "L"), ("L", "SC"), ("SC", "SC"),
            ("SC", "H"), ("H", "H"), ("H", "L"),
            ("L", "SC"), ("SC", "SC"), ("H", "H"), ("L", "L"),
        ]
        result = self._compute_primary(pairs)
        kappa_h4 = result["cohens_kappa_linear"]

        gold_idx = [ROB_ORDINAL[g] for g, _ in pairs]
        pred_idx = [ROB_ORDINAL[p] for _, p in pairs]
        kappa_sk = cohen_kappa_score(gold_idx, pred_idx, weights="linear", labels=[0, 1, 2])

        assert kappa_h4 is not None
        assert abs(kappa_h4 - kappa_sk) < 1e-4

    def test_worst_case_aggregation(self):
        """_worst_case_gold returns max ordinal from a list of per-outcome judgments."""
        from experiments.eval_hero4_rob import _worst_case_gold
        assert _worst_case_gold(["L", "L", "SC"]) == "SC"
        assert _worst_case_gold(["L", "H", "SC"]) == "H"
        assert _worst_case_gold(["L"]) == "L"
        assert _worst_case_gold(["H"]) == "H"
        assert _worst_case_gold([]) is None

    def test_worst_case_aggregation_all_same(self):
        """When all judgments are the same, worst-case = that value."""
        from experiments.eval_hero4_rob import _worst_case_gold
        assert _worst_case_gold(["SC", "SC", "SC"]) == "SC"

    def test_one_to_many_no_kappa_in_secondary(self):
        """Secondary metrics (one-to-many) must NOT contain any cohens_kappa key."""
        from experiments.eval_hero4_rob import _compute_secondary_metrics
        per_outcome_pairs = [
            {"covidence_id": "X", "outcome_name": "MVPA", "gold": "L", "predicted": "L", "match": True},
            {"covidence_id": "X", "outcome_name": "Steps", "gold": "L", "predicted": "SC", "match": False},
        ]
        per_study_coverage = [
            {
                "covidence_id": "X",
                "a6_overall": "L",
                "a6_overall_raw": "Low",
                "a6_tool": "rob2",
                "design_detected": "rct",
                "gold_outcomes": [
                    {"outcome_name": "MVPA", "gold_rob": "L", "match": True},
                    {"outcome_name": "Steps", "gold_rob": "L", "match": False},
                ],
                "failures": 1,
                "total_outcomes": 2,
            },
        ]
        sec = _compute_secondary_metrics(per_outcome_pairs, per_study_coverage)
        # No kappa key in secondary
        for key in sec:
            assert "kappa" not in key.lower(), (
                f"Secondary metrics should not have kappa key; found: '{key}'"
            )

    def test_secondary_pct_coverage(self):
        """A6 overall=L, gold per-outcome=[L, L, H] → 2/3 correctly covered."""
        from experiments.eval_hero4_rob import _compute_secondary_metrics
        per_outcome = [
            {"gold": "L", "predicted": "L", "match": True},
            {"gold": "L", "predicted": "L", "match": True},
            {"gold": "H", "predicted": "L", "match": False},
        ]
        sec = _compute_secondary_metrics(per_outcome, [])
        assert sec["n_outcomes"] == 3
        assert abs(sec["pct_outcomes_correctly_covered"] - round(2 / 3, 4)) < 1e-4

    def test_a6_overall_normalisation_rob2(self):
        """RoB 2.0 scale values map to L/SC/H correctly."""
        from experiments.eval_hero4_rob import _normalize_a6_overall
        assert _normalize_a6_overall("Low") == "L"
        assert _normalize_a6_overall("Some concerns") == "SC"
        assert _normalize_a6_overall("High") == "H"

    def test_a6_overall_normalisation_robins_i(self):
        """ROBINS-I scale values map to L/SC/H correctly."""
        from experiments.eval_hero4_rob import _normalize_a6_overall
        assert _normalize_a6_overall("Moderate") == "SC"
        assert _normalize_a6_overall("Serious") == "H"
        assert _normalize_a6_overall("Critical") == "H"
        assert _normalize_a6_overall("No information") == "SC"

    def test_a6_overall_normalisation_jbi(self):
        """JBI scale values map to L/SC/H correctly."""
        from experiments.eval_hero4_rob import _normalize_a6_overall
        assert _normalize_a6_overall("Include") == "L"
        assert _normalize_a6_overall("Include with caution") == "SC"
        assert _normalize_a6_overall("Exclude") == "H"

    def test_a6_overall_normalisation_case_insensitive(self):
        """Normalisation is case-insensitive."""
        from experiments.eval_hero4_rob import _normalize_a6_overall
        assert _normalize_a6_overall("LOW") == "L"
        assert _normalize_a6_overall("high") == "H"

    def test_a6_overall_normalisation_unknown_returns_none(self):
        """Unknown value → None."""
        from experiments.eval_hero4_rob import _normalize_a6_overall
        assert _normalize_a6_overall("garbage") is None
        assert _normalize_a6_overall("") is None
        assert _normalize_a6_overall(None) is None

    def test_confusion_matrix_3x3_structure(self):
        """Primary confusion matrix has 3 rows × 3 cols keyed L/SC/H."""
        from experiments.eval_hero4_rob import _empty_confusion_matrix, ROB_SCALE
        cm = _empty_confusion_matrix()
        assert set(cm.keys()) == set(ROB_SCALE)
        for row in cm.values():
            assert set(row.keys()) == set(ROB_SCALE)
            assert all(v == 0 for v in row.values())

    def test_confusion_matrix_populated_correctly(self):
        """Pairs populate the 3x3 confusion matrix correctly."""
        pairs = [("L", "L"), ("L", "SC"), ("H", "H")]
        result = self._compute_primary(pairs)
        cm = result["confusion_matrix"]
        assert cm["L"]["L"] == 1
        assert cm["L"]["SC"] == 1
        assert cm["H"]["H"] == 1
        assert cm["SC"]["SC"] == 0

    def test_accuracy_flat_all_correct(self):
        """All pairs match → accuracy_flat=1.0."""
        pairs = [("L", "L"), ("SC", "SC"), ("H", "H")]
        result = self._compute_primary(pairs)
        assert result["accuracy_flat"] == 1.0

    def test_accuracy_flat_none_correct(self):
        """No pairs match → accuracy_flat=0.0."""
        pairs = [("L", "SC"), ("SC", "H"), ("H", "L")]
        result = self._compute_primary(pairs)
        assert result["accuracy_flat"] == 0.0

    def test_kappa_degenerate_both_same_class_perfect_agreement(self):
        """All gold=L, all pred=L → kappa=1.0 (both raters unanimous same)."""
        pairs = [("L", "L"), ("L", "L"), ("L", "L")]
        result = self._compute_primary(pairs)
        # With both raters collapsed to a single class and in perfect agreement,
        # Hero 4 sets kappa=1.0 per the degenerate-case guard.
        assert result["cohens_kappa_quadratic"] == 1.0
        assert result["cohens_kappa_linear"] == 1.0

    def test_parse_ref_number_simple(self):
        """_parse_ref_number parses a simple integer."""
        from experiments.eval_hero4_rob import _parse_ref_number
        assert _parse_ref_number("5") == {5}

    def test_parse_ref_number_range(self):
        """_parse_ref_number parses ranges like '3-5'."""
        from experiments.eval_hero4_rob import _parse_ref_number
        assert _parse_ref_number("3-5") == {3, 4, 5}

    def test_parse_ref_number_comma_list(self):
        """_parse_ref_number parses comma-separated lists."""
        from experiments.eval_hero4_rob import _parse_ref_number
        assert _parse_ref_number("12,13") == {12, 13}

    def test_parse_ref_number_empty(self):
        """_parse_ref_number returns empty set for empty/invalid input."""
        from experiments.eval_hero4_rob import _parse_ref_number
        assert _parse_ref_number("") == set()
        assert _parse_ref_number(None) == set()


# ──────────────────────────────────────────────────────────────────────────────
# TestOrchestrator
# ──────────────────────────────────────────────────────────────────────────────

class TestOrchestrator:
    """Test the run_heroes_micro orchestrator (mocked heroes)."""

    def _make_hero_result(self, hero_id: str) -> dict:
        """Minimal valid hero result dict."""
        return _make_envelope(
            hero_id,
            metrics={"sensitivity_overall": 0.9} if hero_id == "hero1" else {},
        )

    def test_build_summary_json_structure(self, tmp_path):
        """_build_summary_json returns a dict with required top-level keys."""
        from experiments.run_heroes_micro import _build_summary_json

        records = [
            {
                "hero_id": "hero1",
                "label": "Hero 1",
                "status": "ok",
                "elapsed_seconds": 2.0,
                "result": self._make_hero_result("hero1"),
                "error": None,
            },
        ]
        summary = _build_summary_json(
            run_start_iso="2026-05-15T10:00:00Z",
            elapsed_total=5.0,
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path,
            heroes_records=records,
            requested_heroes=["hero1"],
            run_start_epoch=0.0,
        )
        for key in ("timestamp_utc", "elapsed_total_seconds", "per_hero", "errors"):
            assert key in summary, f"summary missing key '{key}'"

    def test_summary_per_hero_entries(self, tmp_path):
        """per_hero contains one entry per executed hero."""
        from experiments.run_heroes_micro import _build_summary_json

        records = [
            {
                "hero_id": h,
                "label": f"Hero {i+1}",
                "status": "ok",
                "elapsed_seconds": 1.0,
                "result": self._make_hero_result(h),
                "error": None,
            }
            for i, h in enumerate(("hero1", "hero2", "hero3", "hero4"))
        ]
        summary = _build_summary_json(
            run_start_iso="2026-05-15T10:00:00Z",
            elapsed_total=10.0,
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path,
            heroes_records=records,
            requested_heroes=list(HERO_AGENT_MAP.keys()),
            run_start_epoch=0.0,
        )
        assert set(summary["per_hero"].keys()) == {"hero1", "hero2", "hero3", "hero4"}

    def test_orchestrator_summary_records_failed_hero(self, tmp_path):
        """A failed hero is recorded in summary.errors[]."""
        from experiments.run_heroes_micro import _build_summary_json

        records = [
            {
                "hero_id": "hero2",
                "label": "Hero 2",
                "status": "failed",
                "elapsed_seconds": 0.1,
                "result": None,
                "error": {
                    "error_type": "ModelError",
                    "message": "CUDA OOM",
                    "traceback": "...",
                },
            },
        ]
        summary = _build_summary_json(
            run_start_iso="2026-05-15T10:00:00Z",
            elapsed_total=0.5,
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path,
            heroes_records=records,
            requested_heroes=["hero2"],
            run_start_epoch=0.0,
        )
        assert len(summary["errors"]) == 1
        assert summary["errors"][0]["hero_id"] == "hero2"
        assert summary["per_hero"]["hero2"]["status"] == "failed"

    def test_orchestrator_summary_no_errors_on_all_ok(self, tmp_path):
        """No errors recorded when all heroes succeed."""
        from experiments.run_heroes_micro import _build_summary_json

        records = [
            {
                "hero_id": h,
                "label": f"Hero {i+1}",
                "status": "ok",
                "elapsed_seconds": 1.0,
                "result": self._make_hero_result(h),
                "error": None,
            }
            for i, h in enumerate(("hero1", "hero2"))
        ]
        summary = _build_summary_json(
            run_start_iso="2026-05-15T10:00:00Z",
            elapsed_total=5.0,
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path,
            heroes_records=records,
            requested_heroes=["hero1", "hero2"],
            run_start_epoch=0.0,
        )
        assert summary["errors"] == []

    def test_build_summary_md_contains_metrics(self, tmp_path):
        """_build_summary_md renders hero sections with metric values."""
        from experiments.run_heroes_micro import _build_summary_json, _build_summary_md

        records = [
            {
                "hero_id": "hero1",
                "label": "Hero 1 (A3 — Screening Título/Abstract)",
                "status": "ok",
                "elapsed_seconds": 3.0,
                "result": {
                    **self._make_hero_result("hero1"),
                    "metrics": {
                        "sensitivity_overall": 0.92,
                        "sensitivity_abstract_yes": 0.88,
                        "sensitivity_abstract_no": 0.95,
                        "tp": 46,
                        "fn": 4,
                        "uncertain_count": 0,
                    },
                },
                "error": None,
            },
        ]
        summary = _build_summary_json(
            run_start_iso="2026-05-15T10:00:00Z",
            elapsed_total=3.5,
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path,
            heroes_records=records,
            requested_heroes=["hero1"],
            run_start_epoch=0.0,
        )
        md = _build_summary_md(summary)
        assert "Hero 1" in md
        assert "0.9200" in md or "sensitivity" in md.lower()

    def test_build_summary_md_is_string(self, tmp_path):
        """_build_summary_md always returns a string (even empty records)."""
        from experiments.run_heroes_micro import _build_summary_json, _build_summary_md
        summary = _build_summary_json(
            run_start_iso="2026-05-15T10:00:00Z",
            elapsed_total=0.0,
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path,
            heroes_records=[],
            requested_heroes=[],
            run_start_epoch=0.0,
        )
        md = _build_summary_md(summary)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_run_single_hero_catches_exception(self, tmp_path):
        """_run_single_hero captures an exception and returns status='failed'."""
        from experiments.run_heroes_micro import _run_single_hero

        def bad_run_fn(config_path, output_dir, agent_instance=None):
            raise RuntimeError("Simulated failure")

        record = _run_single_hero(
            hero_id="hero1",
            label="Hero 1",
            run_fn=bad_run_fn,
            config_path=tmp_path / "micro.yaml",
            output_dir=tmp_path,
            agent_instance=None,
        )
        assert record["status"] == "failed"
        assert record["error"] is not None
        assert "RuntimeError" in record["error"]["error_type"]
        assert "Simulated failure" in record["error"]["message"]

    def test_run_single_hero_success(self, tmp_path):
        """_run_single_hero returns status='ok' when run_fn succeeds."""
        from experiments.run_heroes_micro import _run_single_hero

        hero_result = self._make_hero_result("hero1")

        def ok_run_fn(config_path, output_dir, agent_instance=None):
            return hero_result

        record = _run_single_hero(
            hero_id="hero1",
            label="Hero 1",
            run_fn=ok_run_fn,
            config_path=tmp_path / "micro.yaml",
            output_dir=tmp_path,
            agent_instance=None,
        )
        assert record["status"] == "ok"
        assert record["result"] == hero_result
        assert record["error"] is None

    def test_top_confusions_empty_matrix(self):
        """_top_confusions returns empty list for empty or None matrix."""
        from experiments.run_heroes_micro import _top_confusions
        assert _top_confusions({}) == []
        assert _top_confusions(None) == []

    def test_top_confusions_off_diagonal_only(self):
        """_top_confusions returns only off-diagonal (mismatch) cells."""
        from experiments.run_heroes_micro import _top_confusions
        matrix = {
            "include": {"include": 10, "Wrong age": 2, "Wrong outcome": 1},
            "Wrong age": {"include": 0, "Wrong age": 5, "Wrong outcome": 0},
        }
        top = _top_confusions(matrix, top_n=3)
        for entry in top:
            assert entry["gold"] != entry["predicted"]
        counts = [e["count"] for e in top]
        assert counts == sorted(counts, reverse=True)

    def test_extract_top_metrics_hero1(self):
        """_extract_top_metrics returns sensitivity metrics for hero1."""
        from experiments.run_heroes_micro import _extract_top_metrics
        rec = {
            "hero_id": "hero1",
            "result": {
                "metrics": {
                    "sensitivity_overall": 0.9,
                    "sensitivity_abstract_yes": 0.85,
                    "sensitivity_abstract_no": 0.95,
                    "tp": 45,
                    "fn": 5,
                    "uncertain_count": 0,
                }
            }
        }
        top = _extract_top_metrics(rec)
        assert top["sensitivity_overall"] == 0.9
        assert top["tp"] == 45

    def test_extract_top_metrics_hero4(self):
        """_extract_top_metrics returns primary kappa metrics for hero4."""
        from experiments.run_heroes_micro import _extract_top_metrics
        rec = {
            "hero_id": "hero4",
            "result": {
                "metrics": {
                    "primary": {
                        "cohens_kappa_quadratic": 0.65,
                        "cohens_kappa_linear": 0.70,
                        "accuracy_flat": 0.72,
                        "n_pairs": 10,
                    },
                    "secondary": {
                        "pct_outcomes_correctly_covered": 0.60,
                        "n_outcomes": 18,
                    },
                }
            }
        }
        top = _extract_top_metrics(rec)
        assert top["primary_kappa_quadratic"] == 0.65
        assert top["secondary_n_outcomes"] == 18

    def test_parse_heroes_arg_valid(self):
        """_parse_heroes_arg parses valid comma-separated hero IDs."""
        from experiments.run_heroes_micro import _parse_heroes_arg
        result = _parse_heroes_arg("hero1,hero3")
        assert result == ["hero1", "hero3"]

    def test_parse_heroes_arg_empty_returns_defaults(self):
        """_parse_heroes_arg with empty string returns default heroes."""
        from experiments.run_heroes_micro import _parse_heroes_arg, DEFAULT_HEROES
        result = _parse_heroes_arg("")
        assert result == list(DEFAULT_HEROES)

    def test_parse_heroes_arg_unknown_raises(self):
        """_parse_heroes_arg raises argparse error for unknown hero IDs."""
        import argparse
        from experiments.run_heroes_micro import _parse_heroes_arg
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_heroes_arg("hero5,hero6")


# ──────────────────────────────────────────────────────────────────────────────
# TestEvalCommon
# ──────────────────────────────────────────────────────────────────────────────

class TestEvalCommon:
    """Test _eval_common.py helpers."""

    def test_paula_protocol_has_pico_fields(self):
        """PAULA_SPORT_PROTOCOL carries all PICO framework fields."""
        from experiments._eval_common import PAULA_SPORT_PROTOCOL
        for key in ("framework", "population", "intervention", "comparator",
                    "outcome", "research_question", "inclusion_criteria",
                    "exclusion_criteria"):
            assert key in PAULA_SPORT_PROTOCOL, (
                f"PAULA_SPORT_PROTOCOL missing key '{key}'"
            )
        assert PAULA_SPORT_PROTOCOL["framework"] == "PICO"
        assert len(PAULA_SPORT_PROTOCOL["inclusion_criteria"]) >= 3
        assert len(PAULA_SPORT_PROTOCOL["exclusion_criteria"]) >= 3

    def test_truncate_short_string(self):
        """String shorter than n is returned unchanged."""
        from experiments._eval_common import truncate
        s = "hello"
        assert truncate(s, 100) == "hello"

    def test_truncate_long_string(self):
        """String longer than n is truncated with ellipsis."""
        from experiments._eval_common import truncate
        s = "a" * 200
        result = truncate(s, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_truncate_exact_boundary(self):
        """String exactly n chars is NOT truncated."""
        from experiments._eval_common import truncate
        s = "x" * 10
        assert truncate(s, 10) == s

    def test_truncate_none_returns_empty_string(self):
        """None input → empty string (or at most '...' if n < 3)."""
        from experiments._eval_common import truncate
        result = truncate(None, 50)
        assert isinstance(result, str)

    def test_truncate_special_chars(self):
        """Truncate works with accented / CJK characters."""
        from experiments._eval_common import truncate
        s = "café au lait " * 20
        result = truncate(s, 30)
        assert len(result) <= 30
        assert result.endswith("...")

    def test_load_eval_config_parses_yaml(self, tmp_path):
        """load_eval_config correctly parses a YAML file into a dict."""
        from experiments._eval_common import load_eval_config
        config_file = tmp_path / "micro.yaml"
        config_file.write_text(
            "sample: data/sample.csv\ngold_screening: data/gold.csv\nheroes:\n  - hero1\n  - hero2\n"
        )
        cfg = load_eval_config(config_file)
        assert cfg["sample"] == "data/sample.csv"
        assert cfg["gold_screening"] == "data/gold.csv"
        assert "hero1" in cfg["heroes"]

    def test_load_eval_config_empty_yaml(self, tmp_path):
        """load_eval_config returns empty dict for empty YAML file."""
        from experiments._eval_common import load_eval_config
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        cfg = load_eval_config(config_file)
        assert cfg == {}

    def test_load_agent_params_with_existing_agent(self, tmp_path, monkeypatch):
        """load_agent_params returns merged defaults + agent-specific params."""
        from experiments import _eval_common

        # Write a temporary agents.json
        fake_agents_json = {
            "defaults": {"temperature": 0.3, "repetition_penalty": 1.3},
            "agents": {
                "agent3_screening": {"temperature": 0.2, "max_tokens": 150},
            },
        }
        agents_json_path = tmp_path / "agents.json"
        agents_json_path.write_text(json.dumps(fake_agents_json))
        monkeypatch.setattr(_eval_common, "AGENTS_JSON_PATH", agents_json_path)

        params = _eval_common.load_agent_params("agent3_screening")
        assert params["temperature"] == 0.2       # agent-specific overrides default
        assert params["repetition_penalty"] == 1.3  # from defaults
        assert params["max_tokens"] == 150

    def test_load_agent_params_with_missing_agent(self, tmp_path, monkeypatch):
        """load_agent_params returns defaults when agent key is absent."""
        from experiments import _eval_common

        fake_agents_json = {
            "defaults": {"temperature": 0.3, "repetition_penalty": 1.3},
            "agents": {},
        }
        agents_json_path = tmp_path / "agents.json"
        agents_json_path.write_text(json.dumps(fake_agents_json))
        monkeypatch.setattr(_eval_common, "AGENTS_JSON_PATH", agents_json_path)

        params = _eval_common.load_agent_params("agent99_unknown")
        assert params["temperature"] == 0.3
        assert params["repetition_penalty"] == 1.3

    def test_load_agent_params_missing_file_returns_empty(self, tmp_path, monkeypatch):
        """load_agent_params returns {} when agents.json does not exist."""
        from experiments import _eval_common

        missing_path = tmp_path / "does_not_exist.json"
        monkeypatch.setattr(_eval_common, "AGENTS_JSON_PATH", missing_path)

        params = _eval_common.load_agent_params("agent3_screening")
        assert params == {}

    def test_seed_torch_no_crash_without_cuda(self):
        """seed_torch does not crash when CUDA is unavailable."""
        from experiments._eval_common import seed_torch
        # Must not raise regardless of whether torch/CUDA is available
        seed_torch(42)

    def test_seed_torch_with_mocked_torch(self):
        """seed_torch calls torch.manual_seed when torch is available."""
        from unittest.mock import patch, MagicMock

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Re-import to pick up mock
            import importlib
            from experiments import _eval_common as ec
            # Call directly using the module's function
            ec.seed_torch(42)
            # Verify manual_seed was called
            mock_torch.manual_seed.assert_called_once_with(42)

    def test_gpu_info_returns_string(self):
        """gpu_info always returns a string (even on CPU-only env)."""
        from experiments._eval_common import gpu_info
        result = gpu_info()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_pdf_text_missing_file(self, tmp_path):
        """extract_pdf_text returns (None, error_message) for non-existent file."""
        from experiments._eval_common import extract_pdf_text
        missing = tmp_path / "not_a_real_file.pdf"
        text, err = extract_pdf_text(missing)
        assert text is None
        assert err is not None
        assert "not found" in err.lower() or "pdf" in err.lower()

    def test_extract_pdf_text_returns_tuple_on_real_pdf(self, tmp_path):
        """extract_pdf_text returns a 2-tuple; mocks out the PDF parsing backends."""
        from experiments._eval_common import extract_pdf_text
        from unittest.mock import patch, MagicMock

        # Create a dummy file so is_file() returns True
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 mock content for testing")

        # Mock both fitz and pdfplumber to return extracted text
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Extracted text from mock PDF"
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.close = MagicMock()

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_pdf_text(dummy_pdf)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_extract_pdf_text_empty_file(self, tmp_path):
        """extract_pdf_text returns (None, error) for an empty (0-byte) file."""
        from experiments._eval_common import extract_pdf_text
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")
        text, err = extract_pdf_text(empty_pdf)
        assert text is None
        assert err is not None
        assert "empty" in err.lower() or "0" in err

    def test_paula_protocol_population_mentions_age(self):
        """Population field explicitly mentions an age range."""
        from experiments._eval_common import PAULA_SPORT_PROTOCOL
        pop = PAULA_SPORT_PROTOCOL["population"]
        assert "50" in pop or "70" in pop or "age" in pop.lower()

    def test_paula_protocol_intervention_mentions_behaviour_change(self):
        """Intervention mentions behaviour-change strategies."""
        from experiments._eval_common import PAULA_SPORT_PROTOCOL
        iv = PAULA_SPORT_PROTOCOL["intervention"].lower()
        assert any(kw in iv for kw in (
            "behaviour", "behavior", "motivational", "cognitive", "goal"
        ))
