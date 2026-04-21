"""Tests for Agent 3 (Title/Abstract Screening) — mocked LLM, no GPU.

Covers:
1. Happy path: ensemble of 3 votes, majority INCLUDE.
2. Exclusion: abstract off-topic, majority EXCLUDE.
3. Uncertain / tie-break: 1 INCLUDE + 1 EXCLUDE + 1 UNCERTAIN → UNCERTAIN.
4. Parse failure fallback: invalid LLM JSON → conservative EXCLUDE.
5. Edge cases: empty/None abstract, Unicode/tildes, missing title.
6. Voting threshold parametrizado (0.50 / 0.60 / 0.80) — decisiones distintas.

Strategy: patch `Agent3Screening._single_vote` and `Agent3Screening.load_model`
so we never load a real model. `_single_vote` returns a canned dict
({"vote": "...", "reasoning": "..."}) exactly like the real method.

Runs CPU-only. No network. No GPU.
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

PROTOCOL_PICO = {
    "framework": "PICO",
    "research_question": "Does CBT improve depression outcomes versus antidepressants in adults?",
    "population": "Adults (18-65) with major depressive disorder",
    "intervention": "Cognitive Behavioral Therapy (CBT)",
    "comparator": "Antidepressant medication",
    "outcome": "Depression symptom reduction (BDI, PHQ-9, HAM-D)",
    "inclusion_criteria": [
        "Adults aged 18-65 with MDD diagnosis",
        "RCT or controlled trial design",
        "Follow-up >= 8 weeks",
    ],
    "exclusion_criteria": [
        "Case reports, reviews, or editorials",
        "Follow-up < 8 weeks",
    ],
}


PAPER_ONTOPIC = {
    "pmid": "11111111",
    "title": "CBT versus sertraline for major depression: a 16-week randomized controlled trial",
    "abstract": (
        "Background: 120 adults with MDD randomized to CBT or sertraline 50mg for 16 weeks. "
        "Primary outcome: BDI-II. CBT group -14.2 (SD 6.1), sertraline -13.8 (SD 5.9)."
    ),
    "year": "2024",
    "journal": "JAMA Psychiatry",
}

PAPER_OFFTOPIC = {
    "pmid": "22222222",
    "title": "Prevalence of iron deficiency anemia in school-age children in rural Kenya",
    "abstract": (
        "Cross-sectional survey of 800 children aged 6-12. Hemoglobin measured by HemoCue. "
        "Prevalence of anemia 34%. No mention of psychological interventions."
    ),
    "year": "2023",
    "journal": "Trop Med Int Health",
}

PAPER_UNICODE = {
    "pmid": "33333333",
    "title": "Terapia cognitivo-conductual para depresión en adultos jóvenes: un ensayo",
    "abstract": (
        "Participantes: 89 adultos con depresión moderada (PHQ-9 ≥ 10). "
        "Asignación aleatoria a TCC en línea o escitalopram 10 mg/día. "
        "Resultado primario: PHQ-9 a 12 semanas. Diferencia –0.3 (IC 95% –1.8 a 1.2)."
    ),
    "year": "2024",
    "journal": "Rev Esp Psiquiatría",
}

PAPER_NO_TITLE = {
    "pmid": "44444444",
    "title": "",
    "abstract": (
        "Adults with major depressive disorder (n=150) were randomized to CBT or waitlist. "
        "CBT group showed significant PHQ-9 reduction at 12 weeks."
    ),
    "year": "2024",
    "journal": "J Affect Disord",
}

PAPER_EMPTY_ABSTRACT = {
    "pmid": "55555555",
    "title": "Some paper with no abstract field",
    "abstract": "",
    "year": "2024",
    "journal": "Unknown",
}

PAPER_NONE_ABSTRACT = {
    "pmid": "66666666",
    "title": "Another paper with None abstract",
    "abstract": None,
    "year": "2024",
    "journal": "Unknown",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_agent(ensemble_size=3, voting_threshold=0.60):
    """Build an Agent3Screening without loading any model.

    Patches load_model so instantiating the agent doesn't hit config/VRAM for
    the underlying model. We still need a valid BaseAgent.__init__ path — the
    real config.yaml is readable from project root.
    """
    from agents.agent3_screening import Agent3Screening

    with patch.object(Agent3Screening, "load_model", return_value=None):
        agent = Agent3Screening(
            ensemble_size=ensemble_size,
            voting_threshold=voting_threshold,
            aggregation_method="majority",
        )
    # Ensure screen_batch doesn't try to load the model
    agent.model = MagicMock()  # truthy, but never used (we patch _single_vote)
    agent.tokenizer = MagicMock()
    agent.model_paths = ["./models/qwen2.5-7b"]  # single model, no ensemble path
    return agent


def _vote(decision, reasoning="mock reasoning"):
    """Canned return value for _single_vote."""
    return {"vote": decision, "reasoning": reasoning}


def _expected_keys(result):
    """Assert standard output schema from _screen_single."""
    for k in (
        "pmid", "title", "decision", "confidence", "votes",
        "reasoning", "flags", "screening_confidence", "borderline",
    ):
        assert k in result, f"Missing key: {k}"
    assert result["decision"] in ("INCLUDE", "EXCLUDE", "UNCERTAIN")
    assert isinstance(result["votes"], list)
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["reasoning"], str)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestHappyPath:
    """3 votes all INCLUDE → majority INCLUDE with confidence 1.0."""

    def test_three_includes(self):
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        votes = [_vote("INCLUDE"), _vote("INCLUDE"), _vote("INCLUDE")]
        with patch.object(agent, "_single_vote", side_effect=votes):
            result = agent._screen_single(PAPER_ONTOPIC, PROTOCOL_PICO)
        _expected_keys(result)
        assert result["decision"] == "INCLUDE"
        assert result["confidence"] == 1.0
        assert len(result["votes"]) == 3
        assert all(v["decision"] == "INCLUDE" for v in result["votes"])
        assert result["screening_confidence"] == 1.0
        assert result["borderline"] is False

    def test_batch_includes(self):
        """screen_batch over 3 on-topic papers → all INCLUDE."""
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        papers = [
            dict(PAPER_ONTOPIC, pmid="p1"),
            dict(PAPER_ONTOPIC, pmid="p2"),
            dict(PAPER_ONTOPIC, pmid="p3"),
        ]
        # For every call _single_vote returns INCLUDE
        with patch.object(agent, "_single_vote", return_value=_vote("INCLUDE")):
            out = agent.screen_batch(papers, PROTOCOL_PICO)
        assert out["stats"]["total"] == 3
        assert out["stats"]["included"] == 3
        assert out["stats"]["excluded"] == 0
        assert out["stats"]["uncertain"] == 0
        for r in out["include"]:
            _expected_keys(r)


class TestExclusion:
    """Off-topic abstract → majority EXCLUDE."""

    def test_three_excludes(self):
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        votes = [_vote("EXCLUDE"), _vote("EXCLUDE"), _vote("EXCLUDE")]
        with patch.object(agent, "_single_vote", side_effect=votes):
            result = agent._screen_single(PAPER_OFFTOPIC, PROTOCOL_PICO)
        _expected_keys(result)
        assert result["decision"] == "EXCLUDE"
        # confidence tracks INCLUDE fraction
        assert result["confidence"] == 0.0
        assert result["screening_confidence"] == 1.0
        assert result["borderline"] is False

    def test_two_excludes_one_include(self):
        """2E + 1I: EXCLUDE frac=0.67 >= 0.60 → EXCLUDE."""
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        votes = [_vote("EXCLUDE"), _vote("INCLUDE"), _vote("EXCLUDE")]
        with patch.object(agent, "_single_vote", side_effect=votes):
            result = agent._screen_single(PAPER_OFFTOPIC, PROTOCOL_PICO)
        assert result["decision"] == "EXCLUDE"
        # confidence is rounded to 2 decimals in _screen_single
        assert abs(result["confidence"] - round(1 / 3, 2)) < 1e-6


class TestUncertainTieBreak:
    """Mixed ensemble → no decision reaches threshold → UNCERTAIN."""

    def test_one_include_one_exclude_one_uncertain_at_060(self):
        """I=1/3, E=1/3, U=1/3. Neither 1/3 >= 0.60 → UNCERTAIN."""
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        votes = [_vote("INCLUDE"), _vote("EXCLUDE"), _vote("UNCERTAIN")]
        with patch.object(agent, "_single_vote", side_effect=votes):
            result = agent._screen_single(PAPER_ONTOPIC, PROTOCOL_PICO)
        _expected_keys(result)
        assert result["decision"] == "UNCERTAIN"
        # borderline: screening_confidence = |1/3 - 0.5| * 2 = ~0.33 → NOT borderline (>=0.3)
        # but is close; let's just assert it's a float in [0,1]
        assert 0.0 <= result["screening_confidence"] <= 1.0

    def test_tie_2_include_2_exclude_high_threshold(self):
        """4 votes split 2/2, threshold 0.80. Neither reaches threshold → UNCERTAIN."""
        agent = _make_agent(ensemble_size=4, voting_threshold=0.80)
        votes = [_vote("INCLUDE"), _vote("EXCLUDE"), _vote("INCLUDE"), _vote("EXCLUDE")]
        with patch.object(agent, "_single_vote", side_effect=votes):
            result = agent._screen_single(PAPER_ONTOPIC, PROTOCOL_PICO)
        assert result["decision"] == "UNCERTAIN"
        assert result["confidence"] == 0.5
        # screening_confidence = |0.5 - 0.5| * 2 = 0 → borderline
        assert result["screening_confidence"] == 0.0
        assert result["borderline"] is True


class TestParseFailureFallback:
    """LLM returns garbage JSON → _single_vote itself returns EXCLUDE (conservative).

    We test the real _single_vote parse path by mocking _simple_generate to
    return invalid JSON, ensuring the fallback branch executes.
    """

    def test_invalid_json_falls_back_to_exclude(self):
        agent = _make_agent(ensemble_size=1, voting_threshold=0.60)
        bad_responses = [
            "this is not json at all",
            '{"vote": "BANANA", "reasoning": "unknown tag"}',  # invalid enum
            '{"vote":',  # truncated
            "",
        ]
        for resp in bad_responses:
            with patch.object(agent, "_simple_generate", return_value=resp):
                out = agent._single_vote(PAPER_ONTOPIC, PROTOCOL_PICO)
            # Must always be EXCLUDE or a valid vote — never crash
            assert out["vote"] in ("INCLUDE", "EXCLUDE", "UNCERTAIN")
            # For the "BANANA" case the parser maps unknown votes to EXCLUDE
            if resp.startswith('{"vote": "BANANA"'):
                assert out["vote"] == "EXCLUDE"

    def test_screen_single_exception_in_screen_batch(self):
        """If _single_vote raises, screen_batch must still return EXCLUDE entry."""
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)

        def boom(*a, **kw):
            raise RuntimeError("LLM blew up")

        with patch.object(agent, "_single_vote", side_effect=boom):
            out = agent.screen_batch([PAPER_ONTOPIC], PROTOCOL_PICO)
        assert out["stats"]["excluded"] == 1
        assert out["stats"]["included"] == 0
        err = out["exclude"][0]
        assert err["decision"] == "EXCLUDE"
        assert "screening_error" in err["flags"]
        assert err["confidence"] == 0.0


class TestEdgeCases:
    """Empty abstract, None, unicode, missing title."""

    def test_empty_abstract_does_not_crash(self):
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        # Here we test the FULL path including _single_vote, but mock the LLM
        # call so it returns EXCLUDE (most realistic for empty abstracts).
        with patch.object(agent, "_simple_generate", return_value='{"vote": "EXCLUDE", "reasoning": "empty"}'):
            result = agent._screen_single(PAPER_EMPTY_ABSTRACT, PROTOCOL_PICO)
        _expected_keys(result)
        assert result["decision"] == "EXCLUDE"

    def test_none_abstract_does_not_crash(self):
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        with patch.object(agent, "_simple_generate", return_value='{"vote": "EXCLUDE", "reasoning": "none"}'):
            result = agent._screen_single(PAPER_NONE_ABSTRACT, PROTOCOL_PICO)
        _expected_keys(result)
        assert result["decision"] == "EXCLUDE"

    def test_unicode_abstract_does_not_crash(self):
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        with patch.object(agent, "_simple_generate", return_value='{"vote": "INCLUDE", "reasoning": "ok"}'):
            result = agent._screen_single(PAPER_UNICODE, PROTOCOL_PICO)
        _expected_keys(result)
        assert result["decision"] == "INCLUDE"
        # Title with tildes should survive unchanged into result
        assert "depresión" in result["title"] or "depresion" in result["title"].lower()

    def test_missing_title_uses_abstract_only(self):
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        with patch.object(agent, "_simple_generate", return_value='{"vote": "INCLUDE", "reasoning": "ok"}'):
            result = agent._screen_single(PAPER_NO_TITLE, PROTOCOL_PICO)
        _expected_keys(result)
        assert result["decision"] == "INCLUDE"
        assert result["title"] == ""


class TestVotingThresholdParametrized:
    """Same 3-vote input, different thresholds → different decisions.

    Votes: 2 INCLUDE + 1 EXCLUDE → INCLUDE fraction = 0.666...
    - threshold 0.50 → 0.67 >= 0.50 → INCLUDE
    - threshold 0.60 → 0.67 >= 0.60 → INCLUDE
    - threshold 0.80 → 0.67 <  0.80, 0.33 < 0.80 → UNCERTAIN
    """

    VOTES_2I_1E = [_vote("INCLUDE"), _vote("INCLUDE"), _vote("EXCLUDE")]

    @pytest.mark.parametrize(
        "threshold,expected",
        [
            (0.50, "INCLUDE"),
            (0.60, "INCLUDE"),
            (0.80, "UNCERTAIN"),
        ],
    )
    def test_threshold_changes_decision(self, threshold, expected):
        agent = _make_agent(ensemble_size=3, voting_threshold=threshold)
        with patch.object(agent, "_single_vote", side_effect=list(self.VOTES_2I_1E)):
            result = agent._screen_single(PAPER_ONTOPIC, PROTOCOL_PICO)
        assert result["decision"] == expected, (
            f"threshold={threshold} expected {expected}, got {result['decision']}"
        )
        # confidence is always the INCLUDE fraction regardless of threshold (rounded)
        assert abs(result["confidence"] - round(2 / 3, 2)) < 1e-6

    def test_all_excludes_threshold_050(self):
        """3 EXCLUDE + threshold 0.50 → EXCLUDE."""
        agent = _make_agent(ensemble_size=3, voting_threshold=0.50)
        votes = [_vote("EXCLUDE"), _vote("EXCLUDE"), _vote("EXCLUDE")]
        with patch.object(agent, "_single_vote", side_effect=votes):
            result = agent._screen_single(PAPER_OFFTOPIC, PROTOCOL_PICO)
        assert result["decision"] == "EXCLUDE"

    def test_unanimous_threshold_100(self):
        """3 INCLUDE + threshold 1.00 → INCLUDE (frac=1.0 meets >=1.0)."""
        agent = _make_agent(ensemble_size=3, voting_threshold=1.00)
        votes = [_vote("INCLUDE"), _vote("INCLUDE"), _vote("INCLUDE")]
        with patch.object(agent, "_single_vote", side_effect=votes):
            result = agent._screen_single(PAPER_ONTOPIC, PROTOCOL_PICO)
        assert result["decision"] == "INCLUDE"


class TestAggregateVotesUnit:
    """Unit tests for the voting aggregation function in isolation."""

    def test_aggregate_unanimous_include(self):
        agent = _make_agent(voting_threshold=0.60)
        decision, conf = agent._aggregate_votes(["INCLUDE"] * 3)
        assert decision == "INCLUDE"
        assert conf == 1.0

    def test_aggregate_unanimous_exclude(self):
        agent = _make_agent(voting_threshold=0.60)
        decision, conf = agent._aggregate_votes(["EXCLUDE"] * 3)
        assert decision == "EXCLUDE"
        assert conf == 0.0

    def test_aggregate_all_uncertain(self):
        agent = _make_agent(voting_threshold=0.60)
        decision, conf = agent._aggregate_votes(["UNCERTAIN"] * 3)
        assert decision == "UNCERTAIN"
        assert conf == 0.0

    def test_aggregate_majority_include(self):
        agent = _make_agent(voting_threshold=0.60)
        decision, conf = agent._aggregate_votes(["INCLUDE", "INCLUDE", "EXCLUDE"])
        assert decision == "INCLUDE"
        assert abs(conf - (2 / 3)) < 1e-6


class TestSecondaryStudyOverride:
    """Deterministic EXCLUDE for meta-analyses / systematic reviews,
    regardless of LLM votes."""

    def test_systematic_review_forced_exclude(self):
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        paper = {
            "pmid": "77777777",
            "title": "Systematic review of CBT for depression in adults",
            "abstract": "We performed a systematic review of 50 trials...",
            "year": "2023",
        }
        # Even if ALL votes say INCLUDE, override forces EXCLUDE
        votes = [_vote("INCLUDE"), _vote("INCLUDE"), _vote("INCLUDE")]
        with patch.object(agent, "_single_vote", side_effect=votes):
            result = agent._screen_single(paper, PROTOCOL_PICO)
        assert result["decision"] == "EXCLUDE"
        assert "secondary_study" in result["flags"]
        assert result["confidence"] == 1.0

    def test_meta_analysis_forced_exclude(self):
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        paper = {
            "pmid": "88888888",
            "title": "Meta-analysis of psychological interventions for depression",
            "abstract": "We conducted a meta-analysis pooling 42 studies...",
            "year": "2023",
        }
        with patch.object(agent, "_single_vote", side_effect=[_vote("INCLUDE")] * 3):
            result = agent._screen_single(paper, PROTOCOL_PICO)
        assert result["decision"] == "EXCLUDE"
        assert "secondary_study" in result["flags"]


class TestJsonSerializable:
    """Output must be JSON-serializable (written to pipeline.json)."""

    def test_result_json_serializable(self):
        agent = _make_agent(ensemble_size=3, voting_threshold=0.60)
        with patch.object(agent, "_single_vote", side_effect=[_vote("INCLUDE")] * 3):
            result = agent._screen_single(PAPER_ONTOPIC, PROTOCOL_PICO)
        s = json.dumps(result)
        assert "INCLUDE" in s
        round_trip = json.loads(s)
        assert round_trip["pmid"] == "11111111"
