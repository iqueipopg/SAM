# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for T17.A2 prompt_template kwarg injection into Agent1Protocol.generate_pubmed_query.

Verifies:
  1. With prompt_template=None the prompt passed to _simple_generate is byte-identical
     to the current hardcoded prompt.
  2. With a custom template the formatted string is passed correctly.
  3. Custom template goes through the same postprocessing (Spanish check, AND-count,
     MeSH validation, fallback) as the default path.
  4. Spanish output from a custom template triggers _build_fallback_query.
  5. A template with a missing placeholder raises KeyError explicitly.

No GPU — load_model and _simple_generate are always mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent1_protocol import Agent1Protocol


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_agent() -> Agent1Protocol:
    """Return an Agent1Protocol instance with model loading disabled."""
    with patch("agents.base.AutoModelForCausalLM.from_pretrained"):
        agent = Agent1Protocol.__new__(Agent1Protocol)
        # Minimal BaseAgent state so _simple_generate / generate_pubmed_query work.
        agent.model = MagicMock()
        agent.tokenizer = MagicMock()
        agent.do_sample = True
        agent.top_p = 0.95
        agent.config = {}
        agent.logger = MagicMock()
        # Suppress groq detection — no groq in tests.
        agent._is_groq = MagicMock(return_value=False)
        return agent


# A minimal PICO protocol dict (used as the ``protocol`` argument).
_PICO_PROTOCOL = {
    "framework": "PICO",
    "population": "adults",
    "intervention": "CBT",
    "comparator": "Placebo",
    "outcome": "depression",
}

# A valid Boolean query that passes all postprocessing filters.
_VALID_QUERY = (
    '("Cognitive Behavioral Therapy"[MeSH] OR CBT[Title/Abstract]) '
    'AND ("Depression"[MeSH] OR depression[Title/Abstract])'
)


def _build_expected_default_prompt(user_query: str, protocol: dict) -> str:
    """Reproduce the T17.A2 winning prompt (V3+add_mesh_hint) hardcoded in
    agent1_protocol.py. Drift detector: any change to the default must update
    both source and this fixture in the same commit."""
    framework = protocol.get("framework", "PICO")
    design_filter = {
        "PICO": "randomized controlled trial",
        "SPIDER": "qualitative",
        "PECO": "cohort",
        "PCC": "prevalence",
    }.get(framework, "randomized controlled trial")

    intv = protocol.get("intervention", protocol.get("exposure", ""))
    comp = protocol.get("comparator", "")
    outcome = protocol.get("outcome", "")

    intv_label = "Exposure" if framework == "PECO" else "Intervention"

    return (
        "You are a PubMed search expert. Build a structured Boolean search query.\n\n"
        f'Research question: "{user_query}"\n'
        f"{intv_label}: {intv}\n"
        f"Comparator: {comp}\n"
        f"Outcome/Condition: {outcome}\n"
        f"Design: {design_filter}\n\n"
        "Rules:\n"
        "- Use MAXIMUM 3 AND clauses (intervention AND condition AND comparator)\n"
        "- Do NOT add population (adults/children) as a separate AND clause — it over-restricts\n"
        "- Use Boolean: AND between concepts, OR between synonyms\n"
        "- ATOMICITY: every [MeSH] tag must hold a single canonical NCBI MeSH descriptor "
        "(e.g. \"Depression\"[MeSH], \"Dental Implants\"[MeSH]). NEVER use multi-word "
        "descriptive phrases as MeSH terms (forbidden: \"Different types of dental implants\"[MeSH], "
        "\"Effectiveness of CBT\"[MeSH]). If you cannot find an atomic MeSH heading, drop "
        "the [MeSH] tag and keep the term as [Title/Abstract] only.\n"
        "- Include 3-5 free-text synonyms per concept with [Title/Abstract] tag for recall\n"
        "- Group synonyms with parentheses and OR\n"
        "- Output ONLY the search string, one line, no explanation\n"
        "- All terms in English\n"
        "- HINT: prefer broader MeSH descriptors (e.g. \"Neoplasms\"[MeSH] over "
        "\"Triple Negative Breast Neoplasms\"[MeSH]) — over-specific MeSH terms "
        "drastically reduce recall.\n\n"
        "Examples:\n"
        'Question: "Does smoking increase MI risk in diabetics?"\n'
        'Output: ("Smoking"[MeSH] OR "Tobacco Use"[MeSH] OR smoking[Title/Abstract] OR cigarette[Title/Abstract] OR tobacco[Title/Abstract]) AND ("Myocardial Infarction"[MeSH] OR "myocardial infarction"[Title/Abstract] OR "heart attack"[Title/Abstract] OR "acute coronary syndrome"[Title/Abstract]) AND ("Diabetes Mellitus"[MeSH] OR diabetes[Title/Abstract] OR diabetic[Title/Abstract])\n\n'
        'Question: "Is CBT more effective than medication for depression in adults?"\n'
        'Output: ("Cognitive Behavioral Therapy"[MeSH] OR "cognitive behavioral therapy"[Title/Abstract] OR CBT[Title/Abstract] OR "cognitive therapy"[Title/Abstract] OR "behavioral therapy"[Title/Abstract]) AND ("Depression"[MeSH] OR "Depressive Disorder"[MeSH] OR depression[Title/Abstract] OR depressive[Title/Abstract] OR MDD[Title/Abstract]) AND ("Antidepressive Agents"[MeSH] OR "Drug Therapy"[MeSH] OR antidepressant[Title/Abstract] OR pharmacotherapy[Title/Abstract] OR medication[Title/Abstract] OR SSRI[Title/Abstract])\n\n'
        'Question: "Does mindfulness-based stress reduction reduce anxiety in college students?"\n'
        'Output: ("Mindfulness"[MeSH] OR mindfulness[Title/Abstract] OR MBSR[Title/Abstract] OR "mindfulness-based stress reduction"[Title/Abstract] OR meditation[Title/Abstract]) AND ("Anxiety"[MeSH] OR "Anxiety Disorders"[MeSH] OR anxiety[Title/Abstract] OR anxious[Title/Abstract] OR "anxiety symptoms"[Title/Abstract]) AND ("Students"[MeSH] OR student[Title/Abstract] OR undergraduate[Title/Abstract] OR university[Title/Abstract] OR college[Title/Abstract])\n\n'
        'Question: "Does laparoscopic versus open cholecystectomy reduce postoperative pain?"\n'
        'Output: ("Cholecystectomy, Laparoscopic"[MeSH] OR laparoscopic[Title/Abstract] OR "minimally invasive"[Title/Abstract] OR keyhole[Title/Abstract]) AND ("Cholecystectomy"[MeSH] OR cholecystectomy[Title/Abstract] OR "gallbladder removal"[Title/Abstract] OR "open surgery"[Title/Abstract]) AND ("Pain, Postoperative"[MeSH] OR "postoperative pain"[Title/Abstract] OR "post-operative pain"[Title/Abstract] OR "surgical pain"[Title/Abstract])\n\n'
        'Question: "Does mammography screening reduce breast cancer mortality?"\n'
        'Output: ("Mammography"[MeSH] OR mammography[Title/Abstract] OR mammogram[Title/Abstract] OR "breast screening"[Title/Abstract] OR "screening mammography"[Title/Abstract]) AND ("Breast Neoplasms"[MeSH] OR "breast cancer"[Title/Abstract] OR "breast carcinoma"[Title/Abstract] OR "breast tumor"[Title/Abstract]) AND ("Mortality"[MeSH] OR mortality[Title/Abstract] OR death[Title/Abstract] OR survival[Title/Abstract])\n\n'
        'Question: "Does HPV vaccination prevent cervical cancer in adolescents?"\n'
        'Output: ("Papillomavirus Vaccines"[MeSH] OR "HPV vaccine"[Title/Abstract] OR "HPV vaccination"[Title/Abstract] OR "papillomavirus vaccine"[Title/Abstract] OR Gardasil[Title/Abstract]) AND ("Uterine Cervical Neoplasms"[MeSH] OR "cervical cancer"[Title/Abstract] OR "cervical neoplasm"[Title/Abstract] OR "cervical carcinoma"[Title/Abstract]) AND ("Primary Prevention"[MeSH] OR prevention[Title/Abstract] OR prophylaxis[Title/Abstract] OR "preventive"[Title/Abstract])\n\n'
        "Output:"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — default prompt is bit-identical to the hardcoded string
# ──────────────────────────────────────────────────────────────────────────────

def test_default_prompt_unchanged_bit_identical():
    """With prompt_template=None the prompt passed to _simple_generate equals the
    current hardcoded string character-for-character."""
    agent = _make_agent()
    user_query = "Is CBT more effective than medication for depression in adults?"
    expected_prompt = _build_expected_default_prompt(user_query, _PICO_PROTOCOL)

    captured: list[str] = []

    def fake_simple_generate(prompt, max_tokens=300):
        captured.append(prompt)
        return _VALID_QUERY

    with patch.object(Agent1Protocol, "_simple_generate", side_effect=fake_simple_generate):
        agent.generate_pubmed_query(user_query, _PICO_PROTOCOL, prompt_template=None)

    assert len(captured) == 1, "Expected exactly one call to _simple_generate"
    actual_prompt = captured[0]
    assert actual_prompt == expected_prompt, (
        f"Prompt mismatch.\n"
        f"EXPECTED:\n{expected_prompt!r}\n\n"
        f"ACTUAL:\n{actual_prompt!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — custom template is formatted and passed to _simple_generate
# ──────────────────────────────────────────────────────────────────────────────

def test_custom_template_formatted_correctly():
    """With a custom template, _simple_generate receives the formatted string."""
    agent = _make_agent()
    user_query = "Is CBT more effective than medication for depression in adults?"
    template = "Q: {user_query} I: {intv} C: {comp} O: {outcome} D: {design_filter}"

    captured: list[str] = []

    def fake_simple_generate(prompt, max_tokens=300):
        captured.append(prompt)
        return _VALID_QUERY

    with patch.object(Agent1Protocol, "_simple_generate", side_effect=fake_simple_generate):
        agent.generate_pubmed_query(user_query, _PICO_PROTOCOL, prompt_template=template)

    assert len(captured) == 1
    actual_prompt = captured[0]
    # The formatted string must contain the literal values (not the placeholders).
    assert f"Q: {user_query}" in actual_prompt
    assert "I: CBT" in actual_prompt
    assert "C: Placebo" in actual_prompt
    assert "O: depression" in actual_prompt
    assert "D: randomized controlled trial" in actual_prompt
    # No raw placeholder braces should remain.
    assert "{user_query}" not in actual_prompt
    assert "{intv}" not in actual_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — custom template passes through postprocessing unchanged
# ──────────────────────────────────────────────────────────────────────────────

def test_custom_template_preserves_postprocessing():
    """A custom template whose output is a valid Boolean query passes validation
    and is returned as-is (same behaviour as the default-prompt path)."""
    agent = _make_agent()
    user_query = "Is CBT more effective than medication for depression in adults?"
    template = "Minimal template: {user_query} {intv} {comp} {outcome} {design_filter}"

    with patch.object(Agent1Protocol, "_simple_generate", return_value=_VALID_QUERY):
        result = agent.generate_pubmed_query(user_query, _PICO_PROTOCOL, prompt_template=template)

    # The valid query passes all checks and is returned directly.
    assert result == _VALID_QUERY


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — custom template with Spanish output triggers fallback
# ──────────────────────────────────────────────────────────────────────────────

def test_custom_template_invalid_output_triggers_fallback():
    """When _simple_generate returns a Spanish phrase, _build_fallback_query is invoked."""
    agent = _make_agent()
    user_query = "Is CBT more effective than medication for depression in adults?"
    template = "Custom: {user_query} {intv} {comp} {outcome} {design_filter}"

    spanish_output = "depresión adultos"  # triggers the Spanish guard

    fallback_called: list[bool] = []
    _original_fallback = agent._build_fallback_query

    def fake_fallback(protocol, design_filter):
        fallback_called.append(True)
        return _VALID_QUERY

    with patch.object(Agent1Protocol, "_simple_generate", return_value=spanish_output), \
         patch.object(Agent1Protocol, "_build_fallback_query", side_effect=fake_fallback):
        agent.generate_pubmed_query(user_query, _PICO_PROTOCOL, prompt_template=template)

    assert fallback_called, "_build_fallback_query must be called when output is in Spanish"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — template with missing placeholder raises KeyError
# ──────────────────────────────────────────────────────────────────────────────

def test_template_with_missing_placeholder_raises():
    """A template referencing an unknown placeholder raises KeyError from .format()."""
    agent = _make_agent()
    user_query = "Is CBT more effective than medication for depression in adults?"
    bad_template = "Q: {user_query} {nonexistent}"

    # _simple_generate should NOT be called; the error happens at format time.
    with patch.object(Agent1Protocol, "_simple_generate") as mock_sg:
        with pytest.raises(KeyError):
            agent.generate_pubmed_query(user_query, _PICO_PROTOCOL, prompt_template=bad_template)

    mock_sg.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — PECO framework uses "Exposure" label, not "Intervention"
# ──────────────────────────────────────────────────────────────────────────────

def test_default_prompt_peco_uses_exposure_label():
    """For PECO framework with prompt_template=None, label is 'Exposure' not 'Intervention'."""
    agent = _make_agent()
    peco_protocol = {
        "framework": "PECO",
        "population": "smokers",
        "exposure": "smoking",
        "comparator": "non-smokers",
        "outcome": "lung cancer",
    }
    user_query = "Does smoking increase lung cancer risk?"

    captured: list[str] = []

    def fake_simple_generate(prompt, max_tokens=300):
        captured.append(prompt)
        return _VALID_QUERY

    with patch.object(Agent1Protocol, "_simple_generate", side_effect=fake_simple_generate):
        agent.generate_pubmed_query(user_query, peco_protocol, prompt_template=None)

    assert len(captured) == 1
    assert "Exposure:" in captured[0]
    assert "Intervention:" not in captured[0]


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 — too many AND clauses in custom template output triggers fallback
# ──────────────────────────────────────────────────────────────────────────────

def test_custom_template_too_many_ands_triggers_fallback():
    """Output with more than 3 top-level AND clauses triggers fallback regardless
    of whether a custom template or the default prompt was used."""
    agent = _make_agent()
    user_query = "Is CBT more effective than medication for depression in adults?"
    template = "Custom: {user_query} {intv} {comp} {outcome} {design_filter}"

    # Five AND-joined groups → four ") AND (" separators → n_ands=4 > 3.
    # The regex counts )\s*AND\s*( separators, so N groups → N-1 separators.
    # We need n_ands > 3, i.e. 5 groups (4 separators).
    four_and_query = (
        '("A"[MeSH]) AND ("B"[MeSH]) AND ("C"[MeSH]) AND ("D"[MeSH]) AND ("E"[MeSH])'
    )

    fallback_called: list[bool] = []

    def fake_fallback(protocol, design_filter):
        fallback_called.append(True)
        return _VALID_QUERY

    with patch.object(Agent1Protocol, "_simple_generate", return_value=four_and_query), \
         patch.object(Agent1Protocol, "_build_fallback_query", side_effect=fake_fallback):
        agent.generate_pubmed_query(user_query, _PICO_PROTOCOL, prompt_template=template)

    assert fallback_called, "_build_fallback_query must be called when AND count > 3"
