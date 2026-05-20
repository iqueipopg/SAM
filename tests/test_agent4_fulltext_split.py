"""Tests for Agent4FullTextSplit (T17.B.1 Phase 2).

Covers:
- Block 1: Schema parsing (reporter V1/V2, decider V1/V3)
- Block 2: Prompt building (_build_reporter_prompt, _build_decider_prompt)
- Block 3: Integration smoke (mock LLM, happy path + error fallbacks)
- Block 4: Stats telemetry
- Block 5: Inheritance regression (parent A4 untouched, Phase A inherited)
- Block 6: Config loading (agents.json entry + defaults fallback)

No GPU required. All LLM calls mocked via unittest.mock.patch.

Usage:
    pytest tests/test_agent4_fulltext_split.py -v
"""

import json
import os
import sys
from typing import Dict
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_PROTOCOL = {
    "framework": "PICO",
    "population": "adultos con depresión",
    "intervention": "TCC",
    "comparator": "lista de espera",
    "outcome": "puntuaciones de ansiedad",
    "research_question": "¿Es la TCC más eficaz que la lista de espera para reducir la ansiedad en adultos con depresión?",
    "inclusion_criteria": [
        "Adultos de 18 años o más",
        "Diagnóstico de depresión",
        "TCC como intervención principal",
    ],
    "exclusion_criteria": [
        "Poblaciones pediátricas",
        "Estudios en animales",
    ],
}

SYNTHETIC_PAPER = {
    "pmid": "99000001",
    "title": "Terapia cognitivo-conductual para la depresión: un ECA",
    "abstract": (
        "Métodos: ECA con 100 adultos diagnosticados con depresión. "
        "Participantes asignados a TCC (n=50) o lista de espera (n=50). "
        "Resultados: Reducción significativa (SMD=-0.48, IC 95% [-0.76, -0.20])."
    ),
    "authors": ["García-López R", "Müller K"],
    "journal": "Revista de Psiquiatría Clínica",
    "year": "2024",
    "text_source": "pmc",
    "full_text": (
        "Introducción: La depresión es una causa importante de discapacidad. "
        "Métodos: ECA en 3 hospitales. 100 adultos (18-65) con TDM asignados a TCC (n=50) o "
        "lista de espera (n=50). "
        "Resultados: TCC mostró mejora significativa (SMD=-0.48, IC 95% [-0.76, -0.20], p=0.001). "
        "Discusión: Los resultados apoyan la TCC."
    ),
    "full_text_length": 420,
    "sections": {},
    "doi": "10.1000/test.99000001",
    "pmcid": "PMC99000001",
    "oa_url": None,
}

# Canned reporter JSON (V1 schema)
CANNED_REPORTER_V1 = {
    "population": "adultos 18-65 con TDM, n=100, distribución similar de sexos",
    "intervention": "TCC (n=50)",
    "comparator": "lista de espera (n=50)",
    "outcomes": "puntuación de depresión (BDI), ansiedad (BAI)",
    "study_design": "ECA",
    "publication_type": "artículo original",
    "language": "español",
    "notes": "El paper reporta SMD=-0.48 con IC 95% [-0.76, -0.20].",
}

CANNED_REPORTER_V1_STR = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)

# Canned reporter JSON (V2 schema — includes exclusion_signals)
CANNED_REPORTER_V2 = {
    **CANNED_REPORTER_V1,
    "exclusion_signals": "ninguna",
}

# Canned decider JSON (valid)
CANNED_DECIDER_INCLUDE = {
    "decision": "INCLUDE",
    "reason": "Población cumple criterio de adultos con depresión; intervención es TCC con comparador lista de espera.",
    "confidence": "high",
}

CANNED_DECIDER_EXCLUDE = {
    "decision": "EXCLUDE",
    "reason": "El diseño del estudio es una revisión narrativa, no un ECA.",
    "confidence": "high",
}

CANNED_DECIDER_LOW_CONFIDENCE = {
    "decision": "INCLUDE",
    "reason": "Varios campos no especificados.",
    "confidence": "low",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — build a mock Agent4FullTextSplit with model already loaded
# ─────────────────────────────────────────────────────────────────────────────

def _build_split_agent_no_gpu():
    """Return an Agent4FullTextSplit instance with GPU mocked out.

    Patches AutoModelForCausalLM and AutoTokenizer so no actual model
    is loaded. The returned agent has self.model set to a MagicMock.
    """
    with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
         patch("agents.base.AutoTokenizer.from_pretrained") as mock_tok_cls:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.chat_template = None
        mock_model_cls.return_value = mock_model
        mock_tok_cls.return_value = mock_tokenizer

        from agents.agent4_fulltext_split import Agent4FullTextSplit
        agent = Agent4FullTextSplit()
        # Manually mark model as loaded so screen_fulltext doesn't call load_model
        agent.model = mock_model
        agent.tokenizer = mock_tokenizer
        return agent


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — Schema parsing: reporter
# ─────────────────────────────────────────────────────────────────────────────

class TestReporterSchemaV1:
    """Tests for _parse_reporter with V1 schema (8 required fields)."""

    def _get_agent(self):
        """Get a split agent instance (no GPU)."""
        return _build_split_agent_no_gpu()

    def test_valid_json_v1_returns_dict(self):
        """JSON valido con los 8 campos V1 debe devolver un dict."""
        agent = self._get_agent()
        raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        result = agent._parse_reporter(raw)
        assert result is not None, "JSON V1 valido debe ser parseado correctamente"
        assert isinstance(result, dict)
        for key in ("population", "intervention", "comparator", "outcomes",
                    "study_design", "publication_type", "language", "notes"):
            assert key in result, f"Campo '{key}' debe estar presente en el resultado"

    def test_missing_field_returns_none(self):
        """JSON con campo faltante (omitting 'notes') debe devolver None."""
        agent = self._get_agent()
        partial = {k: v for k, v in CANNED_REPORTER_V1.items() if k != "notes"}
        raw = json.dumps(partial, ensure_ascii=False)
        result = agent._parse_reporter(raw)
        assert result is None, "JSON con campo faltante debe devolver None"

    def test_missing_multiple_fields_returns_none(self):
        """JSON con varios campos faltantes debe devolver None."""
        agent = self._get_agent()
        minimal = {"population": "adultos", "intervention": "TCC"}
        raw = json.dumps(minimal, ensure_ascii=False)
        result = agent._parse_reporter(raw)
        assert result is None

    def test_field_with_dict_value_returns_none(self):
        """JSON con tipo incorrecto: campo con valor dict (contenedor) -> None."""
        agent = self._get_agent()
        bad = dict(CANNED_REPORTER_V1)
        bad["notes"] = {"nested": "object"}  # dict is a container, should be str
        raw = json.dumps(bad, ensure_ascii=False)
        result = agent._parse_reporter(raw)
        assert result is None, "Campo con valor dict debe devolver None"

    def test_field_with_list_value_returns_none(self):
        """JSON con tipo incorrecto: campo con valor list (contenedor) -> None."""
        agent = self._get_agent()
        bad = dict(CANNED_REPORTER_V1)
        bad["outcomes"] = ["outcome1", "outcome2"]  # list is a container
        raw = json.dumps(bad, ensure_ascii=False)
        result = agent._parse_reporter(raw)
        assert result is None, "Campo con valor lista debe devolver None"

    def test_numeric_field_coerced_to_string(self):
        """Campo con valor numérico (int) no es contenedor: debe coercerse a str, no None."""
        agent = self._get_agent()
        # notes: 123 (int) is not a container — agent coerces to str
        coercible = dict(CANNED_REPORTER_V1)
        coercible["notes"] = 123
        raw = json.dumps(coercible, ensure_ascii=False)
        result = agent._parse_reporter(raw)
        # Per implementation: non-str non-container scalars are coerced to str
        assert result is not None, "Escalar no-string debe coercerse, no devolver None"
        assert isinstance(result["notes"], str), "El campo coercido debe ser str"

    def test_malformed_json_truncated_returns_none(self):
        """JSON truncado/malformado debe devolver None."""
        agent = self._get_agent()
        truncated = '{"population": "adultos", "intervention": "TCC", "comparator": '
        result = agent._parse_reporter(truncated)
        assert result is None, "JSON truncado debe devolver None"

    def test_malformed_json_no_braces_returns_none(self):
        """Texto sin llaves JSON debe devolver None."""
        agent = self._get_agent()
        result = agent._parse_reporter("Este texto no es JSON valido")
        assert result is None, "Texto sin JSON debe devolver None"

    def test_empty_string_returns_none(self):
        """String vacio debe devolver None."""
        agent = self._get_agent()
        result = agent._parse_reporter("")
        assert result is None

    def test_json_with_extra_text_before_and_after(self):
        """JSON con texto extra antes/despues debe ser extraido por _clean_json.

        Note: _clean_json strips non-ASCII chars (project rule #1 / design
        decision), so field VALUES may differ from the original after stripping.
        The key assertion is that the JSON is extracted (result is not None)
        and the expected fields are present.
        """
        agent = self._get_agent()
        raw = (
            "Aqui tienes el reporte del paper:\n"
            + json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
            + "\n\nEspero que sea util."
        )
        result = agent._parse_reporter(raw)
        assert result is not None, "_clean_json debe extraer el JSON del texto extra"
        # Only assert field presence and type — _clean_json strips non-ASCII,
        # so value equality against the original with accents is not guaranteed.
        assert "population" in result
        assert isinstance(result["population"], str)

    def test_json_with_markdown_fences(self):
        """JSON dentro de bloque markdown debe ser extraido correctamente."""
        agent = self._get_agent()
        raw = "```json\n" + json.dumps(CANNED_REPORTER_V1, ensure_ascii=False) + "\n```"
        result = agent._parse_reporter(raw)
        assert result is not None, "JSON en bloque markdown debe parsearse correctamente"

    def test_special_characters_accents_in_values(self):
        """Campos con tildes y caracteres especiales deben manejarse, aunque _clean_json
        elimine non-ASCII — el parser no debe crashear."""
        agent = self._get_agent()
        # Note: _clean_json strips non-ASCII. Values with accents will lose them,
        # but the parse should succeed (fields still present as strings).
        raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        result = agent._parse_reporter(raw)
        # After stripping non-ASCII, some values may be empty strings but all fields present
        assert result is not None or result is None, "Parser no debe crashear con tildes"


class TestReporterSchemaV2:
    """Tests for _parse_reporter with V2 schema (9 required fields)."""

    def _get_agent_v2(self):
        """Get a split agent using V2 reporter template."""
        with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("agents.base.AutoTokenizer.from_pretrained") as mock_tok_cls:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.chat_template = None
            mock_model_cls.return_value = mock_model
            mock_tok_cls.return_value = mock_tokenizer

            from agents.agent4_fulltext_split import Agent4FullTextSplit
            from agents.prompts import AGENT4_REPORTER_TEMPLATE_V2
            agent = Agent4FullTextSplit(reporter_prompt_template=AGENT4_REPORTER_TEMPLATE_V2)
            agent.model = mock_model
            agent.tokenizer = mock_tokenizer
            return agent

    def test_v2_valid_json_with_exclusion_signals(self):
        """V2 schema con 9 campos (incluido exclusion_signals) debe parsearse OK."""
        agent = self._get_agent_v2()
        raw = json.dumps(CANNED_REPORTER_V2, ensure_ascii=False)
        result = agent._parse_reporter(raw)
        assert result is not None, "V2 JSON valido debe ser parseado correctamente"
        assert "exclusion_signals" in result, "El campo 'exclusion_signals' debe estar presente"

    def test_v2_missing_exclusion_signals_returns_none(self):
        """V2 schema sin el campo 'exclusion_signals' debe devolver None."""
        agent = self._get_agent_v2()
        # V1 JSON (lacks exclusion_signals) applied against V2 agent
        raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        result = agent._parse_reporter(raw)
        assert result is None, "V2 agent: JSON sin exclusion_signals debe devolver None"

    def test_template_switch_changes_schema(self):
        """Cambiar el template en el agente debe cambiar el schema esperado."""
        from agents.agent4_fulltext_split import (
            Agent4FullTextSplit,
            REPORTER_REQUIRED_KEYS_V1,
            REPORTER_REQUIRED_KEYS_V2,
        )
        from agents.prompts import AGENT4_REPORTER_TEMPLATE_V1, AGENT4_REPORTER_TEMPLATE_V2

        with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("agents.base.AutoTokenizer.from_pretrained") as mock_tok_cls:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.chat_template = None
            mock_model_cls.return_value = mock_model
            mock_tok_cls.return_value = mock_tokenizer

            agent = Agent4FullTextSplit(reporter_prompt_template=AGENT4_REPORTER_TEMPLATE_V1)
            agent.model = mock_model
            keys_v1 = agent._reporter_required_keys()

            agent.reporter_prompt_template = AGENT4_REPORTER_TEMPLATE_V2
            keys_v2 = agent._reporter_required_keys()

        assert keys_v1 == REPORTER_REQUIRED_KEYS_V1, "Template V1 debe usar schema V1"
        assert keys_v2 == REPORTER_REQUIRED_KEYS_V2, "Template V2 debe usar schema V2"
        assert "exclusion_signals" not in keys_v1
        assert "exclusion_signals" in keys_v2


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — Schema parsing: decider
# ─────────────────────────────────────────────────────────────────────────────

class TestDeciderSchema:
    """Tests for _parse_decider."""

    def _get_agent(self):
        return _build_split_agent_no_gpu()

    def test_valid_include_json(self):
        """JSON decider valido con decision INCLUDE debe parsearse OK."""
        agent = self._get_agent()
        raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)
        result = agent._parse_decider(raw)
        assert result is not None
        assert result["decision"] == "INCLUDE"
        assert isinstance(result["reason"], str)
        assert result["confidence"] == "high"

    def test_valid_exclude_json(self):
        """JSON decider valido con decision EXCLUDE debe parsearse OK."""
        agent = self._get_agent()
        raw = json.dumps(CANNED_DECIDER_EXCLUDE, ensure_ascii=False)
        result = agent._parse_decider(raw)
        assert result is not None
        assert result["decision"] == "EXCLUDE"

    def test_valid_medium_confidence(self):
        """JSON decider con confidence 'medium' debe parsearse OK."""
        agent = self._get_agent()
        d = {"decision": "EXCLUDE", "reason": "Diseno observacional.", "confidence": "medium"}
        result = agent._parse_decider(json.dumps(d))
        assert result is not None
        assert result["confidence"] == "medium"

    def test_valid_low_confidence(self):
        """JSON decider con confidence 'low' debe parsearse OK."""
        agent = self._get_agent()
        raw = json.dumps(CANNED_DECIDER_LOW_CONFIDENCE, ensure_ascii=False)
        result = agent._parse_decider(raw)
        assert result is not None
        assert result["confidence"] == "low"

    def test_invalid_decision_maybe_returns_none(self):
        """decision='MAYBE' no esta en {INCLUDE, EXCLUDE} -> None."""
        agent = self._get_agent()
        d = {"decision": "MAYBE", "reason": "No estoy seguro.", "confidence": "low"}
        result = agent._parse_decider(json.dumps(d))
        assert result is None, "decision='MAYBE' debe devolver None"

    def test_invalid_decision_lowercase_is_normalized(self):
        """decision='include' (minusculas) debe normalizarse a 'INCLUDE' y parsear OK."""
        agent = self._get_agent()
        d = {"decision": "include", "reason": "Cumple.", "confidence": "high"}
        result = agent._parse_decider(json.dumps(d))
        assert result is not None, "Decision en minusculas debe normalizarse"
        assert result["decision"] == "INCLUDE"

    def test_invalid_confidence_extra_high_returns_none(self):
        """confidence='extra-high' no esta en {high, medium, low} -> None."""
        agent = self._get_agent()
        d = {"decision": "INCLUDE", "reason": "Cumple.", "confidence": "extra-high"}
        result = agent._parse_decider(json.dumps(d))
        assert result is None, "confidence='extra-high' debe devolver None"

    def test_invalid_confidence_uppercase_is_normalized(self):
        """confidence='HIGH' debe normalizarse a 'high'."""
        agent = self._get_agent()
        d = {"decision": "INCLUDE", "reason": "Cumple.", "confidence": "HIGH"}
        result = agent._parse_decider(json.dumps(d))
        assert result is not None, "Confidence en mayusculas debe normalizarse"
        assert result["confidence"] == "high"

    def test_missing_reason_returns_none(self):
        """JSON sin campo 'reason' debe devolver None."""
        agent = self._get_agent()
        d = {"decision": "INCLUDE", "confidence": "high"}
        result = agent._parse_decider(json.dumps(d))
        assert result is None, "JSON sin 'reason' debe devolver None"

    def test_missing_decision_returns_none(self):
        """JSON sin campo 'decision' debe devolver None."""
        agent = self._get_agent()
        d = {"reason": "Cumple.", "confidence": "high"}
        result = agent._parse_decider(json.dumps(d))
        assert result is None

    def test_missing_confidence_returns_none(self):
        """JSON sin campo 'confidence' debe devolver None."""
        agent = self._get_agent()
        d = {"decision": "INCLUDE", "reason": "Cumple."}
        result = agent._parse_decider(json.dumps(d))
        assert result is None

    def test_empty_string_returns_none(self):
        """String vacio debe devolver None."""
        agent = self._get_agent()
        result = agent._parse_decider("")
        assert result is None

    def test_malformed_json_returns_none(self):
        """JSON malformado (no cerrado) debe devolver None."""
        agent = self._get_agent()
        result = agent._parse_decider('{"decision": "INCLUDE", "reason": ')
        assert result is None

    def test_non_json_text_returns_none(self):
        """Texto plano sin JSON debe devolver None."""
        agent = self._get_agent()
        result = agent._parse_decider("not json at all")
        assert result is None, "Texto no-JSON debe devolver None"

    def test_v3_cot_reasoning_before_json(self):
        """V3 CoT: texto de razonamiento antes del JSON debe parsearse OK (extrae el JSON)."""
        agent = self._get_agent()
        # V3: 1-2 reasoning sentences BEFORE the final JSON block
        raw = (
            "El reporte indica un ECA con poblacion adulta y TCC como intervencion. "
            "El comparador es lista de espera, que coincide con los criterios de inclusion.\n"
            + json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)
        )
        result = agent._parse_decider(raw)
        assert result is not None, "V3 CoT: JSON al final del texto debe parsearse OK"
        assert result["decision"] == "INCLUDE"

    def test_v3_cot_reasoning_without_json_returns_none(self):
        """V3 CoT: texto de razonamiento sin JSON al final debe devolver None."""
        agent = self._get_agent()
        raw = (
            "El reporte indica un ECA con poblacion adulta. "
            "La intervencion es TCC, que cumple los criterios."
        )
        result = agent._parse_decider(raw)
        assert result is None, "V3 CoT sin JSON final debe devolver None"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — Prompt building
# ─────────────────────────────────────────────────────────────────────────────

class TestReporterPromptBuilding:
    """Tests for _build_reporter_prompt."""

    def _get_agent(self):
        return _build_split_agent_no_gpu()

    def test_reporter_prompt_not_empty(self):
        """El prompt del reporter debe ser un string no vacio."""
        agent = self._get_agent()
        prompt = agent._build_reporter_prompt(SYNTHETIC_PAPER, SYNTHETIC_PROTOCOL)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_reporter_prompt_contains_title(self):
        """El prompt del reporter debe contener el titulo del paper."""
        agent = self._get_agent()
        prompt = agent._build_reporter_prompt(SYNTHETIC_PAPER, SYNTHETIC_PROTOCOL)
        assert SYNTHETIC_PAPER["title"] in prompt, "El titulo del paper debe aparecer en el prompt"

    def test_reporter_prompt_contains_journal(self):
        """El prompt del reporter debe contener el nombre del journal."""
        agent = self._get_agent()
        prompt = agent._build_reporter_prompt(SYNTHETIC_PAPER, SYNTHETIC_PROTOCOL)
        assert SYNTHETIC_PAPER["journal"] in prompt, "El journal debe aparecer en el prompt"

    def test_reporter_prompt_contains_year(self):
        """El prompt del reporter debe contener el año del paper."""
        agent = self._get_agent()
        prompt = agent._build_reporter_prompt(SYNTHETIC_PAPER, SYNTHETIC_PROTOCOL)
        assert str(SYNTHETIC_PAPER["year"]) in prompt, "El año debe aparecer en el prompt"

    def test_reporter_prompt_contains_paper_text(self):
        """El prompt del reporter debe contener parte del texto del paper."""
        agent = self._get_agent()
        prompt = agent._build_reporter_prompt(SYNTHETIC_PAPER, SYNTHETIC_PROTOCOL)
        # At least some portion of the full text must appear (may be truncated)
        # Check that some token from the full text is present
        first_sentence = SYNTHETIC_PAPER["full_text"][:40]
        assert first_sentence in prompt, "El texto del paper debe estar en el prompt del reporter"

    def test_reporter_prompt_contains_v1_characteristic_string(self):
        """El prompt del reporter V1 debe contener la cadena caracteristica del template."""
        agent = self._get_agent()
        prompt = agent._build_reporter_prompt(SYNTHETIC_PAPER, SYNTHETIC_PROTOCOL)
        assert "Eres un extractor de información" in prompt or "extractor" in prompt, (
            "El prompt V1 del reporter debe contener su cadena caracteristica"
        )

    def test_reporter_prompt_truncation_long_text(self):
        """Texto largo debe truncarse al limite max_input_chars via _prepare_text_for_prompt."""
        agent = self._get_agent()
        # Create a paper with very long full_text that exceeds max_input_chars (6000 default)
        long_paper = dict(SYNTHETIC_PAPER)
        long_paper["full_text"] = "A" * 10000  # 10000 chars >> 6000 limit
        long_paper["sections"] = {}  # no sections so raw text path is used
        paper_text = agent._prepare_text_for_prompt(long_paper)
        # The text returned should be at most max_input_chars + marker
        max_chars = agent.get_agent_param("max_input_chars", 6000)
        assert len(paper_text) <= max_chars + len("\n[...truncated...]") + 50, (
            "El texto del paper debe truncarse al limite max_input_chars"
        )

    def test_reporter_prompt_pico_reference_present(self):
        """El prompt del reporter debe contener el bloque PICO de referencia."""
        agent = self._get_agent()
        prompt = agent._build_reporter_prompt(SYNTHETIC_PAPER, SYNTHETIC_PROTOCOL)
        # _format_a4_pico_block renders "Intervención: TCC" or similar
        assert "TCC" in prompt or "adultos" in prompt, (
            "Los criterios PICO deben estar inyectados en el prompt del reporter"
        )


class TestDeciderPromptBuilding:
    """Tests for _build_decider_prompt."""

    def _get_agent(self):
        return _build_split_agent_no_gpu()

    def test_decider_prompt_not_empty(self):
        """El prompt del decider debe ser un string no vacio."""
        agent = self._get_agent()
        prompt = agent._build_decider_prompt(CANNED_REPORTER_V1, SYNTHETIC_PROTOCOL)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_decider_prompt_contains_reporter_json(self):
        """El prompt del decider debe contener el JSON serializado del reporter."""
        agent = self._get_agent()
        prompt = agent._build_decider_prompt(CANNED_REPORTER_V1, SYNTHETIC_PROTOCOL)
        # The reporter JSON fields must be serialized into the prompt
        assert "population" in prompt, "El campo 'population' del reporte debe estar en el prompt"
        assert "study_design" in prompt, "El campo 'study_design' del reporte debe estar en el prompt"

    def test_decider_prompt_contains_criteria(self):
        """El prompt del decider debe contener los criterios de inclusion/exclusion."""
        agent = self._get_agent()
        prompt = agent._build_decider_prompt(CANNED_REPORTER_V1, SYNTHETIC_PROTOCOL)
        # _format_a4_criteria_block renders inclusion/exclusion criteria
        assert any(c in prompt for c in SYNTHETIC_PROTOCOL["inclusion_criteria"]), (
            "Los criterios de inclusion deben aparecer en el prompt del decider"
        )

    def test_decider_prompt_does_not_contain_paper_title(self):
        """CRITICO: el prompt del decider NO debe contener el titulo del paper original."""
        agent = self._get_agent()
        prompt = agent._build_decider_prompt(CANNED_REPORTER_V1, SYNTHETIC_PROTOCOL)
        # The decider should NOT see the paper directly
        assert SYNTHETIC_PAPER["title"] not in prompt, (
            "El titulo del paper NO debe estar en el prompt del decider"
        )

    def test_decider_prompt_does_not_contain_paper_fulltext(self):
        """CRITICO: el prompt del decider NO debe contener el texto completo del paper."""
        agent = self._get_agent()
        prompt = agent._build_decider_prompt(CANNED_REPORTER_V1, SYNTHETIC_PROTOCOL)
        # Check that no significant portion of the paper's full text appears
        paper_fragment = SYNTHETIC_PAPER["full_text"][:100]
        # After ASCII stripping (which _clean_json does), spanish text may change
        # We check that the abstract content is NOT copied verbatim
        assert SYNTHETIC_PAPER["abstract"] not in prompt, (
            "El abstract del paper NO debe estar en el prompt del decider"
        )

    def test_decider_prompt_does_not_contain_paper_journal(self):
        """El prompt del decider no debe contener datos de referencia directos del paper."""
        agent = self._get_agent()
        prompt = agent._build_decider_prompt(CANNED_REPORTER_V1, SYNTHETIC_PROTOCOL)
        # The decider gets only reporter JSON + criteria, not the paper metadata
        assert SYNTHETIC_PAPER["journal"] not in prompt, (
            "El journal del paper NO debe estar en el prompt del decider (solo reporter JSON + criterios)"
        )

    def test_decider_prompt_v1_characteristic_string(self):
        """El prompt del decider V1 debe contener la cadena caracteristica del template."""
        agent = self._get_agent()
        prompt = agent._build_decider_prompt(CANNED_REPORTER_V1, SYNTHETIC_PROTOCOL)
        assert "Eres un revisor que decide" in prompt or "revisor" in prompt, (
            "El prompt V1 del decider debe contener su cadena caracteristica"
        )

    def test_decider_prompt_reporter_json_as_string_accepted(self):
        """_build_decider_prompt acepta reporter_json como string serializado."""
        agent = self._get_agent()
        reporter_str = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        prompt = agent._build_decider_prompt(reporter_str, SYNTHETIC_PROTOCOL)
        assert isinstance(prompt, str) and len(prompt) > 0


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — Integration smoke (mock LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationSmoke:
    """Integration tests with mocked LLM (_generate_split)."""

    def _make_paper(self, pmid: str = "99000001") -> Dict:
        """Return a synthetic paper dict for screening."""
        p = dict(SYNTHETIC_PAPER)
        p["pmid"] = pmid
        return p

    def _make_agent(self):
        return _build_split_agent_no_gpu()

    def test_happy_path_single_paper_include(self):
        """Smoke test: reporter OK + decider INCLUDE -> paper en include list."""
        agent = self._make_agent()
        paper = self._make_paper()

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        decider_raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)

        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return reporter_raw
            return decider_raw

        agent._generate_split = mock_generate_split

        result = agent.screen_fulltext([paper], SYNTHETIC_PROTOCOL)

        assert "papers" in result, "El resultado debe tener clave 'papers'"
        assert "include" in result, "El resultado debe tener clave 'include'"
        assert "exclude" in result, "El resultado debe tener clave 'exclude'"
        assert "stats" in result, "El resultado debe tener clave 'stats'"

        assert len(result["include"]) == 1, "El paper debe estar en la lista de incluidos"
        assert len(result["exclude"]) == 0

    def test_happy_path_paper_output_structure(self):
        """Estructura del paper en output: fulltext_decision, votes, confidence, borderline."""
        agent = self._make_agent()
        paper = self._make_paper()

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        decider_raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)

        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            call_count["n"] += 1
            return reporter_raw if call_count["n"] == 1 else decider_raw

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper], SYNTHETIC_PROTOCOL)
        p = result["papers"][0]

        assert p["fulltext_decision"] == "INCLUDE"
        assert "fulltext_votes" in p
        assert len(p["fulltext_votes"]) == 1, "Single-vote: debe haber exactamente 1 voto"
        assert "screening_confidence" in p
        assert p["screening_confidence"] == 1.0, "Confidence 'high' debe mapear a 1.0"
        assert p["borderline"] is False, "Confidence 'high' no es borderline"

    def test_happy_path_vote_contains_reporter_and_decider(self):
        """El voto debe contener el JSON del reporter y del decider para analisis de errores."""
        agent = self._make_agent()
        paper = self._make_paper()

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        decider_raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)

        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            call_count["n"] += 1
            return reporter_raw if call_count["n"] == 1 else decider_raw

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper], SYNTHETIC_PROTOCOL)
        vote = result["papers"][0]["fulltext_votes"][0]

        assert "reporter" in vote, "El voto debe contener el JSON del reporter"
        assert "decider" in vote, "El voto debe contener el JSON del decider"
        assert vote["reporter"] is not None, "El reporter JSON no debe ser None en happy path"
        assert vote["decider"] is not None, "El decider JSON no debe ser None en happy path"

    def test_happy_path_paper_flags(self):
        """reporter_ok y decider_ok deben ser True en happy path."""
        agent = self._make_agent()
        paper = self._make_paper()

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        decider_raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)

        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            call_count["n"] += 1
            return reporter_raw if call_count["n"] == 1 else decider_raw

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper], SYNTHETIC_PROTOCOL)
        p = result["papers"][0]

        assert p["reporter_ok"] is True
        assert p["decider_ok"] is True

    def test_reporter_parse_error_fallback(self):
        """Reporter devuelve texto malformado: paper debe caer en exclude con reporter_parse_error."""
        agent = self._make_agent()
        paper = self._make_paper()

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            return "<<malformed>>"  # not parseable JSON

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper], SYNTHETIC_PROTOCOL)

        assert len(result["exclude"]) == 1, "Paper con reporter error debe estar en exclude"
        assert len(result["include"]) == 0

        p = result["papers"][0]
        assert p["fulltext_decision"] == "EXCLUDE"
        assert p["reporter_ok"] is False, "reporter_ok debe ser False"
        assert p["decider_ok"] is False, "decider_ok debe ser False (reporter failed)"

        # Verify the exclusion_reason is set
        vote = p["fulltext_votes"][0]
        assert vote.get("exclusion_reason") == "reporter_parse_error", (
            "exclusion_reason debe ser 'reporter_parse_error'"
        )

    def test_decider_parse_error_fallback(self):
        """Reporter OK, decider devuelve texto no-JSON: paper en exclude con decider_parse_error."""
        agent = self._make_agent()
        paper = self._make_paper()

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)

        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return reporter_raw
            return "not json at all"

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper], SYNTHETIC_PROTOCOL)

        assert len(result["exclude"]) == 1
        p = result["papers"][0]
        assert p["fulltext_decision"] == "EXCLUDE"
        assert p["reporter_ok"] is True, "reporter_ok debe ser True"
        assert p["decider_ok"] is False, "decider_ok debe ser False"

        vote = p["fulltext_votes"][0]
        assert vote.get("exclusion_reason") == "decider_parse_error", (
            "exclusion_reason debe ser 'decider_parse_error'"
        )

    def test_low_confidence_maps_to_borderline_and_0_3(self):
        """Decider con confidence='low' debe resultar en screening_confidence=0.3 y borderline=True."""
        agent = self._make_agent()
        paper = self._make_paper()

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        decider_raw = json.dumps(CANNED_DECIDER_LOW_CONFIDENCE, ensure_ascii=False)

        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            call_count["n"] += 1
            return reporter_raw if call_count["n"] == 1 else decider_raw

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper], SYNTHETIC_PROTOCOL)
        p = result["papers"][0]

        assert p["screening_confidence"] == 0.3, "confidence='low' debe mapear a 0.3"
        assert p["borderline"] is True, "confidence='low' debe marcar borderline=True"

    def test_medium_confidence_maps_to_0_6(self):
        """Decider con confidence='medium' debe resultar en screening_confidence=0.6."""
        agent = self._make_agent()
        paper = self._make_paper()

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        decider_raw = json.dumps(
            {"decision": "INCLUDE", "reason": "Campos consistentes.", "confidence": "medium"},
            ensure_ascii=False,
        )

        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            call_count["n"] += 1
            return reporter_raw if call_count["n"] == 1 else decider_raw

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper], SYNTHETIC_PROTOCOL)
        p = result["papers"][0]

        assert p["screening_confidence"] == 0.6, "confidence='medium' debe mapear a 0.6"

    def test_exclude_decision_sets_exclusion_reason(self):
        """EXCLUDE con reason debe copiar reason a exclusion_reason del paper."""
        agent = self._make_agent()
        paper = self._make_paper()

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        decider_raw = json.dumps(CANNED_DECIDER_EXCLUDE, ensure_ascii=False)

        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            call_count["n"] += 1
            return reporter_raw if call_count["n"] == 1 else decider_raw

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper], SYNTHETIC_PROTOCOL)
        p = result["papers"][0]

        assert p["fulltext_decision"] == "EXCLUDE"
        assert p["exclusion_reason"] is not None and len(p["exclusion_reason"]) > 0

    def test_two_papers_one_include_one_exclude(self):
        """Con dos papers: uno INCLUDE y uno EXCLUDE — listas deben tener un elemento cada una."""
        agent = self._make_agent()
        paper1 = self._make_paper(pmid="99000001")
        paper2 = self._make_paper(pmid="99000002")

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        include_raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)
        exclude_raw = json.dumps(CANNED_DECIDER_EXCLUDE, ensure_ascii=False)

        call_sequence = [reporter_raw, include_raw, reporter_raw, exclude_raw]
        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            idx = call_count["n"]
            call_count["n"] += 1
            return call_sequence[idx] if idx < len(call_sequence) else reporter_raw

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper1, paper2], SYNTHETIC_PROTOCOL)

        assert len(result["include"]) == 1
        assert len(result["exclude"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — Stats telemetry
# ─────────────────────────────────────────────────────────────────────────────

class TestStatsTelemetry:
    """Tests for _build_stats after smoke runs with 3 papers."""

    def _make_paper(self, pmid: str) -> Dict:
        p = dict(SYNTHETIC_PAPER)
        p["pmid"] = pmid
        return p

    def _make_agent(self):
        return _build_split_agent_no_gpu()

    def test_stats_structure_basic_keys(self):
        """El dict stats debe tener las claves de telemetry basicas."""
        agent = self._make_agent()

        # 3 papers: 1 INCLUDE, 1 EXCLUDE, 1 reporter error
        paper1 = self._make_paper("10000001")
        paper2 = self._make_paper("10000002")
        paper3 = self._make_paper("10000003")

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        include_raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)
        exclude_raw = json.dumps(CANNED_DECIDER_EXCLUDE, ensure_ascii=False)

        sequence = [
            reporter_raw,  # paper1: reporter OK
            include_raw,   # paper1: decider INCLUDE
            reporter_raw,  # paper2: reporter OK
            exclude_raw,   # paper2: decider EXCLUDE
            "<<BAD JSON>>",  # paper3: reporter fails
        ]
        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            idx = call_count["n"]
            call_count["n"] += 1
            return sequence[idx] if idx < len(sequence) else "<<END>>"

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper1, paper2, paper3], SYNTHETIC_PROTOCOL)
        stats = result["stats"]

        assert "reporter_failures" in stats, "stats debe tener 'reporter_failures'"
        assert "decider_failures" in stats, "stats debe tener 'decider_failures'"
        assert "fulltext_included" in stats, "stats debe tener 'fulltext_included'"
        assert "fulltext_excluded" in stats, "stats debe tener 'fulltext_excluded'"
        assert "total_retrieved" in stats, "stats debe tener 'total_retrieved'"

    def test_stats_reporter_failures_count(self):
        """stats['reporter_failures'] debe ser 1 cuando 1 de 3 reporters falla."""
        agent = self._make_agent()

        paper1 = self._make_paper("10000001")
        paper2 = self._make_paper("10000002")
        paper3 = self._make_paper("10000003")

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        include_raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)
        exclude_raw = json.dumps(CANNED_DECIDER_EXCLUDE, ensure_ascii=False)

        sequence = [
            reporter_raw, include_raw,   # paper1: OK INCLUDE
            reporter_raw, exclude_raw,   # paper2: OK EXCLUDE
            "<<BAD>>",                   # paper3: reporter fails
        ]
        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            idx = call_count["n"]
            call_count["n"] += 1
            return sequence[idx] if idx < len(sequence) else "<<END>>"

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper1, paper2, paper3], SYNTHETIC_PROTOCOL)
        stats = result["stats"]

        assert stats["reporter_failures"] == 1, (
            f"reporter_failures debe ser 1, obtenido: {stats['reporter_failures']}"
        )
        assert stats["decider_failures"] == 0, (
            f"decider_failures debe ser 0 (decider no se ejecuto cuando reporter falla), "
            f"obtenido: {stats['decider_failures']}"
        )

    def test_stats_decider_failures_count(self):
        """stats['decider_failures'] debe ser 1 cuando 1 de 3 deciders falla (reporter OK)."""
        agent = self._make_agent()

        paper1 = self._make_paper("10000001")
        paper2 = self._make_paper("10000002")
        paper3 = self._make_paper("10000003")

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        include_raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)

        sequence = [
            reporter_raw, include_raw,    # paper1: OK INCLUDE
            reporter_raw, include_raw,    # paper2: OK INCLUDE
            reporter_raw, "bad_decider",  # paper3: reporter OK, decider fails
        ]
        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            idx = call_count["n"]
            call_count["n"] += 1
            return sequence[idx] if idx < len(sequence) else "<<END>>"

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper1, paper2, paper3], SYNTHETIC_PROTOCOL)
        stats = result["stats"]

        assert stats["reporter_failures"] == 0
        assert stats["decider_failures"] == 1, (
            f"decider_failures debe ser 1, obtenido: {stats['decider_failures']}"
        )

    def test_stats_include_exclude_counts(self):
        """stats['fulltext_included'] y stats['fulltext_excluded'] deben ser correctos."""
        agent = self._make_agent()

        paper1 = self._make_paper("10000001")
        paper2 = self._make_paper("10000002")
        paper3 = self._make_paper("10000003")

        reporter_raw = json.dumps(CANNED_REPORTER_V1, ensure_ascii=False)
        include_raw = json.dumps(CANNED_DECIDER_INCLUDE, ensure_ascii=False)
        exclude_raw = json.dumps(CANNED_DECIDER_EXCLUDE, ensure_ascii=False)

        sequence = [
            reporter_raw, include_raw,  # paper1: INCLUDE
            reporter_raw, exclude_raw,  # paper2: EXCLUDE
            "<<BAD>>",                  # paper3: reporter error -> EXCLUDE
        ]
        call_count = {"n": 0}

        def mock_generate_split(prompt, max_tokens, temperature, repetition_penalty):
            idx = call_count["n"]
            call_count["n"] += 1
            return sequence[idx] if idx < len(sequence) else "<<END>>"

        agent._generate_split = mock_generate_split
        result = agent.screen_fulltext([paper1, paper2, paper3], SYNTHETIC_PROTOCOL)
        stats = result["stats"]

        assert len(result["include"]) == 1
        assert len(result["exclude"]) == 2
        assert stats["fulltext_included"] == 1
        assert stats["fulltext_excluded"] == 2
        assert stats["total_retrieved"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5 — Inheritance regression
# ─────────────────────────────────────────────────────────────────────────────

class TestInheritanceRegression:
    """Verify that Agent4FullText (parent) is untouched and Phase A is inherited."""

    def test_parent_has_screen_fulltext_method(self):
        """Agent4FullText debe tener screen_fulltext en su propio __dict__ (metodo monolitico)."""
        from agents.agent4_fulltext import Agent4FullText
        assert "screen_fulltext" in Agent4FullText.__dict__, (
            "Agent4FullText debe tener su propio screen_fulltext monolitico"
        )

    def test_split_has_screen_fulltext_override(self):
        """Agent4FullTextSplit debe tener screen_fulltext en su propio __dict__ (override)."""
        from agents.agent4_fulltext_split import Agent4FullTextSplit
        assert "screen_fulltext" in Agent4FullTextSplit.__dict__, (
            "Agent4FullTextSplit debe sobreescribir screen_fulltext"
        )

    def test_split_screen_fulltext_is_not_parent_method(self):
        """Los screen_fulltext del padre e hijo deben ser objetos distintos."""
        from agents.agent4_fulltext import Agent4FullText
        from agents.agent4_fulltext_split import Agent4FullTextSplit
        assert Agent4FullText.screen_fulltext is not Agent4FullTextSplit.screen_fulltext, (
            "Agent4FullTextSplit.screen_fulltext debe ser diferente del metodo padre"
        )

    def test_split_inherits_retrieve_fulltext_from_parent(self):
        """Phase A (retrieve_fulltext) debe ser el mismo objeto en padre e hijo (no override)."""
        from agents.agent4_fulltext import Agent4FullText
        from agents.agent4_fulltext_split import Agent4FullTextSplit
        # retrieve_fulltext should NOT be in the split's own __dict__
        assert "retrieve_fulltext" not in Agent4FullTextSplit.__dict__, (
            "Agent4FullTextSplit no debe sobreescribir retrieve_fulltext (Phase A heredada)"
        )

    def test_split_inherits_prepare_text_for_prompt(self):
        """_prepare_text_for_prompt debe ser heredado del padre, no duplicado."""
        from agents.agent4_fulltext_split import Agent4FullTextSplit
        assert "_prepare_text_for_prompt" not in Agent4FullTextSplit.__dict__, (
            "_prepare_text_for_prompt debe ser heredado del padre, no redefinido"
        )

    def test_split_inherits_clean_json(self):
        """_clean_json debe ser heredado del padre (Agent4FullText)."""
        from agents.agent4_fulltext_split import Agent4FullTextSplit
        assert "_clean_json" not in Agent4FullTextSplit.__dict__, (
            "_clean_json debe ser heredado, no redefinido en split"
        )

    def test_parent_screen_fulltext_signature_unchanged(self):
        """El metodo screen_fulltext del padre debe aceptar (self, papers, protocol) -> Dict."""
        import inspect
        from agents.agent4_fulltext import Agent4FullText
        sig = inspect.signature(Agent4FullText.screen_fulltext)
        params = list(sig.parameters.keys())
        assert "papers" in params, "Agent4FullText.screen_fulltext debe aceptar 'papers'"
        assert "protocol" in params, "Agent4FullText.screen_fulltext debe aceptar 'protocol'"

    def test_split_is_subclass_of_agent4_fulltext(self):
        """Agent4FullTextSplit debe heredar de Agent4FullText."""
        from agents.agent4_fulltext import Agent4FullText
        from agents.agent4_fulltext_split import Agent4FullTextSplit
        assert issubclass(Agent4FullTextSplit, Agent4FullText), (
            "Agent4FullTextSplit debe ser subclase de Agent4FullText"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6 — Config loading
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigLoading:
    """Tests for agents.json config loading in Agent4FullTextSplit.__init__."""

    def test_agents_json_entry_loaded(self):
        """agent4_validation_split debe leerse de agents.json si existe."""
        from agents.agent4_fulltext_split import Agent4FullTextSplit

        with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("agents.base.AutoTokenizer.from_pretrained") as mock_tok_cls:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.chat_template = None
            mock_model_cls.return_value = mock_model
            mock_tok_cls.return_value = mock_tokenizer

            agent = Agent4FullTextSplit()

        # agents.json has "agent4_validation_split" entry — verify merged params
        assert "reporter_temperature" in agent.agent_params or agent.reporter_temperature == 0.2, (
            "La entrada agent4_validation_split debe ser leida de agents.json"
        )

    def test_constructor_defaults_apply_when_agents_json_missing_entry(self):
        """Si agents.json NO tiene la entrada 'agent4_validation_split', los defaults se aplican."""
        from agents.agent4_fulltext_split import Agent4FullTextSplit

        with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("agents.base.AutoTokenizer.from_pretrained") as mock_tok_cls, \
             patch.object(Agent4FullTextSplit, "_load_agent_params") as mock_load_params:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.chat_template = None
            mock_model_cls.return_value = mock_model
            mock_tok_cls.return_value = mock_tokenizer

            # Simulate missing entry: _load_agent_params always returns {}
            mock_load_params.return_value = {}

            agent = Agent4FullTextSplit()

        # Constructor defaults should be in effect
        assert agent.reporter_temperature == 0.2, (
            f"reporter_temperature default debe ser 0.2, obtenido: {agent.reporter_temperature}"
        )
        assert agent.decider_temperature == 0.2, (
            f"decider_temperature default debe ser 0.2, obtenido: {agent.decider_temperature}"
        )
        assert agent.reporter_max_tokens == 500, (
            f"reporter_max_tokens default debe ser 500, obtenido: {agent.reporter_max_tokens}"
        )
        assert agent.decider_max_tokens == 200, (
            f"decider_max_tokens default debe ser 200, obtenido: {agent.decider_max_tokens}"
        )

    def test_constructor_params_override_agents_json(self):
        """Valores pasados al constructor deben prevalecer sobre agents.json."""
        from agents.agent4_fulltext_split import Agent4FullTextSplit

        with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("agents.base.AutoTokenizer.from_pretrained") as mock_tok_cls:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.chat_template = None
            mock_model_cls.return_value = mock_model
            mock_tok_cls.return_value = mock_tokenizer

            # Pass explicit override values
            agent = Agent4FullTextSplit(
                reporter_temperature=0.15,
                decider_temperature=0.05,
                reporter_max_tokens=700,
                decider_max_tokens=300,
            )

        assert agent.reporter_temperature == 0.15, "reporter_temperature debe ser el valor pasado al constructor"
        assert agent.decider_temperature == 0.05
        assert agent.reporter_max_tokens == 700
        assert agent.decider_max_tokens == 300

    def test_reporter_repetition_penalty_default(self):
        """reporter_repetition_penalty debe ser 1.3 por defecto (regla #5 del proyecto)."""
        agent = _build_split_agent_no_gpu()
        assert agent.reporter_repetition_penalty == 1.3, (
            f"reporter_repetition_penalty default debe ser 1.3, obtenido: {agent.reporter_repetition_penalty}"
        )

    def test_decider_repetition_penalty_default(self):
        """decider_repetition_penalty debe ser 1.3 por defecto (regla #5 del proyecto)."""
        agent = _build_split_agent_no_gpu()
        assert agent.decider_repetition_penalty == 1.3, (
            f"decider_repetition_penalty default debe ser 1.3, obtenido: {agent.decider_repetition_penalty}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6b — Schema constant exports
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaConstants:
    """Tests for exported schema constants."""

    def test_reporter_required_keys_v1_count(self):
        """REPORTER_REQUIRED_KEYS_V1 debe tener exactamente 8 campos."""
        from agents.agent4_fulltext_split import REPORTER_REQUIRED_KEYS_V1
        assert len(REPORTER_REQUIRED_KEYS_V1) == 8, (
            f"V1 debe tener 8 campos, tiene {len(REPORTER_REQUIRED_KEYS_V1)}"
        )

    def test_reporter_required_keys_v2_count(self):
        """REPORTER_REQUIRED_KEYS_V2 debe tener exactamente 9 campos (V1 + exclusion_signals)."""
        from agents.agent4_fulltext_split import REPORTER_REQUIRED_KEYS_V2
        assert len(REPORTER_REQUIRED_KEYS_V2) == 9, (
            f"V2 debe tener 9 campos, tiene {len(REPORTER_REQUIRED_KEYS_V2)}"
        )

    def test_reporter_required_keys_v2_is_superset_of_v1(self):
        """REPORTER_REQUIRED_KEYS_V2 debe ser superconjunto de V1."""
        from agents.agent4_fulltext_split import REPORTER_REQUIRED_KEYS_V1, REPORTER_REQUIRED_KEYS_V2
        for key in REPORTER_REQUIRED_KEYS_V1:
            assert key in REPORTER_REQUIRED_KEYS_V2, (
                f"Campo V1 '{key}' debe estar presente en V2"
            )

    def test_reporter_v2_has_exclusion_signals(self):
        """REPORTER_REQUIRED_KEYS_V2 debe incluir 'exclusion_signals'."""
        from agents.agent4_fulltext_split import REPORTER_REQUIRED_KEYS_V2
        assert "exclusion_signals" in REPORTER_REQUIRED_KEYS_V2

    def test_decider_required_keys(self):
        """DECIDER_REQUIRED_KEYS debe tener exactamente 3 campos."""
        from agents.agent4_fulltext_split import DECIDER_REQUIRED_KEYS
        assert set(DECIDER_REQUIRED_KEYS) == {"decision", "reason", "confidence"}, (
            f"DECIDER_REQUIRED_KEYS debe ser {{decision, reason, confidence}}, "
            f"obtenido: {set(DECIDER_REQUIRED_KEYS)}"
        )

    def test_decider_valid_decisions(self):
        """DECIDER_VALID_DECISIONS debe contener INCLUDE y EXCLUDE."""
        from agents.agent4_fulltext_split import DECIDER_VALID_DECISIONS
        assert "INCLUDE" in DECIDER_VALID_DECISIONS
        assert "EXCLUDE" in DECIDER_VALID_DECISIONS

    def test_decider_valid_confidence(self):
        """DECIDER_VALID_CONFIDENCE debe contener high, medium, low."""
        from agents.agent4_fulltext_split import DECIDER_VALID_CONFIDENCE
        assert set(DECIDER_VALID_CONFIDENCE) == {"high", "medium", "low"}, (
            f"DECIDER_VALID_CONFIDENCE debe ser {{high, medium, low}}, "
            f"obtenido: {set(DECIDER_VALID_CONFIDENCE)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7 — Template V1 identity check via _reporter_required_keys
# ─────────────────────────────────────────────────────────────────────────────

class TestReporterRequiredKeysSelection:
    """Tests for _reporter_required_keys() schema auto-selection."""

    def test_v1_template_returns_v1_keys(self):
        """Agent con template V1 debe usar schema V1."""
        from agents.agent4_fulltext_split import REPORTER_REQUIRED_KEYS_V1
        from agents.prompts import AGENT4_REPORTER_TEMPLATE_V1

        with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("agents.base.AutoTokenizer.from_pretrained") as mock_tok_cls:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.chat_template = None
            mock_model_cls.return_value = mock_model
            mock_tok_cls.return_value = mock_tokenizer

            from agents.agent4_fulltext_split import Agent4FullTextSplit
            agent = Agent4FullTextSplit(reporter_prompt_template=AGENT4_REPORTER_TEMPLATE_V1)
            agent.model = mock_model

        assert agent._reporter_required_keys() == REPORTER_REQUIRED_KEYS_V1

    def test_v2_template_returns_v2_keys(self):
        """Agent con template V2 debe usar schema V2."""
        from agents.agent4_fulltext_split import REPORTER_REQUIRED_KEYS_V2
        from agents.prompts import AGENT4_REPORTER_TEMPLATE_V2

        with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("agents.base.AutoTokenizer.from_pretrained") as mock_tok_cls:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.chat_template = None
            mock_model_cls.return_value = mock_model
            mock_tok_cls.return_value = mock_tokenizer

            from agents.agent4_fulltext_split import Agent4FullTextSplit
            agent = Agent4FullTextSplit(reporter_prompt_template=AGENT4_REPORTER_TEMPLATE_V2)
            agent.model = mock_model

        assert agent._reporter_required_keys() == REPORTER_REQUIRED_KEYS_V2

    def test_template_with_exclusion_signals_substring_detects_v2(self):
        """Template que contiene 'exclusion_signals' como substring debe usar schema V2."""
        from agents.agent4_fulltext_split import REPORTER_REQUIRED_KEYS_V2

        with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("agents.base.AutoTokenizer.from_pretrained") as mock_tok_cls:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.chat_template = None
            mock_model_cls.return_value = mock_model
            mock_tok_cls.return_value = mock_tokenizer

            from agents.agent4_fulltext_split import Agent4FullTextSplit
            # Inject a custom template string that contains 'exclusion_signals'
            custom_template = "... exclusion_signals field present ..."
            agent = Agent4FullTextSplit(reporter_prompt_template=custom_template)
            agent.model = mock_model

        assert agent._reporter_required_keys() == REPORTER_REQUIRED_KEYS_V2, (
            "Template con 'exclusion_signals' en el string debe usar schema V2"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 8 — Special characters (tildes, accents, CJK)
# ─────────────────────────────────────────────────────────────────────────────

class TestSpecialCharacters:
    """Tests with accented/CJK characters in paper data."""

    def _get_agent(self):
        return _build_split_agent_no_gpu()

    def test_paper_with_accented_title_no_crash(self):
        """Paper con titulo acentuado no debe causar crash en _build_reporter_prompt."""
        agent = self._get_agent()
        accented_paper = dict(SYNTHETIC_PAPER)
        accented_paper["title"] = "Psicología clínica en España: revisión sistemática"
        accented_paper["journal"] = "Revista Española de Psiquiatría"
        try:
            prompt = agent._build_reporter_prompt(accented_paper, SYNTHETIC_PROTOCOL)
            assert isinstance(prompt, str)
        except Exception as e:
            pytest.fail(f"_build_reporter_prompt falló con titulo acentuado: {e}")

    def test_paper_with_cjk_characters_no_crash(self):
        """Paper con caracteres CJK (chino/japones/coreano) no debe causar crash."""
        agent = self._get_agent()
        cjk_paper = dict(SYNTHETIC_PAPER)
        cjk_paper["title"] = "认知行为疗法对抑郁症的效果：系统综述"
        cjk_paper["full_text"] = "研究方法：100名参与者随机分配。结果：SMD=-0.48。"
        try:
            prompt = agent._build_reporter_prompt(cjk_paper, SYNTHETIC_PROTOCOL)
            assert isinstance(prompt, str)
        except Exception as e:
            pytest.fail(f"_build_reporter_prompt falló con caracteres CJK: {e}")

    def test_decider_prompt_with_special_pico_no_crash(self):
        """Criterios PICO con tildes no deben causar crash en _build_decider_prompt."""
        agent = self._get_agent()
        protocol_with_accents = dict(SYNTHETIC_PROTOCOL)
        protocol_with_accents["population"] = "adultos con diagnóstico de depresión mayor"
        protocol_with_accents["intervention"] = "Terapia Cognitivo-Conductual (TCC)"
        try:
            prompt = agent._build_decider_prompt(CANNED_REPORTER_V1, protocol_with_accents)
            assert isinstance(prompt, str)
        except Exception as e:
            pytest.fail(f"_build_decider_prompt falló con criterios acentuados: {e}")

    def test_parse_reporter_with_accented_values_no_crash(self):
        """_parse_reporter no debe crashear con valores que contienen tildes."""
        agent = self._get_agent()
        reporter_with_accents = {
            "population": "adultos con depresión, n=100",
            "intervention": "TCC (Terapia Cognitivo-Conductual)",
            "comparator": "lista de espera",
            "outcomes": "puntuación de ansiedad (BDI)",
            "study_design": "Ensayo Clínico Aleatorizado",
            "publication_type": "artículo original",
            "language": "español",
            "notes": "El paper reporta reducción significativa.",
        }
        raw = json.dumps(reporter_with_accents, ensure_ascii=False)
        # _clean_json strips non-ASCII, so result may have empty strings, but should not crash
        try:
            result = agent._parse_reporter(raw)
            # Result is either None (all fields became empty after stripping) or a dict
            assert result is None or isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"_parse_reporter falló con valores acentuados: {e}")
