"""Tests para experiments/eval_a1_pico.py — evaluador PICO de A1 vs LEADSInstruct.

Cubre: pico_element_similarity, normalize_a1_framework_to_pico, _aggregate,
_hash_template, run_eval E2E con A1 y embedder mockeados, manejo de errores,
y reproducibilidad de metadatos.

Sin GPU, sin red — todo mockeado con unittest.mock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_a1_pico import (
    _aggregate,
    _hash_template,
    _utc_now_iso,
    normalize_a1_framework_to_pico,
    pico_element_similarity,
    run_eval,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers compartidos
# ══════════════════════════════════════════════════════════════════════════════

def _make_cos_sim_mock(value: float):
    """Devuelve un mock de sentence_transformers.util.cos_sim que retorna el valor indicado."""
    result = MagicMock()
    result.item.return_value = value
    return result


def _make_embedder_mock(embed_values: dict[str, np.ndarray] | None = None) -> MagicMock:
    """Crea un mock de SentenceTransformer.encode que devuelve arrays numpy fijos.

    Si embed_values es None, devuelve arrays de ceros (dim=4) para cualquier texto.
    """
    embedder = MagicMock()
    if embed_values is None:
        embedder.encode.return_value = np.zeros(4)
    else:
        def _encode_side_effect(text, **kwargs):
            return embed_values.get(text, np.zeros(4))
        embedder.encode.side_effect = _encode_side_effect
    return embedder


def _write_sample_file(path: Path, samples: list[dict]) -> None:
    """Escribe un fichero de samples mínimo en formato LEADSInstruct."""
    path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")


MINIMAL_SAMPLE = {
    "id": "leads_001",
    "selected_index": 0,
    "origin": {
        "pmid": "12345678",
        "title": "Effect of CBT on depression in adults",
    },
    "pico": {
        "P": "adults with depression",
        "I": "cognitive behavioral therapy",
        "C": "waitlist control",
        "O": "depression scores",
    },
    "publication_pmids": ["111", "222"],
    "publication_pmids_count": 2,
}

FIXED_PROTOCOL = {
    "framework": "PICO",
    "population": "adults with depression",
    "intervention": "cognitive behavioral therapy",
    "comparator": "waitlist",
    "outcome": "depression scores",
}


def _make_mock_a1(protocol=None):
    """Crea un mock de Agent1Protocol con respuesta fija."""
    mock_a1 = MagicMock()
    mock_a1.generate_protocol.return_value = protocol or FIXED_PROTOCOL
    return mock_a1


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 1 — pico_element_similarity: vectores idénticos
# ══════════════════════════════════════════════════════════════════════════════

def test_pico_similarity_identical_strings():
    """Predicted == gold en los 4 campos → cada cos ≈ 1.0, mean ≈ 1.0."""
    predicted = {
        "P": "adults with depression",
        "I": "cognitive behavioral therapy",
        "C": "waitlist control",
        "O": "depression scores",
    }
    gold = dict(predicted)

    # Para vectores idénticos cos_sim = 1.0
    fixed_vec = np.array([1.0, 0.0, 0.0, 0.0])
    embedder = _make_embedder_mock()
    embedder.encode.return_value = fixed_vec

    # util se importa localmente dentro de pico_element_similarity; patchear en sentence_transformers
    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 1.0
        mock_cos_sim.return_value = cos_result

        result = pico_element_similarity(predicted, gold, embedder)

    assert result["P"] == pytest.approx(1.0)
    assert result["I"] == pytest.approx(1.0)
    assert result["C"] == pytest.approx(1.0)
    assert result["O"] == pytest.approx(1.0)
    assert result["mean"] == pytest.approx(1.0)
    # Verificar que cos_sim se llama 4 veces (una por campo)
    assert mock_cos_sim.call_count == 4


def test_pico_similarity_no_nan_on_distant_vectors():
    """Vectores semánticamente distantes → coseno bajo, no NaN."""
    predicted = {"P": "hospital food carts", "I": "x-ray", "C": "nothing", "O": "satisfaction"}
    gold = {"P": "elderly diabetic patients", "I": "insulin therapy", "C": "metformin", "O": "HbA1c"}

    embedder = _make_embedder_mock()
    embedder.encode.return_value = np.array([0.5, 0.5, 0.0, 0.0])

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.05
        mock_cos_sim.return_value = cos_result

        result = pico_element_similarity(predicted, gold, embedder)

    for key in ("P", "I", "C", "O"):
        assert isinstance(result[key], float), f"Expected float for {key}"
        assert not (result[key] != result[key]), f"Got NaN for {key}"  # NaN check
    assert isinstance(result["mean"], float)
    assert not (result["mean"] != result["mean"])


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 2 — pico_element_similarity: campos vacíos
# ══════════════════════════════════════════════════════════════════════════════

def test_pico_similarity_empty_predicted_field_returns_zero():
    """Predicted I="" → cos = 0.0 para ese campo, sin NaN ni excepción."""
    predicted = {"P": "adults", "I": "", "C": "control", "O": "outcomes"}
    gold = {"P": "adults", "I": "CBT", "C": "control", "O": "outcomes"}

    embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.9
        mock_cos_sim.return_value = cos_result

        result = pico_element_similarity(predicted, gold, embedder)

    # I debe ser 0.0 porque predicted I está vacío
    assert result["I"] == 0.0
    # P, C, O no están vacíos → se llama embedder y cos_sim
    assert isinstance(result["mean"], float)
    assert not (result["mean"] != result["mean"])


def test_pico_similarity_empty_gold_field_returns_zero():
    """Gold C="" → cos = 0.0 para ese campo, sin NaN ni excepción."""
    predicted = {"P": "patients", "I": "intervention", "C": "waitlist", "O": "pain"}
    gold = {"P": "patients", "I": "intervention", "C": "", "O": "pain"}

    embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.8
        mock_cos_sim.return_value = cos_result

        result = pico_element_similarity(predicted, gold, embedder)

    assert result["C"] == 0.0
    assert isinstance(result["mean"], float)


def test_pico_similarity_all_empty_returns_zeros_not_nan():
    """Todos los campos vacíos en predicted → mean=0.0, no NaN, no exception."""
    predicted = {"P": "", "I": "", "C": "", "O": ""}
    gold = {"P": "adults", "I": "CBT", "C": "control", "O": "depression"}

    embedder = _make_embedder_mock()

    result = pico_element_similarity(predicted, gold, embedder)

    for key in ("P", "I", "C", "O"):
        assert result[key] == 0.0, f"Expected 0.0 for {key}"
    assert result["mean"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 3 — pico_element_similarity: embedder mockeado con vectores fijos
# ══════════════════════════════════════════════════════════════════════════════

def test_pico_similarity_cos_sim_called_four_times():
    """Embedder mockeado: cos_sim llamado exactamente 4 veces con campos no vacíos."""
    predicted = {"P": "adults", "I": "CBT", "C": "waitlist", "O": "depression"}
    gold = {"P": "adults", "I": "CBT", "C": "waitlist", "O": "depression"}

    fixed_vec = np.array([1.0, 0.0, 0.0])
    embedder = _make_embedder_mock()
    embedder.encode.return_value = fixed_vec

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.95
        mock_cos_sim.return_value = cos_result

        pico_element_similarity(predicted, gold, embedder)

    assert mock_cos_sim.call_count == 4, (
        f"Expected 4 cos_sim calls, got {mock_cos_sim.call_count}"
    )


def test_pico_similarity_with_fixed_numpy_vectors():
    """Verificar que el valor devuelto por cos_sim.item() se almacena correctamente."""
    predicted = {"P": "adults", "I": "CBT", "C": "control", "O": "anxiety"}
    gold = {"P": "patients", "I": "therapy", "C": "placebo", "O": "stress"}

    embedder = _make_embedder_mock()
    embedder.encode.return_value = np.array([0.5, 0.5])

    expected_cosines = [0.7, 0.8, 0.3, 0.6]
    call_count = [0]

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        def make_cos_result(a, b):
            idx = call_count[0]
            call_count[0] += 1
            r = MagicMock()
            r.item.return_value = expected_cosines[idx]
            return r
        mock_cos_sim.side_effect = make_cos_result

        result = pico_element_similarity(predicted, gold, embedder)

    assert result["P"] == pytest.approx(0.7)
    assert result["I"] == pytest.approx(0.8)
    assert result["C"] == pytest.approx(0.3)
    assert result["O"] == pytest.approx(0.6)
    expected_mean = round((0.7 + 0.8 + 0.3 + 0.6) / 4, 6)
    assert result["mean"] == pytest.approx(expected_mean)


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 4 — normalize_a1_framework_to_pico
# ══════════════════════════════════════════════════════════════════════════════

def test_normalize_pico_direct_mapping():
    """PICO con population/intervention/comparator/outcome → mapping directo."""
    md = {
        "framework": "PICO",
        "population": "adults with depression",
        "intervention": "cognitive behavioral therapy",
        "comparator": "waitlist control",
        "outcome": "depression scores",
    }
    result = normalize_a1_framework_to_pico(md)
    assert result["P"] == "adults with depression"
    assert result["I"] == "cognitive behavioral therapy"
    assert result["C"] == "waitlist control"
    assert result["O"] == "depression scores"


def test_normalize_peco_exposure_as_intervention():
    """PECO con exposure en lugar de intervention → exposure extraído como I."""
    md = {
        "framework": "PECO",
        "population": "workers in chemical plants",
        "exposure": "benzene exposure",
        "comparator": "unexposed workers",
        "outcome": "leukemia incidence",
    }
    result = normalize_a1_framework_to_pico(md)
    assert result["P"] == "workers in chemical plants"
    assert result["I"] == "benzene exposure"
    assert result["C"] == "unexposed workers"
    assert result["O"] == "leukemia incidence"


def test_normalize_peco_exposure_takes_priority_over_intervention():
    """PECO con exposure Y intervention → exposure tiene precedencia sobre intervention."""
    md = {
        "framework": "PECO",
        "population": "adults",
        "exposure": "smoking",
        "intervention": "should_be_ignored",
        "comparator": "non-smokers",
        "outcome": "lung cancer",
    }
    result = normalize_a1_framework_to_pico(md)
    assert result["I"] == "smoking"


def test_normalize_spider_phenomenon_as_intervention():
    """SPIDER con phenomenon_of_interest → extraído como I."""
    md = {
        "framework": "SPIDER",
        "population": "nurses",
        "phenomenon_of_interest": "burnout experiences",
        "comparator": "",
        "outcome": "wellbeing",
    }
    result = normalize_a1_framework_to_pico(md)
    assert result["P"] == "nurses"
    assert result["I"] == "burnout experiences"


def test_normalize_spider_sample_as_population_fallback():
    """SPIDER con sample pero sin population → sample se usa como P."""
    md = {
        "framework": "SPIDER",
        "sample": "elderly patients",
        "phenomenon_of_interest": "pain management",
        "comparator": "",
        "outcome": "quality of life",
    }
    result = normalize_a1_framework_to_pico(md)
    assert result["P"] == "elderly patients"
    assert result["I"] == "pain management"


def test_normalize_spider_population_takes_priority_over_sample():
    """SPIDER: si hay population Y sample, population tiene prioridad."""
    md = {
        "framework": "SPIDER",
        "population": "adults",
        "sample": "elderly",
        "phenomenon_of_interest": "experience",
        "outcome": "quality",
    }
    result = normalize_a1_framework_to_pico(md)
    assert result["P"] == "adults"


def test_normalize_pcc_concept_as_intervention():
    """PCC con concept → extraído como I."""
    md = {
        "framework": "PCC",
        "population": "urban adults",
        "concept": "physical activity",
        "comparator": "",
        "outcome": "mental health",
    }
    result = normalize_a1_framework_to_pico(md)
    assert result["P"] == "urban adults"
    assert result["I"] == "physical activity"


def test_normalize_pcc_outcome_takes_priority_over_context_for_o():
    """PCC: si hay outcome, se usa como O; si falta, usa context como fallback."""
    md_with_outcome = {
        "framework": "PCC",
        "population": "adults",
        "concept": "yoga",
        "context": "urban setting",
        "outcome": "stress reduction",
    }
    result_with = normalize_a1_framework_to_pico(md_with_outcome)
    assert result_with["O"] == "stress reduction"

    md_no_outcome = {
        "framework": "PCC",
        "population": "adults",
        "concept": "yoga",
        "context": "urban setting",
    }
    result_without = normalize_a1_framework_to_pico(md_no_outcome)
    assert result_without["O"] == "urban setting"


def test_normalize_missing_fields_returns_empty_strings():
    """Campos ausentes → strings vacíos, no KeyError."""
    md = {"framework": "PICO"}
    result = normalize_a1_framework_to_pico(md)
    assert result["P"] == ""
    assert result["I"] == ""
    assert result["C"] == ""
    assert result["O"] == ""


def test_normalize_unknown_framework_falls_back_to_pico():
    """Framework desconocido → fallback a mapping PICO."""
    md = {
        "framework": "UNKNOWN_XYZ",
        "population": "adults",
        "intervention": "therapy",
        "comparator": "control",
        "outcome": "scores",
    }
    result = normalize_a1_framework_to_pico(md)
    assert result["P"] == "adults"
    assert result["I"] == "therapy"
    assert result["C"] == "control"
    assert result["O"] == "scores"


def test_normalize_none_framework_falls_back_to_pico():
    """Framework None → fallback a mapping PICO sin excepción."""
    md = {
        "framework": None,
        "population": "children",
        "intervention": "exercise",
        "comparator": "sedentary",
        "outcome": "obesity",
    }
    result = normalize_a1_framework_to_pico(md)
    assert result["P"] == "children"
    assert result["I"] == "exercise"


def test_normalize_none_input_returns_empty_dict():
    """Input None → todas las claves devuelven strings vacíos, no excepción."""
    result = normalize_a1_framework_to_pico(None)
    for key in ("P", "I", "C", "O"):
        assert result[key] == ""


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 5 — _aggregate
# ══════════════════════════════════════════════════════════════════════════════

def test_aggregate_macro_mean_correct():
    """Tres samples con pico_similarity válido → macro_mean correcto."""
    records = [
        {"pico_similarity": {"P": 0.8, "I": 0.7, "C": 0.6, "O": 0.9, "mean": 0.75}},
        {"pico_similarity": {"P": 0.5, "I": 0.6, "C": 0.4, "O": 0.7, "mean": 0.55}},
        {"pico_similarity": {"P": 0.9, "I": 0.8, "C": 0.7, "O": 0.6, "mean": 0.75}},
    ]
    result = _aggregate(records)
    expected_mean = round((0.75 + 0.55 + 0.75) / 3, 6)
    assert result["macro_mean"] == pytest.approx(expected_mean, abs=1e-5)


def test_aggregate_pct_samples_3_of_4_above_05():
    """Verifica pct_samples_3_of_4_above_0.5 con data controlada."""
    # Sample 1: P=0.6, I=0.7, C=0.8, O=0.9 → 4/4 > 0.5 → cuenta
    # Sample 2: P=0.3, I=0.7, C=0.8, O=0.9 → 3/4 > 0.5 → cuenta
    # Sample 3: P=0.3, I=0.2, C=0.8, O=0.9 → 2/4 > 0.5 → no cuenta
    records = [
        {"pico_similarity": {"P": 0.6, "I": 0.7, "C": 0.8, "O": 0.9, "mean": 0.75}},
        {"pico_similarity": {"P": 0.3, "I": 0.7, "C": 0.8, "O": 0.9, "mean": 0.675}},
        {"pico_similarity": {"P": 0.3, "I": 0.2, "C": 0.8, "O": 0.9, "mean": 0.55}},
    ]
    result = _aggregate(records)
    # 2 de 3 samples cumplen → pct = 2/3
    assert result["pct_samples_3_of_4_above_0.5"] == pytest.approx(2 / 3, abs=1e-5)


def test_aggregate_empty_records_returns_none_values():
    """Lista vacía → macro_mean=None, median=None, n_with_similarity=0."""
    result = _aggregate([])
    assert result["macro_mean"] is None
    assert result["median"] is None
    assert result["pct_samples_3_of_4_above_0.5"] is None
    assert result["n_with_similarity"] == 0


def test_aggregate_records_with_errors_skipped():
    """Records con error (sin pico_similarity) no se incluyen en el cálculo."""
    records = [
        {"error": "A1 failure", "pico_similarity": None},
        {"pico_similarity": {"P": 0.8, "I": 0.7, "C": 0.6, "O": 0.9, "mean": 0.75}},
    ]
    result = _aggregate(records)
    assert result["n_with_similarity"] == 1
    assert result["macro_mean"] == pytest.approx(0.75, abs=1e-5)


def test_aggregate_per_element_breakdown():
    """per_element contiene P_mean, I_mean, C_mean, O_mean."""
    records = [
        {"pico_similarity": {"P": 0.8, "I": 0.6, "C": 0.4, "O": 0.7, "mean": 0.625}},
        {"pico_similarity": {"P": 0.6, "I": 0.4, "C": 0.6, "O": 0.5, "mean": 0.525}},
    ]
    result = _aggregate(records)
    pe = result["per_element"]
    assert pe["P_mean"] == pytest.approx(0.7, abs=1e-5)
    assert pe["I_mean"] == pytest.approx(0.5, abs=1e-5)
    assert pe["C_mean"] == pytest.approx(0.5, abs=1e-5)
    assert pe["O_mean"] == pytest.approx(0.6, abs=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 6 — _hash_template
# ══════════════════════════════════════════════════════════════════════════════

def test_hash_template_none_returns_baseline():
    """_hash_template(None) → 'baseline'."""
    assert _hash_template(None) == "baseline"


def test_hash_template_deterministic():
    """Mismo template produce el mismo hash."""
    t = "some template {query}"
    assert _hash_template(t) == _hash_template(t)


def test_hash_template_length():
    """Hash tiene exactamente 16 caracteres hex."""
    h = _hash_template("any template string")
    assert len(h) == 16
    int(h, 16)  # valida hex válido


def test_hash_template_different_templates_differ():
    """Templates distintos producen hashes distintos."""
    h1 = _hash_template("template A {query}")
    h2 = _hash_template("template B {query}")
    assert h1 != h2


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 7 — run_eval E2E con A1 y embedder mockeados
# ══════════════════════════════════════════════════════════════════════════════

def test_run_eval_e2e_minimal_structure(tmp_path):
    """E2E mínimo: 1 sample, A1 y embedder mockeados, estructura del JSON de salida."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "output.json"

    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.8
        mock_cos_sim.return_value = cos_result

        result = run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
        )

    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert "metadata" in loaded
    assert "per_sample" in loaded
    assert "aggregates" in loaded

    # metadata debe tener campos requeridos
    meta = loaded["metadata"]
    assert meta["tag"] == "pre-T17.A3"
    assert meta["N"] == 1
    assert "embedder_model" in meta
    assert "git_commit" in meta
    assert isinstance(meta["git_commit"], str)

    # per_sample: 1 record con campos requeridos
    assert len(loaded["per_sample"]) == 1
    rec = loaded["per_sample"][0]
    assert "id" in rec
    assert "framework_chosen" in rec
    assert "predicted_pico_normalized" in rec
    assert "gold_pico" in rec
    assert "pico_similarity" in rec

    # aggregates: macro_mean, median, per_element, pct_samples_3_of_4_above_0.5
    agg = loaded["aggregates"]
    assert "macro_mean" in agg
    assert "median" in agg
    assert "per_element" in agg
    assert "pct_samples_3_of_4_above_0.5" in agg


def test_run_eval_generate_protocol_called_with_title(tmp_path):
    """A1.generate_protocol se invoca con el title del origin del sample."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "output.json"

    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.7
        mock_cos_sim.return_value = cos_result

        run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
        )

    # generate_protocol llamado con el título del sample
    mock_a1.generate_protocol.assert_called_once()
    args, kwargs = mock_a1.generate_protocol.call_args
    assert args[0] == MINIMAL_SAMPLE["origin"]["title"]


def test_run_eval_passes_metadata_template_kwarg(tmp_path):
    """metadata_prompt_template=None → generate_protocol recibe kwarg None."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "output_none.json"

    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.7
        mock_cos_sim.return_value = cos_result

        run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
            metadata_prompt_template=None,
        )

    _, kwargs = mock_a1.generate_protocol.call_args
    assert "metadata_prompt_template" in kwargs
    assert kwargs["metadata_prompt_template"] is None


def test_run_eval_passes_custom_template_kwarg(tmp_path):
    """metadata_prompt_template=string → generate_protocol recibe ese string."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "output_custom.json"

    custom_template = "Extract PICO from: {query}"
    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.7
        mock_cos_sim.return_value = cos_result

        run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
            metadata_prompt_template=custom_template,
        )

    _, kwargs = mock_a1.generate_protocol.call_args
    assert kwargs.get("metadata_prompt_template") == custom_template


def test_run_eval_multiple_samples_correct_per_sample_count(tmp_path):
    """3 samples → per_sample con 3 records."""
    second_sample = {
        "id": "leads_002",
        "origin": {"pmid": "99", "title": "CBT for anxiety in youth"},
        "pico": {"P": "youth", "I": "CBT", "C": "control", "O": "anxiety"},
        "publication_pmids": [],
    }
    third_sample = {
        "id": "leads_003",
        "origin": {"pmid": "88", "title": "Exercise and depression"},
        "pico": {"P": "adults", "I": "exercise", "C": "sedentary", "O": "mood"},
        "publication_pmids": [],
    }
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE, second_sample, third_sample])
    output_path = tmp_path / "output_multi.json"

    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.6
        mock_cos_sim.return_value = cos_result

        result = run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
        )

    assert result["metadata"]["N"] == 3
    assert len(result["per_sample"]) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 8 — Manejo de errores
# ══════════════════════════════════════════════════════════════════════════════

def test_a1_failure_annotates_error_and_continues(tmp_path):
    """A1 lanza excepción en sample 1 → error anotado; sample 2 procesa normalmente."""
    second_sample = {
        "id": "leads_002",
        "origin": {"pmid": "77", "title": "Second study"},
        "pico": {"P": "p", "I": "i", "C": "c", "O": "o"},
        "publication_pmids": [],
    }
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE, second_sample])
    output_path = tmp_path / "output_error.json"

    failing_a1 = MagicMock()
    failing_a1.generate_protocol.side_effect = [
        RuntimeError("LLM failure"),
        FIXED_PROTOCOL,
    ]
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.75
        mock_cos_sim.return_value = cos_result

        result = run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=failing_a1,
            embedder=mock_embedder,
        )

    assert len(result["per_sample"]) == 2

    first = result["per_sample"][0]
    assert first["error"] is not None
    assert "A1 failure" in first["error"]
    assert first["a1_protocol"] is None

    second = result["per_sample"][1]
    assert second["error"] is None
    assert second["pico_similarity"] is not None


def test_run_eval_fails_if_output_exists(tmp_path):
    """output ya existe → FileExistsError."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "existing.json"
    output_path.write_text("{}")

    with pytest.raises(FileExistsError):
        run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=_make_mock_a1(),
            embedder=_make_embedder_mock(),
        )


def test_run_eval_fails_if_sample_file_missing(tmp_path):
    """sample_file no existe → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_eval(
            sample_file=tmp_path / "nonexistent.json",
            output=tmp_path / "output.json",
            a1_instance=_make_mock_a1(),
            embedder=_make_embedder_mock(),
        )


def test_run_eval_raises_if_no_output_path(tmp_path):
    """Sin output ni output_path → TypeError."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    with pytest.raises(TypeError):
        run_eval(sample_file=sample_file, a1_instance=_make_mock_a1())


def test_run_eval_backward_compat_output_path_alias(tmp_path):
    """output_path como alias de output funciona correctamente."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "via_alias.json"

    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.6
        mock_cos_sim.return_value = cos_result

        result = run_eval(
            sample_file=sample_file,
            output_path=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
        )

    assert output_path.exists()
    assert "per_sample" in result


def test_run_eval_sample_missing_pico_field(tmp_path):
    """Sample sin campo 'pico' → gold_pico={}, similitudes=0.0 o error capturado."""
    sample_no_pico = {
        "id": "leads_nopico",
        "origin": {"pmid": "55", "title": "Study without PICO"},
        "publication_pmids": [],
        # sin 'pico' key
    }
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [sample_no_pico])
    output_path = tmp_path / "output_nopico.json"

    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.5
        mock_cos_sim.return_value = cos_result

        result = run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
        )

    # Debe terminar sin abortar y generar 1 record
    assert len(result["per_sample"]) == 1
    rec = result["per_sample"][0]
    # gold_pico debe ser {} (campo ausente) — no exception
    assert rec.get("gold_pico") == {} or rec.get("gold_pico") is None or rec.get("error") is not None


def test_run_eval_sample_missing_origin_title(tmp_path):
    """Sample sin origin.title → A1 recibe string vacío, no KeyError."""
    sample_no_title = {
        "id": "leads_notitle",
        "origin": {},
        "pico": {"P": "adults", "I": "therapy", "C": "control", "O": "outcomes"},
        "publication_pmids": [],
    }
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [sample_no_title])
    output_path = tmp_path / "output_notitle.json"

    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.5
        mock_cos_sim.return_value = cos_result

        result = run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
        )

    assert len(result["per_sample"]) == 1
    args, _ = mock_a1.generate_protocol.call_args
    # El primer arg debe ser string (posiblemente vacío), no None ni error
    assert isinstance(args[0], str)


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 9 — Caché A1 output
# ══════════════════════════════════════════════════════════════════════════════

def test_run_eval_a1_cache_hit_skips_generate(tmp_path):
    """Si la caché ya tiene el entry, generate_protocol NO se invoca."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "output_cache_hit.json"
    cache_path = tmp_path / "a1_cache.json"

    # Pre-poblar la caché con el sample_id + hash de template None ("baseline")
    template_hash = "baseline"
    cache_key = f"{MINIMAL_SAMPLE['id']}::{template_hash}"
    cache_path.write_text(
        json.dumps({cache_key: {"protocol": FIXED_PROTOCOL, "template_hash": "baseline"}}),
        encoding="utf-8",
    )

    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.8
        mock_cos_sim.return_value = cos_result

        result = run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
            a1_output_cache=cache_path,
        )

    # A1 NO debe invocarse — la caché resuelve
    assert mock_a1.generate_protocol.call_count == 0
    assert result["per_sample"][0]["a1_output_source"] == "cache"


def test_run_eval_a1_cache_miss_writes_entry(tmp_path):
    """Cache miss → generate_protocol invocado y entry escrito en caché."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "output_cache_miss.json"
    cache_path = tmp_path / "a1_cache_new.json"

    mock_a1 = _make_mock_a1()
    mock_embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.7
        mock_cos_sim.return_value = cos_result

        result = run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_instance=mock_a1,
            embedder=mock_embedder,
            a1_output_cache=cache_path,
        )

    assert mock_a1.generate_protocol.call_count == 1
    assert cache_path.exists()
    # La caché tiene exactamente 1 entry
    cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(cache_data) == 1
    meta = result["metadata"]
    assert meta["a1_output_cache_state"]["fresh_writes"] == 1
    assert meta["a1_output_cache_state"]["cache_hits"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 10 — _utc_now_iso
# ══════════════════════════════════════════════════════════════════════════════

def test_utc_now_iso_format():
    """_utc_now_iso devuelve string UTC ISO 8601 terminado en Z."""
    ts = _utc_now_iso()
    assert isinstance(ts, str)
    assert ts.endswith("Z")
    assert "T" in ts


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 11 — Caracteres especiales en strings PICO
# ══════════════════════════════════════════════════════════════════════════════

def test_pico_normalize_with_accented_characters():
    """Campos con tildes y caracteres especiales no causan excepción."""
    md = {
        "framework": "PICO",
        "population": "adultos mayores con depresión",
        "intervention": "terapia cognitivo-conductual",
        "comparator": "lista de espera",
        "outcome": "puntuación de ansiedad",
    }
    result = normalize_a1_framework_to_pico(md)
    assert "depresión" in result["P"]
    assert "cognitivo-conductual" in result["I"]


def test_pico_similarity_with_unicode_strings():
    """Strings con tildes/CJK no causan excepción en pico_element_similarity."""
    predicted = {"P": "患者", "I": "疗法", "C": "对照", "O": "结果"}
    gold = {"P": "patients", "I": "therapy", "C": "control", "O": "outcomes"}

    embedder = _make_embedder_mock()

    with patch("sentence_transformers.util.cos_sim") as mock_cos_sim:
        cos_result = MagicMock()
        cos_result.item.return_value = 0.3
        mock_cos_sim.return_value = cos_result

        result = pico_element_similarity(predicted, gold, embedder)

    assert isinstance(result["mean"], float)
    assert not (result["mean"] != result["mean"])  # no NaN
