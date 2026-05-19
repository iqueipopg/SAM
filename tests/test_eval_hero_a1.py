"""Tests para experiments/eval_hero_a1.py — evaluador Hero A1 sobre LEADSInstruct.

Cubre: recall_at_k, _hash_query, _aggregate, _serialize_esearch, caché,
E2E con A1 mockeado, manejo de errores y reproducibilidad de metadatos.

Sin GPU, sin red — todo mockeado con unittest.mock.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_hero_a1 import (
    _aggregate,
    _hash_query,
    _serialize_esearch,
    _utc_now_iso,
    recall_at_k,
    run_eval,
)


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 1 — recall_at_k
# ══════════════════════════════════════════════════════════════════════════════

def test_recall_partial_intersection():
    """Intersección parcial: A presente en gold, B y C ausentes → recall=0.5."""
    assert recall_at_k(["A", "B", "C"], ["A", "D"]) == pytest.approx(0.5)


def test_recall_empty_intersection():
    """Sin coincidencias → recall=0.0."""
    assert recall_at_k(["X", "Y"], ["A", "B"]) == 0.0


def test_recall_full_single_gold():
    """Gold con un elemento y devuelto coincide → recall=1.0."""
    assert recall_at_k(["A"], ["A"]) == 1.0


def test_recall_long_tailed_gold():
    """Gold con 537 PMIDs, 50 solapan en returned (100) → recall ≈ 50/537."""
    gold = [str(i) for i in range(537)]
    returned = [str(i) for i in range(100)]  # cubre ids 0..99
    result = recall_at_k(returned, gold)
    assert result == pytest.approx(100 / 537)


def test_recall_empty_gold_returns_zero():
    """Gold vacío → recall=0.0 sin ZeroDivisionError (comportamiento defensivo del script)."""
    result = recall_at_k(["A"], [])
    assert result == 0.0


def test_recall_returned_with_duplicates():
    """Returned con duplicados: la implementación usa sets, así que recall=1.0."""
    assert recall_at_k(["A", "A", "B"], ["A", "B"]) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 2 — _hash_query
# ══════════════════════════════════════════════════════════════════════════════

def test_hash_query_deterministic():
    """Misma query produce el mismo hash en dos llamadas."""
    q = "CBT AND depression AND anxiety"
    assert _hash_query(q) == _hash_query(q)


def test_hash_query_length():
    """Hash devuelto tiene exactamente 16 caracteres hex."""
    h = _hash_query("some query string")
    assert len(h) == 16
    int(h, 16)  # valida que son hex válidos


def test_hash_query_space_sensitivity():
    """Queries que difieren en un espacio producen hashes distintos."""
    h1 = _hash_query("CBT depression")
    h2 = _hash_query("CBT  depression")
    assert h1 != h2


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 3 — _aggregate
# ══════════════════════════════════════════════════════════════════════════════

def test_aggregate_full_floats():
    """Lista completa de floats: mean=0.2, median=0.2, values lista original."""
    records = [
        {"recall_at_1000": 0.1, "esearch_count": 0.1},
        {"recall_at_1000": 0.2, "esearch_count": 0.2},
        {"recall_at_1000": 0.3, "esearch_count": 0.3},
    ]
    result = _aggregate(records)
    assert result["recall_at_1000"]["mean"] == pytest.approx(0.2, abs=1e-5)
    assert result["recall_at_1000"]["median"] == pytest.approx(0.2, abs=1e-5)
    assert result["recall_at_1000"]["values"] == [0.1, 0.2, 0.3]


def test_aggregate_mixed_with_none():
    """Lista mixta con None: None excluido del cálculo, pero preservado en values."""
    records = [
        {"recall_at_1000": 0.1, "esearch_count": 10},
        {"recall_at_1000": None, "esearch_count": None},
        {"recall_at_1000": 0.3, "esearch_count": 30},
    ]
    result = _aggregate(records)
    assert result["recall_at_1000"]["mean"] == pytest.approx(0.2, abs=1e-5)
    assert result["recall_at_1000"]["median"] == pytest.approx(0.2, abs=1e-5)
    # None debe aparecer en values
    assert None in result["recall_at_1000"]["values"]


def test_aggregate_all_none():
    """Lista todo None: mean=None, median=None, values=[None, None]."""
    records = [
        {"recall_at_1000": None, "esearch_count": None},
        {"recall_at_1000": None, "esearch_count": None},
    ]
    result = _aggregate(records)
    assert result["recall_at_1000"]["mean"] is None
    assert result["recall_at_1000"]["median"] is None
    assert result["recall_at_1000"]["values"] == [None, None]


def test_aggregate_empty_list():
    """Lista vacía: mean=None, median=None, values=[]."""
    result = _aggregate([])
    assert result["recall_at_1000"]["mean"] is None
    assert result["recall_at_1000"]["median"] is None
    assert result["recall_at_1000"]["values"] == []


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 4 — _serialize_esearch
# ══════════════════════════════════════════════════════════════════════════════

class _EntrezResultMock(dict):
    """Mock dict-like que imita Bio.Entrez.read result."""
    pass


def test_serialize_esearch_structure():
    """Entrez mock → dict con IdList, Count, RetMax, RetStart, query, fetched_at_utc."""
    mock_result = _EntrezResultMock({
        "IdList": ["1", "2"],
        "Count": "42",
        "RetMax": "1000",
        "RetStart": "0",
    })
    serialized = _serialize_esearch(mock_result, "test query")

    assert serialized["IdList"] == ["1", "2"]
    assert serialized["Count"] == "42"
    assert serialized["RetMax"] == "1000"
    assert serialized["RetStart"] == "0"
    assert serialized["query"] == "test query"
    assert "fetched_at_utc" in serialized
    # IdList debe ser list de str
    assert isinstance(serialized["IdList"], list)
    assert all(isinstance(x, str) for x in serialized["IdList"])


def test_serialize_esearch_json_roundtrip():
    """Round-trip json.dumps/json.loads preserva todos los valores."""
    mock_result = _EntrezResultMock({
        "IdList": ["111", "222", "333"],
        "Count": "99",
        "RetMax": "1000",
        "RetStart": "0",
    })
    serialized = _serialize_esearch(mock_result, "round-trip query")
    dumped = json.dumps(serialized)
    loaded = json.loads(dumped)

    assert loaded["IdList"] == ["111", "222", "333"]
    assert loaded["Count"] == "99"
    assert loaded["query"] == "round-trip query"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers para crear sample files y fixtures A1/Entrez mockeados
# ══════════════════════════════════════════════════════════════════════════════

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
    "intervention": "CBT",
    "comparator": "waitlist",
    "outcome": "depression scores",
    "pubmed_query": '("CBT") AND depression',
}

FIXED_QUERY = '("CBT") AND depression'


def _make_mock_a1():
    """Crea un mock de Agent1Protocol con respuestas fijas."""
    mock_a1 = MagicMock()
    mock_a1.generate_protocol.return_value = FIXED_PROTOCOL
    mock_a1.generate_pubmed_query.return_value = FIXED_QUERY
    return mock_a1


def _make_mock_esearch_result(id_list=None, count="50"):
    """Crea un mock de handle Entrez.esearch + Entrez.read."""
    if id_list is None:
        id_list = ["111", "222", "333"]
    mock_handle = MagicMock()
    mock_handle.close = MagicMock()

    raw_result = _EntrezResultMock({
        "IdList": id_list,
        "Count": count,
        "RetMax": "1000",
        "RetStart": "0",
    })
    return mock_handle, raw_result


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 5 — Cache hit/miss
# ══════════════════════════════════════════════════════════════════════════════

def test_cache_miss_calls_entrez_and_writes_file(tmp_path):
    """Cache miss: esearch invocado una vez, fichero de caché escrito en disco."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "output.json"
    cache_dir = tmp_path / "cache"

    mock_handle, raw_result = _make_mock_esearch_result(["111", "222"], count="50")

    # Agent1Protocol se importa dentro de run_eval, patch en agents.agent1_protocol
    with patch("agents.agent1_protocol.Agent1Protocol", return_value=_make_mock_a1()), \
         patch("experiments.eval_hero_a1.Entrez.esearch", return_value=mock_handle) as mock_esearch, \
         patch("experiments.eval_hero_a1.Entrez.read", return_value=raw_result), \
         patch("experiments.eval_hero_a1.time.sleep"):

        result = run_eval(sample_file=sample_file, output_path=output_path, cache_dir=cache_dir)

    assert mock_esearch.call_count == 1
    # Fichero de caché escrito
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    # Recall calculado: gold=["111","222"], returned=["111","222"] → 1.0
    assert result["per_sample"][0]["recall_at_1000"] == pytest.approx(1.0)


def test_cache_hit_does_not_call_entrez_again(tmp_path):
    """Cache hit: segunda invocación no llama a Entrez.esearch."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    mock_handle, raw_result = _make_mock_esearch_result(["111", "222"], count="50")

    # Primera pasada: sin caché — esearch llamado una vez
    with patch("agents.agent1_protocol.Agent1Protocol", return_value=_make_mock_a1()), \
         patch("experiments.eval_hero_a1.Entrez.esearch", return_value=mock_handle) as mock_esearch, \
         patch("experiments.eval_hero_a1.Entrez.read", return_value=raw_result), \
         patch("experiments.eval_hero_a1.time.sleep"):

        run_eval(
            sample_file=sample_file,
            output_path=tmp_path / "out1.json",
            cache_dir=cache_dir,
        )

    first_call_count = mock_esearch.call_count  # debe ser 1

    # Segunda pasada: caché presente — esearch NO debe invocarse
    with patch("agents.agent1_protocol.Agent1Protocol", return_value=_make_mock_a1()), \
         patch("experiments.eval_hero_a1.Entrez.esearch", return_value=mock_handle) as mock_esearch2, \
         patch("experiments.eval_hero_a1.Entrez.read", return_value=raw_result), \
         patch("experiments.eval_hero_a1.time.sleep"):

        run_eval(
            sample_file=sample_file,
            output_path=tmp_path / "out2.json",
            cache_dir=cache_dir,
        )

    assert first_call_count == 1
    assert mock_esearch2.call_count == 0  # caché sirve la respuesta


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 6 — A1 mockeado E2E (sin GPU)
# ══════════════════════════════════════════════════════════════════════════════

def test_run_eval_e2e_minimal(tmp_path):
    """E2E mínimo: 1 sample, A1 y Entrez mockeados, verifica estructura de salida."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "smoke_test.json"
    cache_dir = tmp_path / "cache"

    mock_handle, raw_result = _make_mock_esearch_result(
        ["111", "222", "333"], count="50"
    )

    with patch("agents.agent1_protocol.Agent1Protocol", return_value=_make_mock_a1()), \
         patch("experiments.eval_hero_a1.Entrez.esearch", return_value=mock_handle), \
         patch("experiments.eval_hero_a1.Entrez.read", return_value=raw_result), \
         patch("experiments.eval_hero_a1.time.sleep"):

        result = run_eval(sample_file=sample_file, output_path=output_path, cache_dir=cache_dir)

    # JSON output existe y parsea
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert "metadata" in loaded
    assert "per_sample" in loaded
    assert "aggregates" in loaded

    # metadata: tag, K, sort
    meta = loaded["metadata"]
    assert meta["tag"] == "pre-T17.A1-v2"
    assert meta["K"] == 1000
    assert meta["sort"] == "relevance"

    # per_sample: 1 entry con campos requeridos
    assert len(loaded["per_sample"]) == 1
    sample_rec = loaded["per_sample"][0]
    assert "recall_at_1000" in sample_rec
    assert sample_rec["esearch_count"] == 50
    # a1_query_hash: 16 caracteres hex
    assert isinstance(sample_rec["a1_query_hash"], str)
    assert len(sample_rec["a1_query_hash"]) == 16

    # aggregates.completeness.values == [None] (anotación manual pendiente)
    assert loaded["aggregates"]["completeness"]["values"] == [None]

    # git_commit presente y no vacío — si no hay git, degrada a string vacío o "unknown"
    assert "git_commit" in meta
    assert isinstance(meta["git_commit"], str)
    # No crashea aunque sea string vacío (en CI sin git también es válido)


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 7 — Manejo de errores
# ══════════════════════════════════════════════════════════════════════════════

def test_a1_failure_annotates_error_continues(tmp_path):
    """A1 lanza excepción en el primer sample; el script anota error y continúa."""
    second_sample = {
        "id": "leads_002",
        "selected_index": 1,
        "origin": {"pmid": "99999999", "title": "Second study"},
        "pico": {"P": "p", "I": "i", "C": "c", "O": "o"},
        "publication_pmids": ["444"],
        "publication_pmids_count": 1,
    }
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE, second_sample])
    output_path = tmp_path / "out_error.json"
    cache_dir = tmp_path / "cache"

    # A1 falla en el primer sample, tiene éxito en el segundo
    failing_a1 = MagicMock()
    failing_a1.generate_protocol.side_effect = [
        RuntimeError("LLM failure"),  # primer sample
        FIXED_PROTOCOL,              # segundo sample
    ]
    failing_a1.generate_pubmed_query.return_value = FIXED_QUERY

    mock_handle, raw_result = _make_mock_esearch_result(["444"], count="10")

    with patch("agents.agent1_protocol.Agent1Protocol", return_value=failing_a1), \
         patch("experiments.eval_hero_a1.Entrez.esearch", return_value=mock_handle), \
         patch("experiments.eval_hero_a1.Entrez.read", return_value=raw_result), \
         patch("experiments.eval_hero_a1.time.sleep"):

        result = run_eval(sample_file=sample_file, output_path=output_path, cache_dir=cache_dir)

    assert len(result["per_sample"]) == 2

    # Primer sample: error anotado, a1_protocol null, a1_query null
    first = result["per_sample"][0]
    assert first["error"] is not None
    assert "A1 failure" in first["error"]
    assert first["a1_protocol"] is None
    assert first["a1_query"] is None

    # Segundo sample: procesado con éxito
    second = result["per_sample"][1]
    assert second["error"] is None
    assert second["recall_at_1000"] is not None


def test_entrez_retry_on_first_failure(tmp_path):
    """Entrez falla en el primer intento, tiene éxito en el reintento → call_count==2."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "out_retry.json"
    cache_dir = tmp_path / "cache"

    mock_handle, raw_result = _make_mock_esearch_result(["111"], count="5")

    # Primera llamada falla, segunda tiene éxito
    mock_esearch = MagicMock(side_effect=[
        ConnectionError("NCBI timeout"),  # intento 0 falla
        mock_handle,                       # intento 1 éxito
    ])

    with patch("agents.agent1_protocol.Agent1Protocol", return_value=_make_mock_a1()), \
         patch("experiments.eval_hero_a1.Entrez.esearch", mock_esearch), \
         patch("experiments.eval_hero_a1.Entrez.read", return_value=raw_result), \
         patch("experiments.eval_hero_a1.time.sleep"):

        result = run_eval(sample_file=sample_file, output_path=output_path, cache_dir=cache_dir)

    assert mock_esearch.call_count == 2
    # Sample procesado con éxito a pesar del fallo inicial
    assert result["per_sample"][0]["error"] is None
    assert result["per_sample"][0]["recall_at_1000"] is not None


def test_run_eval_fails_if_output_exists(tmp_path):
    """output_path ya existe → FileExistsError."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "already_exists.json"
    output_path.write_text("{}")  # ya existe

    with pytest.raises(FileExistsError):
        run_eval(
            sample_file=sample_file,
            output_path=output_path,
            cache_dir=tmp_path / "cache",
        )


def test_run_eval_fails_if_sample_missing(tmp_path):
    """sample_file no existe → FileNotFoundError."""
    sample_file = tmp_path / "nonexistent.json"
    output_path = tmp_path / "output.json"

    with pytest.raises(FileNotFoundError):
        run_eval(
            sample_file=sample_file,
            output_path=output_path,
            cache_dir=tmp_path / "cache",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 8 — Reproducibilidad metadata
# ══════════════════════════════════════════════════════════════════════════════

def test_utc_now_iso_format():
    """_utc_now_iso devuelve string UTC ISO 8601 terminado en Z o +00:00."""
    ts = _utc_now_iso()
    assert isinstance(ts, str)
    assert ts.endswith("Z") or ts.endswith("+00:00")
    # Formato básico: contiene T separando fecha y hora
    assert "T" in ts


def test_metadata_entrez_email_always_set(tmp_path):
    """entrez_email_set siempre True porque hay email por defecto."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "out_meta.json"
    cache_dir = tmp_path / "cache"

    mock_handle, raw_result = _make_mock_esearch_result(["111"], count="2")

    # Sin NCBI_API_KEY en el entorno
    env_without_key = {k: v for k, v in os.environ.items() if k not in ("NCBI_API_KEY",)}
    with patch("agents.agent1_protocol.Agent1Protocol", return_value=_make_mock_a1()), \
         patch("experiments.eval_hero_a1.Entrez.esearch", return_value=mock_handle), \
         patch("experiments.eval_hero_a1.Entrez.read", return_value=raw_result), \
         patch("experiments.eval_hero_a1.time.sleep"), \
         patch.dict(os.environ, env_without_key, clear=True):

        result = run_eval(sample_file=sample_file, output_path=output_path, cache_dir=cache_dir)

    meta = result["metadata"]
    assert meta["entrez_email_set"] is True
    assert meta["ncbi_api_key_set"] is False


def test_metadata_ncbi_api_key_set_when_env_present(tmp_path):
    """ncbi_api_key_set=True cuando NCBI_API_KEY está en el entorno."""
    sample_file = tmp_path / "samples.json"
    _write_sample_file(sample_file, [MINIMAL_SAMPLE])
    output_path = tmp_path / "out_meta2.json"
    cache_dir = tmp_path / "cache"

    mock_handle, raw_result = _make_mock_esearch_result(["111"], count="2")

    with patch("agents.agent1_protocol.Agent1Protocol", return_value=_make_mock_a1()), \
         patch("experiments.eval_hero_a1.Entrez.esearch", return_value=mock_handle), \
         patch("experiments.eval_hero_a1.Entrez.read", return_value=raw_result), \
         patch("experiments.eval_hero_a1.time.sleep"), \
         patch.dict(os.environ, {"NCBI_API_KEY": "fake_key_for_test"}):

        result = run_eval(sample_file=sample_file, output_path=output_path, cache_dir=cache_dir)

    assert result["metadata"]["ncbi_api_key_set"] is True
