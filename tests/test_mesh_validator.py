"""Unit tests for ``agents/mesh_validator.py`` (T17.A1.1 / Fase E.2).

Locked by pre-registration ``docs/iterations/T17.A1.1_hypothesis.md``
(tag ``pre-T17.A1.1``). All tests use a small fixture catalog at
``tests/fixtures/mesh_catalog_test.json``. No network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from agents import mesh_validator

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mesh_catalog_test.json"


@pytest.fixture
def fixture_catalog() -> set[str]:
    """Lowercase descriptor set from the test fixture."""
    return mesh_validator.load_catalog(FIXTURE_PATH)


@pytest.fixture(autouse=True)
def _reset_module_cache():
    """Reset the module-level catalog cache between tests."""
    mesh_validator._CATALOG_CACHE.update(
        {"path": None, "mtime": None, "set": None, "sorted_list": None}
    )
    yield
    mesh_validator._CATALOG_CACHE.update(
        {"path": None, "mtime": None, "set": None, "sorted_list": None}
    )


# ---------------------------------------------------------------------------
# load_catalog
# ---------------------------------------------------------------------------


def test_load_catalog_returns_nonempty_set():
    """Fixture catalog loads and contains the expected descriptors."""
    cat = mesh_validator.load_catalog(FIXTURE_PATH)
    assert isinstance(cat, set)
    assert len(cat) == 20
    # All values are lowercased + stripped.
    assert all(d == d.lower().strip() for d in cat)
    # Representative case-folded membership checks.
    assert "cognitive behavioral therapy" in cat
    assert "dental implants" in cat
    assert "depression" in cat
    assert "schizophrenia" in cat


# ---------------------------------------------------------------------------
# validate_query — behavior matrix
# ---------------------------------------------------------------------------


def test_validate_query_canonical_unchanged(fixture_catalog):
    """A canonical descriptor must keep its ``[MeSH]`` tag."""
    q = '"Cognitive Behavioral Therapy"[MeSH]'
    out, n, nm = mesh_validator.validate_query(q, fixture_catalog)
    assert out == q
    assert n == 0
    assert nm == []


def test_validate_query_non_canonical_downgraded(fixture_catalog):
    """A non-canonical phrase gets downgraded to ``[Title/Abstract]``."""
    q = '"Different types of dental implants"[MeSH]'
    out, n, nm = mesh_validator.validate_query(q, fixture_catalog)
    assert out == '"Different types of dental implants"[Title/Abstract]'
    assert n == 1
    assert len(nm) == 1
    assert nm[0]["phrase"] == "Different types of dental implants"
    # Nearest canonical may or may not hit the 0.7 cutoff. If it does, it
    # should be "dental implants". If not, None is acceptable per spec.
    assert nm[0]["nearest_canonical"] in {"dental implants", None}


def test_validate_query_no_mesh_tags_unchanged(fixture_catalog):
    """Queries with only ``[Title/Abstract]`` tags pass through bit-identical."""
    q = '("Depression"[Title/Abstract] OR "Anxiety"[Title/Abstract])'
    out, n, nm = mesh_validator.validate_query(q, fixture_catalog)
    assert out == q
    assert n == 0
    assert nm == []


def test_validate_query_mixed(fixture_catalog):
    """Mixed canonical + non-canonical: only the non-canonical is downgraded."""
    q = (
        '("Cognitive Behavioral Therapy"[MeSH] OR '
        '"Different types of dental implants"[MeSH]) '
        'AND "Depression"[MeSH]'
    )
    out, n, nm = mesh_validator.validate_query(q, fixture_catalog)
    # Canonical terms remain [MeSH]; only the non-canonical one is downgraded.
    assert '"Cognitive Behavioral Therapy"[MeSH]' in out
    assert '"Depression"[MeSH]' in out
    assert '"Different types of dental implants"[Title/Abstract]' in out
    assert '"Different types of dental implants"[MeSH]' not in out
    assert n == 1
    assert len(nm) == 1


def test_validate_query_empty(fixture_catalog):
    """Empty string returns (\"\", 0, [])."""
    out, n, nm = mesh_validator.validate_query("", fixture_catalog)
    assert out == ""
    assert n == 0
    assert nm == []


def test_validate_query_majr_untouched(fixture_catalog):
    """``[MAJR]`` is not ``[MeSH]`` and must remain untouched."""
    q = '"Different types of dental implants"[MAJR]'
    out, n, nm = mesh_validator.validate_query(q, fixture_catalog)
    assert out == q
    assert n == 0
    assert nm == []


def test_validate_query_phrase_with_parens(fixture_catalog):
    """Phrases with regex-special chars are correctly escaped."""
    q = '"Type 2 (special) study (variant)"[MeSH]'
    out, n, nm = mesh_validator.validate_query(q, fixture_catalog)
    assert out == '"Type 2 (special) study (variant)"[Title/Abstract]'
    assert n == 1
    assert len(nm) == 1
    assert nm[0]["phrase"] == "Type 2 (special) study (variant)"


def test_validate_query_case_insensitive_match(fixture_catalog):
    """Case-folded lookup: uppercase canonical term is still a hit."""
    q = '"COGNITIVE BEHAVIORAL THERAPY"[MeSH]'
    out, n, nm = mesh_validator.validate_query(q, fixture_catalog)
    assert out == q
    assert n == 0
    assert nm == []


# ---------------------------------------------------------------------------
# download_catalog — mocked HTTP
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, text: str = "") -> MagicMock:
    """Build a ``requests.Response`` MagicMock that mimics raise_for_status."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    if status_code >= 400:
        err = requests.exceptions.HTTPError(f"{status_code} error")
        err.response = resp
        resp.raise_for_status.side_effect = err
    else:
        resp.raise_for_status.return_value = None
    return resp


_ASCII_3RECORDS = """\
*NEWRECORD
MH = Cognitive Behavioral Therapy
UI = D015928

*NEWRECORD
MH = Depression
UI = D003863

*NEWRECORD
MH = Dental Implants
UI = D015921
"""


def test_download_catalog_mocked(tmp_path):
    """Primary URL returns 3 ASCII records → catalog persisted correctly."""
    target = tmp_path / "mesh_catalog.json"

    def _fake_get(url, timeout=60):  # noqa: ARG001
        return _mock_response(200, _ASCII_3RECORDS)

    with patch("agents.mesh_validator.requests.get", side_effect=_fake_get):
        mesh_validator.download_catalog(target)

    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["count"] == 3
    assert set(data["descriptors"]) == {
        "Cognitive Behavioral Therapy",
        "Depression",
        "Dental Implants",
    }
    assert set(data["descriptors_lowercase"]) == {
        "cognitive behavioral therapy",
        "depression",
        "dental implants",
    }
    assert data["source_file"] == "d2025.bin"
    assert "generated_at_utc" in data


def test_download_catalog_primary_fail_fallback_used(tmp_path):
    """Primary 404 → fallback 200 succeeds; source_file reflects fallback."""
    target = tmp_path / "mesh_catalog.json"

    call_log: list[str] = []

    def _fake_get(url, timeout=60):  # noqa: ARG001
        call_log.append(url)
        if url.endswith("d2025.bin"):
            return _mock_response(404)
        if url.endswith("d2024.bin"):
            return _mock_response(200, _ASCII_3RECORDS)
        return _mock_response(500)

    with patch("agents.mesh_validator.requests.get", side_effect=_fake_get):
        mesh_validator.download_catalog(target)

    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["source_file"] == "d2024.bin"
    assert data["count"] == 3
    # Both URLs were tried.
    assert any("d2025.bin" in u for u in call_log)
    assert any("d2024.bin" in u for u in call_log)


def test_download_catalog_all_fail_raises_and_writes_blocked(
    tmp_path, monkeypatch
):
    """All URLs fail → BLOCKED_mesh.md written and RuntimeError raised."""
    target = tmp_path / "mesh_catalog.json"
    # Redirect the hardcoded BLOCKED path to tmp_path so we don't touch the
    # real outputs/ dir.
    monkeypatch.chdir(tmp_path)

    def _fake_get(url, timeout=60):  # noqa: ARG001
        return _mock_response(500)

    # Also ensure the PyPI fallback returns None (no installed pkg).
    with patch("agents.mesh_validator.requests.get", side_effect=_fake_get), \
         patch(
             "agents.mesh_validator._try_pypi_fallback", return_value=None
         ):
        with pytest.raises(RuntimeError):
            mesh_validator.download_catalog(target)

    blocked = tmp_path / "outputs" / "eval" / "T17.A1" / "BLOCKED_mesh.md"
    assert blocked.exists()
    content = blocked.read_text(encoding="utf-8")
    assert "BLOCKED" in content
    assert "d2025.bin" in content
    assert "d2024.bin" in content
    # Catalog file must not have been written on failure.
    assert not target.exists()
