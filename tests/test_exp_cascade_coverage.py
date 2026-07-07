# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/exp_cascade_coverage.py.

Covers the public functions exposed by the cascade coverage analysis script:
  - normalize_source
  - compute_distribution
  - write_csv
  - write_json
  - plot_distribution
  - main (via monkeypatching module-level path constants)

No GPU, no network, no real checkpoint files required.
"""

from __future__ import annotations

import csv
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure repo root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scripts.exp_cascade_coverage as ecc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_papers(*sources) -> list[dict]:
    """Build minimal paper dicts with the given text_source values."""
    papers = []
    for i, src in enumerate(sources):
        p: dict = {"pmid": str(i), "title": f"Paper {i}"}
        if src is not None:
            p["text_source"] = src
        papers.append(p)
    return papers


# ---------------------------------------------------------------------------
# 1. test_count_text_sources_basic
# ---------------------------------------------------------------------------

def test_count_text_sources_basic():
    """Verify that compute_distribution counts text_source values correctly."""
    papers = _make_papers("pmc", "pmc", "unpaywall_pdf", "europepmc", "pmc")
    total, rows = ecc.compute_distribution(papers)

    assert total == 5

    row_map = {r["source"]: r["n"] for r in rows}
    assert row_map["pmc"] == 3
    assert row_map["unpaywall_pdf"] == 1
    assert row_map["europepmc"] == 1


# ---------------------------------------------------------------------------
# 2. test_none_category_for_null_and_empty
# ---------------------------------------------------------------------------

def test_none_category_for_null_and_empty():
    """Papers with text_source None, '', or absent must go to 'none' bucket."""
    papers = [
        {"pmid": "1", "text_source": None},
        {"pmid": "2", "text_source": ""},
        {"pmid": "3", "text_source": "   "},     # whitespace-only -> strip -> ""
        {"pmid": "4"},                            # key absent
        {"pmid": "5", "text_source": "pmc"},     # control: should NOT go to none
    ]
    total, rows = ecc.compute_distribution(papers)

    assert total == 5

    row_map = {r["source"]: r["n"] for r in rows}
    assert row_map.get(ecc.NONE_LABEL, 0) == 4, (
        f"Expected 4 'none' entries, got {row_map}"
    )
    assert row_map.get("pmc", 0) == 1


# ---------------------------------------------------------------------------
# 3. test_percentages_sum_to_100
# ---------------------------------------------------------------------------

def test_percentages_sum_to_100():
    """The sum of pct values across all rows must be within [99.9, 100.1]."""
    papers = _make_papers(
        "pmc", "pmc", "pmc",
        "unpaywall_pdf", "unpaywall_pdf",
        "europepmc",
        "abstract_only",
        None,
        "",
    )
    _, rows = ecc.compute_distribution(papers)

    pct_sum = sum(r["pct"] for r in rows)
    assert 99.9 <= pct_sum <= 100.1, f"pct sum out of range: {pct_sum}"


# ---------------------------------------------------------------------------
# 4. test_outputs_generated
# ---------------------------------------------------------------------------

def test_outputs_generated(tmp_path, monkeypatch):
    """CSV, JSON, and PNG files must be created when main() is called.

    The module-level path constants are redirected to tmp_path, and the real
    checkpoint is replaced with a tiny in-memory JSON written to tmp_path.
    """
    # Build a synthetic checkpoint and write it to tmp_path.
    checkpoint = {
        "include": [
            {"pmid": "1", "text_source": "pmc"},
            {"pmid": "2", "text_source": "unpaywall_pdf"},
        ],
        "exclude": [
            {"pmid": "3", "text_source": "abstract_only"},
        ],
    }
    checkpoint_file = tmp_path / "checkpoint_a4_fulltext.json"
    checkpoint_file.write_text(json.dumps(checkpoint), encoding="utf-8")

    out_dir = tmp_path / "exp_out"

    # Monkeypatch module-level constants.
    monkeypatch.setattr(ecc, "CHECKPOINT_PATH", checkpoint_file)
    monkeypatch.setattr(ecc, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(ecc, "CSV_PATH", out_dir / "cascade_coverage.csv")
    monkeypatch.setattr(ecc, "JSON_PATH", out_dir / "cascade_coverage.json")
    monkeypatch.setattr(ecc, "PNG_PATH", out_dir / "cascade_coverage.png")

    ecc.main()

    assert (out_dir / "cascade_coverage.csv").exists(), "CSV not created"
    assert (out_dir / "cascade_coverage.json").exists(), "JSON not created"
    assert (out_dir / "cascade_coverage.png").exists(), "PNG not created"


# ---------------------------------------------------------------------------
# 5. test_csv_columns_and_ordering
# ---------------------------------------------------------------------------

def test_csv_columns_and_ordering(tmp_path):
    """CSV must have columns source,n,pct and rows ordered by n descending."""
    papers = _make_papers(
        "europepmc",
        "pmc", "pmc", "pmc",
        "unpaywall_pdf", "unpaywall_pdf",
    )
    _, rows = ecc.compute_distribution(papers)

    csv_path = tmp_path / "test_cascade.csv"
    ecc.write_csv(rows, csv_path)

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        data_rows = list(reader)

    assert header is not None
    assert list(header) == ["source", "n", "pct"], f"Unexpected columns: {header}"

    # Rows must be ordered by n descending.
    ns = [int(r["n"]) for r in data_rows]
    assert ns == sorted(ns, reverse=True), f"Rows not ordered by n desc: {ns}"


# ---------------------------------------------------------------------------
# 6. test_json_shape
# ---------------------------------------------------------------------------

def test_json_shape(tmp_path):
    """JSON output must have 'total_papers' (int) and 'sources' (list of dicts)."""
    papers = _make_papers("pmc", "pmc", "abstract_only", None)
    total, rows = ecc.compute_distribution(papers)

    json_path = tmp_path / "test_cascade.json"
    ecc.write_json(total, rows, json_path)

    with json_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    assert "total_papers" in payload, "Missing 'total_papers' key"
    assert isinstance(payload["total_papers"], int), "'total_papers' must be int"
    assert payload["total_papers"] == total

    assert "sources" in payload, "Missing 'sources' key"
    assert isinstance(payload["sources"], list), "'sources' must be a list"
    assert len(payload["sources"]) == len(rows), "sources length mismatch"

    # Each entry must have source, n, pct.
    for entry in payload["sources"]:
        assert "source" in entry
        assert "n" in entry
        assert "pct" in entry


# ---------------------------------------------------------------------------
# Edge-case: empty paper list
# ---------------------------------------------------------------------------

def test_empty_papers():
    """compute_distribution on an empty list must return total=0, rows=[]."""
    total, rows = ecc.compute_distribution([])
    assert total == 0
    assert rows == []


# ---------------------------------------------------------------------------
# Edge-case: special characters in text_source (accents, CJK)
# ---------------------------------------------------------------------------

def test_special_characters_in_text_source():
    """text_source values with tildes, accents, and CJK must be kept as-is."""
    papers = [
        {"pmid": "1", "text_source": "fuente_española"},
        {"pmid": "2", "text_source": "来源_中文"},
        {"pmid": "3", "text_source": "fuente_española"},
    ]
    total, rows = ecc.compute_distribution(papers)

    assert total == 3
    row_map = {r["source"]: r["n"] for r in rows}
    assert row_map.get("fuente_española", 0) == 2
    assert row_map.get("来源_中文", 0) == 1
