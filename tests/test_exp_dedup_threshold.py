"""Tests for scripts/exp_dedup_threshold.py.

Covers:
  - run_threshold_sweep (threshold grid iteration)
  - write_csv / write_json / plot_sweep
  - load_or_fetch_raw_pool (cache hit path)
  - main() integration (with cache + mocked deduplicate)

No GPU, no network. Agent2Search is mocked to avoid SentenceTransformer
downloads and HTTP calls.
"""

from __future__ import annotations

import csv
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

# Ensure repo root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scripts.exp_dedup_threshold as edt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_papers(n: int) -> list[dict]:
    """Create n distinct synthetic paper dicts (no duplicate PMIDs/DOIs)."""
    return [
        {
            "pmid": str(10000 + i),
            "doi": f"10.9999/test.{i:04d}",
            "title": f"Mindfulness study {i} on anxiety outcomes",
            "abstract": f"Abstract text for paper {i}. Different content here.",
            "source": "pubmed" if i % 2 == 0 else "europepmc",
        }
        for i in range(n)
    ]


def _write_cache(tmp_path: Path, papers: list[dict]) -> Path:
    """Write a synthetic raw-pool cache file and return its path."""
    cache_path = tmp_path / "a2_raw_predudup.json"
    payload = {
        "source_run": "outputs/run_test",
        "user_query": "mindfulness anxiety test",
        "pubmed_query": "mindfulness anxiety test",
        "search_terms": "mindfulness anxiety",
        "min_per_db": 5,
        "databases": ["pubmed", "europepmc"],
        "n_raw": len(papers),
        "papers": papers,
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return cache_path


def _make_mock_agent2(dedup_side_effect=None) -> MagicMock:
    """Return a MagicMock that mimics an Agent2Search instance.

    The SentenceTransformer is never loaded because we never call the real
    __init__.
    """
    mock = MagicMock()
    mock.deduplicate.side_effect = dedup_side_effect
    return mock


# ---------------------------------------------------------------------------
# 1. test_sweep_iterates_all_thresholds
# ---------------------------------------------------------------------------

def test_sweep_iterates_all_thresholds(monkeypatch, tmp_path):
    """run_threshold_sweep must call deduplicate exactly 6 times, once per
    threshold in THRESHOLD_GRID = {0.85, 0.88, 0.90, 0.92, 0.95, 0.98}."""
    raw_papers = _make_raw_papers(10)

    # Track which thresholds were passed to deduplicate.
    called_thresholds: list[float] = []

    def fake_dedup(papers_copy, threshold):
        called_thresholds.append(threshold)
        # Return the full list (no dedup) to keep the mock simple.
        return list(papers_copy)

    mock_agent = _make_mock_agent2(dedup_side_effect=fake_dedup)

    # Patch Agent2Search so the constructor never loads SentenceTransformer.
    with patch("agents.agent2_search.Agent2Search") as MockClass:
        MockClass.return_value = mock_agent
        rows = edt.run_threshold_sweep(raw_papers)

    assert len(called_thresholds) == 6, (
        f"Expected 6 deduplicate calls, got {len(called_thresholds)}"
    )
    assert set(called_thresholds) == set(edt.THRESHOLD_GRID), (
        f"Threshold mismatch: {set(called_thresholds)} != {set(edt.THRESHOLD_GRID)}"
    )
    assert len(rows) == 6


# ---------------------------------------------------------------------------
# 2. test_outputs_generated
# ---------------------------------------------------------------------------

def test_outputs_generated(monkeypatch, tmp_path):
    """CSV, JSON, and PNG must exist after main() completes.

    The raw pool cache is injected into tmp_path so no HTTP calls are made.
    Agent2Search.deduplicate is mocked to avoid SentenceTransformer.
    """
    raw_papers = _make_raw_papers(8)
    cache_path = _write_cache(tmp_path, raw_papers)

    out_dir = tmp_path / "exp_out"
    csv_path = out_dir / "dedup_threshold.csv"
    json_path = out_dir / "dedup_threshold.json"
    png_path = out_dir / "dedup_threshold.png"

    # Redirect module-level path constants.
    monkeypatch.setattr(edt, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(edt, "RAW_CACHE_PATH", cache_path)
    monkeypatch.setattr(edt, "CSV_PATH", csv_path)
    monkeypatch.setattr(edt, "JSON_PATH", json_path)
    monkeypatch.setattr(edt, "PNG_PATH", png_path)

    def fake_dedup(papers_copy, threshold):
        # Simulate slight dedup: higher threshold -> keep more papers.
        cutoff = int(len(papers_copy) * threshold)
        return papers_copy[:max(1, cutoff)]

    mock_agent = _make_mock_agent2(dedup_side_effect=fake_dedup)

    with patch("agents.agent2_search.Agent2Search") as MockClass:
        MockClass.return_value = mock_agent
        edt.main()

    assert csv_path.exists(), "CSV not created"
    assert json_path.exists(), "JSON not created"
    assert png_path.exists(), "PNG not created"


# ---------------------------------------------------------------------------
# 3. test_csv_columns
# ---------------------------------------------------------------------------

def test_csv_columns(tmp_path):
    """write_csv must produce a file with the v2 header (6 columns, strict order).

    Expected columns (v2):
        threshold, unique, collapsed_pairs,
        unique_natural, collapsed_natural, synthetic_pairs_collapsed

    synthetic_pairs_collapsed is serialized as a pipe-joined string of
    pair_ids (e.g. "1|3") or an empty string when no pairs collapsed.
    """
    rows = [
        {
            "threshold": 0.85,
            "unique": 9,
            "collapsed_pairs": 1,
            "unique_natural": 7,
            "collapsed_natural": 1,
            "synthetic_pairs_collapsed": [1],
        },
        {
            "threshold": 0.90,
            "unique": 8,
            "collapsed_pairs": 2,
            "unique_natural": 6,
            "collapsed_natural": 2,
            "synthetic_pairs_collapsed": [1, 3],
        },
        {
            "threshold": 0.95,
            "unique": 7,
            "collapsed_pairs": 3,
            "unique_natural": 5,
            "collapsed_natural": 3,
            "synthetic_pairs_collapsed": [],
        },
    ]
    csv_path = tmp_path / "dedup_threshold.csv"
    edt.write_csv(rows, csv_path)

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        data_rows = list(reader)

    assert header is not None
    assert list(header) == [
        "threshold",
        "unique",
        "collapsed_pairs",
        "unique_natural",
        "collapsed_natural",
        "synthetic_pairs_collapsed",
    ], f"Unexpected columns: {header}"
    assert len(data_rows) == 3

    # Verify synthetic_pairs_collapsed serialization.
    assert data_rows[0]["synthetic_pairs_collapsed"] == "1", (
        f"Row 0: expected '1', got '{data_rows[0]['synthetic_pairs_collapsed']}'"
    )
    assert data_rows[1]["synthetic_pairs_collapsed"] == "1|3", (
        f"Row 1: expected '1|3', got '{data_rows[1]['synthetic_pairs_collapsed']}'"
    )
    assert data_rows[2]["synthetic_pairs_collapsed"] == "", (
        f"Row 2: expected empty string, got '{data_rows[2]['synthetic_pairs_collapsed']}'"
    )


# ---------------------------------------------------------------------------
# 4. test_json_shape
# ---------------------------------------------------------------------------

def test_json_shape(tmp_path):
    """write_json must produce JSON with 'raw_total' (int) and 'sweep' (6 entries)."""
    rows = [
        {"threshold": t, "unique": 10 - i, "collapsed_pairs": i}
        for i, t in enumerate(edt.THRESHOLD_GRID)
    ]
    raw_total = 10
    json_path = tmp_path / "dedup_threshold.json"
    edt.write_json(raw_total, rows, json_path)

    with json_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    assert "raw_total" in payload, "Missing 'raw_total' key"
    assert isinstance(payload["raw_total"], int), "'raw_total' must be int"
    assert payload["raw_total"] == raw_total

    assert "sweep" in payload, "Missing 'sweep' key"
    assert isinstance(payload["sweep"], list), "'sweep' must be a list"
    assert len(payload["sweep"]) == len(edt.THRESHOLD_GRID), (
        f"Expected {len(edt.THRESHOLD_GRID)} sweep entries, "
        f"got {len(payload['sweep'])}"
    )

    for entry in payload["sweep"]:
        assert "threshold" in entry
        assert "unique" in entry
        assert "collapsed_pairs" in entry


# ---------------------------------------------------------------------------
# 5. test_unique_monotonic_with_synthetic
# ---------------------------------------------------------------------------

def test_unique_monotonic_with_synthetic(monkeypatch, tmp_path):
    """When deduplicate returns more papers as threshold rises, the 'unique'
    column in sweep rows must be non-decreasing with threshold."""
    raw_papers = _make_raw_papers(20)

    # Simulate: lower threshold collapses more; higher threshold collapses less.
    # Map each threshold to a fixed return size (non-decreasing).
    threshold_to_n = {
        0.85: 12,
        0.88: 13,
        0.90: 14,
        0.92: 15,
        0.95: 17,
        0.98: 19,
    }

    def fake_dedup(papers_copy, threshold):
        n = threshold_to_n[round(threshold, 2)]
        return papers_copy[:n]

    mock_agent = _make_mock_agent2(dedup_side_effect=fake_dedup)

    with patch("agents.agent2_search.Agent2Search") as MockClass:
        MockClass.return_value = mock_agent
        rows = edt.run_threshold_sweep(raw_papers)

    uniques = [r["unique"] for r in rows]
    for i in range(1, len(uniques)):
        assert uniques[i] >= uniques[i - 1], (
            f"unique not non-decreasing at position {i}: "
            f"{uniques[i]} < {uniques[i-1]}"
        )


# ---------------------------------------------------------------------------
# 6. test_no_network_no_gpu
# ---------------------------------------------------------------------------

def test_no_network_no_gpu(monkeypatch, tmp_path):
    """When the raw cache exists, _search_pubmed and _search_europepmc must
    never be called (no HTTP, no GPU).

    Strategy:
      - Write a valid cache file in tmp_path.
      - Redirect RAW_CACHE_PATH to that file.
      - Patch Agent2Search so any instantiation returns a mock.
      - Verify that the network methods are never called.
    """
    raw_papers = _make_raw_papers(6)
    cache_path = _write_cache(tmp_path, raw_papers)

    out_dir = tmp_path / "exp_no_net"
    csv_path = out_dir / "dedup_threshold.csv"
    json_path = out_dir / "dedup_threshold.json"
    png_path = out_dir / "dedup_threshold.png"

    monkeypatch.setattr(edt, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(edt, "RAW_CACHE_PATH", cache_path)
    monkeypatch.setattr(edt, "CSV_PATH", csv_path)
    monkeypatch.setattr(edt, "JSON_PATH", json_path)
    monkeypatch.setattr(edt, "PNG_PATH", png_path)

    # The mock agent: deduplicate returns a stable subset; search methods track calls.
    def fake_dedup(papers_copy, threshold):
        return papers_copy

    mock_agent = _make_mock_agent2(dedup_side_effect=fake_dedup)
    # Explicitly track network methods.
    mock_agent._search_pubmed = MagicMock(return_value=[])
    mock_agent._search_europepmc = MagicMock(return_value=[])

    with patch("agents.agent2_search.Agent2Search") as MockClass:
        MockClass.return_value = mock_agent
        edt.main()

    # Cache was present, so network methods must not have been called.
    mock_agent._search_pubmed.assert_not_called()
    mock_agent._search_europepmc.assert_not_called()


# ---------------------------------------------------------------------------
# Edge-case: empty raw pool aborts sweep
# ---------------------------------------------------------------------------

def test_empty_raw_pool_returns_no_rows():
    """run_threshold_sweep on an empty pool must return an empty list."""
    with patch("agents.agent2_search.Agent2Search") as MockClass:
        MockClass.return_value = MagicMock()
        rows = edt.run_threshold_sweep([])

    assert rows == [], f"Expected [], got {rows}"


# ---------------------------------------------------------------------------
# Edge-case: THRESHOLD_GRID has exactly 6 entries
# ---------------------------------------------------------------------------

def test_threshold_grid_has_six_entries():
    """The threshold grid must have exactly 6 entries as documented."""
    assert len(edt.THRESHOLD_GRID) == 6, (
        f"Expected 6 thresholds, got {len(edt.THRESHOLD_GRID)}"
    )
    assert sorted(edt.THRESHOLD_GRID) == edt.THRESHOLD_GRID, (
        "THRESHOLD_GRID must be in ascending order"
    )
