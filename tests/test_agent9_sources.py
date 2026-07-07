# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""A9 explainability — regression for BUG4: propagate A2 search sources.

The A9 report section §6 "Procedencia de Datos por Paper Incluido" used to
show "?" for the search source because `a2_sources` was never carried through
the pipeline_state. This module validates:

  1. Agent2Search.search() now emits a `search_sources` list in its stats.
  2. Agent9Explainer.run() prints the database list in explainability.md.
  3. When the list is absent, A9 falls back to "(no disponible)" and not "?".

CPU-only, no GPU/LLM/network. Reuses the full_pipeline_data fixture from
conftest.py and tweaks it per scenario.
"""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _run_a9(pipeline_data, tmp_path):
    """Helper: run Agent9Explainer against the fixture pipeline and return
    the generated explainability.md text."""
    from agents.agent9_explainer import Agent9Explainer

    out_dir = tmp_path / "a9_out"
    out_dir.mkdir()
    agent = Agent9Explainer()
    res = agent.run(pipeline_data, run_id="test_run_0001", output_dir=str(out_dir))
    with open(res["report_path"], encoding="utf-8") as f:
        return f.read()


def test_a9_reads_a2_sources_and_lists_them(full_pipeline_data, tmp_path):
    """A9 must enumerate A2 databases when `a2_sources` is present."""
    full_pipeline_data["a2_sources"] = [
        "pubmed", "europepmc", "semantic_scholar", "google_scholar"
    ]
    report = _run_a9(full_pipeline_data, tmp_path)

    assert "## 6. Procedencia de Datos por Paper Incluido" in report
    assert "Bases de datos consultadas por A2" in report
    for db in ("pubmed", "europepmc", "semantic_scholar", "google_scholar"):
        assert db in report


def test_a9_falls_back_to_search_stats_when_a2_sources_missing(
    full_pipeline_data, tmp_path
):
    """If the orchestrator didn't inject `a2_sources`, A9 derives it from
    `search_stats` counts (any DB with >0 results)."""
    # Remove a2_sources but keep per-DB counts in search_stats
    full_pipeline_data.pop("a2_sources", None)
    full_pipeline_data["search_stats"] = {
        "pubmed": 5,
        "europepmc": 2,
        "semantic_scholar": 0,
        "google_scholar": 0,
    }
    report = _run_a9(full_pipeline_data, tmp_path)
    assert "pubmed" in report
    assert "europepmc" in report


def test_a9_shows_no_disponible_when_nothing_available(full_pipeline_data, tmp_path):
    """With both `a2_sources` and per-DB counts missing, fallback string is
    '(no disponible)' — NOT '?' (BUG4 symptom)."""
    full_pipeline_data.pop("a2_sources", None)
    full_pipeline_data["search_stats"] = {}
    # Also strip per-paper `source` so the old "?" branch would fire.
    ft = full_pipeline_data.get("fulltext_retrieval") or {}
    for bucket in ("include", "exclude", "papers"):
        for p in ft.get(bucket, []):
            p.pop("source", None)
            p.pop("sources", None)

    report = _run_a9(full_pipeline_data, tmp_path)
    assert "(no disponible)" in report
    # The per-paper "Fuente de busqueda: ?" line should no longer appear.
    assert "Fuente de busqueda: ?" not in report


def test_agent2_search_emits_search_sources_in_stats():
    """Agent2Search.search() must populate stats['search_sources'] based on
    which DBs actually returned papers. Tested via direct dict manipulation
    to avoid network calls."""
    from agents.agent2_search import Agent2Search
    from unittest.mock import patch

    with patch.object(Agent2Search, "__init__", lambda self: None):
        agent = Agent2Search()
    agent.dedup_model = None  # avoid heavy embedding model load
    agent.db_pubmed = True
    agent.db_europepmc = True
    agent.db_semantic_scholar = False
    agent.db_google_scholar = False
    agent.max_per_db = 100
    agent.dedup_threshold = 0.95
    agent.rate_limit = 0.0
    agent.google_scholar_rate_limit = 0.0

    # Stub each DB call to return a fake paper list
    def fake_pm(*_a, **_kw):
        return [{"pmid": "1", "doi": "", "title": "A", "abstract": "x", "authors": [], "journal": "", "year": "2020", "sources": ["pubmed"]}]

    def fake_epmc(*_a, **_kw):
        return [{"pmid": "2", "doi": "", "title": "B", "abstract": "y", "authors": [], "journal": "", "year": "2021", "sources": ["europepmc"]}]

    def fake_dedup(papers, threshold=0.95):
        return papers

    agent._search_pubmed = fake_pm
    agent._search_europepmc = fake_epmc
    agent._search_semantic_scholar = lambda *a, **k: []
    agent._search_google_scholar = lambda *a, **k: []
    agent._build_search_terms = lambda p: "dummy"
    agent.deduplicate = fake_dedup

    protocol = {"population": "adults", "intervention": "CBT", "outcome": "anxiety",
                "research_question": "dummy"}
    result = agent.search(protocol, pubmed_query="dummy", max_results=10)

    assert "search_sources" in result["stats"]
    assert "pubmed" in result["stats"]["search_sources"]
    assert "europepmc" in result["stats"]["search_sources"]
    # Disabled DBs must not appear
    assert "semantic_scholar" not in result["stats"]["search_sources"]
    assert "google_scholar" not in result["stats"]["search_sources"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
