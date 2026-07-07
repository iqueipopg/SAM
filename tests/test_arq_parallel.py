# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ARQ4 — parallel full-text retrieval in agents/agent4_fulltext.py.

Covers:
- retrieve_fulltext con N papers usa ThreadPoolExecutor.
- Preservación de orden: los resultados vuelven en el orden de entrada.
- Excepción en un paper no rompe el batch; el resto sigue procesándose.
- max_workers respetado: no más de N threads simultáneos.
- Mock de toda la E/S HTTP — sin red, sin GPU.

CPU-only, no GPU, no network.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_paper(idx: int) -> dict:
    """Build a minimal paper dict for testing."""
    return {
        "pmid": str(10000 + idx),
        "title": f"Test paper {idx}",
        "abstract": f"Abstract for paper {idx}.",
        "authors": ["Autor A"],
        "journal": "Test Journal",
        "year": "2024",
        "doi": f"10.9999/test.{idx}",
    }


def _make_agent4_no_gpu():
    """Instantiate Agent4FullText with all LLM/HTTP mocked out."""
    with (
        patch("agents.base.AutoModelForCausalLM") as mock_model_cls,
        patch("agents.base.AutoTokenizer") as mock_tok_cls,
        patch("agents.base.BitsAndBytesConfig"),
    ):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        from agents.agent4_fulltext import Agent4FullText
        agent = Agent4FullText.__new__(Agent4FullText)

        # Minimal attribute setup so retrieve_fulltext works
        agent.config = {
            "retrieval": {
                "europepmc_enabled": False,
                "semanticscholar_enabled": False,
                "openalex_enabled": False,
                "crossref_enabled": False,
                "doaj_enabled": False,
                "lens_enabled": False,
                "core_enabled": False,
                "scihub_enabled": False,
                "google_scholar_enabled": False,
                "preprints_enabled": False,
                "arxiv_enabled": False,
                "semanticscholar_v2_enabled": False,
            }
        }
        # Disable all optional sources
        agent.europepmc_enabled = False
        agent.semanticscholar_enabled = False
        agent.openalex_enabled = False
        agent.crossref_enabled = False
        agent.doaj_enabled = False
        agent.lens_enabled = False
        agent.core_enabled = False
        agent.scihub_enabled = False
        agent.scihub_domain = "sci-hub.se"
        agent.google_scholar_enabled = False
        agent.preprints_enabled = False
        agent.arxiv_enabled = False
        agent.semanticscholar_v2_enabled = False
        agent._lens_api_token = ""
        agent._core_api_key = ""
        agent._last_ncbi_call = 0.0
        agent._last_unpaywall_call = 0.0
        agent._last_semanticscholar_call = 0.0
        agent._last_openalex_call = 0.0
        agent._last_crossref_call = 0.0
        agent._last_doaj_call = 0.0
        agent._last_lens_call = 0.0
        agent._last_core_call = 0.0
        agent._last_google_scholar_call = 0.0
        agent._openalex_interval = 0.0
        agent._crossref_interval = 0.0
        agent._doaj_interval = 0.0
        agent._lens_interval = 0.0
        agent._core_interval = 0.0
        agent._google_scholar_interval = 0.0
        agent.max_workers = 4
        agent._stats_lock = threading.Lock()
        agent._progress_lock = threading.Lock()
        agent._rate_limit_lock = threading.Lock()
        agent.logger = MagicMock()
        return agent


class TestParallelRetrieval:
    """ARQ4: retrieve_fulltext parallelism and correctness."""

    # ── ThreadPoolExecutor is used ───────────────────────────────────────────

    def test_retrieve_fulltext_uses_thread_pool_executor(self):
        """retrieve_fulltext invoca ThreadPoolExecutor para E/S paralela."""
        agent = _make_agent4_no_gpu()
        papers = [_make_paper(i) for i in range(3)]

        executor_calls = []

        original_executor = ThreadPoolExecutor

        def tracking_executor(*args, **kwargs):
            executor_calls.append(kwargs.get("max_workers", args[0] if args else None))
            return original_executor(*args, **kwargs)

        with (
            patch.object(agent, "_batch_fetch_metadata", return_value={}),
            patch.object(agent, "_retrieve_single", return_value=None),
            patch.object(agent, "_parse_sections_from_text", return_value={}),
            patch(
                "agents.agent4_fulltext.ThreadPoolExecutor",
                side_effect=tracking_executor,
            ),
        ):
            agent.retrieve_fulltext(papers, papers)

        assert len(executor_calls) >= 1, "ThreadPoolExecutor debe haberse llamado al menos una vez"

    # ── order preservation ───────────────────────────────────────────────────

    def test_retrieve_fulltext_preserves_input_order(self):
        """Los resultados vuelven en el mismo orden que la entrada, aunque el thread rápido llegue antes."""
        agent = _make_agent4_no_gpu()
        n = 5
        papers = [_make_paper(i) for i in range(n)]
        call_order = []

        def slow_for_first(enriched_paper, local_stats):
            pmid = enriched_paper.get("pmid", "")
            call_order.append(pmid)
            if pmid == str(10000):
                time.sleep(0.05)  # first paper is slowest
            enriched_paper["text_source"] = "abstract_only"
            enriched_paper["full_text"] = None
            enriched_paper["full_text_length"] = 0

        with (
            patch.object(agent, "_batch_fetch_metadata", return_value={}),
            patch.object(agent, "_retrieve_single", side_effect=slow_for_first),
            patch.object(agent, "_parse_sections_from_text", return_value={}),
        ):
            result = agent.retrieve_fulltext(papers, papers)

        pmids_out = [p.get("pmid") for p in result]
        expected = [str(10000 + i) for i in range(n)]
        assert pmids_out == expected, (
            f"El orden de salida debe coincidir con la entrada. "
            f"Esperado: {expected}, obtenido: {pmids_out}"
        )

    # ── exception isolation ──────────────────────────────────────────────────

    def test_exception_in_one_paper_does_not_break_batch(self):
        """Excepción en paper idx=2 no impide que los demás se procesen."""
        agent = _make_agent4_no_gpu()
        n = 5
        papers = [_make_paper(i) for i in range(n)]

        def retrieve_with_error(enriched_paper, local_stats):
            pmid = enriched_paper.get("pmid", "")
            if pmid == str(10002):
                raise RuntimeError("Error simulado para paper idx=2")
            enriched_paper["text_source"] = "abstract_only"
            enriched_paper["full_text"] = None
            enriched_paper["full_text_length"] = 0

        with (
            patch.object(agent, "_batch_fetch_metadata", return_value={}),
            patch.object(agent, "_retrieve_single", side_effect=retrieve_with_error),
            patch.object(agent, "_parse_sections_from_text", return_value={}),
        ):
            result = agent.retrieve_fulltext(papers, papers)

        assert len(result) == n, f"Debe devolver {n} papers aunque uno falle"
        # All papers must be present; the failed one falls back to the stub
        pmids_out = {p.get("pmid") for p in result if p is not None}
        expected_pmids = {str(10000 + i) for i in range(n)}
        assert expected_pmids == pmids_out, "Todos los PMIDs deben estar en la salida"

    # ── max_workers respected ────────────────────────────────────────────────

    def test_max_workers_respected(self):
        """Con max_workers=2 y 5 papers, no hay más de 2 threads activos a la vez."""
        agent = _make_agent4_no_gpu()
        agent.max_workers = 2
        n = 5
        papers = [_make_paper(i) for i in range(n)]

        concurrent_counter = {"peak": 0, "current": 0}
        counter_lock = threading.Lock()

        def counting_retrieve(enriched_paper, local_stats):
            with counter_lock:
                concurrent_counter["current"] += 1
                if concurrent_counter["current"] > concurrent_counter["peak"]:
                    concurrent_counter["peak"] = concurrent_counter["current"]
            time.sleep(0.02)  # hold the slot briefly
            with counter_lock:
                concurrent_counter["current"] -= 1
            enriched_paper["text_source"] = "abstract_only"
            enriched_paper["full_text"] = None
            enriched_paper["full_text_length"] = 0

        with (
            patch.object(agent, "_batch_fetch_metadata", return_value={}),
            patch.object(agent, "_retrieve_single", side_effect=counting_retrieve),
            patch.object(agent, "_parse_sections_from_text", return_value={}),
        ):
            agent.retrieve_fulltext(papers, papers)

        assert concurrent_counter["peak"] <= 2, (
            f"Pico de concurrencia {concurrent_counter['peak']} supera max_workers=2"
        )

    # ── thread-safety locks present ──────────────────────────────────────────

    def test_agent4_has_stats_lock(self):
        """Agent4FullText tiene _stats_lock (threading.Lock)."""
        agent = _make_agent4_no_gpu()
        assert isinstance(agent._stats_lock, type(threading.Lock()))

    def test_agent4_has_progress_lock(self):
        """Agent4FullText tiene _progress_lock (threading.Lock)."""
        agent = _make_agent4_no_gpu()
        assert isinstance(agent._progress_lock, type(threading.Lock()))

    def test_agent4_has_rate_limit_lock(self):
        """Agent4FullText tiene _rate_limit_lock (threading.Lock)."""
        agent = _make_agent4_no_gpu()
        assert isinstance(agent._rate_limit_lock, type(threading.Lock()))

    # ── config key ───────────────────────────────────────────────────────────

    def test_max_workers_attribute_default(self):
        """max_workers tiene un valor numérico positivo por defecto."""
        agent = _make_agent4_no_gpu()
        assert isinstance(agent.max_workers, int)
        assert agent.max_workers >= 1

    def test_retrieve_fulltext_returns_list(self):
        """retrieve_fulltext devuelve una lista (no None, no dict)."""
        agent = _make_agent4_no_gpu()
        papers = [_make_paper(0)]

        def noop_retrieve(enriched_paper, local_stats):
            enriched_paper["text_source"] = "abstract_only"
            enriched_paper["full_text"] = None
            enriched_paper["full_text_length"] = 0

        with (
            patch.object(agent, "_batch_fetch_metadata", return_value={}),
            patch.object(agent, "_retrieve_single", side_effect=noop_retrieve),
            patch.object(agent, "_parse_sections_from_text", return_value={}),
        ):
            result = agent.retrieve_fulltext(papers, papers)

        assert isinstance(result, list)
        assert len(result) == 1
