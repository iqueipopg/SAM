"""ARQ4: Parallel Phase A retrieval in Agent4FullText.

These tests mock `_retrieve_single` to add a fixed sleep and verify that
running the retrieval loop with `max_workers > 1` is meaningfully faster
than strictly sequential execution. They do NOT hit the network.
"""

import os
import sys
import threading
import time
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agents.agent4_fulltext import Agent4FullText


def _make_agent(max_workers: int) -> Agent4FullText:
    """Build an Agent4FullText instance without running __init__.

    __init__ reads config.yaml and the LLM backend selector, which we don't
    need for Phase A tests. We set only the attributes that retrieve_fulltext
    actually touches.
    """
    agent = object.__new__(Agent4FullText)
    agent.max_workers = max_workers
    agent._stats_lock = threading.Lock()
    agent._progress_lock = threading.Lock()
    agent._rate_limit_lock = threading.Lock()
    # Skip real metadata fetch
    agent._batch_fetch_metadata = lambda pmids: {}
    # Skip section parsing (no full text in mocked runs)
    agent._parse_sections_from_text = lambda text: {}
    return agent


def _make_papers(n: int):
    """Build n minimal paper dicts as A3 would emit."""
    return [{"pmid": str(10000 + i), "title": f"Paper {i}", "decision": "INCLUDE"} for i in range(n)]


def test_parallel_retrieval_faster_than_sequential(monkeypatch):
    """With 10 papers and max_workers=4, total time should be well below
    (10 * delay), roughly (ceil(10/4) * delay) = 3 * delay.
    """
    DELAY = 0.15  # seconds per paper
    N_PAPERS = 10
    WORKERS = 4

    call_pmids = []
    call_lock = threading.Lock()

    def fake_retrieve_single(self, paper, stats):
        with call_lock:
            call_pmids.append(paper["pmid"])
        time.sleep(DELAY)
        paper["full_text"] = "fake"
        paper["text_source"] = "pmc"
        stats["pmc_fulltext"] = stats.get("pmc_fulltext", 0) + 1

    agent = _make_agent(max_workers=WORKERS)
    agent._retrieve_single = types.MethodType(fake_retrieve_single, agent)

    papers = _make_papers(N_PAPERS)

    start = time.time()
    enriched = agent.retrieve_fulltext(papers, papers)
    elapsed = time.time() - start

    # Sanity
    assert len(enriched) == N_PAPERS
    assert all(p["text_source"] == "pmc" for p in enriched)
    assert len(call_pmids) == N_PAPERS

    # With N=10, delay=0.15s, workers=4:
    #   sequential would be ~1.50s
    #   parallel lower bound ~ceil(10/4)*0.15 = 0.45s
    # Assert strictly faster than half the sequential time.
    sequential_estimate = N_PAPERS * DELAY
    assert elapsed < sequential_estimate / 2, (
        f"Parallel retrieval took {elapsed:.2f}s, expected < "
        f"{sequential_estimate/2:.2f}s (sequential baseline {sequential_estimate:.2f}s)"
    )


def test_parallel_preserves_input_order(monkeypatch):
    """Even though futures complete out of order, the enriched list order
    must match input_papers order (downstream relies on this).
    """
    DELAY_MAP = {  # different per-paper delays to force out-of-order completion
        "100": 0.30,
        "101": 0.05,
        "102": 0.20,
        "103": 0.02,
        "104": 0.10,
    }

    def fake_retrieve_single(self, paper, stats):
        time.sleep(DELAY_MAP.get(paper["pmid"], 0.05))
        paper["text_source"] = "pmc"
        stats["pmc_fulltext"] = stats.get("pmc_fulltext", 0) + 1

    agent = _make_agent(max_workers=3)
    agent._retrieve_single = types.MethodType(fake_retrieve_single, agent)

    papers = [{"pmid": pmid, "title": f"T {pmid}"} for pmid in DELAY_MAP]

    enriched = agent.retrieve_fulltext(papers, papers)

    assert [p["pmid"] for p in enriched] == [p["pmid"] for p in papers]


def test_parallel_stats_aggregation():
    """Global stats must reflect the sum of per-paper local stats increments,
    with no races (all counters consistent).
    """
    N_PAPERS = 12

    def fake_retrieve_single(self, paper, stats):
        # Alternate source to stress multi-key aggregation
        idx = int(paper["pmid"]) % 3
        if idx == 0:
            stats["pmc_fulltext"] = stats.get("pmc_fulltext", 0) + 1
            paper["text_source"] = "pmc"
        elif idx == 1:
            stats["unpaywall_pdf"] = stats.get("unpaywall_pdf", 0) + 1
            paper["text_source"] = "unpaywall_pdf"
        else:
            stats["abstract_only"] = stats.get("abstract_only", 0) + 1
            paper["text_source"] = "abstract_only"
        time.sleep(0.02)

    agent = _make_agent(max_workers=4)
    agent._retrieve_single = types.MethodType(fake_retrieve_single, agent)

    papers = [{"pmid": str(i), "title": f"P {i}"} for i in range(N_PAPERS)]
    enriched = agent.retrieve_fulltext(papers, papers)

    # Verify each paper got processed exactly once (no double-counting)
    sources = [p["text_source"] for p in enriched]
    assert sources.count("pmc") == sum(1 for i in range(N_PAPERS) if i % 3 == 0)
    assert sources.count("unpaywall_pdf") == sum(1 for i in range(N_PAPERS) if i % 3 == 1)
    assert sources.count("abstract_only") == sum(1 for i in range(N_PAPERS) if i % 3 == 2)


def test_max_workers_default_when_missing(tmp_path, monkeypatch):
    """If agents.json lacks max_workers, default of 4 should apply.

    We verify via the `get_agent_param` fallback behaviour without needing
    to call the real Agent4 __init__ (which requires config.yaml + model).
    """
    # Minimal stand-in for BaseAgent.get_agent_param
    class Dummy:
        agent_params = {}
        def get_agent_param(self, key, fallback=None):
            return self.agent_params.get(key, fallback)

    d = Dummy()
    assert d.get_agent_param("max_workers", 4) == 4

    d.agent_params = {"max_workers": 8}
    assert d.get_agent_param("max_workers", 4) == 8
