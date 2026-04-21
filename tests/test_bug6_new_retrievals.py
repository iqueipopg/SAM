"""BUG6: tests for the new full-text retrieval sources added to Agent4.

Covers:
    - _try_semanticscholar_v2 (Graph API openAccessPdf endpoint)
    - _try_arxiv (export.arxiv.org Atom XML)
    - _try_rxiv   (bioRxiv / medRxiv JSON details API)
    - _try_preprints orchestrator
    - _retrieve_single cascade: only the newly added step succeeds.
    - Error handling: ConnectionError must NOT bubble up.

All HTTP calls are mocked — nothing touches the network.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agents.agent4_fulltext import Agent4FullText


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_agent() -> Agent4FullText:
    """Build an Agent4FullText instance without triggering BaseAgent's model
    / config discovery heavyweight paths.

    BaseAgent.__init__ reads config.yaml and agents.json; both exist in the
    repo, so this is safe. We monkey-patch only the rate-limit method to
    a no-op so tests don't sleep.
    """
    agent = Agent4FullText()
    # Neutralize sleeps inside S2 rate limiter (uses time.sleep through lock)
    agent._rate_limit_semanticscholar = lambda: None
    return agent


def _mock_response(json_data=None, text=None, content=None, status_code=200,
                   headers=None):
    """Build a lightweight stand-in for requests.Response with the
    minimum attrs / methods our code uses.
    """
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_data or {})
    resp.text = text or ""
    resp.content = content if content is not None else b""
    resp.headers = headers or {}

    def raise_for_status():
        if status_code >= 400:
            err = requests.exceptions.HTTPError(f"{status_code} Error")
            err.response = resp
            raise err
    resp.raise_for_status = raise_for_status
    return resp


# ──────────────────────────────────────────────────────────────────────
# Semantic Scholar v2
# ──────────────────────────────────────────────────────────────────────

class TestSemanticScholarV2:
    def test_success_returns_text(self):
        agent = _make_agent()
        paper = {"doi": "10.1000/abc", "title": "Example"}

        api_resp = _mock_response(json_data={
            "openAccessPdf": {"url": "https://example.org/paper.pdf"},
            "externalIds": {"DOI": "10.1000/abc"},
        })
        pdf_resp = _mock_response(content=b"%PDF-1.4 fake", status_code=200,
                                  headers={"Content-Type": "application/pdf"})

        def fake_get(url, *args, **kwargs):
            if "semanticscholar.org" in url:
                return api_resp
            return pdf_resp

        with patch("agents.agent4_fulltext.requests.get", side_effect=fake_get):
            with patch.object(agent, "_extract_pdf_text", return_value="extracted text body"):
                result = agent._try_semanticscholar_v2(paper)

        assert result == "extracted text body"

    def test_no_openaccesspdf_returns_none(self):
        agent = _make_agent()
        paper = {"doi": "10.1000/xyz"}
        api_resp = _mock_response(json_data={"openAccessPdf": None,
                                             "externalIds": {}})

        with patch("agents.agent4_fulltext.requests.get", return_value=api_resp):
            result = agent._try_semanticscholar_v2(paper)
        assert result is None

    def test_connection_error_returns_none(self):
        agent = _make_agent()
        paper = {"doi": "10.1000/fail"}
        # Use retry-friendly config: one retry to keep test fast.
        with patch("agents.agent4_fulltext.requests.get",
                   side_effect=requests.exceptions.ConnectionError("boom")):
            with patch("agents.agent4_fulltext.retry_with_backoff",
                       side_effect=requests.exceptions.ConnectionError("boom")):
                # The function must swallow the exception from the wrapper.
                result = agent._try_semanticscholar_v2(paper)
        assert result is None

    def test_empty_paper_returns_none(self):
        agent = _make_agent()
        # No doi, no paperId -> nothing to look up.
        result = agent._try_semanticscholar_v2({})
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# ArXiv
# ──────────────────────────────────────────────────────────────────────

ARXIV_ATOM_OK = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Example paper on CBT</title>
    <link rel="alternate" href="http://arxiv.org/abs/2401.12345v1" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1.pdf" type="application/pdf"/>
  </entry>
</feed>
"""

ARXIV_ATOM_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""


class TestArxiv:
    def test_success_downloads_pdf(self):
        agent = _make_agent()
        search_resp = _mock_response(text=ARXIV_ATOM_OK, status_code=200)
        pdf_resp = _mock_response(content=b"%PDF-1.4 fake", status_code=200,
                                  headers={"Content-Type": "application/pdf"})

        def fake_get(url, *args, **kwargs):
            if "export.arxiv.org" in url:
                return search_resp
            return pdf_resp

        with patch("agents.agent4_fulltext.requests.get", side_effect=fake_get):
            with patch.object(agent, "_extract_pdf_text", return_value="arxiv body text"):
                result = agent._try_arxiv("Example paper on CBT")
        assert result == "arxiv body text"

    def test_no_match_returns_none(self):
        agent = _make_agent()
        resp = _mock_response(text=ARXIV_ATOM_EMPTY, status_code=200)
        with patch("agents.agent4_fulltext.requests.get", return_value=resp):
            assert agent._try_arxiv("unknown title") is None

    def test_empty_title_returns_none(self):
        agent = _make_agent()
        assert agent._try_arxiv("") is None

    def test_connection_error_does_not_raise(self):
        agent = _make_agent()
        with patch("agents.agent4_fulltext.requests.get",
                   side_effect=requests.exceptions.ConnectionError("boom")):
            # retry_with_backoff will exhaust retries and re-raise; the
            # function must swallow it.
            result = agent._try_arxiv("Example title")
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# bioRxiv / medRxiv
# ──────────────────────────────────────────────────────────────────────

class TestRxiv:
    def test_biorxiv_success(self):
        agent = _make_agent()
        api_resp = _mock_response(json_data={
            "collection": [
                {"doi": "10.1101/2025.01.01.00001", "version": 1}
            ]
        }, status_code=200)
        pdf_resp = _mock_response(content=b"%PDF-1.4 fake", status_code=200,
                                  headers={"Content-Type": "application/pdf"})

        def fake_get(url, *args, **kwargs):
            if "api.biorxiv.org" in url:
                return api_resp
            return pdf_resp

        with patch("agents.agent4_fulltext.requests.get", side_effect=fake_get):
            with patch.object(agent, "_extract_pdf_text", return_value="biorxiv full text"):
                text = agent._try_rxiv("10.1101/2025.01.01.00001",
                                       "Example preprint", server="biorxiv")
        assert text == "biorxiv full text"

    def test_no_doi_returns_none(self):
        agent = _make_agent()
        assert agent._try_rxiv("", "Example preprint", server="biorxiv") is None

    def test_unknown_server_returns_none(self):
        agent = _make_agent()
        assert agent._try_rxiv("10.1101/xxx", "t", server="foobar") is None

    def test_empty_collection_returns_none(self):
        agent = _make_agent()
        resp = _mock_response(json_data={"collection": []}, status_code=200)
        with patch("agents.agent4_fulltext.requests.get", return_value=resp):
            assert agent._try_rxiv("10.1101/xxx", "t", server="biorxiv") is None

    def test_connection_error_does_not_raise(self):
        agent = _make_agent()
        with patch("agents.agent4_fulltext.requests.get",
                   side_effect=requests.exceptions.ConnectionError("boom")):
            text = agent._try_rxiv("10.1101/xxx", "t", server="biorxiv")
        assert text is None


# ──────────────────────────────────────────────────────────────────────
# _try_preprints orchestrator
# ──────────────────────────────────────────────────────────────────────

class TestPreprintOrchestrator:
    def test_arxiv_wins_first(self):
        agent = _make_agent()
        with patch.object(agent, "_try_arxiv", return_value="arxiv text") as m_a, \
             patch.object(agent, "_try_rxiv", return_value="should not be used") as m_r:
            text, src = agent._try_preprints({"title": "t", "doi": "d"})
        assert text == "arxiv text"
        assert src == "arxiv"
        m_a.assert_called_once()
        m_r.assert_not_called()

    def test_biorxiv_fallback(self):
        agent = _make_agent()

        def rxiv_mock(doi, title, server="biorxiv"):
            return "bio content" if server == "biorxiv" else None

        with patch.object(agent, "_try_arxiv", return_value=None), \
             patch.object(agent, "_try_rxiv", side_effect=rxiv_mock):
            text, src = agent._try_preprints({"title": "t", "doi": "10.1101/x"})
        assert text == "bio content"
        assert src == "biorxiv"

    def test_medrxiv_fallback(self):
        agent = _make_agent()

        def rxiv_mock(doi, title, server="biorxiv"):
            return "med content" if server == "medrxiv" else None

        with patch.object(agent, "_try_arxiv", return_value=None), \
             patch.object(agent, "_try_rxiv", side_effect=rxiv_mock):
            text, src = agent._try_preprints({"title": "t", "doi": "10.1101/x"})
        assert text == "med content"
        assert src == "medrxiv"

    def test_all_fail_returns_none_tuple(self):
        agent = _make_agent()
        with patch.object(agent, "_try_arxiv", return_value=None), \
             patch.object(agent, "_try_rxiv", return_value=None):
            text, src = agent._try_preprints({"title": "t", "doi": "d"})
        assert text is None
        assert src == ""


# ──────────────────────────────────────────────────────────────────────
# Cascade test: only ArXiv yields a result
# ──────────────────────────────────────────────────────────────────────

class TestCascadeOnlyArxiv:
    def test_cascade_terminates_in_arxiv(self):
        """When every source upstream of preprints fails, ArXiv must win."""
        agent = _make_agent()
        paper = {
            "pmid": "99999999",
            "title": "Cognitive therapy meta-analysis",
            "abstract": "abstract text",
            "doi": "10.1234/abc",
            "pmcid": "",
        }
        stats = {
            "pmc_fulltext": 0, "unpaywall_pdf": 0, "europepmc_fulltext": 0,
            "semanticscholar_pdf": 0, "openalex_pdf": 0, "crossref_pdf": 0,
            "doaj_pdf": 0, "lens_fulltext": 0, "core_fulltext": 0,
            "google_scholar_pdf": 0, "scihub_pdf": 0, "abstract_only": 0,
        }

        # Short-circuit all earlier cascade steps to None/fail.
        with patch.object(agent, "_fetch_pmc_fulltext", return_value=(None, None)), \
             patch.object(agent, "_fetch_unpaywall_pdf", return_value=(None, None)), \
             patch.object(agent, "_fetch_europepmc_fulltext", return_value=(None, None)), \
             patch.object(agent, "_fetch_semanticscholar_pdf", return_value=None), \
             patch.object(agent, "_try_openalex", return_value=None), \
             patch.object(agent, "_try_crossref", return_value=None), \
             patch.object(agent, "_try_doaj", return_value=None), \
             patch.object(agent, "_try_lens", return_value=None), \
             patch.object(agent, "_try_core", return_value=None), \
             patch.object(agent, "_try_semanticscholar_v2", return_value=None), \
             patch.object(agent, "_try_google_scholar", return_value=None), \
             patch.object(agent, "_fetch_scihub_pdf", return_value=None), \
             patch.object(agent, "_try_preprints",
                          return_value=("preprint body text", "arxiv")):
            agent._retrieve_single(paper, stats)

        assert paper["full_text"] == "preprint body text"
        assert paper["text_source"] == "preprint_arxiv"
        assert stats["preprint_pdf"] == 1


# ──────────────────────────────────────────────────────────────────────
# Config flags: agents.json toggles should be honored
# ──────────────────────────────────────────────────────────────────────

class TestConfigFlags:
    def test_flags_default_to_true(self):
        agent = _make_agent()
        assert agent.preprints_enabled is True
        assert agent.arxiv_enabled is True
        assert agent.semanticscholar_v2_enabled is True
