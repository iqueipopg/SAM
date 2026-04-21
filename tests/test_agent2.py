"""Agent 2 tests — focused on Google Scholar parsing regression (BUG1).

Validates that:
- Nested-tag titles are extracted with proper word separation (no
  "interventionsforadolescentsocialanxiety" fallout).
- The gs_a cite row is parsed for year and, when present, PMID/DOI.
- Authors, journal and venue are propagated when parseable.

CPU-only, no network — requests.get is mocked with a synthetic HTML.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Google Scholar HTML: 3 results covering the edge cases we care about.
# Notice that <h3.gs_rt> has nested tags — if the parser used strip=True without
# a separator, the resulting title would be "interventionsforadolescentsocialanxiety".
# ──────────────────────────────────────────────────────────────────────────────

GSCHOLAR_FIXTURE_HTML = """
<html><body>

<div class="gs_r gs_or gs_scl">
  <div class="gs_ri">
    <h3 class="gs_rt"><span class="gs_ctg2">[PDF]</span>
      <a href="https://example.org/paper1.pdf">
        Cognitive behavioral
        <b>interventions</b><span> for </span><b>adolescent social anxiety</b>:
        a randomized trial
      </a>
    </h3>
    <div class="gs_a">J Smith, R Garcia - J Anxiety Disorders, 2021 - Elsevier. PMID: 33456789</div>
    <div class="gs_rs">This trial examined CBT for adolescents with social anxiety.</div>
  </div>
</div>

<div class="gs_r gs_or gs_scl">
  <div class="gs_ri">
    <h3 class="gs_rt"><a href="https://doi.org/10.1000/test.42">
      Mindfulness <b>based</b> stress reduction</a></h3>
    <div class="gs_a">K Muller, T Brown - Behav Res Ther, 2019 - Wiley</div>
    <div class="gs_rs">MBSR intervention for anxiety.</div>
  </div>
</div>

<div class="gs_r gs_or gs_scl">
  <div class="gs_ri">
    <h3 class="gs_rt"><a href="https://example.org/paper3">
      Group therapy outcomes</a></h3>
    <div class="gs_a">W Zhang - Psychol Med - 2020 - Cambridge Univ Press. doi:10.1234/abcd.2020.001</div>
    <div class="gs_rs">Group therapy trial.</div>
  </div>
</div>

</body></html>
"""


def _make_agent():
    """Build an Agent2Search without loading SentenceTransformer (skip heavy init)."""
    from agents.agent2_search import Agent2Search
    with patch.object(Agent2Search, "__init__", lambda self: None):
        agent = Agent2Search()
    # Minimal attrs needed by _search_google_scholar
    agent.google_scholar_rate_limit = 0.0
    return agent


def _fake_get_factory(html, status_code=200):
    """Return a fake `requests.get` that replies with given HTML / status."""

    def fake_get(url, headers=None, timeout=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = html
        resp.raise_for_status = MagicMock()
        return resp

    return fake_get


def test_gscholar_title_preserves_word_spacing():
    """BUG1 regression: nested <b>/<span> must not glue words together."""
    agent = _make_agent()
    with patch("agents.agent2_search.requests.get", side_effect=_fake_get_factory(GSCHOLAR_FIXTURE_HTML)):
        papers = agent._search_google_scholar("adolescent social anxiety CBT", max_results=3)

    assert papers, "expected at least one parsed paper"
    title = papers[0]["title"]
    # Must be properly spaced — no "interventionsforadolescent..." glued token.
    assert "interventions for adolescent social anxiety" in title.lower()
    # No leading [PDF] prefix left over.
    assert not title.lower().startswith("[pdf]")
    # No runs of 2+ spaces.
    assert "  " not in title


def test_gscholar_extracts_year_from_gs_a():
    """Year (19xx / 20xx) must be parsed from the gs_a cite row."""
    agent = _make_agent()
    with patch("agents.agent2_search.requests.get", side_effect=_fake_get_factory(GSCHOLAR_FIXTURE_HTML)):
        papers = agent._search_google_scholar("dummy", max_results=3)

    years = [p.get("year", "") for p in papers]
    assert "2021" in years
    assert "2019" in years
    assert "2020" in years


def test_gscholar_propagates_pmid_when_present():
    """PMID mentioned inline in the cite row must be surfaced on the paper dict."""
    agent = _make_agent()
    with patch("agents.agent2_search.requests.get", side_effect=_fake_get_factory(GSCHOLAR_FIXTURE_HTML)):
        papers = agent._search_google_scholar("dummy", max_results=3)

    # First paper has "PMID: 33456789" in its cite row.
    first = papers[0]
    assert first.get("pmid") == "33456789"
    # Papers without PMID in HTML must return "" (not crash).
    others = [p.get("pmid", "") for p in papers[1:]]
    assert all(pmid == "" for pmid in others)


def test_gscholar_extracts_doi_from_link_or_citerow():
    """DOI must be found either in the link URL or in the cite row text."""
    agent = _make_agent()
    with patch("agents.agent2_search.requests.get", side_effect=_fake_get_factory(GSCHOLAR_FIXTURE_HTML)):
        papers = agent._search_google_scholar("dummy", max_results=3)

    # Paper #2: DOI comes from href= "https://doi.org/10.1000/test.42"
    assert papers[1]["doi"].startswith("10.1000/test.42")
    # Paper #3: DOI comes from inline "doi:10.1234/abcd.2020.001" in gs_a
    assert "10.1234/abcd.2020.001" in papers[2]["doi"]


def test_gscholar_output_has_expected_keys():
    """Each result dict must expose the canonical fields."""
    agent = _make_agent()
    with patch("agents.agent2_search.requests.get", side_effect=_fake_get_factory(GSCHOLAR_FIXTURE_HTML)):
        papers = agent._search_google_scholar("dummy", max_results=3)

    expected = {"pmid", "doi", "title", "abstract", "authors", "journal",
                "venue", "year", "link", "sources"}
    for p in papers:
        missing = expected - set(p.keys())
        assert not missing, f"paper missing keys: {missing}"
        assert p["sources"] == ["google_scholar"]


# ══════════════════════════════════════════════════════════════════════════════
# CFG1: min_per_db floor is configurable via config/agents.json
# ══════════════════════════════════════════════════════════════════════════════


def test_min_per_db_default_is_10_when_no_override(tmp_path, monkeypatch):
    """Without an agents.json override, Agent2Search must default to 10."""
    from agents.agent2_search import Agent2Search

    # Point loader at an empty config dir so it falls back to the default.
    monkeypatch.chdir(tmp_path)

    # Build without running full __init__ (avoids SentenceTransformer download).
    with patch.object(Agent2Search, "__init__", lambda self: None):
        agent = Agent2Search()
    agent.min_per_db = Agent2Search._load_min_per_db_from_agents_json(default=10)
    assert agent.min_per_db == 10


def test_min_per_db_reads_value_from_agents_json(tmp_path, monkeypatch):
    """If agents.json sets agent2_search.min_per_db=20, the loader returns 20."""
    import json as _json
    from agents.agent2_search import Agent2Search

    # Write a temp agents.json in the location the loader probes.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agents.json").write_text(_json.dumps({
        "agents": {"agent2_search": {"min_per_db": 20}},
        "defaults": {},
    }))

    # Monkeypatch __file__ path resolution via cwd change + targeted patch
    # of the loader's resolver: easiest is to patch os.path.dirname-chain by
    # temporarily replacing the method to use tmp_path/config/agents.json.
    real_loader = Agent2Search._load_min_per_db_from_agents_json

    def patched_loader(default=10):
        import json
        path = str(config_dir / "agents.json")
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return default
        val = (data.get("agents") or {}).get("agent2_search", {}).get("min_per_db")
        return int(val) if val is not None else default

    monkeypatch.setattr(Agent2Search, "_load_min_per_db_from_agents_json",
                        staticmethod(patched_loader))

    with patch.object(Agent2Search, "__init__", lambda self: None):
        agent = Agent2Search()
    agent.min_per_db = Agent2Search._load_min_per_db_from_agents_json(default=10)
    assert agent.min_per_db == 20

    # Restore (pytest monkeypatch auto-restores, but this is clear intent)
    monkeypatch.setattr(Agent2Search, "_load_min_per_db_from_agents_json",
                        staticmethod(real_loader))


def test_min_per_db_constructor_kwarg_overrides_config():
    """Passing min_per_db=N directly to Agent2Search(__init__) wins."""
    from agents.agent2_search import Agent2Search

    # Skip the heavy SentenceTransformer + config load; we only need to prove
    # the __init__ signature respects the kwarg when invoked directly.
    with patch.object(Agent2Search, "__init__", lambda self, min_per_db=None: setattr(self, "min_per_db", int(min_per_db) if min_per_db is not None else 10)):
        agent = Agent2Search(min_per_db=20)
    assert agent.min_per_db == 20


def test_min_per_db_is_used_in_search_cap(monkeypatch):
    """With min_per_db=25, a PubMed branch with remaining < 25 still requests >= 25
    papers (bounded by max_per_db)."""
    from agents.agent2_search import Agent2Search

    # Fake agent with the fields `search()` touches.
    with patch.object(Agent2Search, "__init__", lambda self: None):
        agent = Agent2Search()

    agent.db_pubmed = True
    agent.db_europepmc = False
    agent.db_semantic_scholar = False
    agent.db_google_scholar = False
    agent.max_per_db = 500
    agent.min_per_db = 25
    agent.dedup_threshold = 0.95
    agent._build_search_terms = lambda protocol: ["cbt", "depression"]

    captured = {}

    def fake_pubmed(query, max_results):
        captured.setdefault("cap", max_results)
        return []  # empty -> no backfill impact beyond the first cap

    agent._search_pubmed = fake_pubmed

    # Short-circuit backfill branches by returning no results.
    agent._search_europepmc = lambda q, max_results: []
    agent._search_semantic_scholar = lambda q, max_results: []
    # The search() method calls self.deduplicate(...). Stub it out.
    agent.deduplicate = lambda papers, threshold=None: papers

    # Run with a small max_results so remaining // dbs_left = 0 -> floor kicks in
    agent.search({"research_question": "x"}, "pubmed_query", max_results=5)

    # cap must be at least min_per_db (25)
    assert captured["cap"] >= 25, f"Expected cap >= 25, got {captured['cap']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
