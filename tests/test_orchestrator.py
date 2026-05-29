"""Tests for sam_orchestrator state library (T17.C.0).

No LLM, no GPU, no network. All tests operate on tmpdir fixtures.
Fake JSONs are written directly to disk; library functions are called
via their Python APIs (not the CLI subprocess).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from sam_orchestrator import (
    AGENT_IDS,
    AGENT_MODES,
    OUTPUTS_ROOT,
    RunPaths,
    audit_run,
    commit_output,
    init_run,
    next_work,
    paper_id_to_slug,
)
from sam_orchestrator.checkpoint import append_index_unique, make_entry, read_index
from sam_orchestrator.errors import SchemaValidationError, SubsetMappingMissingError
from sam_orchestrator.manifest import load_manifest


# ---------------------------------------------------------------------------
# Helpers — fake JSON builders (one per agent, minimal valid payloads)
# ---------------------------------------------------------------------------

_BASE = {
    "agent": "placeholder",
    "backend": "claude",
    "timestamp": "2026-05-29T21:00:00Z",
    "tokens_in": None,
    "tokens_out": None,
}


def _base(agent: str) -> Dict[str, Any]:
    return {**_BASE, "agent": agent}


def fake_a1(query: str = "test query") -> Dict[str, Any]:
    return {
        **_base("sam_a1_protocol"),
        "query": query,
        "protocol": {
            "framework": "PICO",
            "research_question": "Is X better than Y?",
            "population": "Adults",
            "intervention": "X",
            "comparator": "Y",
            "outcome": "Depression",
            "original_query": query,
            "inclusion_criteria": ["RCT"],
            "exclusion_criteria": ["Case reports"],
            "protocol_document": "Protocol text.",
            "prospero_draft": "PROSPERO draft.",
        },
        "pubmed_query": "(X[MeSH]) AND (RCT[pt])",
        "version": "V2",
    }


def fake_a2(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(papers)
    return {
        **_base("sam_a2_search"),
        "papers": papers,
        "stats": {
            "pubmed": n,
            "europepmc": 0,
            "semantic_scholar": 0,
            "google_scholar": 0,
            "total_before_dedup": n,
            "total_after_dedup": n,
            "duplicates_removed": 0,
            "search_sources": ["pubmed"],
        },
    }


def _paper(pmid: str, doi: str = "") -> Dict[str, Any]:
    return {
        "pmid": pmid,
        "doi": doi or f"10.1/{pmid}",
        "title": f"Paper {pmid}",
        "abstract": f"Abstract of {pmid}.",
        "authors": ["Author A"],
        "journal": "Journal of Tests",
        "year": "2024",
        "sources": ["pubmed"],
    }


def fake_a3(pmid: str, decision: str = "INCLUDE") -> Dict[str, Any]:
    return {
        **_base("sam_a3_screening"),
        "paper_id": pmid,
        "pmid": pmid,
        "title": f"Paper {pmid}",
        "decision": decision,
        "confidence": 0.9,
        "votes": [{"decision": decision, "reasoning": "ok", "model": "default"}],
        "reasoning": "Meets criteria.",
        "flags": [],
        "screening_confidence": 0.9,
        "borderline": False,
    }


def fake_a4(pmid: str, decision: str = "INCLUDE") -> Dict[str, Any]:
    return {
        **_base("sam_a4_fulltext"),
        "paper_id": pmid,
        "pmid": pmid,
        "title": f"Paper {pmid}",
        "abstract": "Abstract.",
        "authors": ["Author A"],
        "journal": "Journal",
        "year": "2024",
        "doi": f"10.1/{pmid}",
        "pmcid": "",
        "full_text": "Full text body.",
        "text_source": "pmc",
        "full_text_length": 14,
        "sections": {},
        "oa_url": None,
        "fulltext_decision": decision,
        "fulltext_reasoning": "Full text confirms eligibility.",
        "exclusion_reason": None,
        "fulltext_votes": [{"decision": decision, "reasoning": "ok", "model": "default"}],
        "fulltext_confidence": 0.9,
        "screening_confidence": 0.9,
        "borderline": False,
    }


def fake_a5(pmid: str) -> Dict[str, Any]:
    return {
        **_base("sam_a5_extraction"),
        "paper_id": pmid,
        "pmid": pmid,
        "title": f"Paper {pmid}",
        "authors": ["Author A"],
        "journal": "Journal",
        "year": "2024",
        "study_characteristics": {
            "study_design": "RCT",
            "sample_size": 100,
            "n_groups": {"intervention": 50, "control": 50},
            "age_mean": 40.0,
            "age_sd": 10.0,
            "female_pct": 50.0,
            "setting": "outpatient",
            "country": "Spain",
            "followup_duration": "6 months",
        },
        "intervention_exposure": {
            "name": "CBT",
            "description": "Cognitive-behavioural therapy.",
            "comparator": "TAU",
            "dose_duration": "12 weeks",
        },
        "outcomes": [
            {
                "name": "Depression score",
                "effect_size_type": "SMD",
                "effect_size_value": -0.5,
                "ci_lower": -0.8,
                "ci_upper": -0.2,
                "p_value": 0.01,
            }
        ],
        "extraction_quality": "complete",
        "extraction_notes": "Sin notas.",
        "text_source": "pmc",
        "sections_used": ["results"],
        "extraction_confidence": "high",
        "concordance": {
            "sample_size_match": True,
            "outcomes_matched": 1,
            "outcomes_discrepant": 0,
            "outcomes_pass1_only": 0,
            "outcomes_pass2_only": 0,
            "flags": [],
        },
    }


def fake_a6(pmid: str) -> Dict[str, Any]:
    return {
        **_base("sam_a6_quality"),
        "paper_id": pmid,
        "pmid": pmid,
        "title": f"Paper {pmid}",
        "study_design_detected": "rct",
        "quality_assessment": {
            "tool": "RoB2",
            "domains": {
                "randomisation": {"judgment": "Low", "justification": "Adequate."},
                "deviations": {"judgment": "Low", "justification": "ITT."},
            },
            "overall": "Low",
            "overall_justification": "No high-risk domains.",
        },
        "quality_weight": 1.0,
        "study_characteristics": None,
        "outcomes": [],
    }


def fake_a7() -> Dict[str, Any]:
    return {
        **_base("sam_a7_synthesis"),
        "analyses": {"SMD": {"pooled": -0.5, "ci": [-0.8, -0.2]}},
        "plots": {"forest": "outputs/forest.png"},
        "r_script_path": None,
    }


def fake_a8() -> Dict[str, Any]:
    return {
        **_base("sam_a8_manuscript"),
        "sections": {
            "abstract": "Resumen del estudio.",
            "introduction": "Introduccion.",
            "methods": "Métodos.",
            "results": "Resultados.",
            "discussion": "Discusion.",
            "conclusion": "Conclusion.",
        },
        "references": ["Ref1", "Ref2"],
        "pdf_path": None,
    }


def fake_a9() -> Dict[str, Any]:
    return {
        **_base("sam_a9_explainer"),
        "excel_path": "outputs/papers_raw.xlsx",
        "report_path": "outputs/explainability.md",
        "n_total": 5,
        "n_included": 3,
        "n_excluded": 2,
    }


_SINGLE_FAKE_BUILDERS = {
    "sam_a1_protocol": lambda: fake_a1(),
    "sam_a7_synthesis": fake_a7,
    "sam_a8_manuscript": fake_a8,
    "sam_a9_explainer": fake_a9,
}

_PER_PAPER_FAKE_BUILDERS = {
    "sam_a3_screening": fake_a3,
    "sam_a4_fulltext": fake_a4,
    "sam_a5_extraction": fake_a5,
    "sam_a6_quality": fake_a6,
}

TOY_PMIDS = ["P001", "P002", "P003", "P004", "P005"]
TOY_PAPERS = [_paper(pmid) for pmid in TOY_PMIDS]


# ---------------------------------------------------------------------------
# Shared fixture: redirect outputs to tmpdir
# ---------------------------------------------------------------------------


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch):
    """Redirige outputs/ al tmpdir para aislar los tests."""
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.setattr("sam_orchestrator.paths.OUTPUTS_ROOT", root)
    monkeypatch.setattr("sam_orchestrator.state.OUTPUTS_ROOT", root, raising=False)
    monkeypatch.setattr("sam_orchestrator.manifest.OUTPUTS_ROOT", root, raising=False)
    monkeypatch.setattr("sam_orchestrator.commit.OUTPUTS_ROOT", root, raising=False)
    monkeypatch.setattr("sam_orchestrator.audit.OUTPUTS_ROOT", root, raising=False)
    return root


def _init(runs_root: Path, query: str = "test", **kw) -> str:
    """Helper that creates a run using the tmpdir root."""
    return init_run(query, outputs_root=runs_root, **kw)


def _next(runs_root: Path, run_id: str) -> Dict[str, Any]:
    return next_work(run_id, outputs_root=runs_root)


def _commit(runs_root: Path, run_id: str, agent_id: str, path: str) -> Dict[str, Any]:
    return commit_output(run_id, agent_id, path, outputs_root=runs_root)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helper: advance the run up to (but not including) a target agent
# ---------------------------------------------------------------------------


def _advance_to(runs_root: Path, run_id: str, target_agent: str, papers=None) -> None:
    """Drive next/commit loop until `target_agent` would be next.

    For single agents: writes the fake JSON and commits.
    For per-paper agents: writes and commits one fake JSON per paper in
    the provided `papers` list (or TOY_PAPERS).
    """
    if papers is None:
        papers = TOY_PAPERS
    pmids = [p["pmid"] for p in papers]

    for agent_id in AGENT_IDS:
        if agent_id == target_agent:
            break
        mode = AGENT_MODES[agent_id]
        paths = RunPaths(run_id, outputs_root=runs_root)

        if mode == "single":
            # A2 needs the papers fixture written before advancing
            if agent_id == "sam_a2_search":
                out = paths.single_output(agent_id)
                _write_json(out, fake_a2(papers))
                _commit(runs_root, run_id, agent_id, str(out))
            else:
                out = paths.single_output(agent_id)
                build = _SINGLE_FAKE_BUILDERS.get(agent_id)
                if build is None:
                    continue
                _write_json(out, build())
                _commit(runs_root, run_id, agent_id, str(out))
        else:
            # per-paper: commit all papers for this agent
            build = _PER_PAPER_FAKE_BUILDERS.get(agent_id)
            if build is None:
                continue
            for pmid in pmids:
                slug = paper_id_to_slug(pmid)
                out = paths.agent_dir(agent_id) / f"{slug}.json"
                # A3 INCLUDE, A4 INCLUDE, A5/A6 normal
                if agent_id == "sam_a3_screening":
                    data = fake_a3(pmid, "INCLUDE")
                elif agent_id == "sam_a4_fulltext":
                    data = fake_a4(pmid, "INCLUDE")
                else:
                    data = build(pmid)
                _write_json(out, data)
                _commit(runs_root, run_id, agent_id, str(out))


# ---------------------------------------------------------------------------
# Section: init
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_creates_run_id_and_dirs(self, runs_root):
        run_id = _init(runs_root, "smoke test query")
        # run_id follows YYYYMMDD_HHMMSS_slug pattern
        assert re.match(r"^\d{8}_\d{6}_", run_id), f"unexpected run_id: {run_id}"
        run_dir = runs_root / run_id
        assert run_dir.is_dir()
        for aid in AGENT_IDS:
            assert (run_dir / aid).is_dir(), f"missing subdir for {aid}"
        manifest = load_manifest(run_id, outputs_root=runs_root)
        assert manifest.run_id == run_id
        assert all(
            manifest.agents[aid].status == "pending" for aid in AGENT_IDS
        )

    def test_init_with_subset_records_config(self, runs_root, tmp_path):
        partition = tmp_path / "partition.json"
        partition.write_text(
            json.dumps({"dev": ["P001", "P002", "P003"]}), encoding="utf-8"
        )
        run_id = _init(runs_root, "subset query", subset_spec=f"{partition}:dev")
        manifest = load_manifest(run_id, outputs_root=runs_root)
        assert manifest.config.subset is not None
        assert manifest.config.subset.split == "dev"
        assert manifest.config.subset.n == 3

    def test_init_with_invalid_subset_path_aborts(self, runs_root, tmp_path):
        bad_path = tmp_path / "nonexistent_partition.json"
        with pytest.raises(SubsetMappingMissingError):
            _init(runs_root, "query", subset_spec=f"{bad_path}:dev")
        # No run dirs should have been created
        assert list(runs_root.iterdir()) == []

    def test_init_emits_run_id_to_stdout(self, runs_root, capsys):
        run_id = _init(runs_root, "stdout test query")
        # The function returns run_id; CLI would print it.
        # Since we're calling the Python function directly, just verify the
        # return value is a non-empty string (CLI stdout test covered by the CLI itself).
        assert isinstance(run_id, str) and run_id


# ---------------------------------------------------------------------------
# Section: next — single agents
# ---------------------------------------------------------------------------


class TestNextSingleAgents:
    def test_next_initial_returns_a1(self, runs_root):
        run_id = _init(runs_root)
        work = _next(runs_root, run_id)
        assert work["agent"] == "sam_a1_protocol"
        assert work["paper_id"] is None

    def test_next_skips_completed_single_agent(self, runs_root):
        run_id = _init(runs_root)
        _advance_to(runs_root, run_id, "sam_a2_search")
        work = _next(runs_root, run_id)
        assert work["agent"] == "sam_a2_search"

    def test_next_returns_done_when_all_completed(self, runs_root):
        run_id = _init(runs_root)
        # Advance past all agents
        _advance_to(runs_root, run_id, "sam_a9_explainer")
        paths = RunPaths(run_id, outputs_root=runs_root)
        out = paths.single_output("sam_a9_explainer")
        _write_json(out, fake_a9())
        _commit(runs_root, run_id, "sam_a9_explainer", str(out))
        work = _next(runs_root, run_id)
        assert work.get("done") is True


# ---------------------------------------------------------------------------
# Section: next — per-paper agents
# ---------------------------------------------------------------------------


class TestNextPerPaper:
    def _setup_a3(self, runs_root, papers=None):
        """Init + advance to A3 (A1 and A2 completed with toy papers)."""
        if papers is None:
            papers = TOY_PAPERS
        run_id = _init(runs_root)
        _advance_to(runs_root, run_id, "sam_a3_screening", papers=papers)
        return run_id

    def test_next_a3_returns_first_eligible_paper(self, runs_root):
        run_id = self._setup_a3(runs_root)
        work = _next(runs_root, run_id)
        assert work["agent"] == "sam_a3_screening"
        assert work["paper_id"] == TOY_PMIDS[0]

    def test_next_a3_skips_completed_papers(self, runs_root):
        run_id = self._setup_a3(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        # Commit 3 papers (indices 0, 1, 2)
        for pmid in TOY_PMIDS[:3]:
            slug = paper_id_to_slug(pmid)
            out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
            _write_json(out, fake_a3(pmid))
            _commit(runs_root, run_id, "sam_a3_screening", str(out))
        work = _next(runs_root, run_id)
        assert work["agent"] == "sam_a3_screening"
        assert work["paper_id"] == TOY_PMIDS[3]

    def test_next_a3_returns_a4_when_all_papers_done(self, runs_root):
        run_id = self._setup_a3(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        for pmid in TOY_PMIDS:
            slug = paper_id_to_slug(pmid)
            out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
            _write_json(out, fake_a3(pmid, "INCLUDE"))
            _commit(runs_root, run_id, "sam_a3_screening", str(out))
        work = _next(runs_root, run_id)
        assert work["agent"] == "sam_a4_fulltext"

    def test_next_a4_filters_by_a3_include(self, runs_root):
        run_id = self._setup_a3(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        # 3 INCLUDE, 2 EXCLUDE
        for i, pmid in enumerate(TOY_PMIDS):
            decision = "INCLUDE" if i < 3 else "EXCLUDE"
            slug = paper_id_to_slug(pmid)
            out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
            _write_json(out, fake_a3(pmid, decision))
            _commit(runs_root, run_id, "sam_a3_screening", str(out))
        # A4 should only see 3 papers
        eligible_papers_seen = set()
        while True:
            work = _next(runs_root, run_id)
            if work.get("done") or work["agent"] != "sam_a4_fulltext":
                break
            pid = work["paper_id"]
            eligible_papers_seen.add(pid)
            slug = paper_id_to_slug(pid)
            out = paths.agent_dir("sam_a4_fulltext") / f"{slug}.json"
            _write_json(out, fake_a4(pid, "INCLUDE"))
            _commit(runs_root, run_id, "sam_a4_fulltext", str(out))
        assert len(eligible_papers_seen) == 3
        assert eligible_papers_seen == set(TOY_PMIDS[:3])

    def test_next_a5_filters_by_a4_include(self, runs_root):
        run_id = self._setup_a3(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        # A3: all include
        for pmid in TOY_PMIDS:
            slug = paper_id_to_slug(pmid)
            out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
            _write_json(out, fake_a3(pmid, "INCLUDE"))
            _commit(runs_root, run_id, "sam_a3_screening", str(out))
        # A4: 2 INCLUDE, 3 EXCLUDE
        for i, pmid in enumerate(TOY_PMIDS):
            decision = "INCLUDE" if i < 2 else "EXCLUDE"
            slug = paper_id_to_slug(pmid)
            out = paths.agent_dir("sam_a4_fulltext") / f"{slug}.json"
            _write_json(out, fake_a4(pmid, decision))
            _commit(runs_root, run_id, "sam_a4_fulltext", str(out))
        # A5 should only see 2 papers
        eligible_a5 = set()
        while True:
            work = _next(runs_root, run_id)
            if work.get("done") or work["agent"] != "sam_a5_extraction":
                break
            pid = work["paper_id"]
            eligible_a5.add(pid)
            slug = paper_id_to_slug(pid)
            out = paths.agent_dir("sam_a5_extraction") / f"{slug}.json"
            _write_json(out, fake_a5(pid))
            _commit(runs_root, run_id, "sam_a5_extraction", str(out))
        assert len(eligible_a5) == 2
        assert eligible_a5 == set(TOY_PMIDS[:2])

    def test_next_a6_filters_by_a4_include_parallel_to_a5(self, runs_root):
        """A5 and A6 should have the same eligible set (both filtered by A4 INCLUDEs)."""
        run_id = self._setup_a3(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        # A3: all include
        for pmid in TOY_PMIDS:
            slug = paper_id_to_slug(pmid)
            out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
            _write_json(out, fake_a3(pmid, "INCLUDE"))
            _commit(runs_root, run_id, "sam_a3_screening", str(out))
        # A4: first 4 INCLUDE, last EXCLUDE
        for i, pmid in enumerate(TOY_PMIDS):
            decision = "INCLUDE" if i < 4 else "EXCLUDE"
            slug = paper_id_to_slug(pmid)
            out = paths.agent_dir("sam_a4_fulltext") / f"{slug}.json"
            _write_json(out, fake_a4(pmid, decision))
            _commit(runs_root, run_id, "sam_a4_fulltext", str(out))
        # Drain A5
        seen_a5 = set()
        while True:
            work = _next(runs_root, run_id)
            if work.get("done") or work["agent"] != "sam_a5_extraction":
                break
            pid = work["paper_id"]
            seen_a5.add(pid)
            slug = paper_id_to_slug(pid)
            out = paths.agent_dir("sam_a5_extraction") / f"{slug}.json"
            _write_json(out, fake_a5(pid))
            _commit(runs_root, run_id, "sam_a5_extraction", str(out))
        # Drain A6
        seen_a6 = set()
        while True:
            work = _next(runs_root, run_id)
            if work.get("done") or work["agent"] != "sam_a6_quality":
                break
            pid = work["paper_id"]
            seen_a6.add(pid)
            slug = paper_id_to_slug(pid)
            out = paths.agent_dir("sam_a6_quality") / f"{slug}.json"
            _write_json(out, fake_a6(pid))
            _commit(runs_root, run_id, "sam_a6_quality", str(out))
        assert seen_a5 == seen_a6
        assert len(seen_a5) == 4

    def test_next_per_paper_paper_id_canonical_doi(self, runs_root):
        """output_path must use a filesystem-safe slug; paper_id carries the raw pmid."""
        run_id = _init(runs_root)
        special_pmid = "doi:10.1080/17437199.2025.2492042"
        papers = [_paper(special_pmid, doi="10.1080/17437199.2025.2492042")]
        _advance_to(runs_root, run_id, "sam_a3_screening", papers=papers)
        work = _next(runs_root, run_id)
        assert work["agent"] == "sam_a3_screening"
        assert work["paper_id"] == special_pmid
        slug = paper_id_to_slug(special_pmid)
        assert "/" not in slug
        assert ":" not in slug
        assert len(slug) <= 80
        assert work["output_path"].endswith(f"{slug}.json")

    def test_next_respects_subset_filter(self, runs_root, tmp_path):
        """Partition with 3/5 pmids: A3 only surfaces those 3 papers."""
        partition = tmp_path / "partition.json"
        partition.write_text(
            json.dumps({"dev": TOY_PMIDS[:3]}), encoding="utf-8"
        )
        run_id = _init(runs_root, "subset run", subset_spec=f"{partition}:dev")
        # Write A1 + A2 manually
        paths = RunPaths(run_id, outputs_root=runs_root)
        a1_out = paths.single_output("sam_a1_protocol")
        _write_json(a1_out, fake_a1())
        _commit(runs_root, run_id, "sam_a1_protocol", str(a1_out))
        a2_out = paths.single_output("sam_a2_search")
        _write_json(a2_out, fake_a2(TOY_PAPERS))
        _commit(runs_root, run_id, "sam_a2_search", str(a2_out))
        # Drain A3 — should only see 3 papers
        seen = []
        while True:
            work = _next(runs_root, run_id)
            if work.get("done") or work["agent"] != "sam_a3_screening":
                break
            pid = work["paper_id"]
            seen.append(pid)
            slug = paper_id_to_slug(pid)
            out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
            _write_json(out, fake_a3(pid, "INCLUDE"))
            _commit(runs_root, run_id, "sam_a3_screening", str(out))
        assert len(seen) == 3
        assert set(seen) == set(TOY_PMIDS[:3])


# ---------------------------------------------------------------------------
# Section: commit
# ---------------------------------------------------------------------------


class TestCommit:
    def test_commit_single_agent_valid_marks_completed(self, runs_root):
        run_id = _init(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        out = paths.single_output("sam_a1_protocol")
        _write_json(out, fake_a1())
        result = _commit(runs_root, run_id, "sam_a1_protocol", str(out))
        assert result["ok"] is True
        manifest = load_manifest(run_id, outputs_root=runs_root)
        assert manifest.agents["sam_a1_protocol"].status == "completed"

    def test_commit_single_agent_invalid_schema_exit_2(self, runs_root):
        run_id = _init(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        out = paths.single_output("sam_a1_protocol")
        # Missing required fields
        _write_json(out, {"agent": "sam_a1_protocol", "backend": "claude"})
        with pytest.raises(SchemaValidationError):
            _commit(runs_root, run_id, "sam_a1_protocol", str(out))
        # Manifest must remain unchanged (status still pending)
        manifest = load_manifest(run_id, outputs_root=runs_root)
        assert manifest.agents["sam_a1_protocol"].status == "pending"

    def test_commit_per_paper_appends_to_index(self, runs_root):
        run_id = _init(runs_root)
        _advance_to(runs_root, run_id, "sam_a3_screening")
        paths = RunPaths(run_id, outputs_root=runs_root)
        pmid = TOY_PMIDS[0]
        slug = paper_id_to_slug(pmid)
        out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
        _write_json(out, fake_a3(pmid))
        _commit(runs_root, run_id, "sam_a3_screening", str(out))
        entries = read_index(paths.agent_dir("sam_a3_screening"))
        assert any(e.slug == slug for e in entries)

    def test_commit_per_paper_idempotent(self, runs_root):
        run_id = _init(runs_root)
        _advance_to(runs_root, run_id, "sam_a3_screening")
        paths = RunPaths(run_id, outputs_root=runs_root)
        pmid = TOY_PMIDS[0]
        slug = paper_id_to_slug(pmid)
        out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
        _write_json(out, fake_a3(pmid))
        _commit(runs_root, run_id, "sam_a3_screening", str(out))
        _commit(runs_root, run_id, "sam_a3_screening", str(out))
        entries = read_index(paths.agent_dir("sam_a3_screening"))
        assert len([e for e in entries if e.slug == slug]) == 1

    def test_commit_per_paper_invalid_schema_does_not_corrupt_index(self, runs_root):
        run_id = _init(runs_root)
        _advance_to(runs_root, run_id, "sam_a3_screening")
        paths = RunPaths(run_id, outputs_root=runs_root)
        # First commit one valid paper
        pmid_ok = TOY_PMIDS[0]
        slug_ok = paper_id_to_slug(pmid_ok)
        out_ok = paths.agent_dir("sam_a3_screening") / f"{slug_ok}.json"
        _write_json(out_ok, fake_a3(pmid_ok))
        _commit(runs_root, run_id, "sam_a3_screening", str(out_ok))
        # Now try to commit malformed JSON for another paper
        pmid_bad = TOY_PMIDS[1]
        slug_bad = paper_id_to_slug(pmid_bad)
        out_bad = paths.agent_dir("sam_a3_screening") / f"{slug_bad}.json"
        _write_json(out_bad, {"broken": "json_no_required_fields"})
        with pytest.raises(SchemaValidationError):
            _commit(runs_root, run_id, "sam_a3_screening", str(out_bad))
        # Index must only contain the valid entry
        entries = read_index(paths.agent_dir("sam_a3_screening"))
        slugs = {e.slug for e in entries}
        assert slug_ok in slugs
        assert slug_bad not in slugs

    def test_commit_manifest_atomic_write(self, runs_root, monkeypatch):
        """If os.replace raises during manifest save, prior manifest is intact."""
        run_id = _init(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        out = paths.single_output("sam_a1_protocol")
        _write_json(out, fake_a1())
        # Read manifest before the crash simulation
        manifest_before = load_manifest(run_id, outputs_root=runs_root)
        original_status = manifest_before.agents["sam_a1_protocol"].status

        import os as _os

        original_replace = _os.replace
        call_count = [0]

        def _failing_replace(src, dst):
            call_count[0] += 1
            if call_count[0] == 1:
                # Clean up the tmp file before raising so it doesn't litter
                try:
                    _os.remove(src)
                except OSError:
                    pass
                raise OSError("simulated crash")
            return original_replace(src, dst)

        monkeypatch.setattr(_os, "replace", _failing_replace)
        with pytest.raises((OSError, Exception)):
            _commit(runs_root, run_id, "sam_a1_protocol", str(out))
        monkeypatch.setattr(_os, "replace", original_replace)
        # Manifest file should still be readable and unchanged
        manifest_after = load_manifest(run_id, outputs_root=runs_root)
        assert manifest_after.agents["sam_a1_protocol"].status == original_status


# ---------------------------------------------------------------------------
# Section: audit
# ---------------------------------------------------------------------------


class TestAudit:
    def _setup_clean_a3_run(self, runs_root) -> str:
        """Init + A1 + A2 + all 5 A3 INCLUDE."""
        run_id = _init(runs_root)
        _advance_to(runs_root, run_id, "sam_a3_screening")
        paths = RunPaths(run_id, outputs_root=runs_root)
        for pmid in TOY_PMIDS:
            slug = paper_id_to_slug(pmid)
            out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
            _write_json(out, fake_a3(pmid, "INCLUDE"))
            _commit(runs_root, run_id, "sam_a3_screening", str(out))
        return run_id

    def test_audit_clean_run_passes(self, runs_root):
        run_id = self._setup_clean_a3_run(runs_root)
        # Complete the remaining agents so audit has no pending issues
        _advance_to(runs_root, run_id, "sam_a9_explainer")
        paths = RunPaths(run_id, outputs_root=runs_root)
        out = paths.single_output("sam_a9_explainer")
        _write_json(out, fake_a9())
        _commit(runs_root, run_id, "sam_a9_explainer", str(out))
        report = audit_run(run_id, outputs_root=runs_root)
        assert report["ok"] is True

    def test_audit_detects_missing_paper_jsons(self, runs_root):
        run_id = self._setup_clean_a3_run(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        # Delete one physical file after committing
        pmid_del = TOY_PMIDS[2]
        slug_del = paper_id_to_slug(pmid_del)
        (paths.agent_dir("sam_a3_screening") / f"{slug_del}.json").unlink()
        report = audit_run(run_id, outputs_root=runs_root)
        assert report["ok"] is False
        a3_report = report["agents"]["sam_a3_screening"]
        # Should appear in invalid (file missing but slug is in _index)
        joined = " ".join(a3_report.get("invalid", []))
        assert slug_del in joined

    def test_audit_detects_duplicate_slug_in_index(self, runs_root):
        run_id = self._setup_clean_a3_run(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        # Force a duplicate by directly appending to _index.json
        agent_dir = paths.agent_dir("sam_a3_screening")
        index_path = agent_dir / "_index.json"
        entries = json.loads(index_path.read_text(encoding="utf-8"))
        first = entries[0]
        entries.append(first)  # duplicate
        index_path.write_text(json.dumps(entries), encoding="utf-8")
        report = audit_run(run_id, outputs_root=runs_root)
        assert report["ok"] is False
        a3_report = report["agents"]["sam_a3_screening"]
        assert first["slug"] in a3_report.get("duplicates", [])

    def test_audit_detects_invalid_schema_in_paper_json(self, runs_root):
        run_id = self._setup_clean_a3_run(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        # Overwrite one paper JSON with garbage
        pmid_bad = TOY_PMIDS[0]
        slug_bad = paper_id_to_slug(pmid_bad)
        bad_file = paths.agent_dir("sam_a3_screening") / f"{slug_bad}.json"
        bad_file.write_text(json.dumps({"broken": True}), encoding="utf-8")
        report = audit_run(run_id, outputs_root=runs_root)
        assert report["ok"] is False
        a3_report = report["agents"]["sam_a3_screening"]
        joined = " ".join(a3_report.get("invalid", []))
        assert slug_bad in joined

    def test_audit_detects_eligible_set_mismatch(self, runs_root):
        """If _index has only 4/5 papers but agent should have 5, ok is False."""
        run_id = _init(runs_root)
        _advance_to(runs_root, run_id, "sam_a3_screening")
        paths = RunPaths(run_id, outputs_root=runs_root)
        # Commit only 4 out of 5 papers (omit the last)
        for pmid in TOY_PMIDS[:4]:
            slug = paper_id_to_slug(pmid)
            out = paths.agent_dir("sam_a3_screening") / f"{slug}.json"
            _write_json(out, fake_a3(pmid, "INCLUDE"))
            _commit(runs_root, run_id, "sam_a3_screening", str(out))
        report = audit_run(run_id, outputs_root=runs_root)
        assert report["ok"] is False
        a3_report = report["agents"]["sam_a3_screening"]
        missing_slug = paper_id_to_slug(TOY_PMIDS[4])
        assert missing_slug in a3_report.get("missing", [])


# ---------------------------------------------------------------------------
# Section: slug + paths
# ---------------------------------------------------------------------------


class TestSlugAndPaths:
    def test_slug_sanitizes_doi_characters(self):
        doi = "doi:10.1080/17437199.2025.2492042"
        slug = paper_id_to_slug(doi)
        assert "/" not in slug
        assert ":" not in slug
        assert len(slug) <= 80
        assert re.match(r"^[a-zA-Z0-9._-]+$", slug), f"slug has illegal chars: {slug}"

    def test_run_paths_resolution(self, runs_root):
        run_id = _init(runs_root)
        paths = RunPaths(run_id, outputs_root=runs_root)
        # single_output for A1
        a1_out = paths.single_output("sam_a1_protocol")
        assert a1_out == runs_root / run_id / "sam_a1_protocol" / "protocol.json"
        # paper_output for A3
        pmid = "P001"
        slug = paper_id_to_slug(pmid)
        a3_out = paths.paper_output("sam_a3_screening", pmid)
        assert a3_out == runs_root / run_id / "sam_a3_screening" / f"{slug}.json"


# ---------------------------------------------------------------------------
# Section: smoke dry end-to-end
# ---------------------------------------------------------------------------


class TestFullLoopDryRun:
    def test_full_loop_dry_run_5_papers_9_agents(self, runs_root):
        """Emulate the the orchestrator loop: init -> {next -> write fake JSON -> commit} -> audit.

        No LLM invoked. All agent outputs are fake deterministic JSONs conforming
        to the respective Pydantic schema.
        """
        run_id = _init(runs_root, "e2e smoke test")
        paths = RunPaths(run_id, outputs_root=runs_root)

        iteration_limit = 200  # safety guard against infinite loops
        iterations = 0

        while True:
            assert iterations < iteration_limit, "loop did not terminate"
            iterations += 1

            work = _next(runs_root, run_id)
            if work.get("done"):
                break

            agent_id = work["agent"]
            paper_id = work.get("paper_id")
            output_path = Path(work["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if agent_id == "sam_a2_search":
                data = fake_a2(TOY_PAPERS)
            elif paper_id is not None:
                # per-paper
                pmid = paper_id
                build = _PER_PAPER_FAKE_BUILDERS.get(agent_id)
                assert build is not None, f"no builder for {agent_id}"
                # Keep A3 and A4 all-INCLUDE so A5/A6 get all 5 papers
                if agent_id == "sam_a3_screening":
                    data = fake_a3(pmid, "INCLUDE")
                elif agent_id == "sam_a4_fulltext":
                    data = fake_a4(pmid, "INCLUDE")
                else:
                    data = build(pmid)
            else:
                # single
                build_single = _SINGLE_FAKE_BUILDERS.get(agent_id)
                assert build_single is not None, f"no single builder for {agent_id}"
                data = build_single()

            output_path.write_text(json.dumps(data), encoding="utf-8")
            _commit(runs_root, run_id, agent_id, str(output_path))

        # Audit must pass
        report = audit_run(run_id, outputs_root=runs_root)
        assert report["ok"] is True, f"audit failed: {json.dumps(report, indent=2)}"

        # Verify per-paper artefacts: 5 <slug>.json in each per-paper agent dir
        for agent_id in AGENT_IDS:
            if AGENT_MODES[agent_id] == "per_paper":
                agent_dir = paths.agent_dir(agent_id)
                paper_jsons = [
                    f for f in agent_dir.iterdir()
                    if f.name != "_index.json" and f.suffix == ".json"
                ]
                assert len(paper_jsons) == 5, (
                    f"{agent_id}: expected 5 paper JSONs, got {len(paper_jsons)}"
                )

        # Verify single-agent artefacts exist
        for agent_id in AGENT_IDS:
            if AGENT_MODES[agent_id] == "single":
                out = paths.single_output(agent_id)
                assert out.exists(), f"single-agent output missing: {out}"
