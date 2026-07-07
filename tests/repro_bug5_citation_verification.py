# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Reproduction + validation script for BUG5.

Loads outputs/run_20260421_102754/pipeline.json, extracts the already-generated
manuscript text and the included_papers metadata, and runs ONLY the citation
verification code paths (no LLM, no RAG) by bypassing __init__.

Prints a before/after style report:
  - How many citations the verifier finds in the real (degenerate) manuscript.
  - How many in a synthetic well-formed manuscript.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent8_manuscript import Agent8Manuscript


def _new_a8_no_init() -> Agent8Manuscript:
    a8 = Agent8Manuscript.__new__(Agent8Manuscript)
    a8._faiss_index = None
    a8._paper_lookup = {}
    a8._references = []
    a8._cited_pmids = set()
    a8._citation_index = {}
    a8._verification_reports = {}
    a8._language = "en"
    return a8


def _build_paper_lookup(a8: Agent8Manuscript, papers):
    """Replicate the paper_lookup construction from _build_rag_corpus."""
    a8._paper_lookup = {}
    for p in papers:
        pmid = str(p.get("pmid", ""))
        authors = p.get("authors", []) or []
        first_author = "Unknown"
        if authors:
            a0 = authors[0]
            first_author = a0.split(",")[0].split()[-1] if a0 else "Unknown"
        year = p.get("year", "")
        citation = f"{first_author} et al., {year}" if year else f"{first_author} et al."
        # Key by pmid when present; otherwise fall back to title-based synthetic key
        key = pmid or f"nopmid::{(p.get('title') or '')[:80]}"
        a8._paper_lookup[key] = {
            "pmid": pmid,
            "citation": citation,
            "title": p.get("title", ""),
            "journal": p.get("journal", ""),
            "year": year,
        }


def reproduce_real_run():
    print("=" * 72)
    print("CASE A — real run outputs/run_20260421_102754/")
    print("=" * 72)
    path = "~/meta-analisis-automatizado/outputs/run_20260421_102754/pipeline.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data["fulltext_retrieval"].get("include", []) or []
    manuscript = data["manuscript"]["manuscript"]
    print(f"  included_papers: {len(papers)}")
    if papers:
        p0 = papers[0]
        print(f"  sample paper: pmid={p0.get('pmid')!r} year={p0.get('year')!r} authors={p0.get('authors', [])[:3]}")

    a8 = _new_a8_no_init()
    _build_paper_lookup(a8, papers)
    a8._citation_index = a8._build_citation_index(papers)
    print(f"  citation_index entries: {len(a8._citation_index)}")

    references_list = list(a8._paper_lookup.values())

    total_found = 0
    total_verified = 0
    total_removed = 0
    for sec, text in manuscript.items():
        if not text:
            continue
        _, report = a8._verify_citations(text, references_list)
        total_found += report["total_citations"]
        total_verified += report["verified"]
        total_removed += report["removed"]
        print(
            f"    {sec:12s}: total={report['total_citations']:>3}  "
            f"verified={report['verified']:>3}  removed={report['removed']:>3}"
        )
    print(f"  TOTAL: found={total_found}  verified={total_verified}  removed={total_removed}")
    return total_verified


def synthetic_well_formed():
    print()
    print("=" * 72)
    print("CASE B — synthetic manuscript with well-formed (Author, Year) citations")
    print("=" * 72)
    papers = [
        {"pmid": "111", "authors": ["Smith, John", "Doe, Jane"], "year": "2020", "title": "Paper A"},
        {"pmid": "222", "authors": ["Jones, Alice"], "year": "2021", "title": "Paper B"},
        {"pmid": "333", "authors": ["Peter, X", "Srivastava, Y"], "year": "", "title": "Paper C (no year)"},
    ]
    text = (
        "Mindfulness reduced anxiety (Smith et al., 2020). "
        "Other work (Jones, 2021) agreed. "
        "A more recent synthesis (Peter et al.) reached similar conclusions. "
        "A fake cite (Ghost, 1999) should be removed. "
        "Also a phantom bracket [Impostor, 2005] should go."
    )

    a8 = _new_a8_no_init()
    _build_paper_lookup(a8, papers)
    a8._citation_index = a8._build_citation_index(papers)
    print(f"  citation_index entries: {len(a8._citation_index)}")

    refs = list(a8._paper_lookup.values())
    cleaned, report = a8._verify_citations(text, refs)
    print(f"  Input : {text}")
    print(f"  Output: {cleaned}")
    print(f"  report: total={report['total_citations']}  verified={report['verified']}  removed={report['removed']}")
    for d in report["details"]:
        print(f"    - {d}")
    return report["verified"], report["removed"]


def yearless_degenerate_llm_output():
    print()
    print("=" * 72)
    print("CASE C — degenerate LLM output seen in the real run")
    print("=" * 72)
    # Papers with missing year (like the real run)
    papers = [
        {"pmid": "", "authors": ["Peter", "Srivastava"], "year": "", "title": "Paper A"},
        {"pmid": "", "authors": ["Tan", "Martin"], "year": "", "title": "Paper B"},
    ]
    text = (
        "Results used mindfulness (Peter et al.), showing small improvement. "
        "Other approaches like (Tan & Martin) varied widely."
    )

    a8 = _new_a8_no_init()
    _build_paper_lookup(a8, papers)
    a8._citation_index = a8._build_citation_index(papers)
    print(f"  citation_index entries: {len(a8._citation_index)}")

    refs = list(a8._paper_lookup.values())
    cleaned, report = a8._verify_citations(text, refs)
    print(f"  Input : {text}")
    print(f"  Output: {cleaned}")
    print(f"  report: total={report['total_citations']}  verified={report['verified']}  removed={report['removed']}")
    for d in report["details"]:
        print(f"    - {d}")
    return report["verified"]


if __name__ == "__main__":
    a_verified = reproduce_real_run()
    b_verified, b_removed = synthetic_well_formed()
    c_verified = yearless_degenerate_llm_output()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  CASE A (real run)       : verified={a_verified}  (was 0 before fix)")
    print(f"  CASE B (synthetic wellformed): verified={b_verified}  removed={b_removed}")
    print(f"  CASE C (yearless degenerate): verified={c_verified}")

    # Acceptance: at least one case must show verified > 0 for the synthetic well-formed.
    assert b_verified >= 3, f"Expected >=3 verified well-formed citations, got {b_verified}"
    assert b_removed >= 2, f"Expected >=2 removed phantom citations, got {b_removed}"
    assert c_verified >= 2, f"Expected >=2 verified yearless citations, got {c_verified}"
    print("\n[OK] All assertions passed.")
