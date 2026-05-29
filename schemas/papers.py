"""A2 search schema — aligned to Qwen real output.

Per mapping doc:
- Paper: pmid, doi, title, abstract, authors, journal, year (str), sources (list[str]).
- SearchStats: 8 fields (pubmed, europepmc, semantic_scholar, google_scholar,
  total_before_dedup, total_after_dedup, duplicates_removed, search_sources).
- pmid is the canonical paper_id used for slugs.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from .common import AgentResultBase


class Paper(BaseModel):
    """Single paper record from A2 search (Qwen-real layout)."""

    model_config = ConfigDict(extra="forbid")

    pmid: str = Field(..., description="PubMed ID; canonical paper_id for slugs.")
    doi: str = Field(..., description="DOI string. Empty string if unavailable.")
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: str = Field(..., description="Publication year as string (Qwen-real).")
    sources: List[str] = Field(
        ..., description="Databases that returned this paper (e.g. ['pubmed','europepmc'])."
    )


class SearchStats(BaseModel):
    """Aggregated statistics from A2 search."""

    model_config = ConfigDict(extra="forbid")

    pubmed: int
    europepmc: int
    semantic_scholar: int
    google_scholar: int
    total_before_dedup: int
    total_after_dedup: int
    duplicates_removed: int
    search_sources: List[str]


class PapersRawOutput(AgentResultBase):
    """Top-level wrapper for sam_a2_search/papers_raw.json."""

    papers: List[Paper]
    stats: SearchStats
