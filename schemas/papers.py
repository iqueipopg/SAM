"""A2 search schema — aligned to Qwen real output.

Per mapping doc:
- Paper: pmid, doi, title, abstract, authors, journal, year (str), sources (list[str]).
- SearchStats: 8 fields (pubmed, europepmc, semantic_scholar, google_scholar,
  total_before_dedup, total_after_dedup, duplicates_removed, search_sources).
- paper_id = Covidence internal ID (e.g. '#1989'); D4 policy T17.C.1.
- pmid/doi are optional metadata; many gold-standard papers lack them.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import AgentResultBase


class Paper(BaseModel):
    """Single paper record from A2 search (Qwen-real layout)."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(
        ..., description="Covidence internal ID (e.g. '#1989'); canonical id for slugs."
    )
    pmid: Optional[str] = Field(
        default=None, description="PubMed ID if available; None otherwise."
    )
    doi: Optional[str] = Field(
        default=None, description="DOI string; None if unavailable."
    )
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
