"""A4 fulltext schema (per-paper) — aligned to Qwen real output.

CRITICAL: Qwen uses prefixed names — `fulltext_decision`, `fulltext_reasoning`,
`fulltext_votes`, `fulltext_confidence` — NOT generic `decision`/`reasoning`.
Schemas MUST match these literal names. Subagent prompts also enforce this.

Decision enum is binary: INCLUDE | EXCLUDE (no UNCERTAIN in A4).

`paper_id` mirrors `pmid` for orchestrator compatibility.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import AgentResultBase
from .screening import VoteEntry as ScreeningVoteEntry

FulltextDecision = Literal["INCLUDE", "EXCLUDE"]


class FulltextVoteEntry(BaseModel):
    """One vote in the A4 fulltext ensemble (uses binary decision)."""

    model_config = ConfigDict(extra="forbid")

    decision: FulltextDecision
    reasoning: str
    model: str


class FulltextOutput(AgentResultBase):
    """Per-paper output for sam_a4_fulltext/<slug>.json."""

    paper_id: str = Field(
        ..., description="Canonical paper_id (mirrors pmid)."
    )

    # Paper metadata (propagated from A2)
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: str
    doi: str
    pmcid: str = Field(..., description="PMC identifier (empty string if not available).")

    # Fulltext retrieval result
    full_text: str = Field(..., description="Extracted full text (may be empty).")
    text_source: str = Field(
        ..., description="Source key: 'abstract_only' | 'pmc' | 'unpaywall' | etc."
    )
    full_text_length: int
    sections: Dict[str, str] = Field(
        ..., description="Map of section name -> text. May be empty dict."
    )
    oa_url: Optional[str] = Field(
        default=None, description="Open-access URL if discovered; null otherwise."
    )

    # Decision fields (prefixed — Qwen literal)
    fulltext_decision: FulltextDecision
    fulltext_reasoning: str
    exclusion_reason: Optional[str] = Field(
        default=None,
        description="Free-text exclusion rationale; null when INCLUDE.",
    )
    fulltext_votes: List[FulltextVoteEntry]
    fulltext_confidence: float = Field(..., ge=0.0, le=1.0)
    screening_confidence: float = Field(..., ge=0.0, le=1.0)
    borderline: bool


__all__ = [
    "FulltextDecision",
    "FulltextVoteEntry",
    "FulltextOutput",
    "ScreeningVoteEntry",  # re-export convenience
]
