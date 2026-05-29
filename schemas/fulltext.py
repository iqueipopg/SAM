"""A4 fulltext schema (per-paper) — aligned to Qwen real output.

CRITICAL: Qwen uses prefixed names — `fulltext_decision`, `fulltext_reasoning`,
`fulltext_votes`, `fulltext_confidence` — NOT generic `decision`/`reasoning`.
Schemas MUST match these literal names. Subagent prompts also enforce this.

Decision enum is binary: INCLUDE | EXCLUDE (no UNCERTAIN in A4).

`paper_id` is the canonical Covidence internal ID (e.g. "#1989"). D4 policy
(T17.C.1): paper_id = Covidence ID; pmid/doi/pmcid are optional metadata
(many gold-standard papers lack them).

T17.C.1 deterministic I/O separation: `full_text` is intentionally NOT stored
in this schema. Rationale:
  - The LLM cannot reliably reproduce verbatim a ~60k-char full text nor
    self-count its `full_text_length`; making it emit those fields produced
    drift and silent corruption.
  - The Qwen-B per-paper A4 baseline carries NO `full_text` / `text_source` /
    `full_text_length`, so storing `full_text` would break B-parity.
  - The authoritative full text already lives at
    `outputs/<run_id>/_fulltext_cache/<slug>.txt`.
Provenance fields (`text_source`, `full_text_length`, `sections`, `oa_url`) and
all paper metadata are NOT emitted by the subagent: `commit` injects them
deterministically from `papers_raw.json` + `_fulltext_cache/`.
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
        ..., description="Canonical paper_id (Covidence internal ID, e.g. '#1989')."
    )

    # Paper metadata (propagated from A2). Identifiers other than paper_id are
    # optional per D4: many gold-standard papers lack pmid/doi/pmcid.
    pmid: Optional[str] = None
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: str
    doi: Optional[str] = None
    pmcid: Optional[str] = Field(
        default=None, description="PMC identifier (None if not available)."
    )

    # Fulltext retrieval provenance (injected by commit, NOT emitted by the LLM).
    # `full_text` is intentionally absent: see module docstring.
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
