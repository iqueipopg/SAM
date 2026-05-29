"""A1 protocol schema — aligned to Qwen real output.

Qwen real top-level keys: query, protocol, pubmed_query.
The 11 protocol fields are flat inside `protocol` (NOT a nested `pico` block).

Schema decision (per mapping doc):
- protocol: ProtocolBody with 11 flat fields.
- query: str (user question).
- pubmed_query: str (boolean PubMed query).
- version: Literal["V2"] = "V2" (spec-required, defaulted to anchor block C).
- search_date_cutoff / literal_quotes_source: Optional, not emitted by Qwen.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import AgentResultBase

Framework = Literal["PICO", "SPIDER", "PECO", "PCC"]


class ProtocolBody(BaseModel):
    """The 11 flat fields inside Qwen's `protocol` dict."""

    model_config = ConfigDict(extra="forbid")

    framework: Framework
    research_question: str
    population: str
    intervention: str
    comparator: str
    outcome: str
    original_query: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    protocol_document: str
    prospero_draft: str


class ProtocolOutput(AgentResultBase):
    """Top-level wrapper for sam_a1_protocol/protocol.json."""

    query: str = Field(..., description="User's clinical question (verbatim).")
    protocol: ProtocolBody
    pubmed_query: str = Field(..., description="Boolean PubMed query built from protocol.")

    version: Literal["V2"] = Field(
        default="V2",
        description="Protocol version locked for block C.",
    )
    search_date_cutoff: Optional[str] = Field(
        default=None,
        description="Optional ISO date cutoff for search (not emitted by Qwen).",
    )
    literal_quotes_source: Optional[str] = Field(
        default=None,
        description="Optional pointer to source manuscript for literal quotes.",
    )
