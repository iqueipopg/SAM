# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""A3 screening schema (per-paper) — aligned to Qwen real output.

Qwen-real fields: pmid, title, decision, confidence, votes, reasoning,
flags, screening_confidence, borderline.

Divergencia vs spec original: spec proponía `reason` y `criteria_evaluated`;
Qwen usa `reasoning` y `flags: list[str]`. Alineamos a Qwen.

D4 policy (T17.C.1): `paper_id` is the canonical Covidence ID (e.g. "#1989");
`pmid` is optional metadata.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import AgentResultBase

ScreeningDecision = Literal["INCLUDE", "EXCLUDE", "UNCERTAIN"]


class VoteEntry(BaseModel):
    """One vote inside an ensemble decision (per-vote rationale)."""

    model_config = ConfigDict(extra="forbid")

    decision: ScreeningDecision
    reasoning: str
    model: str = Field(..., description="Vote-emitting model identifier (e.g. 'default').")


class ScreeningOutput(AgentResultBase):
    """Per-paper output for sam_a3_screening/<slug>.json."""

    paper_id: str = Field(
        ..., description="Canonical paper_id (Covidence internal ID, e.g. '#1989')."
    )
    pmid: Optional[str] = None
    title: str
    decision: ScreeningDecision
    confidence: float = Field(..., ge=0.0, le=1.0)
    votes: List[VoteEntry]
    reasoning: str
    flags: List[str]
    screening_confidence: float = Field(..., ge=0.0, le=1.0)
    borderline: bool
