"""A4 split V1 schemas (reporter + decider) — T17.C.1 block C.

The A4 split architecture (V1) decomposes the full-text screening decision into
two LLM steps run as separate subagents:

  - `sam_a4_reporter` emits a structured DESCRIPTION of the paper (8 V1 fields),
    reading the complete full text. It does NOT decide. -> ReporterOutput.
  - `sam_a4_decider` reads ONLY the reporter's JSON (never the paper) plus the
    protocol criteria, and emits a binary INCLUDE/EXCLUDE decision with a
    categorical confidence. -> DeciderOutput.

These models are validated by `scripts/run_phase4_dev.py` (the split driver);
they are intentionally NOT wired into `sam_orchestrator/commit.py::SCHEMA_BY_AGENT`.
The driver maps the decider output to a decision-only `FulltextOutput` payload
and commits it into the canonical `sam_a4_fulltext` slot, so the orchestrator's
9-agent pipeline definition stays untouched.

The 8 reporter fields mirror `AGENT4_REPORTER_TEMPLATE_V1` (`agents/prompts.py`)
and `REPORTER_REQUIRED_KEYS_V1`. `notes` defaults to "" to replicate the T17.B.3
schema fix (`notes` made optional) — see docs/iterations/T17.B.3_iteration_report.md §4.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import AgentResultBase

DeciderDecision = Literal["INCLUDE", "EXCLUDE"]
DeciderConfidence = Literal["high", "medium", "low"]


class ReporterOutput(AgentResultBase):
    """Per-paper output for sam_a4_reporter/<slug>.json (V1 description).

    Reporter describes; it does not decide. The decider sees ONLY these fields.
    """

    paper_id: str = Field(
        ..., description="Canonical paper_id (Covidence internal ID, e.g. '#1989')."
    )
    population: str
    intervention: str
    comparator: str
    outcomes: str
    study_design: str
    publication_type: str
    language: str
    notes: str = Field(
        default="",
        description="Optional notes (max 2 sentences). Defaults to '' (T17.B.3 fix).",
    )


class DeciderOutput(AgentResultBase):
    """Per-paper output for sam_a4_decider/<slug>.json (V1 decision).

    Decider works ONLY from the reporter's structured report + protocol criteria;
    it never reads the paper. `confidence` is categorical; the split driver maps
    high/medium/low -> 0.9/0.6/0.3 when assembling the FulltextOutput payload.
    """

    paper_id: str = Field(
        ..., description="Canonical paper_id (Covidence internal ID, e.g. '#1989')."
    )
    decision: DeciderDecision
    reason: str
    confidence: DeciderConfidence


__all__ = [
    "DeciderDecision",
    "DeciderConfidence",
    "ReporterOutput",
    "DeciderOutput",
]
