# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""A6 quality schema (per-paper) — aligned to Qwen real output.

Qwen-real: pmid, title, study_design_detected (lowercase enum),
quality_assessment {tool, domains, overall, overall_justification},
quality_weight, study_characteristics (propagated from A5), outcomes
(propagated from A5).

`domains` is a permissive dict (RoB2 = 5 domains, ROBINS-I = 7, JBI = 10),
each entry {judgment, justification}.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import AgentResultBase
from .extraction import Outcome, StudyCharacteristics

StudyDesignDetected = Literal["rct", "observational", "qualitative"]
QualityTool = Literal["RoB2", "ROBINS-I", "JBI"]
# Applicability of a risk-of-bias instrument to this study. "not_applicable"
# means the study is outside the domain of the intended instrument (e.g. a
# non-randomized study when the review scope is RoB2-only): no judgment is
# fabricated. Defaults to "applicable" so existing records are unchanged.
Applicability = Literal["applicable", "not_applicable"]


class DomainJudgment(BaseModel):
    """One domain rating inside `quality_assessment.domains`."""

    model_config = ConfigDict(extra="forbid")

    judgment: str = Field(
        ..., description="Tool-specific verdict (e.g. 'Low', 'Some concerns', 'High')."
    )
    justification: str


class QualityAssessment(BaseModel):
    """Risk-of-bias / appraisal block."""

    model_config = ConfigDict(extra="forbid")

    tool: QualityTool
    domains: Dict[str, DomainJudgment]
    overall: str = Field(
        ..., description="Overall judgment label (tool-specific vocabulary)."
    )
    overall_justification: str


class QualityOutput(AgentResultBase):
    """Per-paper output for sam_a6_quality/<slug>.json.

    Two shapes, discriminated by `applicability` and enforced by the validator
    below (single representation of each state — no incoherent record possible):

    - applicable     → carries study_design_detected + quality_assessment +
                       quality_weight (the RoB2/ROBINS-I/JBI verdict).
    - not_applicable → study is outside the instrument's domain; NO assessment or
                       weight is fabricated. study_design_category records why.
    """

    paper_id: str = Field(
        ..., description="Canonical paper_id (Covidence internal ID, e.g. '#1989')."
    )

    pmid: Optional[str] = None
    # Optional: bespoke corpora without a metadata layer (papers_raw.json) have no
    # title. Relaxed from required — never rejects a record that carried a title.
    title: Optional[str] = None

    # Applicability of the risk-of-bias instrument (default preserves old shape).
    applicability: Applicability = "applicable"
    # Closed-enum design category from the audited design table — the reason a
    # record is not_applicable (e.g. "single_group_pre_post"). Free-text sentence
    # in `applicability_reason`. Both None for applicable records.
    study_design_category: Optional[str] = None
    applicability_reason: Optional[str] = None

    # Present iff applicable (see validator). None for not_applicable records so
    # no verdict is invented.
    study_design_detected: Optional[StudyDesignDetected] = None
    quality_assessment: Optional[QualityAssessment] = None
    quality_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Propagated from A5
    study_characteristics: Optional[StudyCharacteristics] = None
    outcomes: List[Outcome] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_applicability_consistency(self) -> "QualityOutput":
        """Guarantee one coherent representation per state. Rejects both a
        fabricated NA (carrying an assessment) and an empty applicable record."""
        if self.applicability == "applicable":
            missing = [
                name
                for name, val in (
                    ("study_design_detected", self.study_design_detected),
                    ("quality_assessment", self.quality_assessment),
                    ("quality_weight", self.quality_weight),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    "applicable QualityOutput must carry "
                    + ", ".join(missing)
                )
        else:  # not_applicable
            if self.quality_assessment is not None or self.quality_weight is not None:
                raise ValueError(
                    "not_applicable QualityOutput must NOT carry "
                    "quality_assessment or quality_weight (no fabricated verdict)"
                )
        return self


__all__ = [
    "StudyDesignDetected",
    "QualityTool",
    "Applicability",
    "DomainJudgment",
    "QualityAssessment",
    "QualityOutput",
]
