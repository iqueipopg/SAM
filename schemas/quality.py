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

from pydantic import BaseModel, ConfigDict, Field

from .common import AgentResultBase
from .extraction import Outcome, StudyCharacteristics

StudyDesignDetected = Literal["rct", "observational", "qualitative"]
QualityTool = Literal["RoB2", "ROBINS-I", "JBI"]


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
    """Per-paper output for sam_a6_quality/<slug>.json."""

    paper_id: str = Field(
        ..., description="Canonical paper_id (Covidence internal ID, e.g. '#1989')."
    )

    pmid: Optional[str] = None
    title: str
    study_design_detected: StudyDesignDetected
    quality_assessment: QualityAssessment
    quality_weight: float = Field(..., ge=0.0, le=1.0)

    # Propagated from A5
    study_characteristics: Optional[StudyCharacteristics] = None
    outcomes: List[Outcome] = Field(default_factory=list)


__all__ = [
    "StudyDesignDetected",
    "QualityTool",
    "DomainJudgment",
    "QualityAssessment",
    "QualityOutput",
]
