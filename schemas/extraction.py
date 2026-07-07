# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""A5 extraction schema (per-paper) — aligned to Qwen real output.

Qwen-real fields: pmid, title, authors, journal, year, study_characteristics
(nested), intervention_exposure (nested — NOT flat `intervention`), outcomes
(list), extraction_quality, extraction_notes, text_source, sections_used,
extraction_confidence, concordance.

Numeric fields tolerate None. effect_size_type is permissive Literal extensible
via Outcome.model_config extra='ignore' on subkeys we don't enumerate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import AgentResultBase

ExtractionQuality = Literal["complete", "partial", "failed"]
ExtractionConfidence = Literal["high", "medium", "low"]
EffectSizeType = Literal["MD", "SMD", "OR", "RR", "HR", "PROP", "OTHER"]


class NGroups(BaseModel):
    """Sample-size breakdown per arm."""

    model_config = ConfigDict(extra="forbid")

    intervention: Optional[int] = None
    control: Optional[int] = None


class StudyCharacteristics(BaseModel):
    """Population & design fields extracted from the paper."""

    model_config = ConfigDict(extra="forbid")

    study_design: Optional[str] = Field(
        default=None, description="'RCT' | 'Cohort' | etc."
    )
    sample_size: Optional[int] = None
    n_groups: Optional[NGroups] = None
    age_mean: Optional[float] = None
    age_sd: Optional[float] = None
    female_pct: Optional[float] = None
    setting: Optional[str] = None
    country: Optional[str] = None
    followup_duration: Optional[str] = None


class InterventionExposure(BaseModel):
    """Nested intervention/exposure block (Qwen-real, NOT flat `intervention`)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    comparator: str
    dose_duration: str


class Outcome(BaseModel):
    """Single outcome record.

    extra='allow' because Qwen emits ad-hoc fields per outcome
    (n_intervention, n_control, mean_diff, etc.) and we don't want
    to over-constrain extraction.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    effect_size_type: EffectSizeType = Field(
        default="OTHER",
        description="Canonical effect-size code; defaults to OTHER if unknown.",
    )
    effect_size_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[float] = None


class Concordance(BaseModel):
    """Pass-1 vs pass-2 concordance metrics from extraction QA."""

    model_config = ConfigDict(extra="forbid")

    sample_size_match: bool
    outcomes_matched: int
    outcomes_discrepant: int
    outcomes_pass1_only: int
    outcomes_pass2_only: int
    flags: List[str]


class ExtractionOutput(AgentResultBase):
    """Per-paper output for sam_a5_extraction/<slug>.json."""

    paper_id: str = Field(
        ..., description="Canonical paper_id (Covidence internal ID, e.g. '#1989')."
    )

    pmid: Optional[str] = None
    title: str
    authors: List[str]
    journal: str
    year: str

    study_characteristics: StudyCharacteristics
    intervention_exposure: InterventionExposure
    outcomes: List[Outcome]

    extraction_quality: ExtractionQuality
    extraction_notes: str
    text_source: str
    sections_used: List[str]
    extraction_confidence: ExtractionConfidence
    concordance: Concordance


__all__ = [
    "ExtractionQuality",
    "ExtractionConfidence",
    "EffectSizeType",
    "NGroups",
    "StudyCharacteristics",
    "InterventionExposure",
    "Outcome",
    "Concordance",
    "ExtractionOutput",
]
