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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import AgentResultBase

ExtractionQuality = Literal["complete", "partial", "failed"]
ExtractionConfidence = Literal["high", "medium", "low"]
# "none" flags outcomes that carry only single-arm descriptives (no comparison),
# so downstream analysis never mistakes a bare mean for a mean difference.
EffectSizeType = Literal["MD", "SMD", "OR", "RR", "HR", "PROP", "OTHER", "none"]

# Canonical arm role. A three-arm trial keeps three arms (no collapsing).
ArmType = Literal["intervention", "control", "comparator"]

# Canonical measurement timepoint (replaces free-text PRE/POST fields).
Timepoint = Literal["baseline", "post", "followup"]

# Closed-enum study design category (derived from the free-text study_design).
StudyDesignCategory = Literal[
    "RCT",
    "cluster_RCT",
    "quasi_experimental_controlled",
    "single_group_pre_post",
    "cross_sectional",
    "qualitative",
    "mixed_methods",
    "other",
]


class NGroups(BaseModel):
    """Sample-size breakdown per arm.

    Retained for backward compatibility. `StudyCharacteristics.n_groups` is now a
    derived property that aggregates `arms`; the source of truth is `arms`.
    """

    model_config = ConfigDict(extra="forbid")

    intervention: Optional[int] = None
    control: Optional[int] = None


class Arm(BaseModel):
    """A single study arm (source of truth for group structure)."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., description="Arm label, links to Measurement.arm_label.")
    arm_type: ArmType
    n: Optional[int] = None
    description: Optional[str] = None


class StudyCharacteristics(BaseModel):
    """Population & design fields extracted from the paper."""

    model_config = ConfigDict(extra="forbid")

    study_design: Optional[str] = Field(
        default=None, description="'RCT' | 'Cohort' | etc."
    )
    study_design_category: Optional[StudyDesignCategory] = Field(
        default=None,
        description="Closed-enum design category derived from study_design.",
    )
    sample_size: Optional[int] = None
    arms: List[Arm] = Field(
        default_factory=list,
        description="Study arms; one entry per arm (never collapse multi-arm trials).",
    )
    age_mean: Optional[float] = None
    age_sd: Optional[float] = None
    female_pct: Optional[float] = None
    setting: Optional[str] = None
    country: Optional[str] = None
    followup_duration: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_n_groups(cls, data: Any) -> Any:
        """Convert legacy `n_groups` input into `arms` before validation.

        `StudyCharacteristics` is extra='forbid', so a stray `n_groups` key would
        raise. If `arms` is absent and a legacy `n_groups` dict is present, build
        the equivalent Arm list; either way drop the `n_groups` key.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("arms"):
            data.pop("n_groups", None)
            return data
        ng = data.pop("n_groups", None)
        if isinstance(ng, dict):
            arms: List[Dict[str, Any]] = []
            if "intervention" in ng:
                arms.append(
                    {
                        "label": "intervention",
                        "arm_type": "intervention",
                        "n": ng.get("intervention"),
                    }
                )
            if "control" in ng:
                arms.append(
                    {
                        "label": "control",
                        "arm_type": "control",
                        "n": ng.get("control"),
                    }
                )
            if arms:
                data["arms"] = arms
        return data

    @property
    def n_groups(self) -> NGroups:
        """Derive a legacy NGroups view (sums n by arm_type) for old readers."""

        def _sum(arm_type: str) -> Optional[int]:
            vals = [a.n for a in self.arms if a.arm_type == arm_type and a.n is not None]
            return sum(vals) if vals else None

        return NGroups(intervention=_sum("intervention"), control=_sum("control"))


class InterventionExposure(BaseModel):
    """Nested intervention/exposure block (Qwen-real, NOT flat `intervention`)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    comparator: str
    dose_duration: str


class Measurement(BaseModel):
    """A raw mean/SD/N reading for one arm at one timepoint.

    Replaces free-text PRE/POST fields with a canonical, arm-linked structure.
    """

    model_config = ConfigDict(extra="forbid")

    arm_label: str = Field(..., description="Links to Arm.label.")
    timepoint: Timepoint
    mean: Optional[float] = None
    sd: Optional[float] = None
    n: Optional[int] = None


class DescriptiveStat(BaseModel):
    """A single-arm descriptive statistic (no between-group comparison).

    Used when an outcome reports only a bare mean/proportion for one arm; such
    outcomes carry effect_size_type="none" so they are never treated as an MD.
    """

    model_config = ConfigDict(extra="forbid")

    statistic: str = Field(..., description="e.g. 'mean', 'proportion', 'median'.")
    value: float
    scale: Optional[str] = None
    arm_label: Optional[str] = None


class Outcome(BaseModel):
    """Single outcome record.

    extra='allow' because Qwen emits ad-hoc fields per outcome
    (n_intervention, n_control, mean_diff, etc.) and we don't want
    to over-constrain extraction. The flat legacy fields are consumed by the
    A5 numeric machinery and A7, so they stay untouched; `measurements` and
    `descriptive_stats` are additive canonical structures.
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
    measurements: List[Measurement] = Field(
        default_factory=list,
        description="Arm-linked raw readings (mean/SD/N) per timepoint.",
    )
    descriptive_stats: List[DescriptiveStat] = Field(
        default_factory=list,
        description="Single-arm descriptives when there is no comparison.",
    )


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
    "ArmType",
    "Timepoint",
    "StudyDesignCategory",
    "NGroups",
    "Arm",
    "Measurement",
    "DescriptiveStat",
    "StudyCharacteristics",
    "InterventionExposure",
    "Outcome",
    "Concordance",
    "ExtractionOutput",
]
