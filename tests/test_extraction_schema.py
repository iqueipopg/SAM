# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the A5 extraction schema refactor (``schemas/extraction.py``).

Covers the three new canonical structures added on top of the flat legacy
fields:
- ``Arm`` / ``StudyCharacteristics.arms`` — multi-arm trials are never
  collapsed into a single intervention/control pair (case #1586).
- ``Measurement`` — arm-linked, timepoint-tagged readings replace free-text
  PRE/POST fields (case #2735).
- ``DescriptiveStat`` + ``effect_size_type="none"`` — a bare single-arm mean
  is never forced into an MD comparison (case #1354).

No GPU or model loading involved; this is pure pydantic validation.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.extraction import (
    Arm,
    DescriptiveStat,
    Measurement,
    NGroups,
    Outcome,
    StudyCharacteristics,
)


# ---------------------------------------------------------------------------
# Arm / multi-arm trials (case #1586)
# ---------------------------------------------------------------------------


class TestArmMultiArm:
    def test_three_arms_are_preserved_not_collapsed(self):
        sc = StudyCharacteristics(
            study_design="RCT",
            sample_size=205,
            arms=[
                Arm(label="Imagine young", arm_type="intervention", n=59),
                Arm(label="Imagine old", arm_type="intervention", n=69),
                Arm(label="control", arm_type="control", n=77),
            ],
        )
        assert len(sc.arms) == 3
        assert [a.label for a in sc.arms] == [
            "Imagine young",
            "Imagine old",
            "control",
        ]

    def test_three_arms_n_groups_property_sums_by_arm_type(self):
        sc = StudyCharacteristics(
            arms=[
                Arm(label="Imagine young", arm_type="intervention", n=59),
                Arm(label="Imagine old", arm_type="intervention", n=69),
                Arm(label="control", arm_type="control", n=77),
            ],
        )
        ng = sc.n_groups
        assert isinstance(ng, NGroups)
        assert ng.intervention == 59 + 69
        assert ng.control == 77

    def test_legacy_n_groups_input_yields_two_arms(self):
        """Known legacy limitation: n_groups only ever carried 2 buckets."""
        sc = StudyCharacteristics(n_groups={"intervention": 50, "control": 50})
        assert len(sc.arms) == 2
        assert {a.arm_type for a in sc.arms} == {"intervention", "control"}
        assert sc.n_groups.intervention == 50
        assert sc.n_groups.control == 50

    def test_arms_direct_admits_more_than_two_arms_new_schema_only(self):
        """The new schema (unlike the legacy n_groups compat path) admits
        arbitrarily many arms when supplied directly via `arms`."""
        sc = StudyCharacteristics(
            arms=[
                Arm(label="low dose", arm_type="intervention", n=30),
                Arm(label="high dose", arm_type="intervention", n=30),
                Arm(label="placebo", arm_type="comparator", n=30),
                Arm(label="waitlist", arm_type="control", n=30),
            ],
        )
        assert len(sc.arms) == 4

    def test_arms_present_drops_stray_n_groups_key(self):
        sc = StudyCharacteristics(
            arms=[Arm(label="intervention", arm_type="intervention", n=10)],
            n_groups={"intervention": 999, "control": 999},
        )
        # arms wins; the legacy key is discarded, not merged.
        assert len(sc.arms) == 1
        assert sc.arms[0].n == 10

    def test_arm_forbids_unknown_fields(self):
        with pytest.raises(ValidationError):
            Arm(label="intervention", arm_type="intervention", extra_field="x")

    def test_arm_type_rejects_invalid_value(self):
        with pytest.raises(ValidationError):
            Arm(label="intervention", arm_type="placebo_arm")

    def test_study_characteristics_defaults_to_empty_arms(self):
        sc = StudyCharacteristics()
        assert sc.arms == []
        assert sc.n_groups.intervention is None
        assert sc.n_groups.control is None


# ---------------------------------------------------------------------------
# Measurement / temporal readings (case #2735)
# ---------------------------------------------------------------------------


class TestMeasurementTemporal:
    def test_baseline_and_post_measurements_validate(self):
        outcome = Outcome(
            name="Depression score",
            effect_size_type="MD",
            measurements=[
                Measurement(
                    arm_label="intervention",
                    timepoint="baseline",
                    mean=67.33,
                    sd=10.84,
                    n=84,
                ),
                Measurement(
                    arm_label="intervention",
                    timepoint="post",
                    mean=71.1,
                    sd=11.67,
                    n=84,
                ),
            ],
        )
        assert len(outcome.measurements) == 2
        by_timepoint = {m.timepoint: m for m in outcome.measurements}
        assert by_timepoint["baseline"].mean == 67.33
        assert by_timepoint["post"].mean == 71.1
        # These readings live in the canonical structure, not as ad-hoc extras.
        assert "pretest_mean" not in outcome.model_dump()
        assert "posttest_mean" not in outcome.model_dump()

    def test_measurement_forbids_unknown_fields(self):
        with pytest.raises(ValidationError):
            Measurement(
                arm_label="intervention",
                timepoint="post",
                mean=1.0,
                pretest_mean=2.0,
            )

    def test_measurement_timepoint_rejects_invalid_value(self):
        with pytest.raises(ValidationError):
            Measurement(arm_label="intervention", timepoint="midpoint", mean=1.0)

    def test_measurement_tolerates_missing_numerics(self):
        m = Measurement(arm_label="control", timepoint="followup")
        assert m.mean is None
        assert m.sd is None
        assert m.n is None

    def test_outcome_measurements_default_empty(self):
        outcome = Outcome(name="Anxiety score")
        assert outcome.measurements == []


# ---------------------------------------------------------------------------
# DescriptiveStat / single-arm descriptive (case #1354)
# ---------------------------------------------------------------------------


class TestDescriptiveStatSingleArm:
    def test_none_effect_size_with_descriptive_stat_validates(self):
        outcome = Outcome(
            name="Satisfaction",
            effect_size_type="none",
            descriptive_stats=[
                DescriptiveStat(
                    statistic="mean",
                    value=5.89,
                    scale="1-6",
                    arm_label="intervention",
                )
            ],
        )
        assert outcome.effect_size_type == "none"
        assert len(outcome.descriptive_stats) == 1
        assert outcome.descriptive_stats[0].value == 5.89
        assert outcome.descriptive_stats[0].scale == "1-6"

    def test_none_is_not_coerced_to_md(self):
        """A bare single-arm mean must NOT be silently upgraded to MD."""
        outcome = Outcome(
            name="Satisfaction",
            effect_size_type="none",
            descriptive_stats=[DescriptiveStat(statistic="mean", value=5.89)],
        )
        assert outcome.effect_size_type != "MD"
        assert outcome.effect_size_value is None

    def test_none_is_a_legal_effect_size_type(self):
        outcome = Outcome(name="Satisfaction", effect_size_type="none")
        assert outcome.effect_size_type == "none"

    def test_descriptive_stat_forbids_unknown_fields(self):
        with pytest.raises(ValidationError):
            DescriptiveStat(statistic="mean", value=1.0, effect_size_value=2.0)

    def test_descriptive_stat_requires_statistic_and_value(self):
        with pytest.raises(ValidationError):
            DescriptiveStat(scale="1-6")

    def test_outcome_descriptive_stats_default_empty(self):
        outcome = Outcome(name="Anxiety score")
        assert outcome.descriptive_stats == []

    def test_outcome_still_allows_flat_legacy_fields(self):
        """Outcome keeps extra='allow' so A5/A7 numeric machinery is untouched."""
        outcome = Outcome(
            name="Depression score",
            effect_size_type="SMD",
            effect_size_value=-0.5,
            ci_lower=-0.8,
            ci_upper=-0.2,
            p_value=0.01,
            n_intervention=50,
            n_control=50,
        )
        dumped = outcome.model_dump()
        assert dumped["n_intervention"] == 50
        assert dumped["n_control"] == 50
