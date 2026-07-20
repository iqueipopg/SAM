# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``scripts/migrate_extraction_schema.py``.

Covers the in-memory migration of a legacy A5 extraction document
(study_characteristics.n_groups + a single-arm MD outcome) into the
refactored schema (arms / effect_size_type="none" / descriptive_stats),
and idempotency of running the migration twice.

Pure Python, no GPU, no filesystem I/O (``migrate_document`` operates on
dicts in place; ``migrate_dir`` is the only function that touches disk and
is not exercised here).
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parent.parent / "scripts" / "migrate_extraction_schema.py"
)
_spec = importlib.util.spec_from_file_location(
    "migrate_extraction_schema", SCRIPT_PATH
)
migrate_extraction_schema = importlib.util.module_from_spec(_spec)
sys.modules["migrate_extraction_schema"] = migrate_extraction_schema
_spec.loader.exec_module(migrate_extraction_schema)

migrate_document = migrate_extraction_schema.migrate_document


def _legacy_doc() -> dict:
    """A legacy extraction doc: n_groups + a bare single-arm MD outcome."""
    return {
        "paper_id": "#1354",
        "study_characteristics": {
            "study_design": "RCT",
            "sample_size": 100,
            "n_groups": {"intervention": 50, "control": 50},
        },
        "outcomes": [
            {
                "name": "Satisfaction",
                "effect_size_type": "MD",
                "mean_intervention": 5.89,
            }
        ],
    }


class TestMigrateArmsGap1:
    def test_n_groups_converted_to_arms(self):
        doc = _legacy_doc()
        migrate_document(doc)
        sc = doc["study_characteristics"]
        assert "n_groups" not in sc
        assert "arms" in sc
        arm_types = {a["arm_type"] for a in sc["arms"]}
        assert arm_types == {"intervention", "control"}

    def test_arms_carry_sample_sizes(self):
        doc = _legacy_doc()
        migrate_document(doc)
        arms_by_type = {
            a["arm_type"]: a["n"] for a in doc["study_characteristics"]["arms"]
        }
        assert arms_by_type["intervention"] == 50
        assert arms_by_type["control"] == 50


class TestMigrateSingleArmMeanGap3:
    def test_single_arm_md_becomes_none_plus_descriptive_stat(self):
        doc = _legacy_doc()
        migrate_document(doc)
        outcome = doc["outcomes"][0]
        assert outcome["effect_size_type"] == "none"
        assert outcome["descriptive_stats"] == [
            {"statistic": "mean", "value": 5.89, "arm_label": "intervention"}
        ]

    def test_md_with_control_arm_is_left_as_md(self):
        """A genuine two-arm MD outcome must NOT be reclassified."""
        doc = {
            "study_characteristics": {"sample_size": 100},
            "outcomes": [
                {
                    "name": "Depression score",
                    "effect_size_type": "MD",
                    "mean_intervention": 10.0,
                    "mean_control": 15.0,
                }
            ],
        }
        migrate_document(doc)
        outcome = doc["outcomes"][0]
        assert outcome["effect_size_type"] == "MD"
        assert "descriptive_stats" not in outcome


class TestMigrateIdempotency:
    def test_migrating_twice_yields_no_further_changes(self):
        doc = _legacy_doc()
        first_changes = migrate_document(doc)
        assert first_changes, "expected the first pass to record changes"

        doc_after_first = copy.deepcopy(doc)
        second_changes = migrate_document(doc)
        assert second_changes == []
        assert doc == doc_after_first

    def test_migrating_twice_produces_identical_document(self):
        doc_a = _legacy_doc()
        migrate_document(doc_a)
        migrate_document(doc_a)

        doc_b = _legacy_doc()
        migrate_document(doc_b)

        assert doc_a == doc_b
