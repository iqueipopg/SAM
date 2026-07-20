#!/usr/bin/env python3
# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
migrate_extraction_schema.py — Migrate legacy A5 extraction JSON to the
refactored schema (arms / measurements / descriptive_stats).

Closes three design gaps in already-extracted per-paper JSON:

  GAP 1  study_characteristics.n_groups {"intervention", "control"}
         -> study_characteristics.arms [Arm, ...] (one Arm per group, never
         collapsing multi-arm designs). The legacy n_groups key is dropped.

  GAP 2  PRE/POST readings living as ad-hoc Outcome fields
         (pretest_mean/pretest_sd/posttest_mean/posttest_sd and
         mean_intervention/sd_intervention/n_intervention/
         mean_control/sd_control/n_control) -> outcome.measurements
         [Measurement, ...] with the canonical timepoint (baseline/post) and
         arm_label ("intervention"/"control"). Flat fields are LEFT in place.

  GAP 3  effect_size_type=="MD" over a single-arm mean (a bare mean with no
         control comparison) -> effect_size_type="none" plus a
         DescriptiveStat(statistic="mean", value=<mean>, scale=<inferred>,
         arm_label="intervention"). Never leave MD on a one-arm descriptive.

The script is IDEMPOTENT: re-running detects the new structures and makes no
further changes. Originals are preserved — migrated files are written to a
sibling directory "<dir>_migrated/".

Usage:
    python scripts/migrate_extraction_schema.py <dir> [<dir> ...]

    <dir> is a directory of *.json A5 extraction files.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
#  GAP 1 — n_groups -> arms                                                    #
# --------------------------------------------------------------------------- #
def _migrate_arms(sc: Dict[str, Any]) -> List[str]:
    """Convert study_characteristics.n_groups into arms (in place)."""
    changes: List[str] = []
    if sc.get("arms"):
        # Already migrated; just drop any stray legacy key.
        if "n_groups" in sc:
            sc.pop("n_groups", None)
            changes.append("dropped stale n_groups (arms already present)")
        return changes

    ng = sc.get("n_groups")
    if not isinstance(ng, dict):
        # Nothing to convert (null or absent). Still drop a null n_groups key.
        if "n_groups" in sc:
            sc.pop("n_groups", None)
        return changes

    arms: List[Dict[str, Any]] = []
    if "intervention" in ng:
        arms.append(
            {"label": "intervention", "arm_type": "intervention", "n": ng.get("intervention")}
        )
    if "control" in ng:
        arms.append({"label": "control", "arm_type": "control", "n": ng.get("control")})

    sc.pop("n_groups", None)
    if arms:
        sc["arms"] = arms
        changes.append(f"n_groups -> arms ({len(arms)} arm(s))")
    return changes


# --------------------------------------------------------------------------- #
#  GAP 2 — PRE/POST + arm means -> measurements                                #
# --------------------------------------------------------------------------- #
def _to_num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _to_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return None


def _build_measurements(o: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Derive canonical measurements from flat legacy fields.

    Two mutually compatible sources:
      - PRE/POST (single-group): pretest_* / posttest_* on the intervention arm.
      - Between-group post readings: mean_intervention/sd/n and
        mean_control/sd/n at the 'post' timepoint.
    """
    measurements: List[Dict[str, Any]] = []

    n_int = _to_int(o.get("n_intervention"))
    n_ctrl = _to_int(o.get("n_control"))

    # PRE/POST single-group readings -> baseline + post on intervention arm.
    pre_mean = _to_num(o.get("pretest_mean"))
    post_mean = _to_num(o.get("posttest_mean"))
    if pre_mean is not None:
        measurements.append(
            {
                "arm_label": "intervention",
                "timepoint": "baseline",
                "mean": pre_mean,
                "sd": _to_num(o.get("pretest_sd")),
                "n": n_int,
            }
        )
    if post_mean is not None:
        measurements.append(
            {
                "arm_label": "intervention",
                "timepoint": "post",
                "mean": post_mean,
                "sd": _to_num(o.get("posttest_sd")),
                "n": n_int,
            }
        )

    # Between-group post readings (only when NOT a pre/post row already handled).
    if pre_mean is None and post_mean is None:
        mean_int = _to_num(o.get("mean_intervention"))
        mean_ctrl = _to_num(o.get("mean_control"))
        if mean_int is not None:
            measurements.append(
                {
                    "arm_label": "intervention",
                    "timepoint": "post",
                    "mean": mean_int,
                    "sd": _to_num(o.get("sd_intervention")),
                    "n": n_int,
                }
            )
        if mean_ctrl is not None:
            measurements.append(
                {
                    "arm_label": "control",
                    "timepoint": "post",
                    "mean": mean_ctrl,
                    "sd": _to_num(o.get("sd_control")),
                    "n": n_ctrl,
                }
            )

    return measurements


# --------------------------------------------------------------------------- #
#  GAP 3 — single-arm MD -> none + descriptive_stats                           #
# --------------------------------------------------------------------------- #
_SCALE_RE = re.compile(
    r"(?:escala|rango|scale|range)\s*[:]?\s*(\d+\s*[-–a]\s*\d+)", re.IGNORECASE
)


def _infer_scale(o: Dict[str, Any]) -> Optional[str]:
    """Try to recover a Likert/scale range from name/units free text."""
    for key in ("name", "units", "scale"):
        text = o.get(key)
        if isinstance(text, str):
            m = _SCALE_RE.search(text)
            if m:
                return re.sub(r"\s+", "", m.group(1)).replace("–", "-").replace("a", "-")
    return None


def _is_single_arm_mean(o: Dict[str, Any]) -> bool:
    """True when an MD outcome is really a bare one-arm mean (no comparison).

    A comparison exists if there is control-arm data (mean_control or n_control)
    or a genuine pre/post pair (pretest_mean AND posttest_mean). Absent both,
    an MD value is just a single-arm descriptive mean.
    """
    if o.get("effect_size_type") != "MD":
        return False
    has_control = _to_num(o.get("mean_control")) is not None or _to_int(o.get("n_control")) is not None
    has_pre_post = _to_num(o.get("pretest_mean")) is not None and _to_num(o.get("posttest_mean")) is not None
    if has_control or has_pre_post:
        return False
    # Need an actual mean to describe: mean_intervention or effect_size_value.
    mean_val = _to_num(o.get("mean_intervention"))
    if mean_val is None:
        mean_val = _to_num(o.get("effect_size_value"))
    return mean_val is not None


def _single_arm_mean_value(o: Dict[str, Any]) -> Optional[float]:
    mean_val = _to_num(o.get("mean_intervention"))
    if mean_val is None:
        mean_val = _to_num(o.get("effect_size_value"))
    return mean_val


# --------------------------------------------------------------------------- #
#  Per-outcome migration                                                       #
# --------------------------------------------------------------------------- #
def _migrate_outcome(o: Dict[str, Any]) -> List[str]:
    """Migrate a single outcome dict in place; return change descriptions."""
    changes: List[str] = []

    # GAP 3 first: reclassify single-arm MD to none + descriptive_stats.
    if _is_single_arm_mean(o):
        if not o.get("descriptive_stats"):
            value = _single_arm_mean_value(o)
            stat: Dict[str, Any] = {
                "statistic": "mean",
                "value": value,
                "arm_label": "intervention",
            }
            scale = _infer_scale(o)
            if scale is not None:
                stat["scale"] = scale
            o["descriptive_stats"] = [stat]
            changes.append(
                f"MD single-arm mean -> effect_size_type=none + descriptive_stat(mean={value})"
            )
        o["effect_size_type"] = "none"

    # GAP 2: build measurements if none present yet.
    if not o.get("measurements"):
        measurements = _build_measurements(o)
        if measurements:
            o["measurements"] = measurements
            tps = ", ".join(f"{m['arm_label']}/{m['timepoint']}" for m in measurements)
            changes.append(f"measurements populated ({tps})")

    return changes


# --------------------------------------------------------------------------- #
#  Per-file migration                                                          #
# --------------------------------------------------------------------------- #
def migrate_document(doc: Dict[str, Any]) -> List[str]:
    """Migrate a full extraction document in place; return change log."""
    changes: List[str] = []

    sc = doc.get("study_characteristics")
    if isinstance(sc, dict):
        changes.extend(_migrate_arms(sc))

    outcomes = doc.get("outcomes")
    if isinstance(outcomes, list):
        for idx, o in enumerate(outcomes):
            if isinstance(o, dict):
                for c in _migrate_outcome(o):
                    changes.append(f"outcome[{idx}]: {c}")

    return changes


def migrate_dir(src_dir: Path) -> Tuple[Path, Dict[str, List[str]]]:
    """Migrate every *.json in src_dir into a sibling '<dir>_migrated/'."""
    out_dir = src_dir.parent / f"{src_dir.name}_migrated"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, List[str]] = {}
    for path in sorted(src_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        changes = migrate_document(doc)
        out_path = out_dir / path.name
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        summary[path.name] = changes
    return out_dir, summary


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        src_dir = Path(arg)
        if not src_dir.is_dir():
            print(f"ERROR: not a directory: {src_dir}")
            return 1
        out_dir, summary = migrate_dir(src_dir)
        print(f"\n=== {src_dir} -> {out_dir} ===")
        for fname, changes in summary.items():
            if changes:
                print(f"\n{fname}:")
                for c in changes:
                    print(f"  - {c}")
            else:
                print(f"\n{fname}: (no changes — already migrated)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
