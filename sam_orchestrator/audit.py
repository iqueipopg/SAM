# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Audit step — fully reproducible state check, no LLM.

Per-paper agents: recomputes the eligible set, compares against _index.json,
detects missing slugs (eligible − _index), duplicate slugs in _index, and
invalid per-paper JSONs (file absent OR schema invalid).

Single agents: verifies the canonical output file exists and validates
against its schema; verifies manifest.status == 'completed'.

Returns: {"ok": bool, "agents": {agent_id: {missing, duplicates, invalid}}}.
Stdout from CLI mirrors this dict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .checkpoint import find_duplicates, read_index
from .commit import SCHEMA_BY_AGENT
from .manifest import load_manifest
from .paths import AGENT_IDS, AGENT_MODES, RunPaths, paper_id_to_slug
from .state import (
    _eligible_a3,
    _eligible_a4,
    _eligible_a5,
    _eligible_a6,
)


def _safe_validate(agent_id: str, path: Path) -> Optional[str]:
    """Return None if path validates, else an error message."""
    schema_cls = SCHEMA_BY_AGENT.get(agent_id)
    if schema_cls is None:
        return f"unknown agent_id {agent_id}"
    if not path.exists():
        return f"file missing: {path}"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"
    try:
        schema_cls.model_validate(raw)
        return None
    except ValidationError as e:
        return f"schema validation failed: {e}"


_PER_PAPER_ELIGIBLE_FNS = {
    "sam_a3_screening": _eligible_a3,
    "sam_a4_fulltext": lambda paths, manifest: _eligible_a4(paths),
    "sam_a5_extraction": lambda paths, manifest: _eligible_a5(paths),
    "sam_a6_quality": lambda paths, manifest: _eligible_a6(paths),
}


def _audit_per_paper(
    paths: RunPaths, manifest, agent_id: str
) -> Dict[str, List[str]]:
    elig_fn = _PER_PAPER_ELIGIBLE_FNS[agent_id]
    eligible = elig_fn(paths, manifest)
    eligible_slugs = {paper_id_to_slug(pid): pid for pid in eligible}
    eligible_slug_set = set(eligible_slugs.keys())

    agent_dir = paths.agent_dir(agent_id)
    entries = read_index(agent_dir)
    indexed_slugs = {e.slug for e in entries}

    missing = sorted(eligible_slug_set - indexed_slugs)
    duplicates = sorted(find_duplicates(agent_dir))

    invalid: List[str] = []
    for entry in entries:
        per_paper_path = agent_dir / f"{entry.slug}.json"
        err = _safe_validate(agent_id, per_paper_path)
        if err is not None:
            invalid.append(f"{entry.slug}: {err}")

    # Index entries with slugs that don't belong to the eligible set are also
    # surfaced as `missing` of a different flavour ("unexpected"), but to keep
    # the schema simple we record them under `invalid`.
    unexpected = sorted(indexed_slugs - eligible_slug_set)
    for slug in unexpected:
        invalid.append(f"{slug}: indexed but not in eligible set")

    return {"missing": missing, "duplicates": duplicates, "invalid": invalid}


def _audit_single(paths: RunPaths, manifest, agent_id: str) -> Dict[str, List[str]]:
    out_path = paths.single_output(agent_id)
    invalid: List[str] = []
    err = _safe_validate(agent_id, out_path)
    if err is not None:
        invalid.append(err)

    agent_status = manifest.agents.get(agent_id)
    missing: List[str] = []
    if agent_status is None or agent_status.status != "completed":
        status_str = "unknown" if agent_status is None else agent_status.status
        missing.append(f"manifest status is {status_str!r}, expected 'completed'")

    return {"missing": missing, "duplicates": [], "invalid": invalid}


def audit_run(
    run_id: str,
    outputs_root: Optional[Path] = None,
    only: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the audit for a run_id. Pure function (no mutation).

    When `only` is set it MUST be a valid agent_id; the audit then covers ONLY
    that agent and `ok` reflects only it.
    """
    paths = RunPaths(run_id, outputs_root=outputs_root)
    manifest = load_manifest(run_id, outputs_root=outputs_root)

    if only is not None:
        if only not in AGENT_IDS:
            raise ValueError(
                f"--only must be one of {AGENT_IDS}; got {only!r}"
            )
        agent_ids = [only]
    else:
        agent_ids = AGENT_IDS

    report: Dict[str, Any] = {"agents": {}}
    ok = True
    for agent_id in agent_ids:
        if AGENT_MODES[agent_id] == "per_paper":
            details = _audit_per_paper(paths, manifest, agent_id)
        else:
            details = _audit_single(paths, manifest, agent_id)
        report["agents"][agent_id] = details
        if details["missing"] or details["duplicates"] or details["invalid"]:
            ok = False

    report["ok"] = ok
    return report
