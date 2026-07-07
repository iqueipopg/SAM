# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Path resolution and agent metadata for sam_orchestrator (T17.C.0).

Constants:
- AGENT_IDS: ordered list of all 9 agents (drives `next` order).
- AGENT_MODES: mapping agent_id -> 'single' | 'per_paper'.

Class:
- RunPaths(run_id): resolves manifest, agent dirs, per-paper outputs, indexes.

Function:
- paper_id_to_slug(paper_id): D3 — sanitize paper_id to a filesystem-safe slug
  (<= 80 chars). The canonical paper_id (e.g. raw PMID or DOI) is preserved
  inside the per-paper JSON and in _index.json for auditability.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

# Resolve the repo root from this file: sam_orchestrator/paths.py -> parent.parent
_REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = _REPO_ROOT / "outputs"

# Ordered pipeline (drives `next`).
AGENT_IDS = [
    "sam_a1_protocol",
    "sam_a2_search",
    "sam_a3_screening",
    "sam_a4_fulltext",
    "sam_a5_extraction",
    "sam_a6_quality",
    "sam_a7_synthesis",
    "sam_a8_manuscript",
    "sam_a9_explainer",
]

AgentMode = Literal["single", "per_paper"]

AGENT_MODES: dict[str, AgentMode] = {
    "sam_a1_protocol": "single",
    "sam_a2_search": "single",
    "sam_a3_screening": "per_paper",
    "sam_a4_fulltext": "per_paper",
    "sam_a5_extraction": "per_paper",
    "sam_a6_quality": "per_paper",
    "sam_a7_synthesis": "single",
    "sam_a8_manuscript": "single",
    "sam_a9_explainer": "single",
}

# Conventional single-agent output filenames.
AGENT_OUTPUT_FILENAME = {
    "sam_a1_protocol": "protocol.json",
    "sam_a2_search": "papers_raw.json",
    "sam_a7_synthesis": "results.json",
    "sam_a8_manuscript": "manuscript.json",
    "sam_a9_explainer": "explainer.json",
}


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]")
_SLUG_MAX_LEN = 80


def paper_id_to_slug(paper_id: str) -> str:
    """Sanitize an arbitrary paper_id (DOI, PMID, URL...) to a filesystem-safe slug.

    Steps:
    1. Replace any character outside [a-zA-Z0-9._-] with '_'.
    2. Truncate to 80 chars.

    Example: 'doi:10.1080/17437199.2025.2492042'
           → 'doi_10.1080_17437199.2025.2492042'
    """
    if not isinstance(paper_id, str) or not paper_id:
        raise ValueError("paper_id must be a non-empty string")
    return _SLUG_RE.sub("_", paper_id)[:_SLUG_MAX_LEN]


class RunPaths:
    """Path resolver for a single run_id.

    All paths are absolute. Construction does NOT touch the filesystem — call
    `ensure_dirs()` explicitly to create the directory tree (used by init).
    """

    def __init__(self, run_id: str, outputs_root: Path | None = None) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self.run_id = run_id
        self._outputs_root = (outputs_root or OUTPUTS_ROOT).resolve()

    # --- top-level paths ---

    @property
    def outputs_root(self) -> Path:
        return self._outputs_root

    @property
    def run_dir(self) -> Path:
        return self._outputs_root / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    # --- per-agent paths ---

    def agent_dir(self, agent_id: str) -> Path:
        if agent_id not in AGENT_IDS:
            raise ValueError(f"unknown agent_id: {agent_id!r}")
        return self.run_dir / agent_id

    def index_path(self, agent_id: str) -> Path:
        """Path to _index.json for a per-paper agent."""
        if AGENT_MODES.get(agent_id) != "per_paper":
            raise ValueError(
                f"agent {agent_id!r} is not per-paper; no _index.json"
            )
        return self.agent_dir(agent_id) / "_index.json"

    def paper_output(self, agent_id: str, paper_id: str) -> Path:
        """Path to <slug>.json for a per-paper agent + a given paper_id."""
        if AGENT_MODES.get(agent_id) != "per_paper":
            raise ValueError(
                f"agent {agent_id!r} is not per-paper; no per-paper output"
            )
        slug = paper_id_to_slug(paper_id)
        return self.agent_dir(agent_id) / f"{slug}.json"

    def single_output(self, agent_id: str) -> Path:
        """Path to the canonical single output of a single-agent."""
        if AGENT_MODES.get(agent_id) != "single":
            raise ValueError(
                f"agent {agent_id!r} is per-paper; use paper_output()"
            )
        filename = AGENT_OUTPUT_FILENAME.get(agent_id, f"{agent_id}.json")
        return self.agent_dir(agent_id) / filename

    # --- filesystem ops ---

    def ensure_dirs(self) -> None:
        """Create run_dir + one subdir per agent. Idempotent."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for aid in AGENT_IDS:
            (self.run_dir / aid).mkdir(parents=True, exist_ok=True)
