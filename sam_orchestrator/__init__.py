"""SAM deterministic orchestrator (T17.C.0).

Public API:
- init_run(query, ...) -> run_id
- next_work(run_id) -> dict
- commit_output(run_id, agent_id, output_path) -> dict
- audit_run(run_id) -> dict
- RunPaths(run_id) for path resolution
- paper_id_to_slug(paper_id) for filesystem-safe slugs
- AGENT_IDS / AGENT_MODES constants
"""
from __future__ import annotations

from .audit import audit_run
from .commit import SCHEMA_BY_AGENT, commit_output
from .errors import (
    InvalidWorkStateError,
    ManifestCorruptError,
    OrchestratorError,
    RunNotFoundError,
    SchemaValidationError,
    SubsetMappingMissingError,
    UnknownAgentError,
)
from .paths import (
    AGENT_IDS,
    AGENT_MODES,
    AGENT_OUTPUT_FILENAME,
    OUTPUTS_ROOT,
    RunPaths,
    paper_id_to_slug,
)
from .state import init_run, next_work

__all__ = [
    # public functions
    "init_run",
    "next_work",
    "commit_output",
    "audit_run",
    # paths / slugs
    "RunPaths",
    "AGENT_IDS",
    "AGENT_MODES",
    "AGENT_OUTPUT_FILENAME",
    "OUTPUTS_ROOT",
    "paper_id_to_slug",
    # schema dispatch (for advanced users / tests)
    "SCHEMA_BY_AGENT",
    # errors
    "OrchestratorError",
    "RunNotFoundError",
    "ManifestCorruptError",
    "SchemaValidationError",
    "SubsetMappingMissingError",
    "UnknownAgentError",
    "InvalidWorkStateError",
]
