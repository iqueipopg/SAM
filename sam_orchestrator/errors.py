# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Domain-specific exceptions for the SAM orchestrator (T17.C.0)."""
from __future__ import annotations


class OrchestratorError(Exception):
    """Base class for all sam_orchestrator errors."""


class RunNotFoundError(OrchestratorError):
    """Raised when a run_id directory does not exist."""


class ManifestCorruptError(OrchestratorError):
    """Raised when manifest.json cannot be parsed or fails schema validation."""


class SchemaValidationError(OrchestratorError):
    """Raised when an agent output JSON fails Pydantic validation on commit."""


class SubsetMappingMissingError(OrchestratorError):
    """Raised when --subset references a partition path that does not exist
    or a split name that is absent inside the partition JSON.
    """


class UnknownAgentError(OrchestratorError):
    """Raised when commit/audit is called with an agent_id not in AGENT_IDS."""


class InvalidWorkStateError(OrchestratorError):
    """Raised when next_work cannot determine a coherent work item
    (e.g. eligible set inconsistent with manifest state)."""
