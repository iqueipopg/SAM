# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Manifest schema — on-disk run state at outputs/<run_id>/manifest.json.

The manifest is the single source of truth for resume / audit. It tracks
status per agent, papers_total / papers_done for per-paper agents, and
snapshotted run config (query, subset spec, backend defaults).

Timestamps are ISO 8601 strings serialised by Pydantic v2 default behaviour.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

AgentStatusStr = Literal["pending", "in_progress", "completed", "failed"]


class SubsetSpec(BaseModel):
    """Optional subset specification (`partition.json:split_name`)."""

    model_config = ConfigDict(extra="forbid")

    partition: str = Field(..., description="Path to partition JSON file.")
    split: str = Field(..., description="Split key inside the partition (e.g. 'dev').")
    n: Optional[int] = Field(
        default=None, description="Cardinality of the subset (post-load count)."
    )


class RunConfig(BaseModel):
    """Snapshot of run-time config persisted at init."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="User clinical question.")
    protocol_version: str = Field(
        default="V2", description="Protocol version locked for the run."
    )
    backend_default: Literal["claude", "qwen"] = Field(
        default="claude", description="Default backend for the run."
    )
    backend_overrides: Dict[str, Literal["claude", "qwen"]] = Field(
        default_factory=dict,
        description="Per-agent backend overrides (agent_id -> backend).",
    )
    subset: Optional[SubsetSpec] = None


class AgentStatus(BaseModel):
    """Per-agent status tracked inside Manifest.agents."""

    model_config = ConfigDict(extra="forbid")

    status: AgentStatusStr = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    backend: Optional[Literal["claude", "qwen"]] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None

    # Per-paper agents only
    papers_total: Optional[int] = None
    papers_done: Optional[int] = None
    tokens_cumulative_in: Optional[int] = None
    tokens_cumulative_out: Optional[int] = None

    # Failure metadata (when status='failed')
    error: Optional[str] = None


class Manifest(BaseModel):
    """Top-level on-disk manifest object."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    started_at: str
    last_updated: str
    config: RunConfig
    agents: Dict[str, AgentStatus]


__all__ = [
    "AgentStatusStr",
    "SubsetSpec",
    "RunConfig",
    "AgentStatus",
    "Manifest",
]
