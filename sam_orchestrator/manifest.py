"""Manifest I/O — atomic load/save + status update helpers.

Atomicity: write to <path>.tmp + os.replace(tmp, target). Avoids torn writes
if the process dies mid-write.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from schemas import AgentStatus, Manifest

from .errors import ManifestCorruptError, RunNotFoundError, UnknownAgentError
from .paths import AGENT_IDS, RunPaths


def utc_now_iso() -> str:
    """ISO 8601 UTC timestamp with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def load_manifest(run_id: str, outputs_root: Optional[Path] = None) -> Manifest:
    paths = RunPaths(run_id, outputs_root=outputs_root)
    if not paths.run_dir.exists():
        raise RunNotFoundError(f"run not found: {paths.run_dir}")
    if not paths.manifest_path.exists():
        raise ManifestCorruptError(
            f"manifest.json missing in {paths.run_dir}"
        )
    try:
        raw = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestCorruptError(f"manifest.json invalid JSON: {e}") from e
    try:
        return Manifest.model_validate(raw)
    except ValidationError as e:
        raise ManifestCorruptError(f"manifest schema invalid: {e}") from e


def save_manifest_atomic(
    manifest: Manifest, run_id: str, outputs_root: Optional[Path] = None
) -> None:
    """Write the manifest atomically. Bumps `last_updated`."""
    paths = RunPaths(run_id, outputs_root=outputs_root)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    # Always refresh last_updated on save.
    manifest = manifest.model_copy(update={"last_updated": utc_now_iso()})

    target = paths.manifest_path
    payload = manifest.model_dump_json(indent=2)

    # NamedTemporaryFile in the SAME directory so os.replace is atomic on Linux.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".manifest_", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        # Best-effort cleanup of leftover tmp on failure.
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass
        raise


def _ensure_agent(manifest: Manifest, agent_id: str) -> AgentStatus:
    if agent_id not in AGENT_IDS:
        raise UnknownAgentError(agent_id)
    if agent_id not in manifest.agents:
        # Should not happen on a well-initialised run, but be defensive.
        manifest.agents[agent_id] = AgentStatus()
    return manifest.agents[agent_id]


def update_agent_status(
    manifest: Manifest,
    agent_id: str,
    *,
    status: Optional[str] = None,
    backend: Optional[str] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    error: Optional[str] = None,
) -> Manifest:
    """Mutate manifest in place: update top-level fields for a single agent.

    Returns the manifest (same object) for chaining.
    """
    agent = _ensure_agent(manifest, agent_id)
    if status is not None:
        agent.status = status  # type: ignore[assignment]
        now = utc_now_iso()
        if status == "in_progress" and agent.started_at is None:
            agent.started_at = now
        if status in ("completed", "failed"):
            agent.completed_at = now
    if backend is not None:
        agent.backend = backend  # type: ignore[assignment]
    if tokens_in is not None:
        agent.tokens_in = tokens_in
    if tokens_out is not None:
        agent.tokens_out = tokens_out
    if error is not None:
        agent.error = error
    return manifest


def update_paper_progress(
    manifest: Manifest,
    agent_id: str,
    *,
    papers_total: Optional[int] = None,
    papers_done_increment: int = 0,
    tokens_in_increment: Optional[int] = None,
    tokens_out_increment: Optional[int] = None,
    backend: Optional[str] = None,
) -> Manifest:
    """Mutate manifest in place for per-paper progress accounting."""
    agent = _ensure_agent(manifest, agent_id)

    if papers_total is not None:
        agent.papers_total = papers_total
        if agent.papers_done is None:
            agent.papers_done = 0

    if papers_done_increment:
        agent.papers_done = (agent.papers_done or 0) + papers_done_increment

    if tokens_in_increment is not None:
        agent.tokens_cumulative_in = (
            agent.tokens_cumulative_in or 0
        ) + tokens_in_increment

    if tokens_out_increment is not None:
        agent.tokens_cumulative_out = (
            agent.tokens_cumulative_out or 0
        ) + tokens_out_increment

    if backend is not None:
        agent.backend = backend  # type: ignore[assignment]

    # Transition to in_progress on first increment.
    if agent.status == "pending" and (
        papers_done_increment or papers_total is not None
    ):
        agent.status = "in_progress"
        if agent.started_at is None:
            agent.started_at = utc_now_iso()

    # Auto-complete when papers_done == papers_total (and total is known/>=1).
    if (
        agent.papers_total is not None
        and agent.papers_done is not None
        and agent.papers_done >= agent.papers_total
        and agent.status != "failed"
    ):
        agent.status = "completed"
        agent.completed_at = utc_now_iso()

    return manifest
