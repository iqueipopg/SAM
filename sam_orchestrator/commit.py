"""Commit step — validate a written agent output against its Pydantic schema
and update manifest / _index.json atomically.

Schema dispatch:
  sam_a1_protocol    -> ProtocolOutput
  sam_a2_search      -> PapersRawOutput
  sam_a3_screening   -> ScreeningOutput (per-paper)
  sam_a4_fulltext    -> FulltextOutput (per-paper)
  sam_a5_extraction  -> ExtractionOutput (per-paper)
  sam_a6_quality     -> QualityOutput (per-paper)
  sam_a7_synthesis   -> SynthesisOutput
  sam_a8_manuscript  -> ManuscriptOutput
  sam_a9_explainer   -> ExplainerOutput

If validation fails: raise SchemaValidationError, NO state mutation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from schemas import (
    ExplainerOutput,
    ExtractionOutput,
    FulltextOutput,
    ManuscriptOutput,
    PapersRawOutput,
    ProtocolOutput,
    QualityOutput,
    ScreeningOutput,
    SynthesisOutput,
)

from .checkpoint import append_index_unique, make_entry, read_index
from .errors import SchemaValidationError, UnknownAgentError
from .manifest import (
    load_manifest,
    save_manifest_atomic,
    update_agent_status,
    update_paper_progress,
    utc_now_iso,
)
from .paths import AGENT_IDS, AGENT_MODES, RunPaths, paper_id_to_slug


SCHEMA_BY_AGENT: Dict[str, Type[BaseModel]] = {
    "sam_a1_protocol": ProtocolOutput,
    "sam_a2_search": PapersRawOutput,
    "sam_a3_screening": ScreeningOutput,
    "sam_a4_fulltext": FulltextOutput,
    "sam_a5_extraction": ExtractionOutput,
    "sam_a6_quality": QualityOutput,
    "sam_a7_synthesis": SynthesisOutput,
    "sam_a8_manuscript": ManuscriptOutput,
    "sam_a9_explainer": ExplainerOutput,
}


def _validate(agent_id: str, payload: Any) -> BaseModel:
    schema_cls = SCHEMA_BY_AGENT.get(agent_id)
    if schema_cls is None:
        raise UnknownAgentError(agent_id)
    try:
        return schema_cls.model_validate(payload)
    except ValidationError as e:
        raise SchemaValidationError(
            f"{agent_id} output failed schema validation: {e}"
        ) from e


def _paper_id_from_payload(agent_id: str, payload: Dict[str, Any]) -> str:
    """Extract canonical paper_id from a per-paper payload.

    Falls back from `paper_id` -> `pmid`. Raises SchemaValidationError if both
    are missing (which would also fail Pydantic validation).
    """
    pid = payload.get("paper_id") or payload.get("pmid")
    if not pid:
        raise SchemaValidationError(
            f"{agent_id} per-paper output missing both 'paper_id' and 'pmid'"
        )
    return str(pid)


def commit_output(
    run_id: str,
    agent_id: str,
    output_path: str,
    outputs_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate `output_path` JSON against the schema for `agent_id`, then
    persist state.

    Returns a small dict with the result. Raises SchemaValidationError on
    validation failure (caller should map to exit code 2).
    """
    if agent_id not in AGENT_IDS:
        raise UnknownAgentError(agent_id)

    out_path = Path(output_path)
    if not out_path.exists():
        raise SchemaValidationError(
            f"output file does not exist: {out_path}"
        )

    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SchemaValidationError(
            f"output file is not valid JSON ({out_path}): {e}"
        ) from e

    model = _validate(agent_id, payload)
    mode = AGENT_MODES[agent_id]

    manifest = load_manifest(run_id, outputs_root=outputs_root)
    paths = RunPaths(run_id, outputs_root=outputs_root)

    # Common header fields from AgentResultBase
    backend = getattr(model, "backend", None)
    tokens_in = getattr(model, "tokens_in", None)
    tokens_out = getattr(model, "tokens_out", None)

    result: Dict[str, Any] = {
        "ok": True,
        "agent": agent_id,
        "mode": mode,
        "output_path": str(out_path),
    }

    if mode == "single":
        update_agent_status(
            manifest,
            agent_id,
            status="completed",
            backend=backend,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        save_manifest_atomic(manifest, run_id, outputs_root=outputs_root)
        return result

    # per_paper
    pid = _paper_id_from_payload(agent_id, payload)
    slug = paper_id_to_slug(pid)
    # Sanity: the file must live at the expected slug path.
    expected_path = paths.paper_output(agent_id, pid)
    if expected_path.resolve() != out_path.resolve():
        # Not fatal but worth flagging; we trust the caller's path.
        result["warning"] = (
            f"output_path {out_path} differs from expected {expected_path}"
        )

    entry = make_entry(slug=slug, paper_id=pid, timestamp=utc_now_iso())
    appended = append_index_unique(paths.agent_dir(agent_id), entry)
    result["appended"] = appended

    if appended:
        update_paper_progress(
            manifest,
            agent_id,
            papers_done_increment=1,
            tokens_in_increment=tokens_in if tokens_in is not None else None,
            tokens_out_increment=tokens_out if tokens_out is not None else None,
            backend=backend,
        )
        save_manifest_atomic(manifest, run_id, outputs_root=outputs_root)
    else:
        # Idempotent re-commit: still refresh manifest last_updated.
        save_manifest_atomic(manifest, run_id, outputs_root=outputs_root)

    # Surface progress in the return value
    agent_state = manifest.agents.get(agent_id)
    if agent_state is not None:
        result["papers_done"] = agent_state.papers_done
        result["papers_total"] = agent_state.papers_total
        result["status"] = agent_state.status

    return result
