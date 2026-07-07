# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Commit step — validate a written agent output against its Pydantic schema
and update manifest / _index.json atomically.

T17.C.1 deterministic I/O separation: per-paper subagents (A3/A4/A5/A6) emit
ONLY their decision/content fields + `paper_id`. The commit step is a
deterministic ASSEMBLER: it injects all metadata and provenance fields from
authoritative sources (`papers_raw.json` + `_fulltext_cache/`) BEFORE
validation, then rewrites the on-disk file as the complete record. This fixes
a real failure mode: the LLM cannot reliably reproduce verbatim `full_text`
nor self-count `full_text_length` for ~60k-char texts, and Qwen-B parity
forbids storing `full_text` at all.

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


def _load_json(path: Path) -> Optional[Any]:
    """Best-effort JSON load. Returns None on any failure (never crashes)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_paper_in_raw(
    run_id: str, paper_id: str, outputs_root: Optional[Path]
) -> Dict[str, Any]:
    """Locate the A2 paper record for `paper_id` in papers_raw.json.

    Match by `paper_id` first, then fall back to `pmid`. Returns an empty dict
    if papers_raw.json is missing/unparseable or the paper is not found.
    Never crashes — injection must degrade gracefully (compute what you can).
    """
    paths = RunPaths(run_id, outputs_root=outputs_root)
    raw = _load_json(paths.single_output("sam_a2_search"))
    if not isinstance(raw, dict):
        return {}
    papers = raw.get("papers")
    if not isinstance(papers, list):
        return {}
    for p in papers:
        if not isinstance(p, dict):
            continue
        if p.get("paper_id") == paper_id or (
            p.get("pmid") and str(p.get("pmid")) == paper_id
        ):
            return p
    return {}


def _load_cache_meta(
    run_id: str, paper_id: str, outputs_root: Optional[Path]
) -> Dict[str, Any]:
    """Read `_fulltext_cache/<slug>.meta.json`. Defensive over both meta shapes
    (online builder has sections/oa_url; local builder does not). Returns {} if
    absent/unparseable."""
    paths = RunPaths(run_id, outputs_root=outputs_root)
    slug = paper_id_to_slug(paper_id)
    meta_path = paths.run_dir / "_fulltext_cache" / f"{slug}.meta.json"
    meta = _load_json(meta_path)
    return meta if isinstance(meta, dict) else {}


def _cache_text_length(
    run_id: str, paper_id: str, outputs_root: Optional[Path]
) -> Optional[int]:
    """Length of the authoritative `_fulltext_cache/<slug>.txt`, or None if the
    .txt is absent. This is the SINGLE source of truth for full_text_length —
    the LLM cannot reliably self-count a ~60k-char text."""
    paths = RunPaths(run_id, outputs_root=outputs_root)
    slug = paper_id_to_slug(paper_id)
    txt_path = paths.run_dir / "_fulltext_cache" / f"{slug}.txt"
    try:
        return len(txt_path.read_text(encoding="utf-8"))
    except OSError:
        return None


def _inject_fields(
    agent_id: str,
    payload: Dict[str, Any],
    run_id: str,
    paper_id: str,
    outputs_root: Optional[Path],
) -> Dict[str, Any]:
    """Inject metadata + provenance fields AUTHORITATIVELY into a per-paper
    payload (injected values overwrite whatever the LLM wrote).

    WHY: per-paper subagents emit only decision/content fields. Metadata (title,
    authors, ...) lives in papers_raw.json and provenance (text_source,
    full_text_length, ...) in _fulltext_cache/. Injecting here guarantees
    determinism and B-parity, and avoids trusting the LLM to copy/count large
    text. `paper_id` and all decision/content fields are left untouched.

    Robustness: never crashes on missing cache or papers_raw — falls back to
    the value already in the payload, else a type-appropriate empty.
    """
    paper = _find_paper_in_raw(run_id, paper_id, outputs_root)

    def from_paper(key: str, default: Any) -> Any:
        # papers_raw value wins; else keep what the LLM emitted; else default.
        if key in paper and paper.get(key) is not None:
            return paper.get(key)
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
        return default

    if agent_id == "sam_a3_screening":
        # No fulltext cache for A3: only title + pmid come from papers_raw.
        payload["title"] = from_paper("title", "")
        payload["pmid"] = from_paper("pmid", None)

    elif agent_id == "sam_a4_fulltext":
        # Metadata from papers_raw.
        payload["title"] = from_paper("title", "")
        payload["abstract"] = from_paper("abstract", "")
        payload["authors"] = from_paper("authors", [])
        payload["journal"] = from_paper("journal", "")
        payload["year"] = from_paper("year", "")
        payload["doi"] = from_paper("doi", None)
        payload["pmcid"] = from_paper("pmcid", None)
        payload["pmid"] = from_paper("pmid", None)
        # Provenance from cache. full_text is removed from the schema; the
        # authoritative text lives at _fulltext_cache/<slug>.txt.
        meta = _load_cache_meta(run_id, paper_id, outputs_root)
        payload["text_source"] = meta.get("text_source") or "abstract_only"
        txt_len = _cache_text_length(run_id, paper_id, outputs_root)
        if txt_len is None:
            txt_len = meta.get("full_text_length")
        payload["full_text_length"] = int(txt_len) if txt_len is not None else 0
        sections = meta.get("sections")
        payload["sections"] = sections if isinstance(sections, dict) else {}
        payload["oa_url"] = meta.get("oa_url")
        # Never store full_text (schema field removed; B-parity).
        payload.pop("full_text", None)

    elif agent_id == "sam_a5_extraction":
        payload["title"] = from_paper("title", "")
        payload["authors"] = from_paper("authors", [])
        payload["journal"] = from_paper("journal", "")
        payload["year"] = from_paper("year", "")
        payload["pmid"] = from_paper("pmid", None)
        # text_source is provenance; sections_used reflects what the LLM
        # actually consulted, so it is LEFT as emitted.
        meta = _load_cache_meta(run_id, paper_id, outputs_root)
        payload["text_source"] = meta.get("text_source") or "abstract_only"

    elif agent_id == "sam_a6_quality":
        payload["title"] = from_paper("title", "")
        payload["pmid"] = from_paper("pmid", None)
        # Propagate study_characteristics + outcomes deterministically from the
        # A5 output for the same paper (A7 consumes A6 directly).
        paths = RunPaths(run_id, outputs_root=outputs_root)
        a5 = _load_json(paths.paper_output("sam_a5_extraction", paper_id))
        if isinstance(a5, dict):
            if "study_characteristics" in a5:
                payload["study_characteristics"] = a5.get("study_characteristics")
            if "outcomes" in a5:
                payload["outcomes"] = a5.get("outcomes")

    return payload


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

    mode = AGENT_MODES[agent_id]

    if mode == "single":
        # Single agents (A1/A2/A7/A8/A9): no injection, no rewrite — validate
        # the LLM output as-is.
        model = _validate(agent_id, payload)
        pid = None
    else:
        # per_paper: commit is a deterministic ASSEMBLER. The LLM emitted only
        # decision/content fields; inject metadata + provenance authoritatively,
        # validate the ASSEMBLED record, then rewrite the file as the complete
        # on-disk record. (See module docstring.)
        pid = _paper_id_from_payload(agent_id, payload)
        payload = _inject_fields(
            agent_id, payload, run_id, pid, outputs_root
        )
        model = _validate(agent_id, payload)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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

    # per_paper (pid was computed above before injection)
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
