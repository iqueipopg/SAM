"""Core `next_work` logic — pure, deterministic, no LLM.

Reads manifest + per-agent _index.json + agent outputs to decide what the
next unit of work is. Returns:

  - Per-paper work item: {"agent": "...", "paper_id": "...",
                          "input_paths": [...], "output_path": "..."}
  - Single-agent work item: {"agent": "...", "paper_id": null,
                             "input_paths": [...], "output_path": "..."}
  - Done sentinel: {"done": true}

Per-paper eligible-set rules (Qwen-real schema):
  A3: every paper in papers_raw.json (intersected with subset if present).
  A4: A3 outputs where decision == "INCLUDE".
  A5: A4 outputs where fulltext_decision == "INCLUDE".
  A6: same eligible set as A5 (parallel branch).

Single-agent rules (A1 / A2 / A7 / A8 / A9): pick the next pending one.

`init_run` is also defined here for cohesion (it's the run-creation step).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import (
    AgentStatus,
    Manifest,
    RunConfig,
    SubsetSpec,
)

from .checkpoint import index_slugs, read_index
from .errors import InvalidWorkStateError, RunNotFoundError
from .manifest import load_manifest, save_manifest_atomic, utc_now_iso
from .paths import AGENT_IDS, AGENT_MODES, RunPaths, paper_id_to_slug
from .subset import load_subset_ids, validate_subset_spec


# -----------------------------------------------------------------------------
# init_run
# -----------------------------------------------------------------------------

_SLUG_FILENAME_RE = re.compile(r"[^a-zA-Z0-9]+")


def _build_run_id(query: str, slug_hint: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if slug_hint:
        slug = _SLUG_FILENAME_RE.sub("_", slug_hint).strip("_").lower()[:40] or "run"
    else:
        slug = _SLUG_FILENAME_RE.sub("_", query).strip("_").lower()[:40] or "run"
    return f"{ts}_{slug}"


def init_run(
    query: str,
    *,
    subset_spec: Optional[str] = None,
    backend_default: str = "claude",
    slug_hint: Optional[str] = None,
    outputs_root: Optional[Path] = None,
) -> str:
    """Create a new run directory + manifest. Returns the run_id.

    - Validates --subset (partition.json must exist if provided).
    - Creates 9 subdirs.
    - Writes manifest with all agents `pending`.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    subset_obj: Optional[SubsetSpec] = None
    if subset_spec:
        path, split = validate_subset_spec(subset_spec)
        n: Optional[int] = None
        try:
            ids = load_subset_ids(subset_spec)
            n = len(ids)
        except Exception:
            # Could not pre-count; not fatal — `next` will load again.
            n = None
        subset_obj = SubsetSpec(partition=str(path), split=split, n=n)

    run_id = _build_run_id(query, slug_hint=slug_hint)
    paths = RunPaths(run_id, outputs_root=outputs_root)
    paths.ensure_dirs()

    cfg = RunConfig(
        query=query,
        backend_default=backend_default,  # type: ignore[arg-type]
        backend_overrides={},
        subset=subset_obj,
    )
    now = utc_now_iso()
    manifest = Manifest(
        run_id=run_id,
        started_at=now,
        last_updated=now,
        config=cfg,
        agents={aid: AgentStatus(status="pending") for aid in AGENT_IDS},
    )
    save_manifest_atomic(manifest, run_id, outputs_root=outputs_root)
    return run_id


# -----------------------------------------------------------------------------
# next_work
# -----------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _papers_from_raw(papers_raw_path: Path) -> List[Dict[str, Any]]:
    raw = _read_json(papers_raw_path)
    if not isinstance(raw, dict):
        return []
    papers = raw.get("papers")
    return papers if isinstance(papers, list) else []


def _paper_pmid(paper: Dict[str, Any]) -> str:
    # D4 (T17.C.1): canonical paper_id is Covidence ID. Honor explicit
    # `paper_id` first; fall back to pmid/doi/id for backward compatibility
    # with older runs and Qwen-real outputs that only set pmid.
    pid = paper.get("paper_id")
    if pid:
        return str(pid)
    pmid = paper.get("pmid")
    if pmid:
        return str(pmid)
    return str(paper.get("doi") or paper.get("id") or "")


def _filter_by_subset(
    paper_ids: List[str], subset: Optional[SubsetSpec]
) -> List[str]:
    if subset is None:
        return paper_ids
    spec = f"{subset.partition}:{subset.split}"
    try:
        active = load_subset_ids(spec)
    except Exception:
        # If we can't load the subset at next-time, fall back to no filter
        # so the orchestrator does not silently drop work; audit will catch it.
        return paper_ids
    return [pid for pid in paper_ids if pid in active]


def _eligible_a3(paths: RunPaths, manifest: Manifest) -> List[str]:
    raw_path = paths.single_output("sam_a2_search")
    papers = _papers_from_raw(raw_path)
    ids = [_paper_pmid(p) for p in papers if _paper_pmid(p)]
    return _filter_by_subset(ids, manifest.config.subset)


def _next_per_paper_id(
    paths: RunPaths, agent_id: str, eligible: List[str]
) -> Optional[str]:
    """Return the first eligible paper_id whose slug is NOT yet in _index.json,
    preserving the order in `eligible`.
    """
    done_slugs = index_slugs(paths.agent_dir(agent_id))
    for pid in eligible:
        if paper_id_to_slug(pid) not in done_slugs:
            return pid
    return None


def _includes_from_per_paper(
    paths: RunPaths, agent_id: str, decision_field: str
) -> List[str]:
    """Read every <slug>.json in an agent dir and return paper_ids whose
    `decision_field` == "INCLUDE", preserving _index.json order.
    """
    out: List[str] = []
    agent_dir = paths.agent_dir(agent_id)
    for entry in read_index(agent_dir):
        per_paper_path = agent_dir / f"{entry.slug}.json"
        data = _read_json(per_paper_path)
        if not isinstance(data, dict):
            continue
        if data.get(decision_field) == "INCLUDE":
            out.append(entry.paper_id)
    return out


def _eligible_a4(paths: RunPaths) -> List[str]:
    return _includes_from_per_paper(paths, "sam_a3_screening", "decision")


def _eligible_a5(paths: RunPaths) -> List[str]:
    return _includes_from_per_paper(
        paths, "sam_a4_fulltext", "fulltext_decision"
    )


def _eligible_a6(paths: RunPaths) -> List[str]:
    # Parallel to A5 by design.
    return _eligible_a5(paths)


_PER_PAPER_ELIGIBLE = {
    "sam_a3_screening": _eligible_a3,
    "sam_a4_fulltext": lambda paths, manifest: _eligible_a4(paths),
    "sam_a5_extraction": lambda paths, manifest: _eligible_a5(paths),
    "sam_a6_quality": lambda paths, manifest: _eligible_a6(paths),
}


def _agent_completed(manifest: Manifest, agent_id: str) -> bool:
    return manifest.agents.get(agent_id, AgentStatus()).status == "completed"


def _single_inputs(agent_id: str, paths: RunPaths) -> List[str]:
    """Conventional input paths for single-agent work items."""
    if agent_id == "sam_a1_protocol":
        return []  # query lives in manifest.config
    if agent_id == "sam_a2_search":
        return [str(paths.single_output("sam_a1_protocol"))]
    if agent_id == "sam_a7_synthesis":
        # A5 + A6 indexes feed A7
        return [
            str(paths.agent_dir("sam_a5_extraction")),
            str(paths.agent_dir("sam_a6_quality")),
        ]
    if agent_id == "sam_a8_manuscript":
        return [
            str(paths.single_output("sam_a1_protocol")),
            str(paths.single_output("sam_a2_search")),
            str(paths.agent_dir("sam_a3_screening")),
            str(paths.agent_dir("sam_a4_fulltext")),
            str(paths.agent_dir("sam_a5_extraction")),
            str(paths.agent_dir("sam_a6_quality")),
            str(paths.single_output("sam_a7_synthesis")),
        ]
    if agent_id == "sam_a9_explainer":
        return [str(paths.run_dir)]
    return []


def _per_paper_inputs(agent_id: str, paths: RunPaths) -> List[str]:
    """Conventional input paths for per-paper agents (all share the same
    base inputs; subagent uses paper_id to pick its target)."""
    base = [
        str(paths.single_output("sam_a1_protocol")),
        str(paths.single_output("sam_a2_search")),
    ]
    if agent_id == "sam_a4_fulltext":
        base.append(str(paths.agent_dir("sam_a3_screening")))
    if agent_id == "sam_a5_extraction":
        base.append(str(paths.agent_dir("sam_a4_fulltext")))
    if agent_id == "sam_a6_quality":
        base.append(str(paths.agent_dir("sam_a4_fulltext")))
    return base


def next_work(
    run_id: str, outputs_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Compute the next unit of work for the given run_id.

    Returns a dict ready to be json.dump()-ed to stdout by the CLI.
    """
    paths = RunPaths(run_id, outputs_root=outputs_root)
    if not paths.run_dir.exists():
        raise RunNotFoundError(f"run not found: {paths.run_dir}")
    manifest = load_manifest(run_id, outputs_root=outputs_root)

    for agent_id in AGENT_IDS:
        if _agent_completed(manifest, agent_id):
            continue

        mode = AGENT_MODES[agent_id]

        if mode == "single":
            return {
                "agent": agent_id,
                "paper_id": None,
                "input_paths": _single_inputs(agent_id, paths),
                "output_path": str(paths.single_output(agent_id)),
            }

        # per_paper
        elig_fn = _PER_PAPER_ELIGIBLE.get(agent_id)
        if elig_fn is None:
            raise InvalidWorkStateError(
                f"no eligible-set function for per-paper agent {agent_id}"
            )
        eligible = elig_fn(paths, manifest)

        # If no eligible papers at all, this agent has nothing to do —
        # mark it completed in manifest and move on.
        if not eligible:
            agent_status = manifest.agents.get(agent_id, AgentStatus())
            if agent_status.status != "completed":
                agent_status.status = "completed"
                agent_status.papers_total = 0
                agent_status.papers_done = 0
                if agent_status.started_at is None:
                    agent_status.started_at = utc_now_iso()
                agent_status.completed_at = utc_now_iso()
                manifest.agents[agent_id] = agent_status
                save_manifest_atomic(manifest, run_id, outputs_root=outputs_root)
            continue

        next_pid = _next_per_paper_id(paths, agent_id, eligible)
        if next_pid is None:
            # All eligible papers processed — mark completed and continue.
            agent_status = manifest.agents.get(agent_id, AgentStatus())
            agent_status.status = "completed"
            agent_status.papers_total = len(eligible)
            agent_status.papers_done = len(eligible)
            if agent_status.started_at is None:
                agent_status.started_at = utc_now_iso()
            agent_status.completed_at = utc_now_iso()
            manifest.agents[agent_id] = agent_status
            save_manifest_atomic(manifest, run_id, outputs_root=outputs_root)
            continue

        # Surface papers_total on the first dispatch so audit can compare later.
        agent_status = manifest.agents.get(agent_id, AgentStatus())
        if agent_status.papers_total != len(eligible):
            agent_status.papers_total = len(eligible)
            if agent_status.papers_done is None:
                agent_status.papers_done = 0
            if agent_status.status == "pending":
                agent_status.status = "in_progress"
                if agent_status.started_at is None:
                    agent_status.started_at = utc_now_iso()
            manifest.agents[agent_id] = agent_status
            save_manifest_atomic(manifest, run_id, outputs_root=outputs_root)

        return {
            "agent": agent_id,
            "paper_id": next_pid,
            "input_paths": _per_paper_inputs(agent_id, paths),
            "output_path": str(paths.paper_output(agent_id, next_pid)),
        }

    return {"done": True}
