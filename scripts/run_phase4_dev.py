# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""T17.C.1 Phase 4 — deterministic driver for the A4 dev N=387 runs (mono + split V1).

This script is the DETERMINISTIC glue around the A4 Phase 4 runs. It does NOT
invoke any LLM / subagent (Task() is a harness tool only the orchestrator can call). It:

  - `setup`         : init_run + bootstrap A1/A2/A3 + build _fulltext_cache (local PDFs).
  - `assemble-split`: read reporter+decider JSONs, map to a decision-only
                       FulltextOutput payload, commit into the canonical
                       `sam_a4_fulltext` slot (orchestrator untouched).
  - `status`        : report variant + papers_done/total for sam_a4_fulltext.

The actual `next -> Task(subagent) -> commit` loop is driven by the orchestrator in-session,
in small batches. For the monolithic variant the orchestrator runs Task(sam_a4_fulltext) and
commits directly with `python -m sam_orchestrator commit ...`; for the split
variant the orchestrator runs Task(sam_a4_reporter) then Task(sam_a4_decider) and calls
`assemble-split` here to fold the two into the committed A4 decision.

Reuses (does not reimplement): sam_orchestrator (init_run, commit_output,
RunPaths, paper_id_to_slug), the partition/gold loaders and path constants from
experiments.eval_a4_split, PAULA_SPORT_PROTOCOL_V2 from experiments._eval_common,
and scripts/build_fulltext_cache_local.py (subprocess) for byte-identical PDF
extraction (input-parity with Qwen-B).

NO sacred zones are touched. Usage:
    python scripts/run_phase4_dev.py setup --variant mono [--limit N] [--query STR]
    python scripts/run_phase4_dev.py assemble-split <run_id> <paper_id>
    python scripts/run_phase4_dev.py status <run_id>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sam_orchestrator import (  # noqa: E402
    RunPaths,
    commit_output,
    init_run,
    paper_id_to_slug,
)
from experiments.eval_a4_split import (  # noqa: E402
    DEFAULT_GOLD_CSV,
    DEFAULT_PARTITION,
    DEFAULT_PDFS_DIR,
    DEFAULT_SAMPLE_CSV,
    _load_gold_index,
    _load_partition,
    _load_sample_index,
)
from experiments._eval_common import PAULA_SPORT_PROTOCOL_V2  # noqa: E402
from schemas.papers import Paper, PapersRawOutput, SearchStats  # noqa: E402
from schemas.protocol import ProtocolBody, ProtocolOutput  # noqa: E402
from schemas.reporter import DeciderOutput, ReporterOutput  # noqa: E402
from schemas.screening import ScreeningOutput, VoteEntry  # noqa: E402

_CONFIDENCE_TO_FLOAT = {"high": 0.9, "medium": 0.6, "low": 0.3}
_DEFAULT_QUERY = (
    "¿Mejoran las intervenciones de actividad física con componentes "
    "psicológicos los resultados de actividad física en adultos de edad de "
    "jubilación (50-70 años) que viven en la comunidad?"
)
_SUBSET_SPEC = f"{DEFAULT_PARTITION}:dev"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_year(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    # Drop a trailing ".0" from pandas-parsed numeric years.
    m = re.match(r"^(\d{4})(?:\.0)?$", s)
    return m.group(1) if m else s


def _authors_list(value: Any) -> list[str]:
    if value is None:
        return []
    s = str(value).strip()
    if s.lower() in ("", "nan", "none"):
        return []
    parts = [a.strip() for a in re.split(r"[;]", s) if a.strip()]
    return parts or [s]


def _opt(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none") else None


def _write_and_commit(run_id: str, agent_id: str, output_path: Path, payload: dict) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return commit_output(run_id, agent_id, str(output_path))


# ──────────────────────────────────────────────────────────────────────────────
# setup
# ──────────────────────────────────────────────────────────────────────────────

def cmd_setup(args: argparse.Namespace) -> int:
    variant = args.variant
    query = args.query or _DEFAULT_QUERY

    run_id = init_run(query, subset_spec=_SUBSET_SPEC, backend_default="claude")
    rp = RunPaths(run_id)

    # Variant sidecar (keeps manifest.config.query clean / semantic).
    (rp.run_dir / "_phase4_meta.json").write_text(
        json.dumps(
            {"variant": variant, "task": "T17.C.1_phase4", "created_at": None},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    partition = _load_partition(Path(DEFAULT_PARTITION))
    dev_ids: list[str] = list(partition["dev"])
    if args.limit is not None:
        dev_ids = dev_ids[: args.limit]
    sample_idx = _load_sample_index(Path(DEFAULT_SAMPLE_CSV))
    gold_idx = _load_gold_index(Path(DEFAULT_GOLD_CSV))

    ts = _now()

    # --- A1 protocol (bootstrap; not measured) ---
    body = ProtocolBody(
        framework=PAULA_SPORT_PROTOCOL_V2["framework"],
        research_question=PAULA_SPORT_PROTOCOL_V2["research_question"],
        population=PAULA_SPORT_PROTOCOL_V2["population"],
        intervention=PAULA_SPORT_PROTOCOL_V2["intervention"],
        comparator=PAULA_SPORT_PROTOCOL_V2["comparator"],
        outcome=PAULA_SPORT_PROTOCOL_V2["outcome"],
        original_query=query,
        inclusion_criteria=list(PAULA_SPORT_PROTOCOL_V2["inclusion_criteria"]),
        exclusion_criteria=list(PAULA_SPORT_PROTOCOL_V2["exclusion_criteria"]),
        protocol_document=(
            "Protocolo bootstrap T17.C.1 Phase 4 (PAULA_SPORT_PROTOCOL_V2). "
            "A1 no se ejecuta en esta corrida; se inyecta el protocolo V2 "
            "congelado para alimentar el screening full-text de A4."
        ),
        prospero_draft="PROSPERO CRD42021277712 (Collazo-Castiñeira et al. 2025).",
    )
    proto = ProtocolOutput(
        agent="sam_a1_protocol",
        backend="claude",
        timestamp=ts,
        tokens_in=None,
        tokens_out=None,
        query=query,
        protocol=body,
        pubmed_query="",
        version="V2",
    )
    _write_and_commit(
        run_id, "sam_a1_protocol", rp.single_output("sam_a1_protocol"), proto.model_dump()
    )

    # --- A2 papers_raw (bootstrap from gold metadata) ---
    papers: list[Paper] = []
    for cid in dev_ids:
        g = gold_idx.get(cid, {})
        papers.append(
            Paper(
                paper_id=cid,
                pmid=None,
                doi=_opt(g.get("doi")),
                title=g.get("title", "") or "",
                abstract=g.get("abstract", "") or "",
                authors=_authors_list(g.get("authors")),
                journal="",
                year=_coerce_year(g.get("year")),
                sources=["covidence"],
            )
        )
    n = len(papers)
    praw = PapersRawOutput(
        agent="sam_a2_search",
        backend="claude",
        timestamp=ts,
        tokens_in=None,
        tokens_out=None,
        papers=papers,
        stats=SearchStats(
            pubmed=0,
            europepmc=0,
            semantic_scholar=0,
            google_scholar=0,
            total_before_dedup=n,
            total_after_dedup=n,
            duplicates_removed=0,
            search_sources=["covidence"],
        ),
    )
    _write_and_commit(
        run_id, "sam_a2_search", rp.single_output("sam_a2_search"), praw.model_dump()
    )

    # --- A3 screening = INCLUDE per paper (force all dev papers into A4) ---
    for cid in dev_ids:
        g = gold_idx.get(cid, {})
        scr = ScreeningOutput(
            agent="sam_a3_screening",
            backend="claude",
            timestamp=ts,
            tokens_in=None,
            tokens_out=None,
            paper_id=cid,
            pmid=None,
            title=g.get("title", "") or "",
            decision="INCLUDE",
            confidence=1.0,
            votes=[
                VoteEntry(
                    decision="INCLUDE",
                    reasoning="bootstrap dev (forzar elegibilidad A4)",
                    model="bootstrap",
                )
            ],
            reasoning=(
                "Bootstrap Phase 4: todos los papers del dev pasan a A4 para medir "
                "el screening full-text. No es una decisión de screening real."
            ),
            flags=[],
            screening_confidence=1.0,
            borderline=False,
        )
        _write_and_commit(
            run_id,
            "sam_a3_screening",
            rp.paper_output("sam_a3_screening", cid),
            scr.model_dump(),
        )

    # --- Build _fulltext_cache (local PDFs; abstract_only fallback for no-PDF) ---
    with_pdf_spec: list[dict[str, Any]] = []
    no_pdf: list[tuple[str, str]] = []
    for cid in dev_ids:
        s = sample_idx.get(cid, {})
        g = gold_idx.get(cid, {})
        abstract = g.get("abstract", "") or ""
        if s.get("has_pdf") and _opt(s.get("pdf_filename")):
            pdf_path = Path(DEFAULT_PDFS_DIR) / str(s["pdf_filename"])
            with_pdf_spec.append(
                {
                    "paper_id": cid,
                    "pdf_path": str(pdf_path),
                    "title": g.get("title", "") or "",
                    "abstract": abstract,
                    "doi": g.get("doi", "") or "",
                }
            )
        else:
            no_pdf.append((cid, abstract))

    if with_pdf_spec:
        spec_path = rp.run_dir / "_phase4_papers_spec.json"
        spec_path.write_text(
            json.dumps(with_pdf_spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        builder = _REPO_ROOT / "scripts" / "build_fulltext_cache_local.py"
        ret = subprocess.run(
            [sys.executable, str(builder), str(spec_path), run_id],
            cwd=str(_REPO_ROOT),
        )
        if ret.returncode != 0:
            print(
                f"[run_phase4_dev] WARNING: builder exit {ret.returncode}",
                file=sys.stderr,
            )

    # No-PDF papers: write abstract_only cache with the SAME meta schema as the builder.
    cache_dir = rp.run_dir / "_fulltext_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for cid, abstract in no_pdf:
        slug = paper_id_to_slug(cid)
        (cache_dir / f"{slug}.txt").write_text(abstract, encoding="utf-8")
        meta = {
            "paper_id": cid,
            "slug": slug,
            "text_source": "abstract_only",
            "full_text_length": len(abstract),
            "retrieved_at": _now(),
            "pdf_path": None,
            "extraction_error": None,
        }
        (cache_dir / f"{slug}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(
        f"[run_phase4_dev] setup done: variant={variant} papers={len(dev_ids)} "
        f"with_pdf={len(with_pdf_spec)} abstract_only={len(no_pdf)}",
        file=sys.stderr,
    )
    print(run_id)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# assemble-split
# ──────────────────────────────────────────────────────────────────────────────

def cmd_assemble_split(args: argparse.Namespace) -> int:
    run_id = args.run_id
    paper_id = args.paper_id
    rp = RunPaths(run_id)
    slug = paper_id_to_slug(paper_id)

    reporter_path = rp.run_dir / "sam_a4_reporter" / f"{slug}.json"
    decider_path = rp.run_dir / "sam_a4_decider" / f"{slug}.json"

    rep: Optional[ReporterOutput] = None
    dec: Optional[DeciderOutput] = None
    errors: list[str] = []
    try:
        rep = ReporterOutput.model_validate_json(reporter_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"reporter: {exc}")
    try:
        dec = DeciderOutput.model_validate_json(decider_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"decider: {exc}")

    ts = _now()
    if rep is not None and dec is not None:
        decision = dec.decision
        conf = _CONFIDENCE_TO_FLOAT[dec.confidence]
        reasoning = dec.reason
        exclusion_reason = dec.reason if decision == "EXCLUDE" else None
        votes = [{"decision": decision, "reasoning": dec.reason, "model": "claude-opus-4-8"}]
        borderline = dec.confidence == "low"
    else:
        decision = "EXCLUDE"
        conf = 0.0
        reasoning = (
            "reporter/decider ausente o inválido — exclusión conservadora "
            f"(paridad B). Detalle: {'; '.join(errors)}"
        )
        exclusion_reason = reasoning
        votes = [{"decision": "EXCLUDE", "reasoning": reasoning, "model": "claude-opus-4-8"}]
        borderline = True

    payload = {
        "agent": "sam_a4_fulltext",
        "backend": "claude",
        "timestamp": ts,
        "tokens_in": None,
        "tokens_out": None,
        "paper_id": paper_id,
        "fulltext_decision": decision,
        "fulltext_reasoning": reasoning,
        "exclusion_reason": exclusion_reason,
        "fulltext_votes": votes,
        "fulltext_confidence": conf,
        "screening_confidence": conf,
        "borderline": borderline,
    }
    out_path = rp.paper_output("sam_a4_fulltext", paper_id)
    res = _write_and_commit(run_id, "sam_a4_fulltext", out_path, payload)
    print(json.dumps({"assembled": {"decision": decision, "confidence": conf}, "commit": res},
                     ensure_ascii=False))
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# status
# ──────────────────────────────────────────────────────────────────────────────

def cmd_status(args: argparse.Namespace) -> int:
    rp = RunPaths(args.run_id)
    manifest = json.loads(rp.manifest_path.read_text(encoding="utf-8"))
    sidecar_path = rp.run_dir / "_phase4_meta.json"
    variant = "desconocida"
    if sidecar_path.is_file():
        try:
            variant = json.loads(sidecar_path.read_text(encoding="utf-8")).get(
                "variant", "desconocida"
            )
        except Exception:  # noqa: BLE001
            pass
    a4 = manifest.get("agents", {}).get("sam_a4_fulltext", {})
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "variant": variant,
                "sam_a4_fulltext": {
                    "status": a4.get("status"),
                    "papers_done": a4.get("papers_done"),
                    "papers_total": a4.get("papers_total"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="T17.C.1 Phase 4 deterministic driver.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup", help="init_run + bootstrap A1/A2/A3 + build cache")
    p_setup.add_argument("--variant", choices=["mono", "split"], required=True)
    p_setup.add_argument("--limit", type=int, default=None, help="Process only first N dev papers (smoke).")
    p_setup.add_argument("--query", default=None, help="Override the clinical query.")
    p_setup.set_defaults(func=cmd_setup)

    p_asm = sub.add_parser("assemble-split", help="Fold reporter+decider into a committed A4 decision.")
    p_asm.add_argument("run_id")
    p_asm.add_argument("paper_id")
    p_asm.set_defaults(func=cmd_assemble_split)

    p_status = sub.add_parser("status", help="Report variant + A4 progress.")
    p_status.add_argument("run_id")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
