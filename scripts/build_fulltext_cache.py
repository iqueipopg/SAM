# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Build _fulltext_cache/ for a sam run (D7 input-parity, T17.C.1).

Reuses agents.agent4_fulltext.Agent4FullText.retrieve_fulltext() (Phase A,
no LLM) to extract and persist each paper's full text to
outputs/<run_id>/_fulltext_cache/<slug>.txt. Guarantees that sam_a4
(Claude) and agent4 (Qwen) read identical input text.

agents/agent4_fulltext.py is NOT modified; it is imported as a library.

Usage:
    python scripts/build_fulltext_cache.py papers.json <run_id>

papers.json: JSON list of paper dicts. Each dict must include `paper_id`
(canonical Covidence ID with `#` prefix) and `title`; may include `pmid`,
`doi`, `pmcid`, `abstract`, `authors`, `journal`, `year`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.agent4_fulltext import Agent4FullText  # noqa: E402
from sam_orchestrator.paths import RunPaths, paper_id_to_slug  # noqa: E402


def _load_papers(papers_input: Path) -> list[dict[str, Any]]:
    data = json.loads(papers_input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(
            f"papers_input must be a JSON list; got {type(data).__name__}"
        )
    if not data:
        raise SystemExit("papers_input is an empty list")
    for i, p in enumerate(data):
        if not p.get("paper_id"):
            raise SystemExit(f"papers_input[{i}] missing paper_id")
        if "title" not in p:
            raise SystemExit(f"papers_input[{i}] missing title")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build _fulltext_cache/ for a sam run."
    )
    parser.add_argument(
        "papers_input", type=Path, help="JSON list of paper dicts."
    )
    parser.add_argument(
        "run_id", help="Existing run_id; cache goes under outputs/<run_id>/."
    )
    parser.add_argument(
        "--ensemble-size", type=int, default=1,
        help="Agent4FullText ensemble_size (init only; Phase A doesn't use LLM).",
    )
    args = parser.parse_args()

    if not args.papers_input.exists():
        raise SystemExit(f"papers_input not found: {args.papers_input}")

    papers = _load_papers(args.papers_input)
    run_paths = RunPaths(args.run_id)
    cache_dir = run_paths.run_dir / "_fulltext_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Phase A only; this does NOT load an LLM (verified in T17.C.1 exploration).
    a4 = Agent4FullText(ensemble_size=args.ensemble_size)

    print(
        f"[build_fulltext_cache] run_id={args.run_id} "
        f"papers={len(papers)} cache_dir={cache_dir}"
    )

    # Build enriched stubs and invoke the cascade per-paper. We write the
    # .txt + .meta.json immediately after each paper so progress is visible
    # on disk and partial runs are resumable in spirit.
    # Rationale for bypassing retrieve_fulltext: it indexes papers by PMID
    # and discards `doi`/`pmcid` from the input when PMID is absent. Calling
    # _retrieve_single with a pre-populated enriched_paper preserves
    # DOI/PMCID so the cascade can actually use them. We never modify the
    # sacred module.
    stats_keys = [
        "pmc_fulltext", "unpaywall_pdf", "europepmc_fulltext",
        "semanticscholar_pdf", "openalex_pdf", "crossref_pdf", "doaj_pdf",
        "lens_fulltext", "core_fulltext", "google_scholar_pdf",
        "scihub_pdf", "abstract_only",
    ]
    stats: dict[str, int] = {k: 0 for k in stats_keys}
    summary_by_source: dict[str, int] = {}
    written = 0
    total = len(papers)

    for i, orig in enumerate(papers, start=1):
        paper_id = orig["paper_id"]
        slug = paper_id_to_slug(paper_id)
        text_path = cache_dir / f"{slug}.txt"
        meta_path = cache_dir / f"{slug}.meta.json"
        # Skip already-cached papers (idempotent resumption).
        if meta_path.exists() and text_path.exists():
            try:
                prev = json.loads(meta_path.read_text(encoding="utf-8"))
                src = prev.get("text_source") or "abstract_only"
                summary_by_source[src] = summary_by_source.get(src, 0) + 1
                written += 1
                print(
                    f"[{i}/{total}] {slug:30s}  src={src:20s}  "
                    f"len={prev.get('full_text_length', 0):>7d}  CACHED"
                )
                continue
            except Exception:
                pass  # corrupted meta, redo

        enriched_paper = {
            "pmid": orig.get("pmid") or "",
            "title": orig.get("title") or "",
            "abstract": orig.get("abstract") or "",
            "authors": orig.get("authors") or [],
            "journal": orig.get("journal") or "",
            "year": orig.get("year") or "",
            "doi": orig.get("doi") or "",
            "pmcid": orig.get("pmcid") or "",
            "full_text": None,
            "text_source": "abstract_only",
            "full_text_length": 0,
            "sections": {},
            "oa_url": None,
        }
        try:
            a4._retrieve_single(enriched_paper, stats)
        except Exception as e:
            print(f"  [!] {paper_id}: retrieval error: {e}")
        if (
            enriched_paper.get("full_text")
            and not enriched_paper.get("sections")
            and enriched_paper.get("text_source") != "abstract_only"
        ):
            try:
                secs = a4._parse_sections_from_text(enriched_paper["full_text"])
                if secs:
                    enriched_paper["sections"] = secs
            except Exception:
                pass

        full_text = enriched_paper.get("full_text") or ""
        text_source = enriched_paper.get("text_source") or "abstract_only"
        if not full_text and orig.get("abstract"):
            full_text = orig["abstract"]
            text_source = "abstract_only"

        text_path.write_text(full_text, encoding="utf-8")
        meta = {
            "paper_id": paper_id,
            "slug": slug,
            "text_source": text_source,
            "full_text_length": len(full_text),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pmid": orig.get("pmid") or None,
            "doi": orig.get("doi") or None,
            "oa_url": enriched_paper.get("oa_url"),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        written += 1
        summary_by_source[text_source] = summary_by_source.get(text_source, 0) + 1
        print(
            f"[{i}/{total}] {slug:30s}  src={text_source:20s}  "
            f"len={len(full_text):>7d}  ({paper_id})",
            flush=True,
        )

    print()
    print(f"[build_fulltext_cache] wrote {written} entries to {cache_dir}")
    for src, n in sorted(summary_by_source.items()):
        print(f"  {src}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
