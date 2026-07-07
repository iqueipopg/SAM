# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Build _fulltext_cache/ for a sam run from LOCAL PDFs (D7 input-parity, T17.C.1 PASO 2).

LOCAL-PDF analogue of scripts/build_fulltext_cache.py (the ONLINE-cascade
builder). Instead of running Agent4's retrieval cascade, this script extracts
text from PDFs already on disk, reusing the EXACT SAME extractor that the
Qwen "B" evaluation track (experiments/eval_hero2_fulltext.py) used. Reusing
experiments._eval_common.extract_pdf_text guarantees byte-identical input text
between the Claude "A" track and the Qwen "B" track.

experiments/_eval_common.py is NOT modified; it is imported as a library.

Usage:
    python scripts/build_fulltext_cache_local.py papers_spec.json <run_id> [--force]

papers_spec.json: JSON list of paper dicts. Each dict MUST include `paper_id`
(canonical Covidence ID with `#` prefix) and `pdf_path` (absolute, or relative
to the repo root); may include `title`, `abstract`, `doi`, `authors`,
`journal`, `year`.
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

from experiments._eval_common import extract_pdf_text  # noqa: E402
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
        if not isinstance(p, dict):
            raise SystemExit(f"papers_input[{i}] is not an object")
        if not p.get("paper_id"):
            raise SystemExit(f"papers_input[{i}] missing paper_id")
        if not p.get("pdf_path"):
            raise SystemExit(f"papers_input[{i}] missing pdf_path")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build _fulltext_cache/ for a sam run from local PDFs."
    )
    parser.add_argument(
        "papers_input", type=Path, help="JSON list of paper dicts."
    )
    parser.add_argument(
        "run_id", help="Existing run_id; cache goes under outputs/<run_id>/."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if cache files already exist.",
    )
    args = parser.parse_args()

    if not args.papers_input.exists():
        raise SystemExit(f"papers_input not found: {args.papers_input}")

    papers = _load_papers(args.papers_input)
    run_paths = RunPaths(args.run_id)
    cache_dir = run_paths.run_dir / "_fulltext_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[build_fulltext_cache_local] run_id={args.run_id} "
        f"papers={len(papers)} cache_dir={cache_dir} force={args.force}"
    )

    summary_by_source: dict[str, int] = {}
    written = 0
    total = len(papers)

    for i, orig in enumerate(papers, start=1):
        paper_id = orig["paper_id"]
        slug = paper_id_to_slug(paper_id)
        text_path = cache_dir / f"{slug}.txt"
        meta_path = cache_dir / f"{slug}.meta.json"

        # Skip already-cached papers (idempotent resumption) unless --force.
        if not args.force and meta_path.exists() and text_path.exists():
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

        # Resolve pdf_path relative to the repo root when not absolute.
        pdf_path = Path(orig["pdf_path"])
        if not pdf_path.is_absolute():
            pdf_path = (_REPO_ROOT / pdf_path).resolve()

        text, err = extract_pdf_text(pdf_path)

        if text:
            full_text = text
            text_source = "local_pdf"
            extraction_error = None
        else:
            abstract = orig.get("abstract") or ""
            full_text = abstract
            text_source = "abstract_only"
            extraction_error = err

        text_path.write_text(full_text, encoding="utf-8")
        meta = {
            "paper_id": paper_id,
            "slug": slug,
            "text_source": text_source,
            "full_text_length": len(full_text),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pdf_path": str(pdf_path),
            "extraction_error": extraction_error,
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
    print(f"[build_fulltext_cache_local] wrote {written} entries to {cache_dir}")
    for src, n in sorted(summary_by_source.items()):
        print(f"  {src}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
