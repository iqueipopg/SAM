#!/usr/bin/env python3
"""Select reproducible random sample from LEADSInstruct for T17.A1 Hero A1 evaluation.

This script is part of the pre-registered T17.A1 characterisation iteration of
``agents/agent1_protocol.py``. It does NOT modify A1 itself. It samples N records
from the ``test`` split of ``zifeng-ai/LEADSInstruct`` (config
``study_search_publication``) with a fixed seed so that downstream phases (C.2
eval script and full-run scaling) can reuse exactly the same draws.

Sampling method (pre-registered, immutable):
    rng = numpy.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n, replace=False)

The ``origin.abstract`` field is intentionally NOT persisted. The pre-registered
hypothesis §5.3 excludes it from the sample artefact to avoid abstract-text
leakage into the A1 input (A1 receives only the title).

Output JSON schema is documented inline near ``_build_payload``.

Usage:
    python scripts/select_leads_sample.py --n 3 --seed 42 \
        --output data/groundtruth/leads_smoke.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "groundtruth" / "leads_smoke.json"

DATASET_NAME = "zifeng-ai/LEADSInstruct"
DATASET_CONFIG = "study_search_publication"
DATASET_SPLIT = "test"


def _log(msg: str) -> None:
    """Print a structured log line to stdout (no emojis, regla #12)."""
    print(f"[T17.A1.C1] {msg}", flush=True)


def _get_git_commit() -> str:
    """Return the short HEAD hash; empty string if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def _load_split_streaming() -> list[dict[str, Any]]:
    """Materialise the test split via streaming and return as a list.

    Streaming is used because the non-streaming loader fails on the
    ``prospective_test`` split of this dataset config (schema mismatch in the
    upstream JSON). Streaming bypasses that split entirely. The ``test`` split
    is small enough (~4,217 records) to fit in memory.
    """
    from datasets import load_dataset

    _log(
        f"loading dataset name={DATASET_NAME} config={DATASET_CONFIG} "
        f"split={DATASET_SPLIT} streaming=True"
    )
    ds = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )
    records = list(ds)
    _log(f"materialised split_size={len(records)}")
    return records


def _normalise_origin(raw: Any) -> dict[str, Any]:
    """Coerce ``origin`` into ``{'pmid': str, 'title': str}`` (no abstract).

    The HuggingFace schema stores ``origin`` as a dict with keys
    ``pmid``/``title``/``abstract``. We accept dict-shaped input here and also
    the flat-key fallback (``origin.pmid``/``origin.title``) just in case.
    """
    if isinstance(raw, dict):
        pmid = raw.get("pmid", "")
        title = raw.get("title", "")
    else:
        pmid = ""
        title = ""
    return {"pmid": str(pmid) if pmid is not None else "", "title": title or ""}


def _normalise_pico(raw: Any) -> dict[str, str]:
    """Coerce ``pico`` into ``{'P','I','C','O'}`` with string values."""
    if not isinstance(raw, dict):
        return {"P": "", "I": "", "C": "", "O": ""}
    out: dict[str, str] = {}
    for key in ("P", "I", "C", "O"):
        val = raw.get(key, "")
        out[key] = val if isinstance(val, str) else str(val) if val is not None else ""
    return out


def _normalise_pmids(raw: Any) -> list[str]:
    """Coerce ``publication_pmids`` into a list of string PMIDs."""
    if not isinstance(raw, list):
        return []
    return [str(p) for p in raw if p is not None and str(p).strip() != ""]


def _build_payload(
    records: list[dict[str, Any]],
    indices: Iterable[int],
    *,
    seed: int,
    n: int,
    split_size: int,
) -> dict[str, Any]:
    """Construct the JSON payload to persist.

    Output schema:
        {
          "metadata": {
            "dataset": "<repo>", "config": "<name>", "split": "<split>",
            "split_size": <int>, "seed": <int>, "n": <int>,
            "sampling_method": "<exact code>",
            "numpy_version": "<runtime>", "datasets_version": "<runtime>",
            "timestamp": "<UTC ISO 8601>", "git_commit": "<HEAD short hash>"
          },
          "samples": [
            {
              "selected_index": <int>, "id": "<MLSEP######>",
              "origin": {"pmid": "...", "title": "..."},
              "pico": {"P": "...", "I": "...", "C": "...", "O": "..."},
              "publication_pmids": ["...", ...],
              "publication_pmids_count": <int>
            }, ...
          ]
        }

    ``origin.abstract`` is intentionally omitted (§5.3 of pre-registro).
    """
    import datasets as _datasets_pkg

    samples: list[dict[str, Any]] = []
    for idx in indices:
        rec = records[idx]
        pmids = _normalise_pmids(rec.get("publication_pmids"))
        sample = {
            "selected_index": int(idx),
            "id": rec.get("id", ""),
            "origin": _normalise_origin(rec.get("origin")),
            "pico": _normalise_pico(rec.get("pico")),
            "publication_pmids": pmids,
            "publication_pmids_count": len(pmids),
        }
        samples.append(sample)
        _log(
            f"sampled idx={sample['selected_index']} id={sample['id']} "
            f"pmids={sample['publication_pmids_count']}"
        )

    payload = {
        "metadata": {
            "dataset": DATASET_NAME,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "split_size": split_size,
            "seed": seed,
            "n": n,
            "sampling_method": (
                "numpy.random.default_rng(seed).choice(len(ds), size=n, replace=False)"
            ),
            "numpy_version": np.__version__,
            "datasets_version": _datasets_pkg.__version__,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _get_git_commit(),
        },
        "samples": samples,
    }
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible random sampling from LEADSInstruct test split for "
            "T17.A1 Hero A1 characterisation."
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of records to sample (default 3 for smoke).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for numpy.random.default_rng (default 42, pre-registered).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.n <= 0:
        _log(f"ERROR: --n must be positive (got {args.n})")
        return 2

    records = _load_split_streaming()
    split_size = len(records)

    if args.n > split_size:
        _log(f"ERROR: --n={args.n} exceeds split_size={split_size}")
        return 2

    rng = np.random.default_rng(args.seed)
    raw_indices = rng.choice(split_size, size=args.n, replace=False)
    # Ascending order for human-readable reproducibility; deterministic from
    # the same seed because we sort plain int values.
    indices = sorted(int(i) for i in raw_indices)
    _log(f"seed={args.seed} n={args.n} selected_indices={indices}")

    payload = _build_payload(
        records,
        indices,
        seed=args.seed,
        n=args.n,
        split_size=split_size,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _log(f"wrote artefact path={output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
