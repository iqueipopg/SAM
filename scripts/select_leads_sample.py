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
    sampling_method: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the JSON payload to persist.

    Output schema:
        {
          "metadata": {
            "dataset": "<repo>", "config": "<name>", "split": "<split>",
            "split_size": <int>, "seed": <int>, "n": <int>,
            "sampling_method": "<exact code>",
            "numpy_version": "<runtime>", "datasets_version": "<runtime>",
            "timestamp": "<UTC ISO 8601>", "git_commit": "<HEAD short hash>",
            <optional extras: pairing_strategy, parent_smoke_file, ...>
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

    if sampling_method is None:
        sampling_method = (
            "numpy.random.default_rng(seed).choice(len(ds), size=n, replace=False)"
        )

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

    metadata: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "split_size": split_size,
        "seed": seed,
        "n": n,
        "sampling_method": sampling_method,
        "numpy_version": np.__version__,
        "datasets_version": _datasets_pkg.__version__,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _get_git_commit(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {"metadata": metadata, "samples": samples}


def _load_smoke_indices(smoke_path: Path) -> tuple[list[int], list[str]]:
    """Read smoke artefact and return (selected_indices, sample_ids) in smoke order.

    The smoke file is the source of truth for the hard-prefix of the stitched
    sample. Order is preserved exactly as it appears in ``samples`` (NOT
    re-sorted). The caller uses this order for positions 0..len(smoke)-1 of
    the stitched artefact.
    """
    raw = json.loads(smoke_path.read_text())
    samples = raw.get("samples", [])
    if not samples:
        raise ValueError(f"smoke file has no samples: {smoke_path}")
    indices = [int(s["selected_index"]) for s in samples]
    ids = [str(s.get("id", "")) for s in samples]
    return indices, ids


def _load_parent_indices(parent_path: Path) -> tuple[list[int], list[str]]:
    """Read parent artefact and return (selected_indices, sample_ids).

    Used by ``_run_disjoint_mode`` to compute the complement set for held-out
    sampling. Order is not significant — these are excluded from the draw.
    """
    raw = json.loads(parent_path.read_text())
    samples = raw.get("samples", [])
    if not samples:
        raise ValueError(f"parent file has no samples: {parent_path}")
    indices = [int(s["selected_index"]) for s in samples]
    ids = [str(s.get("id", "")) for s in samples]
    return indices, ids


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
    parser.add_argument(
        "--stitch-from-smoke",
        type=Path,
        default=None,
        help=(
            "If given, build a stitched sample whose first len(smoke) positions "
            "are the smoke records (in smoke order) and the remaining n-len(smoke) "
            "are drawn deterministically from the complement with "
            "numpy.random.default_rng(seed).choice(remaining, size=delta, replace=False), "
            "then sorted ascending. Smoke path stays untouched when this flag is omitted."
        ),
    )
    parser.add_argument(
        "--disjoint-from",
        type=Path,
        default=None,
        help=(
            "If given, build a sample of N indices drawn from the complement of "
            "the parent artefact's selected_index set (held-out from DEV). Uses "
            "numpy.random.default_rng(seed).choice(complement, size=n, replace=False) "
            "then sorts ascending. Mutually exclusive with --stitch-from-smoke."
        ),
    )
    args = parser.parse_args(argv)
    if args.stitch_from_smoke is not None and args.disjoint_from is not None:
        parser.error("--stitch-from-smoke and --disjoint-from are mutually exclusive")
    return args


def _run_smoke_mode(records: list[dict[str, Any]], *, seed: int, n: int) -> tuple[list[int], dict[str, Any]]:
    """Original (pre-stitched) sampling path. Bit-identical to v1 behaviour."""
    split_size = len(records)
    rng = np.random.default_rng(seed)
    raw_indices = rng.choice(split_size, size=n, replace=False)
    indices = sorted(int(i) for i in raw_indices)
    _log(f"seed={seed} n={n} selected_indices={indices}")
    return indices, {}


def _run_stitched_mode(
    records: list[dict[str, Any]],
    *,
    seed: int,
    n: int,
    smoke_path: Path,
) -> tuple[list[int], dict[str, Any], str]:
    """Stitched sampling: hard-prefix smoke + delta draw from complement.

    Returns (indices, extra_metadata, sampling_method_string).
    Positions 0..len(smoke)-1 preserve smoke order (NOT sorted).
    Positions len(smoke)..n-1 are sorted ascending.
    """
    split_size = len(records)
    smoke_indices, smoke_ids = _load_smoke_indices(smoke_path)
    smoke_set = set(smoke_indices)
    if not smoke_set.issubset(range(split_size)):
        raise ValueError(
            f"smoke indices {smoke_indices} contain values outside [0, {split_size})"
        )
    delta = n - len(smoke_indices)
    if delta < 0:
        raise ValueError(
            f"--n={n} smaller than smoke size={len(smoke_indices)}; cannot stitch"
        )

    remaining = sorted(set(range(split_size)) - smoke_set)
    _log(
        f"stitch mode: smoke={len(smoke_indices)} delta={delta} "
        f"complement_size={len(remaining)} seed={seed}"
    )
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(remaining), size=delta, replace=False).tolist()
    delta_indices = sorted(remaining[i] for i in picks)

    indices = list(smoke_indices) + delta_indices  # smoke ORDER preserved
    _log(
        f"stitch result: smoke_prefix={smoke_indices} smoke_ids={smoke_ids} "
        f"delta_sorted={delta_indices}"
    )

    sampling_method = (
        "smoke3_prefix + numpy.random.default_rng(42).choice("
        "remaining_indices_after_smoke, size=17, replace=False)"
    )
    extra_metadata = {
        "pairing_strategy": (
            "stitched: smoke N=3 hard-prefixed, 17 new samples drawn from "
            "complement with seed=42"
        ),
        "parent_smoke_file": str(smoke_path).replace(
            str(PROJECT_ROOT) + "/", ""
        ),
        "smoke_prefix_indices": smoke_indices,
        "smoke_prefix_ids": smoke_ids,
        "delta_indices_sorted": delta_indices,
    }
    return indices, extra_metadata, sampling_method


def _run_disjoint_mode(
    records: list[dict[str, Any]],
    *,
    seed: int,
    n: int,
    parent_path: Path,
) -> tuple[list[int], dict[str, Any], str]:
    """Disjoint sampling: draw N indices from complement of a parent DEV artefact.

    Held-out style for T17.A2: parent artefact is the DEV sample (e.g.
    ``leads_full20.json``); we want N fresh indices that share zero overlap
    with DEV so the winner of prompt iteration can be validated on truly
    unseen data.

    Returns (indices, extra_metadata, sampling_method_string).
    Indices are sorted ascending.
    """
    split_size = len(records)
    parent_indices, parent_ids = _load_parent_indices(parent_path)
    parent_set = set(parent_indices)
    if not parent_set.issubset(range(split_size)):
        raise ValueError(
            f"parent indices contain values outside [0, {split_size}): "
            f"{sorted(parent_set - set(range(split_size)))}"
        )

    complement = sorted(set(range(split_size)) - parent_set)
    if len(complement) < n:
        raise ValueError(
            f"complement size={len(complement)} < n={n}; cannot draw disjoint sample"
        )

    _log(
        f"disjoint mode: parent_n={len(parent_indices)} "
        f"complement_size={len(complement)} drawn={n}"
    )
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(complement), size=n, replace=False).tolist()
    disjoint_indices = sorted(complement[i] for i in picks)

    _log(f"disjoint result: heldout_indices={disjoint_indices}")

    rel_parent = str(parent_path).replace(str(PROJECT_ROOT) + "/", "")
    sampling_method = (
        f"disjoint-from {rel_parent} with "
        f"numpy.random.default_rng({seed}).choice(complement, size={n}, replace=False)"
    )
    extra_metadata = {
        "pairing_strategy": (
            f"disjoint: N={n} drawn from complement of parent DEV "
            f"(N={len(parent_indices)}) with seed={seed}"
        ),
        "parent_dev_file": rel_parent,
        "dev_excluded_ids": parent_ids,
        "dev_excluded_indices": parent_indices,
        "complement_size": len(complement),
    }
    return disjoint_indices, extra_metadata, sampling_method


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

    if args.stitch_from_smoke is not None:
        indices, extra_metadata, sampling_method = _run_stitched_mode(
            records,
            seed=args.seed,
            n=args.n,
            smoke_path=args.stitch_from_smoke,
        )
        payload = _build_payload(
            records,
            indices,
            seed=args.seed,
            n=args.n,
            split_size=split_size,
            sampling_method=sampling_method,
            extra_metadata=extra_metadata,
        )

        # Hard sanity gate: positions 0..len(smoke)-1 must reproduce smoke IDs.
        expected_ids = extra_metadata["smoke_prefix_ids"]
        actual_ids = [s["id"] for s in payload["samples"][: len(expected_ids)]]
        if actual_ids != expected_ids:
            _log(
                f"ERROR: sanity gate failed. expected_prefix={expected_ids} "
                f"got={actual_ids}"
            )
            return 3
        _log(f"sanity gate PASSED: prefix_ids={actual_ids}")
    elif args.disjoint_from is not None:
        indices, extra_metadata, sampling_method = _run_disjoint_mode(
            records,
            seed=args.seed,
            n=args.n,
            parent_path=args.disjoint_from,
        )
        payload = _build_payload(
            records,
            indices,
            seed=args.seed,
            n=args.n,
            split_size=split_size,
            sampling_method=sampling_method,
            extra_metadata=extra_metadata,
        )

        # Hard sanity gate: zero overlap between the new sample and the parent DEV.
        parent_ids_set = set(extra_metadata["dev_excluded_ids"])
        new_ids_set = {s["id"] for s in payload["samples"]}
        overlap = parent_ids_set & new_ids_set
        if overlap:
            _log(
                f"ERROR: disjoint sanity gate failed. overlap_ids={sorted(overlap)}"
            )
            return 3
        _log(
            f"disjoint sanity gate PASSED: zero overlap between heldout "
            f"({len(new_ids_set)} ids) and parent DEV ({len(parent_ids_set)} ids)"
        )
    else:
        indices, _extras = _run_smoke_mode(records, seed=args.seed, n=args.n)
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
