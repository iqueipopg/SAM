# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Subset specification parsing and loading.

Spec syntax: 'path/to/partition.json:split_name', e.g.
'experiments/T17.B.1/partition.json:dev'.

The partition JSON is expected to be either:
  (a) A dict with split keys -> list of IDs:
        {"dev": ["id1","id2",...], "test": [...]}
  (b) A dict with split keys -> dict-of-records (we extract the keys):
        {"dev": {"id1": {...}, "id2": {...}}, ...}

Returned IDs are strings. For C.0 we don't resolve internal-id <-> DOI
mapping; intersection vs eligible papers is performed in state.py using
the values as-is (matched against `pmid` of each paper).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from .errors import SubsetMappingMissingError


def parse_subset_spec(spec: str) -> Tuple[Path, str]:
    """Split 'path:split' into (Path, split_name).

    Raises ValueError if the format is invalid.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"invalid subset spec {spec!r}; expected 'path/to/partition.json:split'"
        )
    # rsplit on the last ':' so Windows-style drive letters don't trip us up.
    path_str, _, split = spec.rpartition(":")
    if not path_str or not split:
        raise ValueError(
            f"invalid subset spec {spec!r}; both path and split must be non-empty"
        )
    return Path(path_str), split


def validate_subset_spec(spec: str) -> Tuple[Path, str]:
    """Parse + check the partition file exists. Returns (Path, split).

    Used by `init` to fail fast.
    """
    path, split = parse_subset_spec(spec)
    if not path.exists():
        raise SubsetMappingMissingError(
            f"partition file not found: {path}"
        )
    return path, split


def load_subset_ids(spec: str) -> set[str]:
    """Load IDs from the indicated split of the partition file.

    Raises SubsetMappingMissingError if path missing or split absent.
    """
    path, split = validate_subset_spec(spec)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SubsetMappingMissingError(
            f"partition JSON unreadable {path}: {e}"
        ) from e

    if not isinstance(data, dict):
        raise SubsetMappingMissingError(
            f"partition JSON must be a top-level object, got {type(data).__name__}"
        )
    if split not in data:
        raise SubsetMappingMissingError(
            f"split {split!r} not present in {path}; "
            f"available: {sorted(data.keys())}"
        )

    block = data[split]
    ids: set[str] = set()
    if isinstance(block, list):
        for item in block:
            ids.add(str(item))
    elif isinstance(block, dict):
        for key in block.keys():
            ids.add(str(key))
    else:
        raise SubsetMappingMissingError(
            f"split {split!r} must be list or dict, got {type(block).__name__}"
        )
    return ids
