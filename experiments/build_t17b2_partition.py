"""
T17.B.2 partition copy — replicates the T17.B.1 dev partition byte-for-byte.

Identical dev/held-out split guarantees comparability across iterations:
T17.B.2 must score on the same 387 dev papers as T17.B.1.

Run:
    python -m experiments.build_t17b2_partition
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Replicate T17.B.1 partition into T17.B.2/")
    p.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "T17.B.1" / "partition.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "T17.B.2" / "partition.json",
    )
    args = p.parse_args(argv)

    if not args.source.is_file():
        print(f"ERROR: source partition not found: {args.source}", file=sys.stderr)
        return 2

    src_bytes = args.source.read_bytes()
    src_hash = _sha256(src_bytes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(src_bytes)

    dst_hash = _sha256(args.output.read_bytes())

    if dst_hash != src_hash:
        print(
            f"ERROR: SHA256 mismatch — src={src_hash}, dst={dst_hash}",
            file=sys.stderr,
        )
        return 1

    print(f"Partición copiada: {args.output}")
    print(f"SHA256: {dst_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
