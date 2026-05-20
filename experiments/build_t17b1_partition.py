"""
T17.B.1 — Phase 1 / Script 1: Build dev/held-out partition.

Generates `experiments/T17.B.1/partition.json` from Paula's N=553 sport corpus.

Logic:
  1. Load the medio sample CSV (566 covidence_ids).
  2. Restrict to the 553 ids that have predictions in the canonical hero2 dump
     (intersection between CSV and dump). The CSV has 13 extra ids that were
     filtered out during the hero2 run; we partition only the 553 actionable
     ones so the baseline-on-dev script can reuse the same subset cleanly.
  3. JOIN with `data/gold_standard.json` to obtain binary `gold_decision`
     (INCLUDE/EXCLUDE) per covidence_id. Gold standard keys lack the `#`
     prefix, so we strip it before lookup. Abort if any id is missing.
  4. Stratified 70/30 split using sklearn (random_state=42, test_size=0.30).
  5. Verify prevalences are within +/- 0.02 of the global prevalence.
  6. Persist partition.json with git HEAD sha and ISO 8601 UTC timestamp.

Usage:
    ~/meta-analisis-automatizado/venv/bin/python \
        experiments/build_t17b1_partition.py
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Constants (paths)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path("~/meta-analisis-automatizado")
CSV_PATH = (
    PROJECT_ROOT
    / "data/gold_standard/deporte/samples/medio/sample_medio_a4.csv"
)
GOLD_STANDARD_PATH = PROJECT_ROOT / "data/gold_standard.json"
HERO2_DUMP_PATH = (
    PROJECT_ROOT / "outputs/eval/medio/hero2_results_20260519_025605.json"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments/T17.B.1"
OUTPUT_PATH = OUTPUT_DIR / "partition.json"

# Expected actionable corpus size (intersection CSV ∩ hero2 dump).
EXPECTED_N = 553
TEST_SIZE = 0.30
SEED = 42
PREVALENCE_TOLERANCE = 0.02


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def git_head_sha() -> str:
    """Return the current git HEAD short sha (full 40 chars)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_csv_ids() -> list[str]:
    """Load covidence_ids from Paula's medio sample CSV."""
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["covidence_id"] for r in rows]


def load_dump_ids() -> set[str]:
    """Load covidence_ids that have predictions in the canonical hero2 dump."""
    with HERO2_DUMP_PATH.open(encoding="utf-8") as f:
        dump = json.load(f)
    return {r["input_id"] for r in dump["raw_predictions"]}


def load_gold_decisions() -> dict[str, str]:
    """Build {covidence_id_no_hash: gold_decision} from gold_standard.json."""
    with GOLD_STANDARD_PATH.open(encoding="utf-8") as f:
        gs = json.load(f)
    return {p["covidence_id"]: p["gold_decision"] for p in gs["papers"]}


def prevalence(labels: list[str], positive: str = "INCLUDE") -> float:
    """Return fraction of `positive` labels in the list."""
    if not labels:
        return 0.0
    return sum(1 for lab in labels if lab == positive) / len(labels)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load sources
    csv_ids = load_csv_ids()
    dump_ids = load_dump_ids()
    gold_map = load_gold_decisions()

    # 2) Restrict to intersection CSV ∩ dump (the 553 actionable corpus)
    actionable_ids = [cid for cid in csv_ids if cid in dump_ids]
    if len(actionable_ids) != EXPECTED_N:
        raise RuntimeError(
            f"Esperaba N={EXPECTED_N} ids actionables (CSV ∩ dump), "
            f"pero encontré {len(actionable_ids)}. Abortando."
        )

    # 3) JOIN with gold_standard.json (strip '#' prefix to match GS keys)
    labels: list[str] = []
    missing: list[str] = []
    for cid in actionable_ids:
        key = cid.lstrip("#")
        decision = gold_map.get(key)
        if decision is None:
            missing.append(cid)
        else:
            labels.append(decision)
    if missing:
        raise RuntimeError(
            "Los siguientes covidence_ids no aparecen en "
            f"gold_standard.json: {missing[:10]} "
            f"(total {len(missing)}). Abortando."
        )
    if not all(d in ("INCLUDE", "EXCLUDE") for d in labels):
        bad = sorted({d for d in labels if d not in ("INCLUDE", "EXCLUDE")})
        raise RuntimeError(
            f"gold_decision contiene valores inesperados: {bad}. Abortando."
        )

    # 4) Stratified 70/30 split
    dev_ids, heldout_ids, dev_labels, heldout_labels = train_test_split(
        actionable_ids,
        labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    # 5) Prevalences
    prev_global = prevalence(labels)
    prev_dev = prevalence(dev_labels)
    prev_heldout = prevalence(heldout_labels)

    # 6) Asserts (verifications)
    assert len(dev_ids) + len(heldout_ids) == EXPECTED_N, (
        f"dev+heldout ({len(dev_ids) + len(heldout_ids)}) != {EXPECTED_N}"
    )
    assert set(dev_ids).isdisjoint(set(heldout_ids)), (
        "dev y held-out se solapan; el split es incorrecto"
    )
    assert abs(prev_dev - prev_global) <= PREVALENCE_TOLERANCE, (
        f"prevalencia dev ({prev_dev:.4f}) fuera de tolerancia "
        f"+/-{PREVALENCE_TOLERANCE} frente al global ({prev_global:.4f})"
    )
    assert abs(prev_heldout - prev_global) <= PREVALENCE_TOLERANCE, (
        f"prevalencia heldout ({prev_heldout:.4f}) fuera de tolerancia "
        f"+/-{PREVALENCE_TOLERANCE} frente al global ({prev_global:.4f})"
    )

    # 7) Persist
    payload = {
        "seed": SEED,
        "n_total": EXPECTED_N,
        "n_dev": len(dev_ids),
        "n_heldout": len(heldout_ids),
        "stratification": "binary gold_decision",
        "prevalence_global": prev_global,
        "prevalence_dev": prev_dev,
        "prevalence_heldout": prev_heldout,
        "git_commit": git_head_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dev": list(dev_ids),
        "heldout": list(heldout_ids),
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 8) stdout summary (en español)
    print("[T17.B.1] Particion dev / held-out generada correctamente.")
    print(f"  Fichero       : {OUTPUT_PATH}")
    print(f"  n_total       : {EXPECTED_N}")
    print(f"  n_dev         : {len(dev_ids)}")
    print(f"  n_heldout     : {len(heldout_ids)}")
    print(f"  prev_global   : {prev_global:.4f}")
    print(f"  prev_dev      : {prev_dev:.4f}")
    print(f"  prev_heldout  : {prev_heldout:.4f}")
    print(f"  seed          : {SEED}")


if __name__ == "__main__":
    main()
