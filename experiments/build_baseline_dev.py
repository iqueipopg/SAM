"""
T17.B.1 — Phase 1 / Script 2: Recompute baseline metrics on the dev subset.

Generates `experiments/T17.B.1/baseline_dev.json` by filtering the canonical
hero2 dump down to the dev partition (~387 covidence_ids) and recomputing the
6 binary classification metrics.

Logic:
  1. Load the canonical hero2 dump
     (outputs/eval/medio/hero2_results_20260519_025605.json).
  2. Load partition.json produced by build_t17b1_partition.py.
  3. Filter raw_predictions to ids in the dev partition. Abort if any dev id
     is missing in the dump (should not happen given the partition is built
     against the same dump).
  4. Build confusion matrix (positive = INCLUDE) and compute:
        sensitivity, specificity, accuracy, precision, F1, kappa.
  5. Persist baseline_dev.json with source sha256, partition pointer, and
     git HEAD sha.

Usage:
    ~/meta-analisis-automatizado/venv/bin/python \
        experiments/build_baseline_dev.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


# --------------------------------------------------------------------------- #
# Constants (paths)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path("~/meta-analisis-automatizado")
HERO2_DUMP_PATH = (
    PROJECT_ROOT / "outputs/eval/medio/hero2_results_20260519_025605.json"
)
PARTITION_PATH = PROJECT_ROOT / "experiments/T17.B.1/partition.json"
OUTPUT_PATH = PROJECT_ROOT / "experiments/T17.B.1/baseline_dev.json"

POSITIVE_LABEL = "INCLUDE"
NEGATIVE_LABEL = "EXCLUDE"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def git_head_sha() -> str:
    """Return the current git HEAD sha (full 40 chars)."""
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


def sha256_of_file(path: Path) -> str:
    """Compute the sha256 hex digest of a file in 64 KiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_label(raw: str) -> str:
    """Normalize a decision string to INCLUDE/EXCLUDE upper-case form."""
    return raw.strip().upper()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # 1) Load canonical dump
    with HERO2_DUMP_PATH.open(encoding="utf-8") as f:
        dump = json.load(f)
    raw_predictions = dump["raw_predictions"]

    # 2) Load partition
    with PARTITION_PATH.open(encoding="utf-8") as f:
        partition = json.load(f)
    dev_ids = set(partition["dev"])

    # 3) Filter
    by_id = {p["input_id"]: p for p in raw_predictions}
    missing = [cid for cid in dev_ids if cid not in by_id]
    if missing:
        raise RuntimeError(
            "Los siguientes covidence_ids del dev partition no aparecen "
            f"en el dump hero2: {missing[:10]} (total {len(missing)}). "
            "Abortando."
        )
    filtered: list[dict] = [by_id[cid] for cid in partition["dev"]]
    if len(filtered) != len(dev_ids):
        raise RuntimeError(
            f"Esperaba {len(dev_ids)} predicciones filtradas, "
            f"obtuve {len(filtered)}. Abortando."
        )

    # 4) Confusion matrix (positive = INCLUDE)
    y_true: list[str] = []
    y_pred: list[str] = []
    for item in filtered:
        gold = normalize_label(item["gold"]["decision"])
        pred = normalize_label(item["predicted"]["decision"])
        if gold not in (POSITIVE_LABEL, NEGATIVE_LABEL):
            raise RuntimeError(
                f"gold decision inesperada para {item['input_id']}: "
                f"{item['gold']['decision']!r}"
            )
        if pred not in (POSITIVE_LABEL, NEGATIVE_LABEL):
            raise RuntimeError(
                f"predicted decision inesperada para {item['input_id']}: "
                f"{item['predicted']['decision']!r}"
            )
        y_true.append(gold)
        y_pred.append(pred)

    tp = sum(
        1
        for t, p in zip(y_true, y_pred)
        if t == POSITIVE_LABEL and p == POSITIVE_LABEL
    )
    fp = sum(
        1
        for t, p in zip(y_true, y_pred)
        if t == NEGATIVE_LABEL and p == POSITIVE_LABEL
    )
    fn = sum(
        1
        for t, p in zip(y_true, y_pred)
        if t == POSITIVE_LABEL and p == NEGATIVE_LABEL
    )
    tn = sum(
        1
        for t, p in zip(y_true, y_pred)
        if t == NEGATIVE_LABEL and p == NEGATIVE_LABEL
    )
    n = tp + fp + fn + tn
    assert n == len(filtered), (
        f"Confusion matrix suma {n}, esperado {len(filtered)}"
    )

    # 5) Metrics (guard against zero-division on degenerate splits)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )
    kappa = float(
        cohen_kappa_score(
            y_true, y_pred, labels=[POSITIVE_LABEL, NEGATIVE_LABEL]
        )
    )

    # 6) Persist
    payload = {
        "source_dump": str(
            HERO2_DUMP_PATH.relative_to(PROJECT_ROOT)
        ),
        "source_dump_sha256": sha256_of_file(HERO2_DUMP_PATH),
        "partition_file": str(PARTITION_PATH.relative_to(PROJECT_ROOT)),
        "n_dev": len(filtered),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "metrics": {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy,
            "precision": precision,
            "F1": f1,
            "kappa": kappa,
        },
        "raw_predictions_subset": filtered,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head_sha(),
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 7) stdout summary (en español)
    print("[T17.B.1] Baseline-on-dev generado correctamente.")
    print(f"  Fichero        : {OUTPUT_PATH}")
    print(f"  n_dev          : {len(filtered)}")
    print(
        f"  Confusion mtx  : TP={tp}, FP={fp}, FN={fn}, TN={tn} "
        f"(suma={n})"
    )
    print("  Metricas:")
    print(f"    sensitivity  : {sensitivity:.4f}")
    print(f"    specificity  : {specificity:.4f}")
    print(f"    accuracy     : {accuracy:.4f}")
    print(f"    precision    : {precision:.4f}")
    print(f"    F1           : {f1:.4f}")
    print(f"    kappa        : {kappa:.4f}")


if __name__ == "__main__":
    main()
