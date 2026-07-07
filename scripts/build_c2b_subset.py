#!/usr/bin/env python3
# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Build the T17.C.2b N=300 evaluation subset (deterministic, no LLM).

Purpose
-------
Produce a single frozen subset of N=300 papers — 150 relevant + 150 irrelevant,
all WITH abstract — that serves as the shared source for the two T17.C.2b tracks.

The "with-abstract" criterion is a byte-exact replica of
``eval_hero1_screening._build_papers`` and ``eval_a3_claude._load_candidates``:
the abstract is taken from the GOLD CSV, ``.strip()``-ped, and considered present
only if non-empty and ``lower()`` not in ("nan", "none"). This guarantees the
subset matches exactly what the downstream evaluation will treat as "con-abstract".

Determinism
-----------
``random.Random(20260531)``; each class pool is sorted by ``covidence_id`` before
sampling; ``rng.sample(pool, 150)`` per class. If a pool has fewer than 150
with-abstract candidates the script ABORTS (no silent truncation).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SEED = 20260531
PER_CLASS = 150
TOTAL = 2 * PER_CLASS

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = (
    REPO_ROOT
    / "data/gold_standard/deporte/samples/medio/sample_medio_a3.csv"
)
GOLD_PATH = (
    REPO_ROOT
    / "data/gold_standard/deporte/unified/gold_screening_unified.csv"
)
OUT_DIR = REPO_ROOT / "experiments/T17.C.2b"
OUT_CSV = OUT_DIR / "c2b_subset_300.csv"
OUT_IDS = OUT_DIR / "c2b_subset_300_ids.txt"
OUT_COMP = OUT_DIR / "c2b_subset_300_composition.md"

_NAN_TOKENS = ("nan", "none")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _has_gold_abstract(cid: str, gold_idx: pd.DataFrame) -> bool:
    """Replica of eval_hero1_screening._build_papers / eval_a3_claude.

    abstract from GOLD, .strip(); present iff non-empty and lower() not in
    ("nan", "none").
    """
    if cid not in gold_idx.index:
        return False
    gold_row = gold_idx.loc[cid]
    if isinstance(gold_row, pd.DataFrame):
        gold_row = gold_row.iloc[0]
    abstract = str(gold_row.get("abstract", "") or "").strip()
    return bool(abstract) and abstract.lower() not in _NAN_TOKENS


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    sample_df = pd.read_csv(SAMPLE_PATH, dtype={"covidence_id": str})
    gold_df = pd.read_csv(GOLD_PATH, dtype={"covidence_id": str})
    gold_idx = gold_df.set_index("covidence_id")

    # 1. with-abstract flag (gold-based) for every sample row.
    with_abs = sample_df["covidence_id"].map(
        lambda cid: _has_gold_abstract(cid, gold_idx)
    )

    label = sample_df["human_label_ta"].astype(str).str.strip().str.lower()

    # 2. Candidate pools: with-abstract, partitioned by human_label_ta.
    rel_mask = with_abs & (label == "relevant")
    irr_mask = with_abs & (label == "irrelevant")

    rel_ids = sorted(sample_df.loc[rel_mask, "covidence_id"].tolist())
    irr_ids = sorted(sample_df.loc[irr_mask, "covidence_id"].tolist())

    print(f"[pools] relevant con-abstract   = {len(rel_ids)}")
    print(f"[pools] irrelevant con-abstract = {len(irr_ids)}")
    print(f"[pools] relevant   sin-abstract = {int((~with_abs & (label == 'relevant')).sum())}")
    print(f"[pools] irrelevant sin-abstract = {int((~with_abs & (label == 'irrelevant')).sum())}")

    # 3. Abort if any pool < PER_CLASS (no silent truncation).
    if len(rel_ids) < PER_CLASS:
        sys.stderr.write(
            f"ERROR: relevant con-abstract pool = {len(rel_ids)} < {PER_CLASS}. "
            "Aborting (no silent truncation).\n"
        )
        return 1
    if len(irr_ids) < PER_CLASS:
        sys.stderr.write(
            f"ERROR: irrelevant con-abstract pool = {len(irr_ids)} < {PER_CLASS}. "
            "Aborting (no silent truncation).\n"
        )
        return 1

    # Deterministic sampling: sorted pools + seeded RNG.
    rng = random.Random(SEED)
    rel_sel = rng.sample(rel_ids, PER_CLASS)
    irr_sel = rng.sample(irr_ids, PER_CLASS)

    # 4. Union of 300, sorted by covidence_id.
    selected_ids = sorted(set(rel_sel) | set(irr_sel))
    assert len(selected_ids) == TOTAL, (
        f"expected {TOTAL} unique ids, got {len(selected_ids)}"
    )

    # Full rows, ordered canonically by covidence_id (matching selected_ids).
    subset = (
        sample_df.set_index("covidence_id")
        .loc[selected_ids]
        .reset_index()[list(sample_df.columns)]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # SALIDA 1: full rows.
    subset.to_csv(OUT_CSV, index=False)

    # SALIDA 2: ids, one per line.
    OUT_IDS.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")

    # SALIDA 3: composition (Spanish).
    n_with_abs = int(
        subset["covidence_id"].map(
            lambda cid: _has_gold_abstract(cid, gold_idx)
        ).sum()
    )
    n_title_only = TOTAL - n_with_abs
    lc = subset["human_label_ta"].astype(str).str.strip().str.lower()
    n_rel = int((lc == "relevant").sum())
    n_irr = int((lc == "irrelevant").sum())

    comp = "\n".join(
        [
            "# Composición del subconjunto T17.C.2b (N=300)",
            "",
            "Subconjunto determinista y congelado, fuente única compartida por los",
            "dos tracks de T17.C.2b. Generado sin LLM por `scripts/build_c2b_subset.py`.",
            "",
            "## Parámetros",
            "",
            f"- Seed: `{SEED}` (`random.Random({SEED})`).",
            f"- Celdas (diseño balanceado): {PER_CLASS} relevant / {PER_CLASS} irrelevant.",
            "- Muestreo: pools ordenados por `covidence_id`, luego "
            f"`rng.sample(pool, {PER_CLASS})` por clase; unión ordenada por `covidence_id`.",
            "",
            "## Totales",
            "",
            f"- Total filas: {TOTAL}.",
            f"- Con abstract (criterio gold): {n_with_abs}.",
            f"- Title-only (sin abstract): {n_title_only}.",
            f"- `human_label_ta`: relevant = {n_rel}, irrelevant = {n_irr}.",
            "",
            "## Pools candidatos (sample_medio_a3 + abstract gold)",
            "",
            f"- relevant con-abstract: {len(rel_ids)}",
            f"- irrelevant con-abstract: {len(irr_ids)}",
            f"- relevant sin-abstract: {int((~with_abs & (label == 'relevant')).sum())}",
            f"- irrelevant sin-abstract: {int((~with_abs & (label == 'irrelevant')).sum())}",
            "",
            "## Criterio \"con-abstract\"",
            "",
            "Réplica exacta de `eval_hero1_screening._build_papers` y",
            "`eval_a3_claude._load_candidates`: el abstract se toma del GOLD",
            "(`gold_screening_unified.csv`), se aplica `.strip()`, y se considera",
            "presente solo si no queda vacío y su `lower()` no está en (\"nan\", \"none\").",
            "Es el mismo criterio que usará la evaluación cuantitativa.",
            "",
            "## Decisión: exclusión de title-only",
            "",
            "Se excluyen del subconjunto cuantitativo los papers sin abstract",
            "(title-only). Rationale: los pools sin-abstract están radicalmente",
            "desbalanceados — 744 relevant sin-abstract frente a solo 9 irrelevant",
            "sin-abstract —, lo que hace imposible equilibrar 150/150 en la celda",
            "title-only sin recortar muy por debajo de N. La política de tratamiento",
            "de title-only sí está igualada en el prompt `c18ab41` (ambos tracks lo",
            "manejan idénticamente), pero queda FUERA de la evaluación cuantitativa",
            "de C.2b para preservar el balance de clases y la paridad de entrada.",
            "",
        ]
    )
    OUT_COMP.write_text(comp, encoding="utf-8")

    # ── stdout report ──────────────────────────────────────────────────────
    print()
    print(f"[out] {OUT_CSV}")
    print(f"[out] {OUT_IDS}")
    print(f"[out] {OUT_COMP}")
    print()
    print(f"[verify] filas CSV                 = {len(subset)}")
    print(f"[verify] human_label_ta relevant   = {n_rel}")
    print(f"[verify] human_label_ta irrelevant = {n_irr}")
    print(f"[verify] con-abstract (gold)       = {n_with_abs}")
    print(f"[verify] title-only                = {n_title_only}")
    all_subset = set(sample_df["covidence_id"])
    subset_ok = set(subset["covidence_id"]).issubset(all_subset)
    print(f"[verify] ids ⊆ sample_medio_a3     = {subset_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
