# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""T17.C.1 Phase 4 — evaluate a committed A4 dev run vs Paula's gold standard.

Reads the committed A4 decisions (`sam_a4_fulltext/<slug>.json::fulltext_decision`)
for the dev N=387, crosses them by Covidence ID against the gold full-text labels,
computes binary metrics (reusing `_compute_binary_metrics` from
experiments.eval_a4_split — NOT reimplemented), classifies the variant into the
T17.C.1 pre-registro bands (§5), prints the 2×2 against the frozen Qwen reference
cells, and reports the text_source distribution (fulltext-real vs abstract_only).

The pre-registro is the authoritative baseline. The official Qwen split-V1 cell is
sens 0.0889 (TP4/FP9/TN333/FN41), from experiments/T17.B.3/runs/V1_dev_20260521_052019.json
(see T17.B.3_iteration_report.md §6). The pre-fix run (sens 0.0667, TP3/FP8/TN334/FN42)
is annotated as an alternative cell — it differs by code version, not stochasticity
(see docs/iterations/T17.C.1_log.md §B).

Usage:
    python scripts/eval_phase4.py <run_id> [--variant mono|split]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sam_orchestrator import RunPaths, paper_id_to_slug  # noqa: E402
from experiments.eval_a4_split import (  # noqa: E402
    DEFAULT_PARTITION,
    DEFAULT_SAMPLE_CSV,
    _compute_binary_metrics,
    _load_partition,
    _load_sample_index,
)

# Frozen Qwen reference cells (pre-registro §2 + T17.C.1_log.md §B).
QWEN_REFERENCE = {
    "mono_v2": {"sensitivity": 0.7556, "specificity": 0.7836, "source": "pre-registro §2 (B)"},
    "split_v1_official": {
        "sensitivity": 0.0889,
        "specificity": 0.9737,
        "confusion": {"TP": 4, "FP": 9, "TN": 333, "FN": 41},
        "source": "B.3 V1_dev_20260521_052019.json (OFICIAL)",
    },
    "split_v1_prefix_alt": {
        "sensitivity": 0.0667,
        "specificity": 0.9766,
        "confusion": {"TP": 3, "FP": 8, "TN": 334, "FN": 42},
        "source": "B.1 V1_dev_20260520_214241.json (pre-fix; versión de código, no varianza)",
    },
}


def _classify_band(sensitivity: float, specificity: float) -> str:
    if specificity < 0.80:
        return "Inválido"
    if sensitivity >= 0.65:
        return "Optimistic"
    if sensitivity >= 0.60:
        return "Borderline-alto"
    if sensitivity >= 0.55:
        return "Borderline-bajo"
    return "Pessimistic"


def _read_variant(rp: RunPaths) -> str:
    sidecar = rp.run_dir / "_phase4_meta.json"
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8")).get("variant", "desconocida")
        except Exception:  # noqa: BLE001
            pass
    return "desconocida"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 4 A4 dev run.")
    parser.add_argument("run_id")
    parser.add_argument("--variant", choices=["mono", "split"], default=None)
    args = parser.parse_args()

    rp = RunPaths(args.run_id)
    variant = args.variant or _read_variant(rp)

    partition = _load_partition(Path(DEFAULT_PARTITION))
    dev_ids: list[str] = list(partition["dev"])
    sample_idx = _load_sample_index(Path(DEFAULT_SAMPLE_CSV))
    cache_dir = rp.run_dir / "_fulltext_cache"

    records: list[dict[str, Any]] = []
    missing_decision = 0
    missing_gold = 0
    text_source_counts: dict[str, int] = {}

    for cid in dev_ids:
        gold = sample_idx.get(cid, {}).get("gold_decision")
        if not gold:
            missing_gold += 1
            continue

        slug = paper_id_to_slug(cid)
        a4_path = rp.paper_output("sam_a4_fulltext", cid)
        predicted: Optional[str] = None
        if a4_path.is_file():
            try:
                predicted = json.loads(a4_path.read_text(encoding="utf-8")).get(
                    "fulltext_decision"
                )
            except Exception:  # noqa: BLE001
                predicted = None
        if predicted is None:
            predicted = "EXCLUDE"
            missing_decision += 1

        records.append(
            {
                "covidence_id": cid,
                "gold_decision": gold,
                "predicted_decision": predicted,
                "reporter_ok": True,
                "decider_ok": True,
            }
        )

        # text_source distribution (from cache meta).
        meta_path = cache_dir / f"{slug}.meta.json"
        src = "missing_cache"
        if meta_path.is_file():
            try:
                src = json.loads(meta_path.read_text(encoding="utf-8")).get(
                    "text_source", "missing_cache"
                )
            except Exception:  # noqa: BLE001
                src = "missing_cache"
        text_source_counts[src] = text_source_counts.get(src, 0) + 1

    metrics = _compute_binary_metrics(records)
    band = _classify_band(metrics["sensitivity"], metrics["specificity"])

    fulltext_real = sum(
        c for s, c in text_source_counts.items() if s not in ("abstract_only", "missing_cache")
    )
    abstract_only = text_source_counts.get("abstract_only", 0)

    result = {
        "run_id": args.run_id,
        "variant": variant,
        "n_evaluated": len(records),
        "missing_decision": missing_decision,
        "missing_gold": missing_gold,
        "metrics": metrics,
        "band": band,
        "text_source_distribution": text_source_counts,
        "fulltext_real": fulltext_real,
        "abstract_only": abstract_only,
        "qwen_reference": QWEN_REFERENCE,
    }

    out_path = rp.run_dir / f"eval_phase4_{variant}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable report (Spanish).
    conf = metrics["confusion"]
    print(f"=== T17.C.1 Phase 4 — eval ({variant}) — run {args.run_id} ===")
    print(f"N evaluados: {len(records)}  (sin decisión A4: {missing_decision}, sin gold: {missing_gold})")
    print(
        f"sens={metrics['sensitivity']}  spec={metrics['specificity']}  "
        f"acc={metrics['accuracy']}  prec={metrics['precision']}  F1={metrics['F1']}  "
        f"kappa={metrics['kappa']}"
    )
    print(f"TP={conf['TP']} FP={conf['FP']} TN={conf['TN']} FN={conf['FN']}")
    print(f"BANDA (pre-registro §5): {band}")
    print()
    print("--- 2×2 (modelo × arquitectura) ---")
    print(f"  mono-Qwen V2        : sens {QWEN_REFERENCE['mono_v2']['sensitivity']}  spec {QWEN_REFERENCE['mono_v2']['specificity']}  [{QWEN_REFERENCE['mono_v2']['source']}]")
    print(f"  split-Qwen V1 (OFIC): sens {QWEN_REFERENCE['split_v1_official']['sensitivity']}  spec {QWEN_REFERENCE['split_v1_official']['specificity']}  [{QWEN_REFERENCE['split_v1_official']['source']}]")
    print(f"  split-Qwen V1 (alt) : sens {QWEN_REFERENCE['split_v1_prefix_alt']['sensitivity']}  [{QWEN_REFERENCE['split_v1_prefix_alt']['source']}]")
    print(f"  {variant}-Claude       : sens {metrics['sensitivity']}  spec {metrics['specificity']}  ← ESTA CORRIDA")
    print()
    print(f"--- text_source: fulltext-real={fulltext_real}  abstract_only={abstract_only} ---")
    print(f"  detalle: {text_source_counts}")
    print()
    print(f"Escrito: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
