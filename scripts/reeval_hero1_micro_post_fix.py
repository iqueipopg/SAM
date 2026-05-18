"""
Re-evaluate Hero 1 micro using the fixed evaluator (T16.E.1.fix Fase 4).

Loads the most recent hero1_results micro JSON (post-D9.A baseline) and
rebuilds the metrics via ``compute_hero1_metrics`` from the fixed evaluator.
Because the micro sample is 100% gold-positive, the legacy fields must match
byte-for-byte; the new fields (specificity, kappa) come out as ``None``.

Outputs:
  - outputs/eval/micro/hero1_results_micro_post_fix.json
  - outputs/eval/micro/hero1_micro_pre_vs_post_fix.md

This is a CPU-only re-evaluation; no LLM is invoked.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_hero1_screening import (  # noqa: E402
    compute_hero1_metrics,
    _normalise_gold_label,
)

MICRO_DIR = PROJECT_ROOT / "outputs" / "eval" / "micro"
SAMPLE_PATH = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "samples" / "micro" / "sample_50.csv"

# Most recent post-D9 result file. T16.D.9.A glossary fix on Hero 1.
PRE_FIX_RESULT = MICRO_DIR / "hero1_results_20260516_140621.json"
POST_FIX_RESULT = MICRO_DIR / "hero1_results_micro_post_fix.json"
COMPARISON_MD = MICRO_DIR / "hero1_micro_pre_vs_post_fix.md"


def main() -> int:
    if not PRE_FIX_RESULT.exists():
        print(f"[ERROR] missing pre-fix result: {PRE_FIX_RESULT}", file=sys.stderr)
        return 1
    if not SAMPLE_PATH.exists():
        print(f"[ERROR] missing sample: {SAMPLE_PATH}", file=sys.stderr)
        return 1

    print(f"[reeval] Cargando pre-fix: {PRE_FIX_RESULT.name}")
    with open(PRE_FIX_RESULT, "r", encoding="utf-8") as f:
        pre = json.load(f)

    sample_df = pd.read_csv(SAMPLE_PATH, dtype={"covidence_id": str})
    gold_label_by_cid = {
        str(r["covidence_id"]): _normalise_gold_label(r.get("human_label_ta"))
        for _, r in sample_df.iterrows()
    }
    abstract_used_by_cid = {}
    for rp in pre.get("raw_predictions", []):
        cid = str(rp.get("input_id", ""))
        details = rp.get("details", {}) or {}
        fields = details.get("field_scores", {}) or {}
        abstract_used_by_cid[cid] = bool(fields.get("abstract_used", False))

    predictions: list = []
    for rp in pre.get("raw_predictions", []):
        cid = str(rp.get("input_id", ""))
        predictions.append({
            "covidence_id": cid,
            "gold": gold_label_by_cid.get(cid, "relevant"),
            "predicted": rp.get("predicted", "irrelevant"),
            "abstract_used": abstract_used_by_cid.get(cid, False),
        })

    metrics, cm, fn_ids, fp_ids = compute_hero1_metrics(predictions)
    metrics["false_negatives"] = [
        {"covidence_id": cid} for cid in fn_ids[:50]
    ]
    metrics["false_positives"] = [
        {"covidence_id": cid} for cid in fp_ids[:50]
    ]

    # Build post-fix envelope mirroring the pre-fix structure.
    post = dict(pre)
    post["metrics"] = metrics
    post["confusion_matrix"] = cm
    post["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    post["notes"] = (
        "T16.E.1.fix re-evaluation. Re-uso de predicciones A3 del run "
        f"{PRE_FIX_RESULT.name}; metricas recalculadas con compute_hero1_metrics "
        "del evaluador fixed. Para sample 100% gold-positive (micro) las "
        "metricas legacy deben coincidir byte-byte con la version pre-fix."
    )

    with open(POST_FIX_RESULT, "w", encoding="utf-8") as f:
        json.dump(post, f, indent=2, ensure_ascii=False)
    print(f"[reeval] Escrito: {POST_FIX_RESULT.name}")

    # Build comparison MD.
    m_pre = pre.get("metrics", {})
    m_post = post["metrics"]
    lines = []
    lines.append("# Hero 1 micro pre vs post T16.E.1.fix")
    lines.append("")
    lines.append(f"Fuente pre-fix: `{PRE_FIX_RESULT.name}`")
    lines.append(f"Fuente post-fix: `{POST_FIX_RESULT.name}`")
    lines.append("")
    lines.append("## Metricas legacy (deben coincidir)")
    lines.append("")
    lines.append("| Metrica | Pre-fix | Post-fix | Delta |")
    lines.append("|---|---|---|---|")
    legacy_keys = [
        "sensitivity_overall",
        "sensitivity_abstract_yes",
        "sensitivity_abstract_no",
        "tp",
        "fn",
        "uncertain_count",
    ]
    for k in legacy_keys:
        v_pre = m_pre.get(k)
        v_post = m_post.get(k)
        delta = "=" if v_pre == v_post else f"changed ({v_pre} -> {v_post})"
        lines.append(f"| `{k}` | `{v_pre}` | `{v_post}` | {delta} |")
    lines.append("")
    lines.append("## Metricas nuevas (T16.E.1.fix)")
    lines.append("")
    lines.append("| Metrica | Valor | Nota |")
    lines.append("|---|---|---|")
    new_keys = [
        ("sensitivity_adjusted", "= tp/(tp+fn) (excluye uncertain)"),
        ("specificity", "None en micro: no hay irrelevant gold"),
        ("precision", "tp/(tp+fp) (puede ser None si no hay FP)"),
        ("f1", "None si specificity es None"),
        ("accuracy", "(tp+tn)/decided"),
        ("cohens_kappa", "None en micro: solo una clase gold"),
        ("n_relevant_gold", "siempre 50 en micro"),
        ("n_irrelevant_gold", "siempre 0 en micro"),
    ]
    for k, note in new_keys:
        v = m_post.get(k)
        lines.append(f"| `{k}` | `{v}` | {note} |")
    lines.append("")
    lines.append("## Veredicto")
    lines.append("")
    delta_legacy = [k for k in legacy_keys if m_pre.get(k) != m_post.get(k)]
    if not delta_legacy:
        lines.append("**Sin regresion.** Todas las metricas legacy coinciden byte-byte. "
                     "Las metricas nuevas (specificity, kappa, etc.) son `None` como "
                     "se espera para un sample 100% gold-positivo.")
    else:
        lines.append(f"**REGRESION DETECTADA** en: {', '.join('`'+k+'`' for k in delta_legacy)}.")
        lines.append("Revisar `compute_hero1_metrics` antes de proceder a Fase 5.")
    lines.append("")

    COMPARISON_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[reeval] Comparison MD: {COMPARISON_MD.name}")

    if delta_legacy:
        print("[REGRESION] Metricas legacy difieren: " + ", ".join(delta_legacy))
        return 2
    print("[OK] No regression — metricas legacy coinciden byte-byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
