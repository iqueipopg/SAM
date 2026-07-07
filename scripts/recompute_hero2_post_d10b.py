# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""T16.D.10.B — Recompute Hero 2 metrics from cached A4 outputs.

Loads the post-D9 Hero 2 results JSON (which carries the raw A4 reasons +
fulltext_reasoning per study) and re-applies the updated normaliser
``_normalise_a4_reason`` (with the \\bage\\b word-boundary fix from
T16.D.10.B). Produces a fresh hero2_results_post_d10b.json and a short
markdown summary. No A4 re-run; no GPU usage.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402  (after sys.path manipulation)

from experiments.eval_hero2_fulltext import (  # noqa: E402
    GOLD_REASON_CATEGORIES,
    _cohens_kappa,
    _normalise_a4_reason,
)

INPUT_JSON = REPO / "outputs/eval/micro/hero2_results_20260516_130759.json"
OUT_JSON = REPO / "outputs/eval/micro/hero2_results_post_d10b.json"
OUT_MD = REPO / "outputs/eval/micro/hero2_results_post_d10b.md"


def _confusion_8x8(pairs: list[tuple[str, str]]) -> dict:
    labels = GOLD_REASON_CATEGORIES + ["unknown_reason"]
    matrix = {row: {col: 0 for col in labels} for row in labels}
    for g, p in pairs:
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1
    return matrix


def _binary_from_decisions(preds: list[dict]) -> tuple[list[int], list[int]]:
    y_true, y_pred = [], []
    for p in preds:
        g = p["gold"].get("decision", "exclude")
        a = p["predicted"].get("decision", "exclude")
        y_true.append(1 if g == "include" else 0)
        y_pred.append(1 if a == "include" else 0)
    return y_true, y_pred


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    accuracy = (tp + tn) / n if n else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) else 0.0)
    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "n": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main() -> int:
    print(f"[1/4] Loading {INPUT_JSON.name}...")
    h2 = json.loads(INPUT_JSON.read_text())
    preds_pre = h2["raw_predictions"]

    print(f"[2/4] Re-normalising A4 reasons over {len(preds_pre)} entries...")
    preds_post: list[dict] = []
    cat_true: list[str] = []
    cat_pred: list[str] = []
    pairs: list[tuple[str, str]] = []
    diff_cases: list[dict] = []

    for p in preds_pre:
        gold_decision = p["gold"]["decision"]
        gold_reason_norm = p["gold"]["reason_norm"]
        a4_decision = p["predicted"]["decision"]
        a4_reason_raw = p["predicted"].get("reason", "") or ""
        a4_reasoning = p["details"]["field_scores"].get("fulltext_reasoning", "") or ""

        if a4_decision == "include":
            new_reason_norm = "include"
        else:
            new_reason_norm = _normalise_a4_reason(a4_reason_raw, a4_reasoning)

        old_reason_norm = p["predicted"]["reason_norm"]
        changed = (new_reason_norm != old_reason_norm)
        if changed:
            diff_cases.append({
                "covidence_id": p["input_id"],
                "gold_reason_norm": gold_reason_norm,
                "pred_reason_norm_pre": old_reason_norm,
                "pred_reason_norm_post": new_reason_norm,
                "raw_reason": a4_reason_raw,
                "reasoning": a4_reasoning[:200],
            })

        new_pred = dict(p)
        new_pred["predicted"] = dict(p["predicted"])
        new_pred["predicted"]["reason_norm"] = new_reason_norm
        match_decision = (gold_decision == a4_decision)
        match_reason = (gold_reason_norm == new_reason_norm)
        new_pred["match"] = match_decision and match_reason
        new_pred["details"] = dict(p["details"])
        fs = dict(new_pred["details"].get("field_scores", {}))
        fs["decision_match"] = match_decision
        fs["reason_match"] = match_reason
        new_pred["details"]["field_scores"] = fs
        preds_post.append(new_pred)

        cat_true.append(gold_reason_norm)
        cat_pred.append(new_reason_norm)
        pairs.append((gold_reason_norm, new_reason_norm))

    y_true, y_pred = _binary_from_decisions(preds_pre)
    bm = _binary_metrics(y_true, y_pred)
    kappa_old = h2["metrics"].get("cohens_kappa", 0.0)
    kappa_pre = _cohens_kappa(cat_true, [p["predicted"]["reason_norm"] for p in preds_pre])
    kappa_post = _cohens_kappa(cat_true, cat_pred)

    matrix_post = _confusion_8x8(pairs)
    matrix_pre = h2.get("confusion_matrix", {})

    metrics = {
        **bm,
        "cohens_kappa": round(kappa_post, 4) if kappa_post is not None else None,
        "cohens_kappa_pre_d10b": round(kappa_pre, 4) if kappa_pre is not None else None,
        "cohens_kappa_pre_d10b_from_json": kappa_old,
        "n_reason_changes": len(diff_cases),
    }

    pred_norm_dist_pre = Counter(p["predicted"]["reason_norm"] for p in preds_pre)
    pred_norm_dist_post = Counter(p["predicted"]["reason_norm"] for p in preds_post)

    print(f"[3/4] Writing {OUT_JSON.name}...")
    envelope = {
        "hero_id": h2.get("hero_id", "hero2"),
        "source_input": str(INPUT_JSON.relative_to(REPO)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fix_applied": "T16.D.10.B — word-boundary on 'age' rule of _normalise_a4_reason",
        "metrics": metrics,
        "confusion_matrix_post": matrix_post,
        "confusion_matrix_pre": matrix_pre,
        "predicted_reason_norm_distribution_pre": dict(pred_norm_dist_pre),
        "predicted_reason_norm_distribution_post": dict(pred_norm_dist_post),
        "changed_cases": diff_cases,
        "raw_predictions": preds_post,
        "notes": h2.get("notes", ""),
    }
    OUT_JSON.write_text(json.dumps(envelope, indent=2, ensure_ascii=False))

    print(f"[4/4] Writing {OUT_MD.name}...")
    OUT_MD.write_text(_render_md(envelope, h2))
    print(f"Done. Δ cases: {len(diff_cases)}. kappa pre→post: {kappa_pre} → {kappa_post}.")
    return 0


def _render_md(env: dict, h2_pre: dict) -> str:
    m = env["metrics"]
    pre = h2_pre["metrics"]
    lines: list[str] = []
    lines.append("# Hero 2 — métricas tras T16.D.10.B (word-boundary `age`)\n")
    lines.append(f"Timestamp: `{env['timestamp_utc']}`\n")
    lines.append(f"Fuente: `{env['source_input']}` (re-normalizado sin re-correr A4)\n")
    lines.append("## Métricas binarias\n")
    lines.append("| Métrica | Pre-D10B | Post-D10B | Δ |")
    lines.append("|---|---:|---:|---:|")
    for k in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
        v_pre = pre.get(k, 0.0)
        v_post = m.get(k, 0.0)
        d = v_post - v_pre if isinstance(v_pre, (int, float)) and isinstance(v_post, (int, float)) else 0.0
        lines.append(f"| {k} | {v_pre} | {v_post} | {d:+.4f} |")
    lines.append("")
    lines.append("## Métricas categóricas\n")
    lines.append("| Métrica | Pre-D10B | Post-D10B |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Cohen kappa | {pre.get('cohens_kappa')} | {m.get('cohens_kappa')} |")
    lines.append(f"| Reason changes | — | {m.get('n_reason_changes')} |")
    lines.append("")
    lines.append("## Casos cuya razón normalizada cambió\n")
    if env["changed_cases"]:
        lines.append("| cid | gold | pre | post | razón raw A4 (truncado) |")
        lines.append("|---|---|---|---|---|")
        for c in env["changed_cases"]:
            raw = (c.get("raw_reason") or "")[:60].replace("|", "/")
            lines.append(f"| {c['covidence_id']} | {c['gold_reason_norm']} | "
                         f"{c['pred_reason_norm_pre']} | {c['pred_reason_norm_post']} | {raw} |")
    else:
        lines.append("Ningún caso modificó su categoría normalizada.")
    lines.append("")
    lines.append("## Distribución de razones predichas\n")
    lines.append("| Razón | Pre-D10B | Post-D10B |")
    lines.append("|---|---:|---:|")
    keys = sorted(set(env["predicted_reason_norm_distribution_pre"].keys())
                  | set(env["predicted_reason_norm_distribution_post"].keys()))
    for k in keys:
        pre_v = env["predicted_reason_norm_distribution_pre"].get(k, 0)
        post_v = env["predicted_reason_norm_distribution_post"].get(k, 0)
        lines.append(f"| {k} | {pre_v} | {post_v} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
