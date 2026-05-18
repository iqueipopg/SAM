"""
Hero 1 evaluator — Agent3 Title/Abstract screening vs Paula's gold standard.

Sprint:  T16.D.5.B (initial), T16.E.1.fix (mixed-sample support)
Schema:  docs/heroes_output_schema.md (section 3.1)
Inputs:
  - config/eval/micro.yaml          : paths to sample and gold CSVs
  - data/gold_standard/deporte/samples/micro/sample_50.csv
  - data/gold_standard/deporte/unified/gold_screening_unified.csv
  - config/agents.json              : per-agent LLM params (agent3_screening)

Output:
  outputs/eval/{scale}/hero1_results_{YYYYMMDD_HHMMSS}.json
    + short stdout report (Spanish)

T16.E.1.fix
-----------
The original evaluator hard-coded ``gold_label = "relevant"`` and assumed a
100% gold-positive sample (the micro composition). The medio sample contains
both relevant and irrelevant rows, so we now read the gold label dynamically
from the sample's ``human_label_ta`` column and compute full binary metrics:
sensitivity, specificity, precision, F1, accuracy, Cohen's kappa.

When the sample is 100% positive (e.g. micro), the legacy fields
(sensitivity_*, tp, fn) match byte-for-byte; new fields (specificity, fp, tn,
etc.) are reported as ``None``.

The script does NOT modify ``agents/agent3_screening.py``, ``base.py``, prompts,
or ``config.yaml``.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is on sys.path so `agents` package imports resolve when
# launching as `python experiments/eval_hero1_screening.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._eval_common import (  # noqa: E402
    PAULA_SPORT_PROTOCOL,
    gpu_info as _gpu_info,
    load_agent_params,
    load_eval_config as _load_eval_config,
    seed_torch as _seed_torch,
    truncate as _truncate,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "eval" / "micro.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "micro"
HERO_ID = "hero1"
AGENT_TAG = "A3"

# Recognised gold labels in the sample's ``human_label_ta`` column.
GOLD_RELEVANT = "relevant"
GOLD_IRRELEVANT = "irrelevant"


def _load_agent3_params() -> dict:
    """Load agent3_screening params from config/agents.json (defaults merged)."""
    return load_agent_params("agent3_screening")


def _normalise_gold_label(raw: object) -> str:
    """Map raw ``human_label_ta`` value -> canonical gold label.

    Falls back to ``relevant`` only if the value is missing AND no explicit
    irrelevant signal is present (legacy behaviour for older sample CSVs).
    Anything that doesn't normalise to ``irrelevant`` is treated as ``relevant``
    so the legacy micro sample (where every row is gold-positive) keeps working
    when ``human_label_ta`` is absent or unfilled.
    """
    if raw is None:
        return GOLD_RELEVANT
    if isinstance(raw, float) and pd.isna(raw):
        return GOLD_RELEVANT
    s = str(raw).strip().lower()
    if s == "irrelevant":
        return GOLD_IRRELEVANT
    return GOLD_RELEVANT


def _build_papers(sample_df: pd.DataFrame, gold_df: pd.DataFrame) -> list:
    """Join sample (covidence_id) with gold (title+abstract). Preserve row order.

    Each output dict has the shape Agent3 expects: title, abstract, pmid. Adds
    internal helpers used downstream by the evaluator:
      - ``_abstract_used`` : True if abstract is non-empty (for stratification).
      - ``_gold_label``    : ``relevant`` or ``irrelevant``, read from the
                             sample's ``human_label_ta`` column (T16.E.1.fix).
    """
    gold_idx = gold_df.set_index("covidence_id")
    papers = []
    for _, row in sample_df.iterrows():
        cid = row["covidence_id"]
        if cid in gold_idx.index:
            gold_row = gold_idx.loc[cid]
            if isinstance(gold_row, pd.DataFrame):
                gold_row = gold_row.iloc[0]
            title = str(gold_row.get("title", "") or "").strip()
            abstract = str(gold_row.get("abstract", "") or "").strip()
        else:
            title = ""
            abstract = ""
        abstract_used = bool(abstract) and abstract.lower() not in ("nan", "none")
        if not abstract_used:
            abstract = ""
        papers.append(
            {
                "pmid": cid,
                "title": title,
                "abstract": abstract,
                "_abstract_used": abstract_used,
                "_has_abstract_csv": bool(row.get("has_abstract", False)),
                "_gold_label": _normalise_gold_label(row.get("human_label_ta")),
            }
        )
    return papers


def _map_decision_to_predicted(decision: str) -> str:
    """Map Agent3 decision (INCLUDE/EXCLUDE/UNCERTAIN) → schema label."""
    if decision == "INCLUDE":
        return "relevant"
    if decision == "EXCLUDE":
        return "irrelevant"
    return "uncertain"


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers (pure-python; importable from tests)
# ──────────────────────────────────────────────────────────────────────────────

def _cohens_kappa(y_true: list, y_pred: list) -> Optional[float]:
    """Compute Cohen's kappa between two equal-length label lists.

    Returns ``None`` for empty inputs or when expected-agreement is 1 (e.g.
    all-one-class predictions and golds). Implementation matches
    ``sklearn.metrics.cohen_kappa_score`` for the binary case used here.
    """
    if not y_true or not y_pred:
        return None
    if len(y_true) != len(y_pred):
        return None
    n = len(y_true)
    labels = sorted({*y_true, *y_pred})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t]][label_to_idx[p]] += 1

    row_totals = [sum(row) for row in cm]
    col_totals = [sum(cm[r][c] for r in range(k)) for c in range(k)]
    po = sum(cm[i][i] for i in range(k)) / n
    pe = sum((row_totals[i] * col_totals[i]) / (n * n) for i in range(k))
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def _safe_div(num: float, den: float) -> Optional[float]:
    """Return ``num/den`` rounded to 4 decimals, or ``None`` if ``den == 0``."""
    if den == 0:
        return None
    return round(num / den, 4)


def compute_hero1_metrics(predictions: list) -> tuple:
    """Compute Hero 1 metrics from a list of prediction dicts.

    Each ``prediction`` dict must contain:
      - ``gold``         : ``"relevant"`` or ``"irrelevant"``
      - ``predicted``    : ``"relevant"`` / ``"irrelevant"`` / ``"uncertain"``
      - ``abstract_used``: ``bool`` (optional; for stratification)
      - ``covidence_id`` : optional, used only when building FN/FP lists

    Returns ``(metrics_dict, confusion_matrix)``.

    The function is purely deterministic and side-effect free; it is tested
    directly in ``tests/test_heroes_evaluators.py``.
    """
    cm = {
        GOLD_RELEVANT:   {"relevant": 0, "irrelevant": 0, "uncertain": 0},
        GOLD_IRRELEVANT: {"relevant": 0, "irrelevant": 0, "uncertain": 0},
    }
    tp = fp = tn = fn = 0
    tp_yes = fn_yes = tp_no = fn_no = 0
    n_with_abs = n_without_abs = 0
    n_relevant_gold = 0
    n_irrelevant_gold = 0
    uncertain_count = 0
    uncertain_among_relevant = 0
    uncertain_among_irrelevant = 0
    false_negatives_ids: list = []
    false_positives_ids: list = []
    y_true_decided: list = []
    y_pred_decided: list = []

    for pr in predictions:
        gold = pr.get("gold", GOLD_RELEVANT)
        predicted = pr.get("predicted", "irrelevant")
        abstract_used = bool(pr.get("abstract_used", False))
        cid = pr.get("covidence_id")

        if abstract_used:
            n_with_abs += 1
        else:
            n_without_abs += 1

        cm[gold][predicted] += 1

        if gold == GOLD_RELEVANT:
            n_relevant_gold += 1
            if predicted == "relevant":
                tp += 1
                if abstract_used:
                    tp_yes += 1
                else:
                    tp_no += 1
                y_true_decided.append(1)
                y_pred_decided.append(1)
            elif predicted == "irrelevant":
                fn += 1
                if abstract_used:
                    fn_yes += 1
                else:
                    fn_no += 1
                if cid is not None:
                    false_negatives_ids.append(cid)
                y_true_decided.append(1)
                y_pred_decided.append(0)
            else:  # uncertain
                uncertain_count += 1
                uncertain_among_relevant += 1
        else:  # irrelevant gold
            n_irrelevant_gold += 1
            if predicted == "relevant":
                fp += 1
                if cid is not None:
                    false_positives_ids.append(cid)
                y_true_decided.append(0)
                y_pred_decided.append(1)
            elif predicted == "irrelevant":
                tn += 1
                y_true_decided.append(0)
                y_pred_decided.append(0)
            else:  # uncertain
                uncertain_count += 1
                uncertain_among_irrelevant += 1

    sample_size = len(predictions)

    # Sensitivity bookkeeping (over relevant gold).
    sensitivity_overall = _safe_div(tp, sample_size)
    sensitivity_adjusted = _safe_div(tp, tp + fn)  # excludes uncertain
    sensitivity_yes = _safe_div(tp_yes, n_with_abs) if n_with_abs > 0 else None
    sensitivity_no = _safe_div(tp_no, n_without_abs) if n_without_abs > 0 else None

    # Specificity / FP rate (over irrelevant gold). None when no negatives.
    if n_irrelevant_gold == 0:
        specificity = None
        fp_rate = None
    else:
        specificity = _safe_div(tn, tn + fp)
        fp_rate = _safe_div(fp, tn + fp)

    # Precision / F1: defined only when at least one positive prediction.
    if (tp + fp) > 0:
        precision = _safe_div(tp, tp + fp)
    else:
        precision = None

    if precision is not None and sensitivity_adjusted not in (None, 0):
        denom = precision + sensitivity_adjusted
        if denom > 0:
            f1 = round(2 * precision * sensitivity_adjusted / denom, 4)
        else:
            f1 = None
    elif precision == 0 or sensitivity_adjusted == 0:
        f1 = 0.0
    else:
        f1 = None

    # Accuracy over decided predictions (excluding uncertain) when both classes
    # are present; otherwise fall back to overall accuracy.
    decided = tp + tn + fp + fn
    if decided > 0:
        accuracy = round((tp + tn) / decided, 4)
    else:
        accuracy = None

    # Cohen's kappa over decided predictions (binary), only when both classes
    # present.
    if n_relevant_gold > 0 and n_irrelevant_gold > 0 and y_true_decided:
        kappa = _cohens_kappa(y_true_decided, y_pred_decided)
        if kappa is not None:
            kappa = round(kappa, 4)
    else:
        kappa = None

    metrics = {
        # Legacy fields kept for backward compatibility with micro consumers.
        "sensitivity_overall": sensitivity_overall if sensitivity_overall is not None else 0.0,
        "sensitivity_abstract_yes": sensitivity_yes,
        "sensitivity_abstract_no": sensitivity_no,
        "tp": tp,
        "fn": fn,
        "tn": tn if n_irrelevant_gold > 0 else None,
        "fp": fp if n_irrelevant_gold > 0 else None,
        "uncertain_count": uncertain_count,
        "stratification": {
            "n_with_abstract": n_with_abs,
            "n_without_abstract": n_without_abs,
            "tp_with_abstract": tp_yes,
            "tp_without_abstract": tp_no,
            "fn_with_abstract": fn_yes,
            "fn_without_abstract": fn_no,
        },
        # T16.E.1.fix additions for mixed-sample evaluation.
        "sensitivity_adjusted": sensitivity_adjusted,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "cohens_kappa": kappa,
        "fp_rate": fp_rate,
        "n_relevant_gold": n_relevant_gold,
        "n_irrelevant_gold": n_irrelevant_gold,
        "uncertain_among_relevant": uncertain_among_relevant,
        "uncertain_among_irrelevant": uncertain_among_irrelevant,
    }

    return metrics, cm, false_negatives_ids, false_positives_ids


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def run_hero1_evaluation(
    config_path: Path,
    output_dir: Path,
    agent_instance=None,
) -> dict:
    """Run the Hero 1 evaluation and return the result dict (also written to disk).

    Args:
        config_path: Path to the eval YAML config.
        output_dir: Directory where ``hero1_results_*.json`` will be written.
        agent_instance: Optional pre-instantiated ``Agent3Screening`` whose model
            is already loaded (used by ``run_heroes_*.py`` to share VRAM across
            heroes). When ``None``, a new instance is created and its VRAM is
            freed before returning (standalone mode).
    """
    _seed_torch(42)

    eval_cfg = _load_eval_config(config_path)
    sample_path = (PROJECT_ROOT / eval_cfg["sample"]).resolve()
    gold_path = (PROJECT_ROOT / eval_cfg["gold_screening"]).resolve()

    # ── 1. Load CSVs ───────────────────────────────────────────────────────
    sample_df = pd.read_csv(sample_path, dtype={"covidence_id": str})
    gold_df = pd.read_csv(gold_path, dtype={"covidence_id": str})

    sample_size = len(sample_df)
    papers = _build_papers(sample_df, gold_df)

    # ── 2. Agent3 setup (single batch call) ────────────────────────────────
    from agents.agent3_screening import Agent3Screening

    a3_params = _load_agent3_params()
    owns_agent = agent_instance is None
    if agent_instance is None:
        agent = Agent3Screening()
    else:
        agent = agent_instance

    llm_config = {
        "temperature": 0.2,
        "top_p": getattr(agent, "top_p", None),
        "max_tokens": a3_params.get("max_tokens", 150),
        "repetition_penalty": a3_params.get("repetition_penalty", 1.3),
        "seed": 42,
        "ensemble_size": agent.ensemble_size,
        "voting_threshold": agent.voting_threshold,
        "max_input_chars": a3_params.get("max_input_chars", 800),
    }

    model_used = agent.model_path or "unknown"
    if isinstance(model_used, str) and "qwen" in model_used.lower():
        model_display = "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)"
    else:
        model_display = str(model_used)

    # ── 3. Run screening (one batch call for all papers) ───────────────────
    start_ts = time.time()
    errors: list = []
    try:
        screen_out = agent.screen_batch(papers, PAULA_SPORT_PROTOCOL)
    except Exception as e:
        errors.append(
            {
                "input_id": "<batch>",
                "error_type": "ModelError",
                "message": f"screen_batch raised: {type(e).__name__}: {e}",
            }
        )
        screen_out = {"include": [], "exclude": [], "uncertain": [], "stats": {}}
    elapsed = round(time.time() - start_ts, 2)

    a3_by_id: dict = {}
    for bucket in ("include", "exclude", "uncertain"):
        for r in screen_out.get(bucket, []):
            a3_by_id[r.get("pmid", "")] = r

    # ── 4. Build predictions + raw_predictions ─────────────────────────────
    raw_predictions: list = []
    predictions: list = []
    paper_meta: dict = {}

    for paper in papers:
        cid = paper["pmid"]
        gold_label = paper["_gold_label"]
        abstract_used = bool(paper.get("_abstract_used", False))
        a3_result = a3_by_id.get(cid)

        if a3_result is None:
            predicted = "irrelevant"
            votes_dump = []
            reasoning = "Missing Agent3 result"
            screening_conf = 0.0
            borderline = False
            errors.append(
                {
                    "input_id": cid,
                    "error_type": "ValidationError",
                    "message": "covidence_id absent from Agent3 output",
                }
            )
        else:
            predicted = _map_decision_to_predicted(a3_result.get("decision", "EXCLUDE"))
            votes_dump = a3_result.get("votes", [])
            reasoning = a3_result.get("reasoning", "")
            screening_conf = a3_result.get("screening_confidence", 0.0)
            borderline = a3_result.get("borderline", False)

        predictions.append(
            {
                "covidence_id": cid,
                "gold": gold_label,
                "predicted": predicted,
                "abstract_used": abstract_used,
            }
        )
        paper_meta[cid] = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "reasoning": reasoning,
            "screening_confidence": screening_conf,
            "borderline": borderline,
            "votes": votes_dump,
        }

        # Match: gold relevant + predicted relevant OR gold irrelevant + pred
        # irrelevant. uncertain never matches.
        match = (gold_label == GOLD_RELEVANT and predicted == "relevant") or (
            gold_label == GOLD_IRRELEVANT and predicted == "irrelevant"
        )

        raw_llm_response = ""
        if votes_dump:
            raw_pieces = []
            for i, v in enumerate(votes_dump):
                d = v.get("decision", "") if isinstance(v, dict) else str(v)
                r = v.get("reasoning", "") if isinstance(v, dict) else ""
                raw_pieces.append(f"vote{i+1}={d}: {r}")
            raw_llm_response = "\n".join(raw_pieces)

        raw_predictions.append(
            {
                "input_id": cid,
                "gold": gold_label,
                "predicted": predicted,
                "match": bool(match),
                "details": {
                    "raw_llm_response": raw_llm_response,
                    "field_scores": {
                        "votes": votes_dump,
                        "reasoning": reasoning,
                        "screening_confidence": screening_conf,
                        "borderline": borderline,
                        "abstract_used": abstract_used,
                    },
                },
            }
        )

    # ── 5. Metrics + confusion matrix ──────────────────────────────────────
    metrics, cm, fn_ids, fp_ids = compute_hero1_metrics(predictions)

    # Promote ids -> rich dicts using paper_meta. Cap at first 50 to keep JSON small.
    false_negatives = [
        {
            "covidence_id": cid,
            "title": _truncate(paper_meta.get(cid, {}).get("title", ""), 100),
            "abstract": _truncate(paper_meta.get(cid, {}).get("abstract", ""), 200),
            "reasoning": paper_meta.get(cid, {}).get("reasoning", ""),
            "screening_confidence": paper_meta.get(cid, {}).get("screening_confidence", 0.0),
            "borderline": paper_meta.get(cid, {}).get("borderline", False),
        }
        for cid in fn_ids[:50]
    ]
    false_positives = [
        {
            "covidence_id": cid,
            "title": _truncate(paper_meta.get(cid, {}).get("title", ""), 100),
            "abstract": _truncate(paper_meta.get(cid, {}).get("abstract", ""), 200),
            "reasoning": paper_meta.get(cid, {}).get("reasoning", ""),
            "screening_confidence": paper_meta.get(cid, {}).get("screening_confidence", 0.0),
            "borderline": paper_meta.get(cid, {}).get("borderline", False),
        }
        for cid in fp_ids[:50]
    ]
    metrics["false_negatives"] = false_negatives
    metrics["false_positives"] = false_positives

    n_relevant = metrics["n_relevant_gold"]
    n_irrelevant = metrics["n_irrelevant_gold"]
    if n_irrelevant == 0:
        notes = (
            f"Sample 100% gold-positive (human_label_ta == 'relevant') -> "
            "solo sensitivity es medible; specificity, precision y F1 no son "
            "calculables porque no hay verdaderos negativos en la muestra. "
            f"Cobertura abstract: {metrics['stratification']['n_with_abstract']} "
            f"con / {metrics['stratification']['n_without_abstract']} sin."
        )
    else:
        notes = (
            f"Sample mixto: {n_relevant} relevant + {n_irrelevant} irrelevant. "
            "Metricas binarias completas (sens/spec/prec/F1/acc/kappa) calculadas "
            "sobre predicciones decididas (excluye uncertain). T16.E.1.fix."
        )
    metrics["notes"] = notes

    # ── 6. Final JSON envelope ─────────────────────────────────────────────
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "hero_id": HERO_ID,
        "agent": AGENT_TAG,
        "sample_path": str(Path(eval_cfg["sample"])),
        "sample_size": sample_size,
        "model_used": model_display,
        "llm_config": llm_config,
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": elapsed,
        "environment": {
            "python_version": platform.python_version(),
            "ollama_version": None,
            "gpu_info": _gpu_info(),
        },
        "metrics": metrics,
        "confusion_matrix": cm,
        "raw_predictions": raw_predictions,
        "errors": errors,
        "notes": notes,
    }

    # ── 7. Persist JSON ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"hero1_results_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _print_short_report(result, out_path)

    if owns_agent:
        try:
            agent.free_vram()
        except Exception:
            pass

    return result


def _print_short_report(result: dict, out_path: Path) -> None:
    """Print a brief Spanish summary to stdout."""
    m = result["metrics"]
    print()
    print("=" * 60)
    print("HERO 1 (A3 — Screening Título/Abstract): RESUMEN")
    print("=" * 60)
    print(f"Sample size            : {result['sample_size']}")
    print(f"Modelo                 : {result['model_used']}")
    print(f"Tiempo total           : {result['elapsed_seconds']}s")
    print()
    print(f"N relevant gold        : {m['n_relevant_gold']}")
    print(f"N irrelevant gold      : {m['n_irrelevant_gold']}")
    print(f"Uncertain total        : {m['uncertain_count']}")
    print()
    print(f"Sensitivity (overall)  : {m['sensitivity_overall']}")
    print(f"Sensitivity (adj.)     : {m['sensitivity_adjusted']}")
    print(f"Specificity            : {m['specificity']}")
    print(f"Precision              : {m['precision']}")
    print(f"F1                     : {m['f1']}")
    print(f"Accuracy               : {m['accuracy']}")
    print(f"Cohen's kappa          : {m['cohens_kappa']}")
    print()
    print(f"TP / FN / FP / TN      : {m['tp']} / {m['fn']} / {m['fp']} / {m['tn']}")
    print(f"Output JSON guardado en: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hero 1 evaluator — Agent3 screening vs Paula gold standard."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to eval YAML config (default: config/eval/micro.yaml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for hero1_results_*.json (default: outputs/eval/micro/).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()

    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2

    run_hero1_evaluation(config_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
