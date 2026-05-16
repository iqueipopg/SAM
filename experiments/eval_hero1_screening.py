"""
Hero 1 evaluator — Agent3 Title/Abstract screening vs Paula's gold standard.

Sprint:  T16.D.5.B
Schema:  docs/heroes_output_schema.md (section 3.1)
Inputs:
  - config/eval/micro.yaml          : paths to sample and gold CSVs
  - data/gold_standard/deporte/samples/micro/sample_50.csv
  - data/gold_standard/deporte/unified/gold_screening_unified.csv
  - config/agents.json              : per-agent LLM params (agent3_screening)

Output:
  outputs/eval/micro/hero1_results_{YYYYMMDD_HHMMSS}.json
    + short stdout report (Spanish)

Notes:
  - Sample is 100% gold-positive (human_label_ta == 'relevant'), so we can only
    measure sensitivity. False positive / true negative cells of the confusion
    matrix are not measurable from this sample; the caveat is documented in
    metrics.notes and at the end of the stdout report.
  - The script does NOT modify agents/agent3_screening.py, base.py, config.yaml.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


def _load_agent3_params() -> dict:
    """Load agent3_screening params from config/agents.json (defaults merged)."""
    return load_agent_params("agent3_screening")


def _build_papers(sample_df: pd.DataFrame, gold_df: pd.DataFrame) -> list:
    """Join sample (covidence_id) with gold (title+abstract). Preserve row order.

    Each output dict has the shape Agent3 expects: title, abstract, pmid.
    Adds an internal `_abstract_used` flag (True if abstract is non-empty) which
    the evaluator uses later to stratify metrics.
    """
    gold_idx = gold_df.set_index("covidence_id")
    papers = []
    for _, row in sample_df.iterrows():
        cid = row["covidence_id"]
        if cid in gold_idx.index:
            gold_row = gold_idx.loc[cid]
            # If the same covidence_id appears multiple times in gold, take the
            # first row (defensive — gold should be deduplicated upstream).
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
                "pmid": cid,                 # use covidence_id as the identifier
                "title": title,
                "abstract": abstract,
                "_abstract_used": abstract_used,
                "_has_abstract_csv": bool(row.get("has_abstract", False)),
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
            is already loaded (used by ``run_heroes_micro.py`` to share VRAM
            across heroes). When ``None``, a new instance is created and its
            VRAM is freed before returning (standalone mode).
    """
    _seed_torch(42)

    eval_cfg = _load_eval_config(config_path)
    sample_path = (PROJECT_ROOT / eval_cfg["sample"]).resolve()
    gold_path = (PROJECT_ROOT / eval_cfg["gold_screening"]).resolve()

    # ── 1. Load CSVs ───────────────────────────────────────────────────────
    sample_df = pd.read_csv(sample_path)
    gold_df = pd.read_csv(gold_path, dtype={"covidence_id": str})

    sample_size = len(sample_df)
    papers = _build_papers(sample_df, gold_df)

    # ── 2. Agent3 setup (single batch call) ────────────────────────────────
    # Import here to avoid heavy CUDA init at module load time.
    from agents.agent3_screening import Agent3Screening

    a3_params = _load_agent3_params()
    owns_agent = agent_instance is None
    if agent_instance is None:
        agent = Agent3Screening()
    else:
        agent = agent_instance

    # Capture the actual generation params A3 will use (it does not accept a
    # temperature override; we record what is effective).
    llm_config = {
        "temperature": 0.2,  # hardcoded inside Agent3._simple_generate
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

    # ── 3. Run screening (one batch call for all 50 papers) ────────────────
    start_ts = time.time()
    errors: list = []
    try:
        screen_out = agent.screen_batch(papers, PAULA_SPORT_PROTOCOL)
    except Exception as e:
        # Surface as a top-level error and bail out with an empty result.
        errors.append(
            {
                "input_id": "<batch>",
                "error_type": "ModelError",
                "message": f"screen_batch raised: {type(e).__name__}: {e}",
            }
        )
        screen_out = {"include": [], "exclude": [], "uncertain": [], "stats": {}}
    elapsed = round(time.time() - start_ts, 2)

    # Index Agent3 results by pmid (== covidence_id) for fast lookup.
    a3_by_id: dict = {}
    for bucket in ("include", "exclude", "uncertain"):
        for r in screen_out.get(bucket, []):
            a3_by_id[r.get("pmid", "")] = r

    # ── 4. Build per-paper rows + metrics ──────────────────────────────────
    raw_predictions = []
    tp = 0
    fn = 0
    uncertain_count = 0
    tp_yes = 0   # TPs among papers with abstract
    fn_yes = 0
    tp_no = 0    # TPs among papers without abstract
    fn_no = 0
    n_with_abs = 0
    n_without_abs = 0
    false_negatives = []

    # Confusion-matrix counts (rows = gold, cols = predicted).
    # Sample is 100% relevant → only the "relevant" row will receive counts.
    cm = {
        "relevant":   {"relevant": 0, "irrelevant": 0, "uncertain": 0},
        "irrelevant": {"relevant": 0, "irrelevant": 0, "uncertain": 0},
    }

    for paper in papers:
        cid = paper["pmid"]
        gold_label = "relevant"  # sample invariant
        a3_result = a3_by_id.get(cid)
        abstract_used = bool(paper.get("_abstract_used", False))

        if abstract_used:
            n_with_abs += 1
        else:
            n_without_abs += 1

        if a3_result is None:
            # Should not happen with screen_batch (it returns conservative
            # EXCLUDEs on internal errors), but be defensive.
            predicted = "irrelevant"
            match = False
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
            match = predicted == "relevant"
            votes_dump = a3_result.get("votes", [])
            reasoning = a3_result.get("reasoning", "")
            screening_conf = a3_result.get("screening_confidence", 0.0)
            borderline = a3_result.get("borderline", False)

        # Confusion matrix
        cm[gold_label][predicted] += 1

        # Tally TP/FN/uncertain
        if predicted == "relevant":
            tp += 1
            if abstract_used:
                tp_yes += 1
            else:
                tp_no += 1
        elif predicted == "uncertain":
            uncertain_count += 1
            # An uncertain prediction is not a TP, but we also do not count it
            # as a FN: it represents an explicit "do not know" answer.
            # However, schema 3.1 only tracks TP / FN / uncertain — we report
            # the uncertain count separately and exclude these from FN totals.
        else:  # irrelevant
            fn += 1
            if abstract_used:
                fn_yes += 1
            else:
                fn_no += 1
            false_negatives.append(
                {
                    "covidence_id": cid,
                    "title": _truncate(paper.get("title", ""), 100),
                    "abstract": _truncate(paper.get("abstract", ""), 200),
                    "reasoning": reasoning,
                    "screening_confidence": screening_conf,
                    "borderline": borderline,
                }
            )

        # raw_llm_response: concatenate each vote's reasoning (Agent3 does not
        # surface the raw LLM string verbatim, but `votes[].reasoning` is the
        # parsed `reasoning` per-vote — sufficient for debugging per task spec).
        if votes_dump:
            raw_pieces = []
            for i, v in enumerate(votes_dump):
                d = v.get("decision", "") if isinstance(v, dict) else str(v)
                r = v.get("reasoning", "") if isinstance(v, dict) else ""
                raw_pieces.append(f"vote{i+1}={d}: {r}")
            raw_llm_response = "\n".join(raw_pieces)
        else:
            raw_llm_response = ""

        raw_predictions.append(
            {
                "input_id": cid,
                "gold": gold_label,
                "predicted": predicted,
                "match": match,
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

    # ── 5. Metrics ─────────────────────────────────────────────────────────
    sens_overall = round(tp / sample_size, 4) if sample_size > 0 else 0.0
    sens_yes = round(tp_yes / n_with_abs, 4) if n_with_abs > 0 else None
    sens_no = round(tp_no / n_without_abs, 4) if n_without_abs > 0 else None

    notes = (
        "Sample is 100% gold-positive (human_label_ta == 'relevant') → "
        "solo sensitivity es medible; specificity, precision y F1 no son "
        "calculables porque no hay verdaderos negativos en la muestra. "
        f"Cobertura por estrato de abstract: {n_with_abs} con abstract / "
        f"{n_without_abs} sin abstract."
    )

    metrics = {
        "sensitivity_overall": sens_overall,
        "sensitivity_abstract_yes": sens_yes,
        "sensitivity_abstract_no": sens_no,
        "tp": tp,
        "fn": fn,
        "tn": None,
        "fp": None,
        "uncertain_count": uncertain_count,
        "false_negatives": false_negatives,
        "stratification": {
            "n_with_abstract": n_with_abs,
            "n_without_abstract": n_without_abs,
            "tp_with_abstract": tp_yes,
            "tp_without_abstract": tp_no,
            "fn_with_abstract": fn_yes,
            "fn_without_abstract": fn_no,
        },
        "notes": notes,
    }

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

    # Free VRAM before exit (Agent3 holds the model). Only when we own the
    # instance — if the caller injected a shared agent, it manages VRAM.
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
    print(f"Sensitivity (overall)  : {m['sensitivity_overall']}")
    print(f"Sensitivity (con abs)  : {m['sensitivity_abstract_yes']} "
          f"({m['stratification']['tp_with_abstract']}/{m['stratification']['n_with_abstract']})")
    print(f"Sensitivity (sin abs)  : {m['sensitivity_abstract_no']} "
          f"({m['stratification']['tp_without_abstract']}/{m['stratification']['n_without_abstract']})")
    print(f"TP / FN / uncertain    : {m['tp']} / {m['fn']} / {m['uncertain_count']}")
    print()
    print("Caveat: la muestra es 100% gold-positive → solo es medible la "
          "sensitivity; specificity / FP / TN no son calculables aquí.")
    print(f"\nOutput JSON guardado en: {out_path}")


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
