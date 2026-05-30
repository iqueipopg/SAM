"""T17.C.2 — deterministic harness (NO LLM) for the Claude track of A3 screening.

This script is the DETERMINISTIC glue around the A3 Claude smoke run. It does
NOT invoke any LLM / subagent (Task() is a harness tool only the orchestrator can call).
It mirrors the bootstrap pattern of ``scripts/run_phase4_dev.py:cmd_setup`` for
A1 + A2, but with three deliberate differences:

  (a) It does NOT stub A3 — A3 is exactly what the Claude track will run.
  (b) ``papers_raw.json`` contains ONLY the N selected papers (not the full dev
      partition), so that the orchestrator's ``_eligible_a3`` set (papers_raw ∩
      subset) is exactly those N papers when ``subset_spec=None``.
  (c) Length-stratified deterministic selection of the N papers.

Two subcommands:
  - ``prepare``  : select N papers (length-stratified, class-balanced, seeded),
                   init_run + bootstrap A1/A2 (NO A3), write an ``_a3_eval_meta.json``
                   sidecar with the gold labels and stratum/class composition.
  - ``evaluate`` : read the committed A3 decisions (one JSON per paper), map them
                   to predicted labels, and compute Hero-1 metrics by IMPORTING
                   ``compute_hero1_metrics`` from ``eval_hero1_screening`` (not a
                   reimplementation). Papers with no A3 JSON yet are counted as
                   ``pending`` and omitted from the predictions.

The actual ``next -> Task(sam_a3_screening) -> commit`` loop is driven by the orchestrator
in-session. This harness only prepares the run and scores it afterwards.

Reuses (does not reimplement): sam_orchestrator (init_run, commit_output,
RunPaths, paper_id_to_slug), PAULA_SPORT_PROTOCOL_V2 from experiments._eval_common,
and the metric/label helpers from experiments.eval_hero1_screening.

NO sacred zones are touched. Usage:
    python experiments/eval_a3_claude.py prepare --n 20 --seed 20260531
    python experiments/eval_a3_claude.py evaluate <run_id> [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sam_orchestrator import (  # noqa: E402
    RunPaths,
    commit_output,
    init_run,
    paper_id_to_slug,
)
from experiments._eval_common import PAULA_SPORT_PROTOCOL_V2  # noqa: E402
from experiments.eval_hero1_screening import (  # noqa: E402
    _map_decision_to_predicted,
    _normalise_gold_label,
    compute_hero1_metrics,
)
from schemas.papers import Paper, PapersRawOutput, SearchStats  # noqa: E402
from schemas.protocol import ProtocolBody, ProtocolOutput  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_SAMPLE = (
    _REPO_ROOT / "data" / "gold_standard" / "deporte" / "samples" / "medio"
    / "sample_medio_a3.csv"
)
_DEFAULT_GOLD = (
    _REPO_ROOT / "data" / "gold_standard" / "deporte" / "unified"
    / "gold_screening_unified.csv"
)
_DEFAULT_QUERY = (
    "¿Mejoran las intervenciones de actividad física con componentes "
    "psicológicos los resultados de actividad física en adultos de edad de "
    "jubilación (50-70 años) que viven en la comunidad?"
)
_META_FILENAME = "_a3_eval_meta.json"
_AGENT_ID = "sam_a3_screening"

# Stratum names ordered from shortest to longest abstract.
_STRATA_ORDER = ["sin_abstract", "corto", "medio", "largo", "muy_largo"]
# Target mix per stratum (sums to 20).
_STRATA_TARGET = {
    "sin_abstract": 3,
    "corto": 5,
    "medio": 6,
    "largo": 4,
    "muy_largo": 2,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stratum_for_len(abstract_len: int) -> str:
    """Map an abstract length to a length stratum."""
    if abstract_len == 0:
        return "sin_abstract"
    if abstract_len <= 800:
        return "corto"
    if abstract_len <= 2000:
        return "medio"
    if abstract_len <= 4000:
        return "largo"
    return "muy_largo"


def _write_and_commit(run_id: str, agent_id: str, output_path: Path, payload: dict) -> dict:
    """Write a payload to ``output_path`` then register it with the orchestrator."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return commit_output(run_id, agent_id, str(output_path))


# ──────────────────────────────────────────────────────────────────────────────
# Candidate loading + join
# ──────────────────────────────────────────────────────────────────────────────

def _load_candidates(sample_path: Path, gold_path: Path) -> list[dict[str, Any]]:
    """JOIN sample (covidence_id) with gold (title/abstract) and build candidates.

    The abstract comes from the GOLD CSV (byte-identical to what Qwen consumed via
    ``eval_hero1_screening._build_papers``): strip; if empty / "nan" / "none" then
    abstract="" and abstract_used=False.
    """
    sample_df = pd.read_csv(sample_path, dtype={"covidence_id": str})
    gold_df = pd.read_csv(gold_path, dtype={"covidence_id": str})
    gold_idx = gold_df.set_index("covidence_id")

    candidates: list[dict[str, Any]] = []
    for _, row in sample_df.iterrows():
        cid = row["covidence_id"]
        if cid in gold_idx.index:
            gold_row = gold_idx.loc[cid]
            if isinstance(gold_row, pd.DataFrame):
                gold_row = gold_row.iloc[0]
            title = str(gold_row.get("title", "") or "").strip()
            abstract = str(gold_row.get("abstract", "") or "").strip()
            doi = str(gold_row.get("doi", "") or "").strip()
            authors = str(gold_row.get("authors", "") or "").strip()
            year = str(gold_row.get("year", "") or "").strip()
        else:
            title = abstract = doi = authors = year = ""

        abstract_used = bool(abstract) and abstract.lower() not in ("nan", "none")
        if not abstract_used:
            abstract = ""

        candidates.append(
            {
                "covidence_id": cid,
                "title": title,
                "abstract": abstract,
                "doi": doi if doi.lower() not in ("", "nan", "none") else "",
                "authors": authors,
                "year": year if year.lower() not in ("", "nan", "none") else "",
                "abstract_len": len(abstract),
                "abstract_used": abstract_used,
                "human_label_ta": row.get("human_label_ta"),
                "gold_label": _normalise_gold_label(row.get("human_label_ta")),
                "stratum": _stratum_for_len(len(abstract)),
            }
        )
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic, length-stratified, class-balanced selection
# ──────────────────────────────────────────────────────────────────────────────

def _select(
    candidates: list[dict[str, Any]], n: int, seed: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select N candidates: length-stratified target mix, class-balanced within.

    Strategy (fully deterministic given ``seed``):
      1. Compute per-stratum quotas from ``_STRATA_TARGET`` scaled to N.
      2. Fill each stratum quota by alternating relevant/irrelevant draws (to push
         the overall mix towards ~50/50), drawing seeded-randomly within each
         stratum x class cell.
      3. If a stratum is short, redistribute the deficit to adjacent strata.
      4. If after step 3 we still have fewer than N (global shortage), top up from
         any remaining candidates (seeded-random), again favouring the
         minority class to keep balance.

    Returns ``(selected, notes)`` where ``notes`` records any redistribution.
    """
    rng = random.Random(seed)
    notes: list[str] = []

    # Bucket candidates by stratum, then by class, sorted by covidence_id for a
    # stable base order before the seeded shuffle.
    by_stratum: dict[str, dict[str, list[dict[str, Any]]]] = {
        s: {"relevant": [], "irrelevant": []} for s in _STRATA_ORDER
    }
    for c in candidates:
        by_stratum[c["stratum"]][c["gold_label"]].append(c)
    for s in _STRATA_ORDER:
        for cls in ("relevant", "irrelevant"):
            cell = by_stratum[s][cls]
            cell.sort(key=lambda c: c["covidence_id"])
            rng.shuffle(cell)

    # Scale the target mix to N (default mix already sums to 20).
    base_total = sum(_STRATA_TARGET.values())
    quotas: dict[str, int] = {}
    if n == base_total:
        quotas = dict(_STRATA_TARGET)
    else:
        # Proportional scaling with largest-remainder rounding.
        raw = {s: _STRATA_TARGET[s] * n / base_total for s in _STRATA_ORDER}
        quotas = {s: int(raw[s]) for s in _STRATA_ORDER}
        remainder = n - sum(quotas.values())
        order = sorted(_STRATA_ORDER, key=lambda s: raw[s] - quotas[s], reverse=True)
        for i in range(remainder):
            quotas[order[i % len(order)]] += 1

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    class_taken = {"relevant": 0, "irrelevant": 0}

    def _draw_from(stratum: str, want_class: str) -> Optional[dict[str, Any]]:
        cell = by_stratum[stratum][want_class]
        while cell:
            cand = cell.pop()
            if cand["covidence_id"] not in selected_ids:
                return cand
        return None

    def _fill_stratum(stratum: str, quota: int) -> int:
        """Fill up to ``quota`` from ``stratum``, alternating minority class first."""
        taken = 0
        while taken < quota:
            # Prefer the currently-minority class to keep ~50/50.
            if class_taken["relevant"] <= class_taken["irrelevant"]:
                primary, secondary = "relevant", "irrelevant"
            else:
                primary, secondary = "irrelevant", "relevant"
            cand = _draw_from(stratum, primary) or _draw_from(stratum, secondary)
            if cand is None:
                break
            selected.append(cand)
            selected_ids.add(cand["covidence_id"])
            class_taken[cand["gold_label"]] += 1
            taken += 1
        return taken

    # Pass 1: fill each stratum to its quota.
    deficit: dict[str, int] = {}
    for s in _STRATA_ORDER:
        got = _fill_stratum(s, quotas[s])
        if got < quotas[s]:
            short = quotas[s] - got
            deficit[s] = short
            notes.append(
                f"estrato '{s}' corto: pedidos {quotas[s]}, obtenidos {got} "
                f"(deficit {short}) -> redistribuido a adyacentes"
            )

    # Pass 2: redistribute deficits to adjacent strata (nearest length first).
    for s, short in deficit.items():
        idx = _STRATA_ORDER.index(s)
        # Build adjacency order by increasing distance from s.
        neighbours: list[str] = []
        for dist in range(1, len(_STRATA_ORDER)):
            for j in (idx - dist, idx + dist):
                if 0 <= j < len(_STRATA_ORDER):
                    neighbours.append(_STRATA_ORDER[j])
        for nb in neighbours:
            if short <= 0:
                break
            got = _fill_stratum(nb, short)
            short -= got

    # Pass 3: global top-up if still short (any stratum, minority-class first).
    if len(selected) < n:
        for s in _STRATA_ORDER:
            if len(selected) >= n:
                break
            _fill_stratum(s, n - len(selected))
        if len(selected) < n:
            notes.append(
                f"escasez global: solicitados {n}, seleccionados {len(selected)} "
                "(no hay mas candidatos disponibles)"
            )

    # Stable, reproducible output order (by covidence_id).
    selected.sort(key=lambda c: c["covidence_id"])
    return selected[:n], notes


# ──────────────────────────────────────────────────────────────────────────────
# prepare
# ──────────────────────────────────────────────────────────────────────────────

def _write_slim_inputs(rp, selected: list[dict]) -> str:
    """Write one slim per-paper input file {paper_id,title,abstract} so each A3
    invocation reads only its own paper (not the whole papers_raw.json). Returns
    the sidecar dir name."""
    slim_dir = rp.run_dir / "_a3_inputs"
    slim_dir.mkdir(parents=True, exist_ok=True)
    for c in selected:
        cid = c["covidence_id"]
        payload = {"paper_id": cid, "title": c["title"], "abstract": c["abstract"]}
        (slim_dir / f"{paper_id_to_slug(cid)}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return "_a3_inputs"


def cmd_prepare(args: argparse.Namespace) -> int:
    sample_path = Path(args.sample) if args.sample else _DEFAULT_SAMPLE
    gold_path = Path(args.gold) if args.gold else _DEFAULT_GOLD
    query = args.query or _DEFAULT_QUERY

    candidates = _load_candidates(sample_path, gold_path)
    selected, notes = _select(candidates, args.n, args.seed)

    strata_counts = Counter(c["stratum"] for c in selected)
    class_counts = Counter(c["gold_label"] for c in selected)

    # --- init_run: subset_spec=None so eligible-A3 = papers_raw exactly ---
    run_id = init_run(query, subset_spec=None, backend_default="claude")
    rp = RunPaths(run_id)
    ts = _now()

    # --- A1 protocol bootstrap (PAULA_SPORT_PROTOCOL_V2, version "V2") ---
    body = ProtocolBody(
        framework=PAULA_SPORT_PROTOCOL_V2["framework"],
        research_question=PAULA_SPORT_PROTOCOL_V2["research_question"],
        population=PAULA_SPORT_PROTOCOL_V2["population"],
        intervention=PAULA_SPORT_PROTOCOL_V2["intervention"],
        comparator=PAULA_SPORT_PROTOCOL_V2["comparator"],
        outcome=PAULA_SPORT_PROTOCOL_V2["outcome"],
        original_query=query,
        inclusion_criteria=list(PAULA_SPORT_PROTOCOL_V2["inclusion_criteria"]),
        exclusion_criteria=list(PAULA_SPORT_PROTOCOL_V2["exclusion_criteria"]),
        protocol_document=(
            "Protocolo bootstrap T17.C.2 (PAULA_SPORT_PROTOCOL_V2). A1 no se "
            "ejecuta en esta corrida; se inyecta el protocolo V2 congelado para "
            "alimentar el screening titulo/abstract de A3."
        ),
        prospero_draft="PROSPERO CRD42021277712 (Collazo-Castiñeira et al. 2025).",
    )
    proto = ProtocolOutput(
        agent="sam_a1_protocol",
        backend="claude",
        timestamp=ts,
        tokens_in=None,
        tokens_out=None,
        query=query,
        protocol=body,
        pubmed_query="",
        version="V2",
    )
    _write_and_commit(
        run_id, "sam_a1_protocol", rp.single_output("sam_a1_protocol"), proto.model_dump()
    )

    # --- A2 papers_raw with ONLY the N selected papers ---
    papers: list[Paper] = []
    for c in selected:
        authors = [a.strip() for a in str(c["authors"]).split(";") if a.strip()]
        papers.append(
            Paper(
                paper_id=c["covidence_id"],
                pmid=None,
                doi=c["doi"] or None,
                title=c["title"],
                abstract=c["abstract"],
                authors=authors,
                journal="",
                year=c["year"],
                sources=["covidence"],
            )
        )
    n = len(papers)
    praw = PapersRawOutput(
        agent="sam_a2_search",
        backend="claude",
        timestamp=ts,
        tokens_in=None,
        tokens_out=None,
        papers=papers,
        stats=SearchStats(
            pubmed=0,
            europepmc=0,
            semantic_scholar=0,
            google_scholar=0,
            total_before_dedup=n,
            total_after_dedup=n,
            duplicates_removed=0,
            search_sources=["covidence"],
        ),
    )
    _write_and_commit(
        run_id, "sam_a2_search", rp.single_output("sam_a2_search"), praw.model_dump()
    )

    # NOTE: A3 is NOT bootstrapped — it is what the Claude track will run.

    # --- Slim per-paper inputs so each A3 invocation reads only its own paper ---
    slim_dir_name = _write_slim_inputs(rp, selected)

    # --- Sidecar with gold labels + composition (for evaluate) ---
    meta = {
        "task": "T17.C.2_smoke_claude_a3",
        "seed": args.seed,
        "n": n,
        "sample_path": str(sample_path),
        "gold_path": str(gold_path),
        "selected": [
            {
                "covidence_id": c["covidence_id"],
                "human_label_ta": (
                    None if c["human_label_ta"] is None
                    or (isinstance(c["human_label_ta"], float) and pd.isna(c["human_label_ta"]))
                    else str(c["human_label_ta"])
                ),
                "gold_label": c["gold_label"],
                "abstract_len": c["abstract_len"],
                "abstract_used": c["abstract_used"],
            }
            for c in selected
        ],
        "strata_counts": {s: strata_counts.get(s, 0) for s in _STRATA_ORDER},
        "class_counts": dict(class_counts),
        "selection_notes": notes,
        "slim_inputs_dir": slim_dir_name,
    }
    (rp.run_dir / _META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- stdout report ---
    print(f"run_id: {run_id}")
    print(f"seleccionados: {n}  (seed={args.seed})")
    print()
    print(f"{'covidence_id':<16} {'gold_label':<12} {'abs_len':>8}  abstract_used")
    print("-" * 56)
    for c in selected:
        print(
            f"{c['covidence_id']:<16} {c['gold_label']:<12} "
            f"{c['abstract_len']:>8}  {c['abstract_used']}"
        )
    print()
    print("conteos por estrato:")
    for s in _STRATA_ORDER:
        print(f"  {s:<14}: {strata_counts.get(s, 0)}")
    print("conteos por clase:")
    for cls in ("relevant", "irrelevant"):
        print(f"  {cls:<14}: {class_counts.get(cls, 0)}")
    print()
    print(f"slim inputs: {rp.run_dir / slim_dir_name}/ ({n} ficheros)")
    if notes:
        print()
        print("notas de seleccion:")
        for note in notes:
            print(f"  - {note}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# evaluate
# ──────────────────────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> int:
    run_id = args.run_id
    rp = RunPaths(run_id)
    meta_path = rp.run_dir / _META_FILENAME
    if not meta_path.is_file():
        print(f"ERROR: no existe {meta_path}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    selected = meta["selected"]

    predictions: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    for entry in selected:
        cid = entry["covidence_id"]
        out_path = rp.paper_output(_AGENT_ID, cid)
        if not out_path.is_file():
            pending_ids.append(cid)
            continue
        a3 = json.loads(out_path.read_text(encoding="utf-8"))
        decision = a3.get("decision", "EXCLUDE")
        predictions.append(
            {
                "covidence_id": cid,
                "gold": entry["gold_label"],
                "predicted": _map_decision_to_predicted(decision),
                "abstract_used": bool(entry["abstract_used"]),
            }
        )

    metrics, cm, fn_ids, fp_ids = compute_hero1_metrics(predictions)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = (
            _REPO_ROOT / "outputs" / "eval" / "medio"
            / f"hero1_claude_smoke_{stamp}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "track": "claude",
        "agent": _AGENT_ID,
        "run_id": run_id,
        "n_selected": len(selected),
        "n_evaluated": len(predictions),
        "n_pending": len(pending_ids),
        "pending_ids": pending_ids,
        "metrics": metrics,
        "confusion": cm,
        "false_negatives": fn_ids,
        "false_positives": fp_ids,
        "timestamp_utc": timestamp_utc,
    }
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- stdout summary ---
    m = metrics
    print("=" * 60)
    print("A3 CLAUDE TRACK — RESUMEN (Hero 1)")
    print("=" * 60)
    print(f"run_id          : {run_id}")
    print(f"seleccionados   : {len(selected)}")
    print(f"evaluados       : {len(predictions)}")
    print(f"pendientes      : {len(pending_ids)}")
    if pending_ids:
        print(f"  ids pendientes: {', '.join(pending_ids)}")
    print()
    print(f"Sensibilidad-aj : {m['sensitivity_adjusted']}")
    print(f"Especificidad   : {m['specificity']}")
    print(f"Cohen's kappa   : {m['cohens_kappa']}")
    print(f"Precision       : {m['precision']}")
    print(f"F1              : {m['f1']}")
    print(f"Accuracy        : {m['accuracy']}")
    print(f"Sensib-overall  : {m['sensitivity_overall']}")
    print()
    print(f"TP / FP / TN / FN : {m['tp']} / {m['fp']} / {m['tn']} / {m['fn']}")
    print(
        f"UNCERTAIN total   : {m['uncertain_count']} "
        f"(relevant={m['uncertain_among_relevant']}, "
        f"irrelevant={m['uncertain_among_irrelevant']})"
    )
    print()
    print(f"Output JSON       : {out_path}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="T17.C.2 deterministic harness for the Claude track of A3."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser(
        "prepare", help="Select N papers + bootstrap A1/A2 (NO A3) + sidecar."
    )
    p_prep.add_argument("--n", type=int, default=20)
    p_prep.add_argument("--seed", type=int, default=20260531)
    p_prep.add_argument("--sample", default=None, help="Override sample CSV path.")
    p_prep.add_argument("--gold", default=None, help="Override gold CSV path.")
    p_prep.add_argument("--query", default=None, help="Override the clinical query.")
    p_prep.set_defaults(func=cmd_prepare)

    p_eval = sub.add_parser(
        "evaluate", help="Score committed A3 decisions via compute_hero1_metrics."
    )
    p_eval.add_argument("run_id")
    p_eval.add_argument("--out", default=None, help="Override output JSON path.")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
