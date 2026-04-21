#!/usr/bin/env python3
"""SAM Rudiger runner — reads rudiger/queries.json and launches the SAM
pipeline (A1..A9) once per query, honouring the --config override that
T16.9 added to batch_run.py.

This exists because batch_run.py has its QUERIES hardcoded. Instead of
patching that module per-server, this thin wrapper:

    1. Loads rudiger/queries.json (schema defined in T16.3).
    2. Reloads the module-level `_CFG` / MAX_RESULTS / ENSEMBLE_SIZE /
       VOTING_THRESHOLD in batch_run from the given --config path.
    3. Iterates queries (or a single --query-id) and calls
       batch_run.run_single_pipeline() for each, writing checkpoints and
       a pipeline.json under outputs/batch_<ts>/run_<id>/.

Usage (do not invoke directly; use rudiger/run_rudiger.sh):

    python rudiger/run_queries.py \
        --config rudiger/config_rudiger.yaml \
        --queries rudiger/queries.json \
        [--query-id paula_nutrition]

Exit codes:
    0  all queries completed (even if some pipelines failed internally)
    1  invalid args / queries.json missing or malformed
    2  requested --query-id not found in queries.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime

# Make sure we can import batch_run when launched from any cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM Rudiger queries runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML (e.g. rudiger/config_rudiger.yaml).",
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="Path to queries JSON (e.g. rudiger/queries.json).",
    )
    parser.add_argument(
        "--query-id",
        default=None,
        help="If set, only run the query with this id. Otherwise run all.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Override output root (default: outputs/batch_rudiger_<ts>/). "
            "One subdirectory per query will be created."
        ),
    )
    return parser.parse_args()


def _load_queries(path: str) -> list[dict]:
    if not os.path.isfile(path):
        print(f"[rudiger] ERROR: no existe queries.json en {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", [])
    if not queries:
        print(f"[rudiger] ERROR: queries.json no contiene 'queries'", file=sys.stderr)
        sys.exit(1)
    return queries


def _select_queries(queries: list[dict], query_id: str | None) -> list[dict]:
    if query_id is None:
        return queries
    matching = [q for q in queries if q.get("id") == query_id]
    if not matching:
        available = ", ".join(q.get("id", "?") for q in queries)
        print(
            f"[rudiger] ERROR: query_id '{query_id}' no encontrada. "
            f"IDs disponibles: {available}",
            file=sys.stderr,
        )
        sys.exit(2)
    return matching


def main() -> None:
    args = _parse_args()

    if not os.path.isfile(args.config):
        print(f"[rudiger] ERROR: no existe config YAML en {args.config}", file=sys.stderr)
        sys.exit(1)

    queries = _select_queries(_load_queries(args.queries), args.query_id)

    # Import AFTER we validate paths so a bad arg does not pay the import cost.
    import batch_run  # type: ignore  # noqa: E402

    # Refresh module-level config cache so run_single_pipeline picks it up
    # (it reads batch_run._CFG via closure). Mirrors what batch_run.main()
    # does after parsing --config.
    batch_run._CFG = batch_run._load_config(args.config)
    batch_run.MAX_RESULTS = batch_run._CFG.get("search", {}).get(
        "max_results_per_database", 150
    )
    batch_run.ENSEMBLE_SIZE = batch_run._CFG.get("screening", {}).get(
        "ensemble_size", 3
    )
    batch_run.VOTING_THRESHOLD = batch_run._CFG.get("screening", {}).get(
        "voting_threshold", 0.60
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = args.output_root or os.path.join("outputs", f"batch_rudiger_{ts}")
    os.makedirs(batch_dir, exist_ok=True)

    print("=" * 70)
    print(f"  SAM Rudiger runner")
    print(f"  Config:   {args.config}")
    print(f"  Queries:  {args.queries} ({len(queries)} seleccionadas)")
    print(f"  Output:   {batch_dir}")
    print(f"  Inicio:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_reports = []
    for i, q in enumerate(queries, 1):
        qid = q.get("id", f"q{i}")
        query = q.get("query", "")
        if not query:
            print(f"[rudiger] WARN: query '{qid}' sin campo 'query', saltada.")
            continue

        run_dir = os.path.join(batch_dir, f"run_{qid}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(queries)}] id={qid} framework={q.get('framework', '?')}")
        print(f"{query}")
        print("=" * 70)

        run_log: list[str] = []
        start = time.time()
        try:
            result = batch_run.run_single_pipeline(query, run_dir, run_log)
        except Exception as e:  # noqa: BLE001
            run_log.append(f"PIPELINE CRASHED: {e}")
            run_log.append(traceback.format_exc())
            result = {"query": query, "run_dir": run_dir, "error": str(e)}
        elapsed = round(time.time() - start, 1)
        run_log.append(f"Run completed in {elapsed}s")

        # Persist per-query log.
        with open(os.path.join(run_dir, "run_log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(run_log))

        report = batch_run.analyze_run(run_dir, result)
        report["elapsed_s"] = elapsed
        report["query_id"] = qid
        all_reports.append(report)

        # Force GPU cleanup between queries (same pattern as batch_run.main).
        batch_run.free_vram()
        gc.collect()

    # Persist batch report.
    with open(os.path.join(batch_dir, "batch_report.json"), "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print(f"  BATCH COMPLETO — {len(all_reports)} queries")
    print(f"  Reportes: {batch_dir}/batch_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
