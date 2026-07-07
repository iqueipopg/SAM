# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Hero A1 evaluator — Agent1Protocol vs LEADSInstruct ground truth.

Sprint:  T17.A1 Fase C.2 (script construction; smoke run is C.3).
Schema:  docs/iterations/T17.A1_hypothesis.md §5.4, §10.2.
Tag:     pre-T17.A1-v2 (pre-registered, immutable thresholds H1=0.70, H2=0.05).

Pipeline (per sample):
  1. Read ``origin.title`` from the LEADSInstruct sample record (no abstract is
     passed; §5.3 of the pre-registration restricts input to the title).
  2. Run ``Agent1Protocol.generate_protocol(title)`` to obtain the structured
     PICO/SPIDER/PECO/PCC protocol JSON.
  3. Run ``Agent1Protocol.generate_pubmed_query(title, protocol)`` to obtain
     the Boolean query string.
  4. Execute the query against PubMed E-utilities (``esearch``) with
     ``retmax=1000`` and ``sort="relevance"``. Cache by SHA-256 hash of the
     query so repeated runs do not hit NCBI again.
  5. Compute ``Recall@1000 = |returned PMIDs ∩ publication_pmids| /
     |publication_pmids|`` and report ``esearch_count`` as the workload proxy.

Completeness PICO (§4 of the pre-registration) is NOT computed by this script;
it requires manual annotation by the human evaluator and is therefore emitted
as ``null`` in the output JSON (both per-sample and aggregate).

Outputs:
  - ``outputs/eval/T17.A1/smoke_{YYYYMMDD_HHMMSS}.json`` — full per-sample
    results plus aggregates (macro recall/workload + null completeness).
  - ``outputs/eval/T17.A1/cache/{sha256_16}.json`` — one cache file per
    distinct Boolean query (gitignored).
  - Structured stdout in Spanish with one line per sample plus a final
    summary table.

Design notes:
  - Entrez configuration replicates the minimal pattern from
    ``agents/agent2_search.py`` (read email from env or ``config.yaml``;
    upgrade to 10 req/s when ``NCBI_API_KEY`` is set). We deliberately do
    NOT import ``Agent2Search`` because that would drag in Google Scholar
    scraping, semantic dedup, and other unused machinery (see hypothesis
    §5.4 and the approved plan ``cached-mixing-dream.md``).
  - The script never modifies ``agents/agent1_protocol.py`` and never writes
    to any of the sacred zones (``app.py``, ``config.yaml``, ``.env``,
    ``config/agents.json``, ``data/gold_standard.json``).
  - Errors on a single sample (A1 generation failure or Entrez network error)
    are captured and stored alongside that sample's record; processing
    continues with the remaining samples (anti-fragile smoke).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so ``agents`` package imports resolve when
# launching as ``python experiments/eval_hero_a1.py``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402
from Bio import Entrez  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SAMPLE_FILE = PROJECT_ROOT / "data" / "groundtruth" / "leads_smoke.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "T17.A1"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "eval" / "T17.A1" / "cache"

PRE_REGISTRATION_TAG = "pre-T17.A1-v2"
SEED = 42
K_RETMAX = 1000
SORT_ORDER = "relevance"
DATASET_ID = "zifeng-ai/LEADSInstruct/study_search_publication/test"

# Rate-limit cushion between live esearch calls (NCBI public limits).
# Without an API key NCBI allows ~3 req/s; with a key ~10 req/s.
SLEEP_WITH_KEY = 0.1
SLEEP_WITHOUT_KEY = 0.4


# ──────────────────────────────────────────────────────────────────────────────
# Entrez configuration (replicates Agent2Search pattern without importing it)
# ──────────────────────────────────────────────────────────────────────────────

def _configure_entrez() -> tuple[bool, bool]:
    """Configure Entrez email + optional API key from env / config.yaml.

    Returns ``(email_set, api_key_set)`` for metadata logging. The function
    never raises: if ``config.yaml`` is unreadable it falls back to the
    documented default email.
    """
    cfg: dict[str, Any] = {}
    cfg_path = PROJECT_ROOT / "config.yaml"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    search_cfg = cfg.get("search") or {}
    Entrez.email = (
        os.environ.get("ENTREZ_EMAIL")
        or search_cfg.get("entrez_email")
        or "meta-analysis@example.com"
    )
    api_key = (os.environ.get("NCBI_API_KEY") or "").strip()
    if api_key:
        Entrez.api_key = api_key
    return bool(Entrez.email), bool(api_key)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    """Return current UTC time formatted as ISO 8601 (``YYYY-MM-DDTHH:MM:SSZ``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> tuple[str, str]:
    """Return ``(short, full)`` HEAD commit hashes; empty strings on failure."""
    try:
        full = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        short = full[:7]
        return short, full
    except Exception:
        return "", ""


def _hash_query(query: str) -> str:
    """Return the first 16 hex chars of SHA-256(query) for cache keys."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def recall_at_k(returned_pmids: list[str], gold_pmids: list[str]) -> float:
    """Compute Recall@K = |returned ∩ gold| / |gold|.

    Returns 0.0 if ``gold_pmids`` is empty (avoids ZeroDivisionError; degenerate
    gold sets are filtered out upstream but the function is defensive).
    """
    if not gold_pmids:
        return 0.0
    intersection = set(returned_pmids) & set(gold_pmids)
    return len(intersection) / len(gold_pmids)


def _serialize_esearch(result: Any, query: str) -> dict[str, Any]:
    """Convert a Bio.Entrez parsed result into a plain JSON-serializable dict.

    The object returned by ``Entrez.read`` behaves like a dict but its values
    can be ``Bio.Entrez.Parser`` element subclasses (Element-like). We cast
    everything we need to plain ``str``/``int``/``list[str]`` so the cache
    survives across Biopython versions.
    """
    return {
        "IdList": [str(x) for x in result.get("IdList", [])],
        "Count": str(result.get("Count", "0")),
        "RetMax": str(result.get("RetMax", K_RETMAX)),
        "RetStart": str(result.get("RetStart", 0)),
        "query": query,
        "fetched_at_utc": _utc_now_iso(),
    }


def _load_a1_output_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    """Load the on-disk A1-output cache (``{sample_id: {"protocol", "query"}}``).

    Returns an empty dict when ``cache_path`` is ``None`` or the file is absent
    or unreadable. The cache file MUST contain a JSON object mapping sample ids
    to dicts; anything else is treated as a fresh start.
    """
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _read_a1_output_cache_entry(
    cache_path: Path | None,
    sample_id: str,
) -> dict[str, Any] | None:
    """Return the cached A1 output for ``sample_id`` or ``None`` on miss.

    Always re-reads the file from disk so concurrent writers don't trample
    entries (cheap insurance during a long overnight run).
    """
    if cache_path is None:
        return None
    cache = _load_a1_output_cache(cache_path)
    entry = cache.get(sample_id)
    if not isinstance(entry, dict):
        return None
    return entry


def _append_a1_output_cache_entry(
    cache_path: Path,
    sample_id: str,
    entry: dict[str, Any],
) -> None:
    """Atomically append a new entry to the A1-output cache file.

    Append-only semantics: existing entries are never overwritten — if
    ``sample_id`` is already present the disk file is left untouched.
    Writes go through a sibling ``.tmp`` file and ``os.replace`` for atomicity.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Re-read the file at the last possible moment to minimise the window
    # where another writer could have appended something we'd overwrite.
    current = _load_a1_output_cache(cache_path)
    if sample_id in current:
        return  # respect append-only invariant
    current[sample_id] = entry
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, cache_path)


def _esearch_with_retry(query: str, has_api_key: bool) -> dict[str, Any]:
    """Run ``Entrez.esearch`` once; on any exception sleep 2 s and retry once.

    Returns the serialized (plain dict) form of the response. Raises the
    second exception if both attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=K_RETMAX,
                sort=SORT_ORDER,
            )
            try:
                raw = Entrez.read(handle)
            finally:
                handle.close()
            serialized = _serialize_esearch(raw, query)
            # Honour rate limit cushion between successive calls.
            time.sleep(SLEEP_WITH_KEY if has_api_key else SLEEP_WITHOUT_KEY)
            return serialized
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == 0:
                # One retry after a short sleep — covers transient 429/5xx.
                time.sleep(2.0)
    assert last_exc is not None  # for type checkers
    raise last_exc


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_sample(
    sample: dict[str, Any],
    a1: Any,
    cache_dir: Path,
    has_api_key: bool,
    slot: int,
    total: int,
    intervention: bool = False,
    a1_output_cache_path: Path | None = None,
    a1_output_cache_state: dict[str, int] | None = None,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    """Run the full A1 → (validator?) → Entrez pipeline on a single sample.

    Errors are captured and returned in the result dict (``error`` field) so
    that the caller can continue with the next sample. The function logs one
    structured stdout line summarising the sample at the end.

    When ``a1_output_cache_path`` is provided, A1 invocations are cached on disk
    keyed by ``sample_id`` so a paired pre/post run reuses the *exact same* A1
    protocol+query (necessary because A1 is non-deterministic — see plan
    ``elegant-tumbling-turtle.md`` §"Critical exploration findings").

    When ``intervention=True``, the (cached-or-fresh) A1 query is post-processed
    by ``agents.mesh_validator.validate_query`` before being sent to Entrez.
    """
    sample_id = sample.get("id", "<unknown>")
    selected_index = sample.get("selected_index")
    origin = sample.get("origin", {}) or {}
    title = (origin.get("title") or "").strip()
    origin_pmid = origin.get("pmid")
    gold_pmids = list(sample.get("publication_pmids", []) or [])

    record: dict[str, Any] = {
        "id": sample_id,
        "selected_index": selected_index,
        "origin_title": title,
        "origin_pmid": origin_pmid,
        "gold_pmids_count": len(gold_pmids),
        "a1_protocol": None,
        "a1_query": None,
        "a1_query_hash": None,
        "esearch_count": None,
        "returned_pmids_count": None,
        "returned_pmids": [],
        "intersection_pmids": [],
        "intersection_count": None,
        "recall_at_1000": None,
        "completeness": None,  # manual annotation, §4.3 hypothesis
        "from_cache": False,
        "error": None,
        # T17.A1 Fase D.2 — paired-eval extension fields.
        "intervention_applied": bool(intervention),
        "a1_output_source": "fresh",
        "validator_modifications": None,
        "validator_near_misses": None,
        "a1_query_pre_validation": None,
    }

    # ── Stage 1: A1 protocol + query (cache-aware) ────────────────────────
    cached_a1 = (
        _read_a1_output_cache_entry(a1_output_cache_path, sample_id)
        if a1_output_cache_path is not None
        else None
    )

    if cached_a1 is not None:
        protocol = cached_a1.get("protocol")
        query = cached_a1.get("query") or ""
        record["a1_output_source"] = "cache"
        if a1_output_cache_state is not None:
            a1_output_cache_state["cache_hits"] = (
                a1_output_cache_state.get("cache_hits", 0) + 1
            )
    else:
        try:
            protocol = a1.generate_protocol(title)
            if prompt_template is not None:
                query = a1.generate_pubmed_query(
                    title, protocol, prompt_template=prompt_template
                )
            else:
                query = a1.generate_pubmed_query(title, protocol)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"A1 failure: {type(exc).__name__}: {exc}"
            _log_sample(slot, total, record)
            return record

        if a1_output_cache_path is not None:
            try:
                _append_a1_output_cache_entry(
                    a1_output_cache_path,
                    sample_id,
                    {"protocol": protocol, "query": query},
                )
                if a1_output_cache_state is not None:
                    a1_output_cache_state["fresh_writes"] = (
                        a1_output_cache_state.get("fresh_writes", 0) + 1
                    )
            except Exception as exc:  # noqa: BLE001
                # Cache write failure is logged but doesn't abort this sample —
                # the run still produces a valid record; the next run will retry.
                print(
                    f"[T17.A1.D] WARNING: no se pudo escribir a1_output_cache "
                    f"para {sample_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    if not query or not query.strip():
        record["a1_protocol"] = protocol
        record["error"] = "A1 returned empty query"
        _log_sample(slot, total, record)
        return record

    # ── Stage 1b: Optional MeSH validator intervention ────────────────────
    if intervention:
        # Local import so unit tests can patch ``agents.mesh_validator.validate_query``.
        from agents import mesh_validator  # noqa: WPS433

        try:
            modified_query, mod_count, near_misses = mesh_validator.validate_query(query)
        except Exception as exc:  # noqa: BLE001
            record["a1_protocol"] = protocol
            record["a1_query"] = query
            record["a1_query_pre_validation"] = query
            record["error"] = (
                f"mesh_validator failure: {type(exc).__name__}: {exc}"
            )
            _log_sample(slot, total, record)
            return record

        record["a1_query_pre_validation"] = query
        record["validator_modifications"] = int(mod_count)
        record["validator_near_misses"] = list(near_misses or [])
        query = modified_query

    record["a1_protocol"] = protocol
    record["a1_query"] = query
    record["a1_query_hash"] = _hash_query(query)

    # ── Stage 2: Entrez esearch (with disk cache) ─────────────────────────
    cache_path = cache_dir / f"{record['a1_query_hash']}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            record["from_cache"] = True
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"Cache read failure: {type(exc).__name__}: {exc}"
            _log_sample(slot, total, record)
            return record
    else:
        try:
            cached = _esearch_with_retry(query, has_api_key)
            cache_path.write_text(
                json.dumps(cached, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"Entrez failure: {type(exc).__name__}: {exc}"
            _log_sample(slot, total, record)
            return record

    # ── Stage 3: Metrics ──────────────────────────────────────────────────
    returned = [str(p) for p in cached.get("IdList", [])]
    try:
        esearch_count = int(cached.get("Count", "0"))
    except (TypeError, ValueError):
        esearch_count = 0

    intersection = sorted(set(returned) & set(gold_pmids))
    recall = recall_at_k(returned, gold_pmids)

    record["esearch_count"] = esearch_count
    record["returned_pmids_count"] = len(returned)
    record["returned_pmids"] = returned
    record["intersection_pmids"] = intersection
    record["intersection_count"] = len(intersection)
    record["recall_at_1000"] = round(recall, 6)

    _log_sample(slot, total, record)
    return record


def _log_sample(slot: int, total: int, record: dict[str, Any]) -> None:
    """Emit one structured stdout line per sample (Spanish, no emojis)."""
    title_preview = (record.get("origin_title") or "").replace("\n", " ")
    if len(title_preview) > 80:
        title_preview = title_preview[:77] + "..."
    if record.get("error"):
        print(
            f"[T17.A1.C2] sample {slot}/{total} id={record['id']} "
            f"title='{title_preview}' ERROR={record['error']}",
            flush=True,
        )
        return
    recall = record.get("recall_at_1000")
    recall_str = f"{recall:.3f}" if isinstance(recall, (int, float)) else "n/a"
    print(
        f"[T17.A1.C2] sample {slot}/{total} id={record['id']} "
        f"title='{title_preview}' "
        f"recall@1000={recall_str} "
        f"workload={record.get('esearch_count')} "
        f"returned={record.get('returned_pmids_count')} "
        f"intersect={record.get('intersection_count')} "
        f"from_cache={record.get('from_cache')}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Aggregates + summary
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute macro-averaged recall and workload aggregates over per-sample records.

    Completeness is always ``null`` (manual annotation per §4 hypothesis).
    Records with an ``error`` contribute ``None`` to the value lists; aggregates
    are computed only over numeric values.
    """
    recall_values: list[float | None] = []
    workload_values: list[float | None] = []
    completeness_values: list[None] = []

    for r in records:
        recall_values.append(
            r["recall_at_1000"] if isinstance(r.get("recall_at_1000"), (int, float)) else None
        )
        workload_values.append(
            r["esearch_count"] if isinstance(r.get("esearch_count"), (int, float)) else None
        )
        completeness_values.append(None)

    def _stats(values: list[float | None]) -> dict[str, Any]:
        numeric = [v for v in values if isinstance(v, (int, float))]
        if not numeric:
            return {"mean": None, "median": None, "values": values}
        return {
            "mean": round(statistics.fmean(numeric), 6),
            "median": round(statistics.median(numeric), 6),
            "values": values,
        }

    return {
        "recall_at_1000": _stats(recall_values),
        "workload": _stats(workload_values),
        "completeness": {
            "mean": None,
            "median": None,
            "values": completeness_values,
        },
    }


def _print_summary(result: dict[str, Any], out_path: Path) -> None:
    """Print a final Spanish summary table to stdout."""
    print()
    print("=" * 72)
    print("HERO A1 (Agent1 — Protocolo + Query PubMed) — RESUMEN SMOKE")
    print("=" * 72)
    meta = result["metadata"]
    print(f"Tag pre-registro       : {meta['tag']}")
    print(f"Dataset                : {meta['dataset']}")
    print(f"N                      : {meta['N']}")
    print(f"K (retmax)             : {meta['K']}")
    print(f"Sort                   : {meta['sort']}")
    print(f"Commit git             : {meta['git_commit']}")
    print(f"Tiempo total (s)       : {meta['elapsed_seconds']}")
    print(f"NCBI API key activa    : {meta['ncbi_api_key_set']}")
    print()
    agg = result["aggregates"]
    print(f"Recall@1000 (media)    : {agg['recall_at_1000']['mean']}")
    print(f"Recall@1000 (mediana)  : {agg['recall_at_1000']['median']}")
    print(f"Workload (media)       : {agg['workload']['mean']}")
    print(f"Workload (mediana)     : {agg['workload']['median']}")
    print("Completitud PICO       : null (anotación manual pendiente)")
    print()
    print(f"{'Slot':<5}{'ID':<14}{'Gold':>6}{'Workload':>10}{'Returned':>10}"
          f"{'Hit':>6}{'Recall@1000':>14}  Title")
    print("-" * 72)
    for i, r in enumerate(result["per_sample"], start=1):
        title_preview = (r.get("origin_title") or "").replace("\n", " ")
        if len(title_preview) > 40:
            title_preview = title_preview[:37] + "..."
        recall = r.get("recall_at_1000")
        recall_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else "n/a"
        workload = r.get("esearch_count")
        workload_str = str(workload) if workload is not None else "n/a"
        returned = r.get("returned_pmids_count")
        returned_str = str(returned) if returned is not None else "n/a"
        hit = r.get("intersection_count")
        hit_str = str(hit) if hit is not None else "n/a"
        print(
            f"{i:<5}{r.get('id', ''):<14}{r.get('gold_pmids_count', 0):>6}"
            f"{workload_str:>10}{returned_str:>10}{hit_str:>6}{recall_str:>14}  "
            f"{title_preview}"
        )
    print("-" * 72)
    print(f"JSON guardado en       : {out_path}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(
    sample_file: Path,
    output: Path | None = None,
    cache_dir: Path | None = None,
    a1_output_cache: Path | None = None,
    intervention: bool = False,
    a1_instance: Any = None,
    prompt_template: str | None = None,
    # Backward-compat aliases (existing tests call with these names).
    output_path: Path | None = None,
    a1_output_cache_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full evaluation and write the JSON output. Returns the metrics dict.

    Parameters
    ----------
    sample_file:
        Path to the LEADSInstruct sample artefact JSON (e.g. leads_full20.json).
    output:
        Path of the result JSON to write. Backward-compat alias ``output_path``
        is accepted (older tests use it).
    cache_dir:
        Directory for Entrez esearch response cache (per-query SHA-256 keyed).
    a1_output_cache:
        Optional path to an append-only JSON cache of A1 outputs (protocol + query)
        keyed by ``sample_id``. Enables deterministic paired pre/post runs.
        Backward-compat alias ``a1_output_cache_path``.
    intervention:
        If True, post-process every A1 query through ``agents.mesh_validator.validate_query``
        before sending to Entrez.
    a1_instance:
        Optional pre-built ``Agent1Protocol`` instance. If provided, it is reused
        across all samples (saves VRAM cycles when looping over prompt variants).
        If None, a fresh instance is constructed.
    prompt_template:
        Optional prompt template string passed through to ``a1.generate_pubmed_query``
        for every sample (used by the T17.A2 prompt-iteration orchestrator).
    """
    # Resolve canonical + alias parameters. New names win when both supplied.
    if output is None and output_path is not None:
        output = output_path
    if a1_output_cache is None and a1_output_cache_path is not None:
        a1_output_cache = a1_output_cache_path
    if output is None:
        raise TypeError("run_eval() requires an output path (kwarg 'output')")
    if cache_dir is None:
        raise TypeError("run_eval() requires a cache_dir path")

    output_path = output
    a1_output_cache_path = a1_output_cache

    # Refuse to overwrite an existing output silently — fail loudly.
    if output_path.exists():
        raise FileExistsError(
            f"El fichero de salida ya existe y no se sobrescribirá: {output_path}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not sample_file.is_file():
        raise FileNotFoundError(f"Sample file no encontrado: {sample_file}")

    with open(sample_file, encoding="utf-8") as f:
        sample_doc = json.load(f)
    samples = sample_doc.get("samples") or []
    if not samples:
        raise ValueError(f"Sample file no contiene 'samples': {sample_file}")

    email_set, api_key_set = _configure_entrez()
    git_short, git_full = _git_commit()

    # T17.A1 Fase D.2 — A1-output cache state (loaded once for metadata).
    initial_cache = _load_a1_output_cache(a1_output_cache_path)
    a1_output_cache_state: dict[str, int] = {
        "loaded": len(initial_cache),
        "fresh_writes": 0,
        "cache_hits": 0,
        "final_size": len(initial_cache),
    }

    # Build or reuse the A1 instance. T17.A2 orchestrator injects a shared one
    # so the model is loaded once per overnight run, not per variant.
    if a1_instance is None:
        # Import here so import-time failures of Bio/yaml don't block A1 unit tests.
        from agents.agent1_protocol import Agent1Protocol  # noqa: WPS433

        a1 = Agent1Protocol()
        owns_a1 = True
    else:
        a1 = a1_instance
        owns_a1 = False

    start_ts = time.time()
    per_sample: list[dict[str, Any]] = []
    try:
        for i, sample in enumerate(samples, start=1):
            record = evaluate_sample(
                sample=sample,
                a1=a1,
                cache_dir=cache_dir,
                has_api_key=api_key_set,
                slot=i,
                total=len(samples),
                intervention=intervention,
                a1_output_cache_path=a1_output_cache_path,
                a1_output_cache_state=a1_output_cache_state,
                prompt_template=prompt_template,
            )
            per_sample.append(record)
    finally:
        # project rule #1: free VRAM after the last A1 call (single GPU).
        # Only free if we own the instance — the caller (T17.A2 orchestrator)
        # may want to reuse it across many variants.
        if owns_a1:
            free_vram = getattr(a1, "free_vram", None)
            if callable(free_vram):
                try:
                    free_vram()
                except Exception:
                    traceback.print_exc()

    elapsed = round(time.time() - start_ts, 2)

    # Refresh ``final_size`` from disk so the metadata reflects any writes
    # appended during the run (and tolerates concurrent appenders).
    a1_output_cache_state["final_size"] = len(
        _load_a1_output_cache(a1_output_cache_path)
    )

    metadata = {
        "tag": PRE_REGISTRATION_TAG,
        "seed": SEED,
        "N": len(samples),
        "K": K_RETMAX,
        "sort": SORT_ORDER,
        "timestamp": _utc_now_iso(),
        "git_commit": git_short,
        "git_commit_full": git_full,
        "dataset": DATASET_ID,
        "sample_file": str(sample_file.resolve()),
        "entrez_email_set": email_set,
        "ncbi_api_key_set": api_key_set,
        "python_version": platform.python_version(),
        "elapsed_seconds": elapsed,
        # T17.A1 Fase D.2 — paired-eval metadata.
        "intervention": bool(intervention),
        "a1_output_cache_path": (
            str(a1_output_cache_path.resolve())
            if a1_output_cache_path is not None
            else None
        ),
        "a1_output_cache_state": a1_output_cache_state,
    }
    # T17.A2 — only add the prompt-iteration marker when a template was passed,
    # so existing CLI invocations produce bit-identical output JSONs.
    if prompt_template is not None:
        metadata["prompt_template_used"] = True

    result = {
        "metadata": metadata,
        "per_sample": per_sample,
        "aggregates": _aggregate(per_sample),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _print_summary(result, output_path)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _default_output_path() -> Path:
    """Build the default output path with current timestamp."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"smoke_{stamp}.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluador Hero A1 — caracteriza Agent1Protocol sobre el smoke "
            "N=3 de LEADSInstruct (tag pre-T17.A1-v2). No modifica A1."
        ),
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help=(
            "Ruta al JSON de samples generado en Fase C.1 (default: "
            "data/groundtruth/leads_smoke.json)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Ruta del JSON de salida (default: "
            "outputs/eval/T17.A1/smoke_{YYYYMMDD_HHMMSS}.json)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=(
            "Directorio de caché para respuestas esearch (default: "
            "outputs/eval/T17.A1/cache/)."
        ),
    )
    parser.add_argument(
        "--intervention",
        action="store_true",
        default=False,
        help=(
            "Activa la validación post-A1 vía agents.mesh_validator. La query "
            "modificada se usa para Entrez; la original se conserva en el "
            "campo a1_query_pre_validation (T17.A1 Fase D.2)."
        ),
    )
    parser.add_argument(
        "--a1-output-cache",
        type=Path,
        default=None,
        help=(
            "Ruta JSON de caché append-only para outputs de A1 "
            "(protocolo + query) por sample_id. Permite pares pre/post "
            "deterministas reutilizando exactamente la misma salida A1 "
            "(T17.A1 Fase D.2)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sample_file = args.sample_file.resolve()
    cache_dir = args.cache_dir.resolve()
    output_path = (args.output or _default_output_path()).resolve()
    a1_output_cache_path = (
        args.a1_output_cache.resolve() if args.a1_output_cache is not None else None
    )

    try:
        run_eval(
            sample_file=sample_file,
            output=output_path,
            cache_dir=cache_dir,
            a1_output_cache=a1_output_cache_path,
            intervention=bool(args.intervention),
        )
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
