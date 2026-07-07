# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Hero A1 PICO evaluator — Agent1Protocol metadata vs LEADSInstruct ground truth.

Sprint:  T17.A3 Fase I.1 (script construction).
Target:  ``Agent1Protocol._generate_metadata`` (entrypoint via
         ``generate_protocol(query, metadata_prompt_template=...)``).
Metric:  per-element semantic cosine similarity (P/I/C/O) using
         ``sentence-transformers/all-MiniLM-L6-v2`` (already cached in
         ``~/.cache/huggingface``).

Differences vs ``eval_hero_a1.py``:
  - No Entrez calls; we evaluate only the protocol's PICO structure.
  - Per-sample metric is a 4-element cosine vector + its arithmetic mean.
  - Aggregates: macro mean of per-sample means, median, per-element breakdown,
    fraction of samples with >= 3/4 elements above similarity 0.5.
  - Reused: sample loader, A1 output cache (append-only by sample_id),
    git/platform metadata, structured logging.

Outputs:
  - ``<output>`` — full per-sample records plus aggregates JSON.
  - ``<a1_output_cache>`` — append-only protocol cache keyed by sample_id with
    a hash of the prompt template (so different templates do not collide).

Errors on a single sample are captured (``error`` field) and the run continues.
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

# Ensure project root is on sys.path so ``agents`` package imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SAMPLE_FILE = PROJECT_ROOT / "data" / "groundtruth" / "leads_smoke.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "T17.A3"
DEFAULT_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PRE_REGISTRATION_TAG = "pre-T17.A3"
SEED = 42
DATASET_ID = "zifeng-ai/LEADSInstruct/study_search_publication/test"

# Threshold used to flag "good enough" per-element similarity in the secondary
# aggregate (pct_samples_3_of_4_above_0.5). Frozen pre-iteration per plan.
ELEMENT_GOOD_THRESHOLD = 0.5
ELEMENT_GOOD_MIN_COUNT = 3  # 3 out of 4 P/I/C/O elements


# ──────────────────────────────────────────────────────────────────────────────
# Embedder singleton (lazy-cached, local — not coupled to agents/)
# ──────────────────────────────────────────────────────────────────────────────

_EMBEDDER_CACHE: dict[str, Any] = {}


def _get_embedder(model_name: str = DEFAULT_EMBEDDER_MODEL) -> Any:
    """Lazy singleton SentenceTransformer (CPU). Loaded once per process.

    We intentionally instantiate the embedder locally rather than reusing the
    one in ``agents/agent2_search.py`` to keep experiments decoupled from
    production agents (no GPU dependency, no Bio import side-effects).
    """
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def pico_element_similarity(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    embedder: Any,
) -> dict[str, float]:
    """Compute per-element cosine similarity over P/I/C/O.

    Both ``predicted`` and ``gold`` are dicts with ``P/I/C/O`` keys (strings).
    Empty or missing values yield 0.0 similarity for that element (never NaN).
    Returns ``{"P": float, "I": float, "C": float, "O": float, "mean": float}``.
    """
    from sentence_transformers import util  # noqa: WPS433

    elements = ("P", "I", "C", "O")
    sims: dict[str, float] = {}
    for key in elements:
        pred_val = (predicted.get(key) or "").strip() if isinstance(predicted, dict) else ""
        gold_val = (gold.get(key) or "").strip() if isinstance(gold, dict) else ""
        if not pred_val or not gold_val:
            sims[key] = 0.0
            continue
        try:
            emb_pred = embedder.encode(pred_val, convert_to_tensor=True, show_progress_bar=False)
            emb_gold = embedder.encode(gold_val, convert_to_tensor=True, show_progress_bar=False)
            cs = util.cos_sim(emb_pred, emb_gold).item()
            sims[key] = float(round(cs, 6))
        except Exception:
            sims[key] = 0.0
    sims["mean"] = float(round(sum(sims[k] for k in elements) / len(elements), 6))
    return sims


def normalize_a1_framework_to_pico(metadata_dict: dict[str, Any]) -> dict[str, str]:
    """Map A1 metadata (any framework) into a uniform P/I/C/O dict.

    Mapping table (documented in T17.A3 hypothesis):
      - PICO  : P=population,           I=intervention,            C=comparator, O=outcome.
      - PECO  : P=population,           I=exposure (or intervention if absent),
                                        C=comparator, O=outcome.
      - SPIDER: P=population or sample, I=phenomenon_of_interest (or intervention),
                                        C=comparator (often empty), O=outcome.
      - PCC   : P=population,           I=concept (or intervention),
                                        C=comparator (often empty),
                                        O=outcome (fallback to context).

    Any missing string becomes ``""``.
    """
    md = metadata_dict or {}
    framework = (md.get("framework") or "PICO").upper()

    population = md.get("population") or ""
    intervention = md.get("intervention") or ""
    comparator = md.get("comparator") or ""
    outcome = md.get("outcome") or ""

    if framework == "PECO":
        # ``exposure`` takes precedence over ``intervention``.
        i_value = md.get("exposure") or intervention or ""
        return {
            "P": population,
            "I": i_value,
            "C": comparator,
            "O": outcome,
        }
    if framework == "SPIDER":
        p_value = population or md.get("sample") or ""
        i_value = md.get("phenomenon_of_interest") or intervention or ""
        return {
            "P": p_value,
            "I": i_value,
            "C": comparator,
            "O": outcome,
        }
    if framework == "PCC":
        i_value = md.get("concept") or intervention or ""
        o_value = outcome or md.get("context") or ""
        return {
            "P": population,
            "I": i_value,
            "C": comparator,
            "O": o_value,
        }
    # Default: PICO (and any unknown framework falls back to PICO mapping).
    return {
        "P": population,
        "I": intervention,
        "C": comparator,
        "O": outcome,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (general)
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


def _hash_template(template: str | None) -> str:
    """Return the first 16 hex chars of SHA-256(template) or 'baseline'."""
    if template is None:
        return "baseline"
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def _load_a1_output_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    """Load the on-disk A1-output cache. Returns ``{}`` on miss or parse error."""
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
    template_hash: str,
) -> dict[str, Any] | None:
    """Return cached A1 protocol entry iff sample_id + template_hash match.

    Different prompts produce different metadata; caching keyed only by
    ``sample_id`` would silently mix variants. We key by
    ``f"{sample_id}::{template_hash}"`` to keep variants isolated.
    """
    if cache_path is None:
        return None
    cache = _load_a1_output_cache(cache_path)
    key = f"{sample_id}::{template_hash}"
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    return entry


def _append_a1_output_cache_entry(
    cache_path: Path,
    sample_id: str,
    template_hash: str,
    entry: dict[str, Any],
) -> None:
    """Atomically append a fresh entry. Append-only (never overwrites)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_a1_output_cache(cache_path)
    key = f"{sample_id}::{template_hash}"
    if key in current:
        return
    current[key] = entry
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, cache_path)


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_sample(
    sample: dict[str, Any],
    a1: Any,
    embedder: Any,
    slot: int,
    total: int,
    metadata_prompt_template: str | None = None,
    a1_output_cache_path: Path | None = None,
    a1_output_cache_state: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run A1 protocol generation + PICO normalization + cosine eval on one sample."""
    sample_id = sample.get("id", "<unknown>")
    origin = sample.get("origin", {}) or {}
    title = (origin.get("title") or "").strip()
    gold_pico = sample.get("pico") or {}
    template_hash = _hash_template(metadata_prompt_template)

    record: dict[str, Any] = {
        "id": sample_id,
        "origin_title": title,
        "framework_chosen": None,
        "a1_protocol": None,
        "predicted_pico_normalized": None,
        "gold_pico": gold_pico,
        "pico_similarity": None,
        "from_cache": False,
        "error": None,
        "a1_output_source": "fresh",
        "metadata_prompt_template_hash": template_hash,
    }

    # ── Stage 1: A1 protocol (cache-aware) ────────────────────────────────
    cached_a1 = (
        _read_a1_output_cache_entry(a1_output_cache_path, sample_id, template_hash)
        if a1_output_cache_path is not None
        else None
    )

    if cached_a1 is not None:
        protocol = cached_a1.get("protocol")
        record["a1_output_source"] = "cache"
        record["from_cache"] = True
        if a1_output_cache_state is not None:
            a1_output_cache_state["cache_hits"] = (
                a1_output_cache_state.get("cache_hits", 0) + 1
            )
    else:
        try:
            protocol = a1.generate_protocol(
                title,
                metadata_prompt_template=metadata_prompt_template,
            )
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"A1 failure: {type(exc).__name__}: {exc}"
            _log_sample(slot, total, record)
            return record

        if a1_output_cache_path is not None:
            try:
                _append_a1_output_cache_entry(
                    a1_output_cache_path,
                    sample_id,
                    template_hash,
                    {"protocol": protocol, "template_hash": template_hash},
                )
                if a1_output_cache_state is not None:
                    a1_output_cache_state["fresh_writes"] = (
                        a1_output_cache_state.get("fresh_writes", 0) + 1
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[T17.A3.I1] WARNING: no se pudo escribir a1_output_cache "
                    f"para {sample_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    if not isinstance(protocol, dict):
        record["error"] = "A1 returned non-dict protocol"
        _log_sample(slot, total, record)
        return record

    record["a1_protocol"] = protocol
    record["framework_chosen"] = protocol.get("framework")

    # ── Stage 2: Normalize to PICO ────────────────────────────────────────
    try:
        predicted_pico = normalize_a1_framework_to_pico(protocol)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"normalize failure: {type(exc).__name__}: {exc}"
        _log_sample(slot, total, record)
        return record

    record["predicted_pico_normalized"] = predicted_pico

    # ── Stage 3: Cosine similarity per element ────────────────────────────
    try:
        sims = pico_element_similarity(predicted_pico, gold_pico, embedder)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"similarity failure: {type(exc).__name__}: {exc}"
        _log_sample(slot, total, record)
        return record

    record["pico_similarity"] = sims
    _log_sample(slot, total, record)
    return record


def _log_sample(slot: int, total: int, record: dict[str, Any]) -> None:
    """Emit one structured stdout line per sample (Spanish, no emojis)."""
    title_preview = (record.get("origin_title") or "").replace("\n", " ")
    if len(title_preview) > 60:
        title_preview = title_preview[:57] + "..."
    if record.get("error"):
        print(
            f"[T17.A3.I1] sample {slot}/{total} id={record['id']} "
            f"title='{title_preview}' ERROR={record['error']}",
            flush=True,
        )
        return
    sim = record.get("pico_similarity") or {}
    mean = sim.get("mean")
    mean_str = f"{mean:.3f}" if isinstance(mean, (int, float)) else "n/a"
    p_str = f"{sim.get('P', 0.0):.2f}" if isinstance(sim.get("P"), (int, float)) else "n/a"
    i_str = f"{sim.get('I', 0.0):.2f}" if isinstance(sim.get("I"), (int, float)) else "n/a"
    c_str = f"{sim.get('C', 0.0):.2f}" if isinstance(sim.get("C"), (int, float)) else "n/a"
    o_str = f"{sim.get('O', 0.0):.2f}" if isinstance(sim.get("O"), (int, float)) else "n/a"
    print(
        f"[T17.A3.I1] sample {slot}/{total} id={record['id']} "
        f"title='{title_preview}' "
        f"framework={record.get('framework_chosen')} "
        f"mean={mean_str} (P={p_str} I={i_str} C={c_str} O={o_str}) "
        f"from_cache={record.get('from_cache')}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Aggregates + summary
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute macro-averaged similarity metrics over per-sample records."""
    means: list[float] = []
    per_element: dict[str, list[float]] = {"P": [], "I": [], "C": [], "O": []}
    samples_above_3_of_4 = 0
    n_with_sim = 0

    for r in records:
        sim = r.get("pico_similarity")
        if not isinstance(sim, dict):
            continue
        n_with_sim += 1
        mean = sim.get("mean")
        if isinstance(mean, (int, float)):
            means.append(float(mean))
        good_count = 0
        for elem in ("P", "I", "C", "O"):
            val = sim.get(elem)
            if isinstance(val, (int, float)):
                per_element[elem].append(float(val))
                if val > ELEMENT_GOOD_THRESHOLD:
                    good_count += 1
        if good_count >= ELEMENT_GOOD_MIN_COUNT:
            samples_above_3_of_4 += 1

    def _safe_mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 6) if values else None

    def _safe_median(values: list[float]) -> float | None:
        return round(statistics.median(values), 6) if values else None

    pct_3_of_4 = (
        round(samples_above_3_of_4 / n_with_sim, 6) if n_with_sim > 0 else None
    )

    return {
        "macro_mean": _safe_mean(means),
        "median": _safe_median(means),
        "values": means,
        "per_element": {
            "P_mean": _safe_mean(per_element["P"]),
            "I_mean": _safe_mean(per_element["I"]),
            "C_mean": _safe_mean(per_element["C"]),
            "O_mean": _safe_mean(per_element["O"]),
        },
        "pct_samples_3_of_4_above_0.5": pct_3_of_4,
        "n_with_similarity": n_with_sim,
    }


def _print_summary(result: dict[str, Any], out_path: Path) -> None:
    """Print final Spanish summary table to stdout."""
    print()
    print("=" * 72)
    print("HERO A1 PICO (Agent1 — extracción metadata PICO) — RESUMEN")
    print("=" * 72)
    meta = result["metadata"]
    print(f"Tag pre-registro       : {meta['tag']}")
    print(f"Dataset                : {meta['dataset']}")
    print(f"N                      : {meta['N']}")
    print(f"Commit git             : {meta['git_commit']}")
    print(f"Tiempo total (s)       : {meta['elapsed_seconds']}")
    print(f"Embedder               : {meta['embedder_model']}")
    print()
    agg = result["aggregates"]
    print(f"Macro mean             : {agg['macro_mean']}")
    print(f"Mediana                : {agg['median']}")
    print(f"P (media)              : {agg['per_element']['P_mean']}")
    print(f"I (media)              : {agg['per_element']['I_mean']}")
    print(f"C (media)              : {agg['per_element']['C_mean']}")
    print(f"O (media)              : {agg['per_element']['O_mean']}")
    print(f"% muestras 3/4 > 0.5   : {agg['pct_samples_3_of_4_above_0.5']}")
    print()
    print(f"{'Slot':<5}{'ID':<14}{'Framework':<10}{'Mean':>8}"
          f"{'P':>7}{'I':>7}{'C':>7}{'O':>7}  Title")
    print("-" * 72)
    for i, r in enumerate(result["per_sample"], start=1):
        title_preview = (r.get("origin_title") or "").replace("\n", " ")
        if len(title_preview) > 30:
            title_preview = title_preview[:27] + "..."
        sim = r.get("pico_similarity") or {}
        m = sim.get("mean")
        m_str = f"{m:.3f}" if isinstance(m, (int, float)) else "n/a"
        p = sim.get("P")
        i_v = sim.get("I")
        c = sim.get("C")
        o = sim.get("O")
        p_str = f"{p:.2f}" if isinstance(p, (int, float)) else "n/a"
        i_str = f"{i_v:.2f}" if isinstance(i_v, (int, float)) else "n/a"
        c_str = f"{c:.2f}" if isinstance(c, (int, float)) else "n/a"
        o_str = f"{o:.2f}" if isinstance(o, (int, float)) else "n/a"
        print(
            f"{i:<5}{r.get('id', ''):<14}{(r.get('framework_chosen') or '-'):<10}"
            f"{m_str:>8}{p_str:>7}{i_str:>7}{c_str:>7}{o_str:>7}  "
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
    a1_instance: Any = None,
    metadata_prompt_template: str | None = None,
    embedder: Any = None,
    embedder_model: str = DEFAULT_EMBEDDER_MODEL,
    a1_output_cache: Path | None = None,
    # Backward-compat aliases mirroring eval_hero_a1 conventions.
    output_path: Path | None = None,
    a1_output_cache_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full PICO evaluation and write the JSON output. Returns the metrics dict.

    Parameters
    ----------
    sample_file:
        Path to the LEADSInstruct sample artefact JSON (e.g. leads_full20.json).
    output:
        Path of the result JSON to write. Backward-compat alias ``output_path``.
    a1_instance:
        Optional pre-built ``Agent1Protocol`` instance. If provided, it is reused
        across all samples (saves VRAM cycles for prompt-iteration loops).
    metadata_prompt_template:
        Optional prompt template (with ``{query}`` placeholder) passed through to
        ``a1.generate_protocol``. ``None`` ⇒ A1 uses its baseline hardcoded prompt.
    embedder:
        Optional pre-loaded SentenceTransformer. Lazy-loaded via ``_get_embedder``
        when ``None``.
    embedder_model:
        Model id for SentenceTransformer (default ``all-MiniLM-L6-v2``).
    a1_output_cache:
        Optional append-only JSON cache for A1 outputs keyed by
        ``{sample_id}::{template_hash}``. Different templates do not collide.
    """
    # Resolve canonical + alias parameters.
    if output is None and output_path is not None:
        output = output_path
    if a1_output_cache is None and a1_output_cache_path is not None:
        a1_output_cache = a1_output_cache_path
    if output is None:
        raise TypeError("run_eval() requires an output path (kwarg 'output')")

    output_path = output
    a1_output_cache_path = a1_output_cache

    if output_path.exists():
        raise FileExistsError(
            f"El fichero de salida ya existe y no se sobrescribirá: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not sample_file.is_file():
        raise FileNotFoundError(f"Sample file no encontrado: {sample_file}")

    with open(sample_file, encoding="utf-8") as f:
        sample_doc = json.load(f)
    samples = sample_doc.get("samples") or []
    if not samples:
        raise ValueError(f"Sample file no contiene 'samples': {sample_file}")

    git_short, git_full = _git_commit()

    # Embedder: lazy singleton — instantiated once per process.
    if embedder is None:
        embedder = _get_embedder(embedder_model)

    initial_cache = _load_a1_output_cache(a1_output_cache_path)
    a1_output_cache_state: dict[str, int] = {
        "loaded": len(initial_cache),
        "fresh_writes": 0,
        "cache_hits": 0,
        "final_size": len(initial_cache),
    }

    # Build or reuse the A1 instance.
    if a1_instance is None:
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
                embedder=embedder,
                slot=i,
                total=len(samples),
                metadata_prompt_template=metadata_prompt_template,
                a1_output_cache_path=a1_output_cache_path,
                a1_output_cache_state=a1_output_cache_state,
            )
            per_sample.append(record)
    finally:
        if owns_a1:
            free_vram = getattr(a1, "free_vram", None)
            if callable(free_vram):
                try:
                    free_vram()
                except Exception:
                    traceback.print_exc()

    elapsed = round(time.time() - start_ts, 2)

    a1_output_cache_state["final_size"] = len(
        _load_a1_output_cache(a1_output_cache_path)
    )

    metadata = {
        "tag": PRE_REGISTRATION_TAG,
        "seed": SEED,
        "N": len(samples),
        "timestamp": _utc_now_iso(),
        "git_commit": git_short,
        "git_commit_full": git_full,
        "dataset": DATASET_ID,
        "sample_file": str(sample_file.resolve()),
        "python_version": platform.python_version(),
        "elapsed_seconds": elapsed,
        "embedder_model": embedder_model,
        "metadata_prompt_template_hash": _hash_template(metadata_prompt_template),
        "metadata_prompt_template_used": metadata_prompt_template is not None,
        "a1_output_cache_path": (
            str(a1_output_cache_path.resolve())
            if a1_output_cache_path is not None
            else None
        ),
        "a1_output_cache_state": a1_output_cache_state,
    }

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
    """Build default output path with current timestamp."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"eval_pico_{stamp}.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluador Hero A1 PICO — caracteriza la extracción de "
            "componentes PICO de Agent1Protocol vs LEADSInstruct."
        ),
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Ruta al JSON de samples LEADSInstruct (default: leads_smoke.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del JSON de salida (default: outputs/eval/T17.A3/eval_pico_*.json).",
    )
    parser.add_argument(
        "--a1-output-cache",
        type=Path,
        default=None,
        help=(
            "Ruta JSON de caché append-only para outputs de A1 (protocolo) "
            "por sample_id + hash del template. Permite pares pre/post "
            "deterministas reutilizando exactamente la misma salida A1."
        ),
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=DEFAULT_EMBEDDER_MODEL,
        help=f"Modelo SentenceTransformer (default: {DEFAULT_EMBEDDER_MODEL}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar a N samples para smoke (default: todos).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sample_file = args.sample_file.resolve()
    output_path = (args.output or _default_output_path()).resolve()
    a1_output_cache_path = (
        args.a1_output_cache.resolve() if args.a1_output_cache is not None else None
    )

    # Honour --limit by writing a temp file with truncated samples list.
    if args.limit is not None and args.limit > 0:
        try:
            with open(sample_file, encoding="utf-8") as f:
                doc = json.load(f)
            doc["samples"] = (doc.get("samples") or [])[: args.limit]
            tmp_sample = sample_file.parent / f".{sample_file.stem}.smoke{args.limit}.json"
            tmp_sample.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            sample_file = tmp_sample
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: no se pudo aplicar --limit: {exc}", file=sys.stderr)
            return 2

    try:
        run_eval(
            sample_file=sample_file,
            output=output_path,
            a1_output_cache=a1_output_cache_path,
            embedder_model=args.embedder_model,
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
