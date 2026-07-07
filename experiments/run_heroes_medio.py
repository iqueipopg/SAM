# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Orchestrator -- run the 4 hero evaluators sequentially on the medio sample.

Sprint:  T16.E.1
Adapted from: experiments/run_heroes_micro.py

Differences vs micro orchestrator
---------------------------------
1. The medio config (config/eval/medio.yaml) declares one sample PER hero
   (Hero 1 = sample_medio_a3.csv, Hero 2 = sample_medio_a4.csv, etc.). This
   orchestrator serialises a per-hero temp YAML before invoking each eval
   function (which expects a single `sample` field).
2. Execution order: hero4 (fastest, ~33min) -> hero2 (~1.5h) -> hero1 (~4h)
   -> hero3 (~3.7h). Failing early on the cheapest hero is the cost-aware
   choice for an overnight ~9.7h run.
3. The VRAM-sharing strategy (single Agent3Screening that loads the model;
   refs injected into A4/A5/A6) is preserved verbatim from the micro
   orchestrator.

The micro orchestrator's summary / markdown rendering code is reused with
``scale=medio`` rather than duplicated; this orchestrator imports its helpers
directly from run_heroes_micro.

Output JSONs go to outputs/eval/medio/. Idempotent caches (e.g.
``_a3_full_outputs.json``) are written by the evaluators themselves.

CLI
---
    ~/meta-analisis-automatizado/venv/bin/python \\
        experiments/run_heroes_medio.py
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

# Ensure project root is on sys.path so `agents` and `experiments` packages
# resolve when launching as `python experiments/run_heroes_medio.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._eval_common import (  # noqa: E402
    gpu_info as _gpu_info,
    load_eval_config,
    seed_torch as _seed_torch,
)
from experiments.run_heroes_micro import (  # noqa: E402
    _build_shared_agents,
    _build_summary_json,
    _build_summary_md,
    _hero_registry,
    _release_shared_agents,
    _run_single_hero,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "eval" / "medio.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "medio"
ALL_HEROES = ("hero1", "hero2", "hero3", "hero4")

# Cost-aware order: shortest hero first so we surface failures before the
# expensive runs. Estimated times (from micro extrapolation):
#   hero4: ~33 min (62 papers x 30s)
#   hero2: ~1.5h  (566 papers x 9.4s)
#   hero1: ~4h    (2000 papers x 7s)
#   hero3: ~3.7h  (64 papers x 200s)
DEFAULT_ORDER = ("hero4", "hero2", "hero1", "hero3")


# ──────────────────────────────────────────────────────────────────────────────
# Per-hero temp config materialisation
# ──────────────────────────────────────────────────────────────────────────────

def _materialise_per_hero_config(
    master_cfg: dict,
    hero_id: str,
    temp_dir: Path,
) -> Path:
    """Write a temporary single-hero YAML compatible with eval_heroX scripts.

    The eval scripts call ``load_eval_config`` and read ``cfg["sample"]`` as
    the single sample path. medio.yaml carries one path per hero under
    ``samples[heroN]``; we flatten that into a per-hero file.
    """
    samples_map = master_cfg.get("samples") or {}
    sample_path = samples_map.get(hero_id)
    if not sample_path:
        raise KeyError(f"medio.yaml missing samples.{hero_id}")
    single = dict(master_cfg)  # shallow copy
    single.pop("samples", None)
    single["sample"] = sample_path
    single["heroes"] = [hero_id]
    out = temp_dir / f"medio_{hero_id}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(single, f, sort_keys=False, allow_unicode=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_heroes_arg(raw) -> list[str]:
    if not raw:
        return list(DEFAULT_ORDER)
    if isinstance(raw, list):
        items = [str(s).strip().lower() for s in raw if str(s).strip()]
    else:
        items = [s.strip().lower() for s in str(raw).split(",") if s.strip()]
    unknown = [h for h in items if h not in ALL_HEROES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Heroes desconocidos: {unknown}. Validos: {list(ALL_HEROES)}"
        )
    return items


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Orquestador T16.E.1 — corre hero4/2/1/3 (orden coste-asc) "
            "sobre los samples medio compartiendo el modelo en VRAM."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to medio YAML config (default: config/eval/medio.yaml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output dir (default: outputs/eval/medio/).",
    )
    p.add_argument(
        "--heroes",
        type=_parse_heroes_arg,
        default=list(DEFAULT_ORDER),
        help=(
            "Comma-separated subset of {hero1,hero2,hero3,hero4}. "
            "Default: hero4,hero2,hero1,hero3 (cost ascending)."
        ),
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()

    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2

    _seed_torch(42)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_cfg = load_eval_config(config_path)
    heroes_yaml = master_cfg.get("heroes") or list(ALL_HEROES)
    heroes_yaml = [str(h).strip().lower() for h in heroes_yaml]
    heroes_to_run = [h for h in args.heroes if h in heroes_yaml]
    if not heroes_to_run:
        heroes_to_run = heroes_yaml

    run_start_epoch = time.time()
    run_start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print()
    print("=" * 72)
    print("ORQUESTADOR T16.E.1 — run_heroes_medio")
    print("=" * 72)
    print(f"Config           : {config_path}")
    print(f"Output dir       : {output_dir}")
    print(f"Heroes a ejecutar (orden): {heroes_to_run}")
    print(f"GPU              : {_gpu_info()}")
    print(f"Timestamp UTC    : {run_start_iso}")

    # Temp dir for per-hero configs.
    temp_dir = Path(tempfile.mkdtemp(prefix="medio_cfgs_"))

    # ── Bootstrap shared agents (loads model once) ─────────────────────────
    agents: dict = {}
    bootstrap_failed = False
    try:
        agents = _build_shared_agents()
    except Exception as exc:
        bootstrap_failed = True
        tb = traceback.format_exc()
        print(f"[bootstrap] FALLO: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)

    registry = _hero_registry()
    heroes_records: list[dict] = []

    if bootstrap_failed:
        for hero_id in heroes_to_run:
            heroes_records.append({
                "hero_id": hero_id,
                "label": registry.get(hero_id, {}).get("label", hero_id),
                "status": "failed",
                "elapsed_seconds": 0.0,
                "result": None,
                "error": {
                    "error_type": "BootstrapError",
                    "message": "Shared model bootstrap failed; hero not executed.",
                    "traceback": "",
                },
            })
    else:
        for hero_id in heroes_to_run:
            info = registry.get(hero_id)
            if info is None:
                heroes_records.append({
                    "hero_id": hero_id,
                    "label": hero_id,
                    "status": "failed",
                    "elapsed_seconds": 0.0,
                    "result": None,
                    "error": {
                        "error_type": "UnknownHero",
                        "message": f"Hero '{hero_id}' not registered.",
                        "traceback": "",
                    },
                })
                continue
            try:
                per_hero_cfg = _materialise_per_hero_config(
                    master_cfg, hero_id, temp_dir
                )
            except Exception as exc:
                heroes_records.append({
                    "hero_id": hero_id,
                    "label": info["label"],
                    "status": "failed",
                    "elapsed_seconds": 0.0,
                    "result": None,
                    "error": {
                        "error_type": "ConfigError",
                        "message": str(exc),
                        "traceback": "",
                    },
                })
                continue
            record = _run_single_hero(
                hero_id=hero_id,
                label=info["label"],
                run_fn=info["run_fn"],
                config_path=per_hero_cfg,
                output_dir=output_dir,
                agent_instance=agents.get(hero_id),
            )
            heroes_records.append(record)

        print()
        print("[teardown] Liberando VRAM compartida...")
        _release_shared_agents(agents)

    elapsed_total = round(time.time() - run_start_epoch, 2)

    summary = _build_summary_json(
        run_start_iso=run_start_iso,
        elapsed_total=elapsed_total,
        config_path=config_path,
        output_dir=output_dir,
        heroes_records=heroes_records,
        requested_heroes=heroes_to_run,
        run_start_epoch=run_start_epoch,
    )
    summary_json_path = output_dir / f"summary_{stamp}.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    md_text = _build_summary_md(summary)
    summary_md_path = output_dir / f"summary_{stamp}.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # Cleanup temp configs.
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except OSError:
        pass

    print()
    print("=" * 72)
    print("RESUMEN DEL RUN")
    print("=" * 72)
    print(f"Duracion total : {elapsed_total}s")
    print(f"Summary JSON   : {summary_json_path}")
    print(f"Summary MD     : {summary_md_path}")
    print()
    for rec in heroes_records:
        print(
            f"  {rec['hero_id']}: status=`{rec['status']}` "
            f"({rec['elapsed_seconds']}s) -> "
            f"{summary['per_hero'].get(rec['hero_id'], {}).get('result_path', 'n/a')}"
        )
    if summary["errors"]:
        print()
        print(f"  {len(summary['errors'])} error(es) registrados - ver summary.md")
    print()

    if all(rec["status"] == "failed" for rec in heroes_records) and heroes_records:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
