# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for hero evaluators (T16.D.5).

Used by:
  - experiments/eval_hero1_screening.py  (A3)
  - experiments/eval_hero2_fulltext.py   (A4)
  - experiments/eval_hero3_extraction.py (A5)  [planned]
  - experiments/eval_hero4_quality.py    (A6)  [planned]

Helpers exposed here are intentionally small and side-effect free (except for
torch seeding). They avoid duplicating boilerplate across hero evaluators:

  * `load_eval_config()`     -- parse config/eval/*.yaml
  * `load_agent_params()`    -- merge defaults + agent block from config/agents.json
  * `gpu_info()`             -- short GPU description string
  * `seed_torch()`           -- best-effort deterministic seeding
  * `truncate()`             -- short-string formatter for human reports
  * `PAULA_SPORT_PROTOCOL`   -- fixed PICO protocol for Paula's sport MA

This module does NOT import heavy ML libraries at top level. `torch` is
imported lazily inside `gpu_info` / `seed_torch` so the evaluators can be
loaded for `--help` without booting CUDA.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_JSON_PATH = PROJECT_ROOT / "config" / "agents.json"

# Fixed PICO protocol for Paula's MA (sport / community-dwelling older adults).
# Used by every hero evaluator (Agent3 / Agent4 / Agent6) when an external
# protocol argument is required. Kept here to guarantee a single source of
# truth — divergence between heroes would invalidate cross-hero comparisons.
#
# DEPRECATED 2026-05-20 (T17.B.2). Conservado para trazabilidad histórica.
# Confound documentado en docs/iterations/T17.B.1_protocol_audit.md. Usar
# PAULA_SPORT_PROTOCOL_V2 (default).
PAULA_SPORT_PROTOCOL_V1 = {
    "framework": "PICO",
    "research_question": (
        "Do physical activity interventions that incorporate psychological "
        "components improve PA outcomes in community-dwelling adults aged 50-70?"
    ),
    "population": (
        "Community-dwelling adults aged 50-70 years (retirement-age, "
        "no major comorbidities, living independently)"
    ),
    "intervention": (
        "Physical activity interventions incorporating psychological components "
        "(motivational interviewing, goal-setting, self-monitoring, "
        "cognitive-behavioural strategies, theory-based behaviour change)"
    ),
    "comparator": (
        "Usual care, waitlist, attention control, or PA intervention without "
        "psychological components"
    ),
    "outcome": (
        "Physical activity outcomes (objective: accelerometry, pedometer; "
        "self-report: IPAQ, GLTEQ, step counts)"
    ),
    "inclusion_criteria": [
        "Adults aged 50-70 years (community-dwelling, retirement-age)",
        "Randomised controlled trial (RCT) design",
        "Intervention incorporates psychological / behaviour-change component",
        "Physical activity reported as primary or secondary outcome",
        "Outcome measured objectively (accelerometer, pedometer) or by validated self-report",
    ],
    "exclusion_criteria": [
        "Population with major chronic disease or institutionalised",
        "Adults younger than 50 or older than 70 years",
        "No psychological / behaviour-change component in the intervention",
        "Physical activity not reported as an outcome",
        "Case reports, case series, editorials, reviews, meta-analyses",
        "Protocol papers without results",
    ],
}


# PAULA_SPORT_PROTOCOL_V2 — corrected against authoritative source.
# Source: Collazo-Castiñeira et al. 2025, Health Psychology Review,
# DOI 10.1080/17437199.2025.2492042, p.3 (Study eligibility → Population +
# Intervention). PROSPERO CRD42021277712.
# Divergence from V1 limited to `population` and `exclusion_criteria[0]`;
# `intervention` refreshed with literal quote. Other fields preserved from V1.
PAULA_SPORT_PROTOCOL_V2 = {
    "framework": "PICO",
    "research_question": (
        "Do physical activity interventions that incorporate psychological "
        "components improve PA outcomes in community-dwelling adults aged 50-70?"
    ),
    "population": (
        "Community-dwelling older adults around retirement age, defined as "
        "participants' mean age between 55 and 65 years, OR both mean age and "
        "standard deviations falling between 50 and 70 years."
    ),
    "intervention": (
        "Eligible interventions targeted the promotion of physical activity "
        "and included psychological components as active ingredients of the "
        "intervention. We defined psychological components as specific "
        "techniques and strategies intentionally designed and implemented "
        "within the intervention to promote behaviour change."
    ),
    "comparator": (
        "Usual care, waitlist, attention control, or PA intervention without "
        "psychological components"
    ),
    "outcome": (
        "Physical activity outcomes (objective: accelerometry, pedometer; "
        "self-report: IPAQ, GLTEQ, step counts)"
    ),
    "inclusion_criteria": [
        "Adults aged 50-70 years (community-dwelling, retirement-age)",
        "Randomised controlled trial (RCT) design",
        "Intervention incorporates psychological / behaviour-change component",
        "Physical activity reported as primary or secondary outcome",
        "Outcome measured objectively (accelerometer, pedometer) or by validated self-report",
    ],
    "exclusion_criteria": [
        (
            "Studies specifically targeting participants with serious mental or "
            "physical illnesses (e.g., COPD, active cancer). Studies targeting "
            "participants with common and prevalent chronic conditions such as "
            "diabetes or cardiac conditions ARE included."
        ),
        "Adults younger than 50 or older than 70 years",
        "No psychological / behaviour-change component in the intervention",
        "Physical activity not reported as an outcome",
        "Case reports, case series, editorials, reviews, meta-analyses",
        "Protocol papers without results",
    ],
}


# Backward-compatible alias: 15+ downstream consumers import the unversioned
# name. T17.B.2 onwards: alias points to V2 (corrected).
PAULA_SPORT_PROTOCOL = PAULA_SPORT_PROTOCOL_V2


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_eval_config(config_path: Path) -> dict:
    """Load eval YAML config (e.g. config/eval/micro.yaml). Returns dict (may be empty)."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_agent_params(agent_key: str) -> dict:
    """Load agent params from config/agents.json: defaults merged with `agents.<agent_key>`.

    Returns an empty dict if the file is missing or malformed (callers should
    treat absent params as soft fall-throughs and rely on their own defaults).
    """
    if not AGENTS_JSON_PATH.is_file():
        return {}
    try:
        with open(AGENTS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    defaults = data.get("defaults", {})
    agent_cfg = data.get("agents", {}).get(agent_key, {})
    merged = dict(defaults)
    merged.update(agent_cfg)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

def gpu_info() -> str:
    """Return a short GPU description string, or 'CPU-only' when no CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            try:
                _free_b, total_b = torch.cuda.mem_get_info(0)
                total_gb = round(total_b / (1024 ** 3), 1)
                return f"{name} {total_gb}GB"
            except Exception:
                return name
        return "CPU-only"
    except ImportError:
        return "torch unavailable"


def seed_torch(seed: int = 42) -> None:
    """Best-effort deterministic seeding. No-op if torch is unavailable."""
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ──────────────────────────────────────────────────────────────────────────────

def truncate(text: Optional[str], n: int) -> str:
    """Truncate `text` to at most n characters, appending '...' if shortened."""
    s = str(text or "")
    return s if len(s) <= n else s[: n - 3] + "..."


# ──────────────────────────────────────────────────────────────────────────────
# PDF text extraction (shared by hero2 / hero3 / hero4)
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path, min_chars: int = 500) -> tuple[Optional[str], Optional[str]]:
    """Extract text from a local PDF using the same backends Agent4 relies on.

    Uses PyMuPDF (``fitz``) as the primary backend and ``pdfplumber`` as a
    fallback when PyMuPDF yields too little text. This helper exists so the
    hero evaluators (hero2 full-text, hero3 extraction, hero4 quality) share a
    single, well-tested extraction routine instead of duplicating it.

    Args:
        pdf_path: absolute or relative path to a PDF file.
        min_chars: if PyMuPDF returns fewer than this many characters, pdfplumber
            is also attempted and the longer of the two outputs is kept.

    Returns:
        ``(text, error_message)``. On success, ``text`` is the extracted string
        and ``error_message`` is ``None``. On failure, ``text`` is ``None`` and
        ``error_message`` contains a short diagnostic describing both backends'
        outcomes.
    """
    if not pdf_path.is_file():
        return None, f"PDF file not found: {pdf_path.name}"

    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError as e:
        return None, f"PDF read failed ({type(e).__name__}): {e}"

    if not pdf_bytes:
        return None, "PDF file is empty (0 bytes)"

    text: Optional[str] = None
    pymupdf_err: Optional[str] = None
    pdfplumber_err: Optional[str] = None

    # Backend 1: PyMuPDF (fitz) — fast, handles most PDFs well
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            page_text = page.get_text()
            if page_text:
                pages_text.append(page_text)
        doc.close()
        if pages_text:
            text = "\n\n".join(pages_text)
    except Exception as e:
        pymupdf_err = f"{type(e).__name__}: {e}"

    # Backend 2: pdfplumber — better with complex layouts / tables
    if not text or len(text) < min_chars:
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                if pages_text:
                    plumber_text = "\n\n".join(pages_text)
                    if not text or len(plumber_text) > len(text):
                        text = plumber_text
        except Exception as e:
            pdfplumber_err = f"{type(e).__name__}: {e}"

    if text and text.strip():
        return text, None

    parts = []
    if pymupdf_err:
        parts.append(f"PyMuPDF: {pymupdf_err}")
    if pdfplumber_err:
        parts.append(f"pdfplumber: {pdfplumber_err}")
    if not parts:
        parts.append("both backends returned empty text")
    return None, "; ".join(parts)
