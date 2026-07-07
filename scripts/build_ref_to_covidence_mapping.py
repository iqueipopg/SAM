#!/usr/bin/env python3
# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
build_ref_to_covidence_mapping.py — Mapping `ref_number` -> `covidence_id`.

Sprint T16.D.4.A: este mapping desbloquea Hero 4 (evaluador A6 RoB) — el dataset
`gold_rob.csv` indexa los estudios por `ref_number` (1..73) y necesita un cruce
para obtener el `covidence_id` correspondiente. La fuente del cruce es el xlsx
`Resumen_Total_PDFs.xlsx`, que vincula filenames con covidence_id para los 66
estudios marcados como `Incluido`.

IMPORTANTE: este mapping resuelve la deuda de Bloque C exclusivamente para
`gold_rob` (que indexa por ref_number); `gold_metrics.csv` ya tiene `covidence_id`
nativo y no requiere mapping. Hero 3 (A5/extraction) NO necesita este mapping.

Estrategia:
  1. Cargar gold_rob.csv y deduplicar por (study_ref, ref_number). Expandir los
     multi-refs (`[3-5]`, `[12,13]`) a ref_numbers individuales.
  2. Cargar Resumen_Total_PDFs.xlsx, filtrar `Clasificacion == "Incluido"`, y
     parsear los filenames descargados para extraer (surname, year, title).
     Un mismo covidence_id puede tener varios filenames; agregamos todos los
     candidates (surname, year, title) como representantes del estudio.
  3. Para cada ref_number, hacer match (surname, year) con tiebreaker por
     titulo cuando hay ambiguedad (Keyserling 2002 vs 2008, Bennett 2007 vs
     2008). Confidence: high/medium/low/unresolved.
  4. Emitir una fila por ref_number (1..73). Refs ausentes en gold_rob
     (presumiblemente ref=52) -> fila unresolved con notes apropiada.

Identifiers y comentarios en ingles. Mensajes/print/docstrings de usuario en
espanol (regla #12).
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_ROB_CSV = PROJECT_ROOT / "data" / "gold_standard_normalized" / "gold_rob.csv"
XLSX_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "Resumen_Total_PDFs.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "unified"
OUTPUT_CSV = OUTPUT_DIR / "ref_to_covidence_mapping.csv"
OUTPUT_REPORT = OUTPUT_DIR / "ref_mapping_report.md"

# ref_number domain (per spec, refs go 1..73).
REF_DOMAIN_MIN = 1
REF_DOMAIN_MAX = 73

# Final column order for the persisted CSV.
OUTPUT_COLUMNS = [
    "ref_number",
    "covidence_id",
    "confidence",
    "study_label",
    "match_method",
    "notes",
]

# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------
# study_ref examples that must parse:
#   "Hartman et al. (2016) [1]"
#   "Thomsen et al. (2017) [3-5]" or "[3–5]"
#   "Balducci et al. (2019) [12,13]"
#   "Liu & Lachman (2020) [18]"          (no "et al.", & conjunction)
#   "Liu Y y Lachman ME (2020) [18]"     (Spanish "y" conjunction)
#   "De Greef et al. (2010) [23]"        (compound surname with space)
#   "Di Loreto et al. (2003) [28]"
#   "Van Keulen et al. (2011) [49,50]"
#   "Da Silva et al. (2015) [69]"
# The surname capture supports an optional second word (e.g. "De Greef")
# that starts with a Latin letter (upper or lower case to also support
# the xlsx variant "van Keulen", "da Silva").
# A surname token may be a normal name word ("Hartman", "Greef") or a
# common compound prefix ("De", "Di", "Van", "Da", "von", "van"). To avoid
# accidentally swallowing "et" (as in "et al."), the second word of a
# compound surname must be the prefix coming FIRST. We model this by
# accepting an OPTIONAL prefix word, followed by a mandatory primary word.
_SURNAME_PREFIX = r"(?:De|Di|Van|van|Da|da|von|Von|Le|La|du|Du|del|Del|al|Al)"
_SURNAME_WORD = r"[A-Za-zÀ-ſ][A-Za-zÀ-ſ'\-]+"
_SURNAME_GROUP = r"(?:" + _SURNAME_PREFIX + r"\s+)?" + _SURNAME_WORD
STUDY_REF_RE = re.compile(
    r"^(?P<surname>" + _SURNAME_GROUP + r")"
    r"(?:\s*(?:&|y|et)\s+" + _SURNAME_GROUP + r")?"
    r"\s+(?:et al\.\s+)?"
    r"\((?P<year>\d{4})\)\s*\[(?P<refs>[^\]]+)\]$"
)

# xlsx filename examples that must parse (Incluido subset):
#   "Hartman SJ et al. - 2016 - Technology- and Phone-Based Weight Loss Interventi_94.pdf"
#   "Hartman-2016-Technology- and Phone-Based Weigh_95.pdf"
#   "Bennell-2020-Behavior-change-text-messages-for-h_14.pdf"
#   "Costanzo C y Walker SN - 2008 - ..."  (Spanish "y" conjunction)
#   "Mouton y Cloes - 2014 - ..."
#   "Liu Y y Lachman ME - 2020 - ..."
# Many irrelevant filenames also exist for the same covidence_id ("PROTOCOLO ...",
# "APPENDIX 1_23.pdf", "intervención costanzo_43.pdf"). These will simply fail
# to parse and we will not register them as candidates - that is fine because
# the covidence_id will still have at least ONE parseable filename in practice.
FILENAME_STD_RE = re.compile(
    # "Surname Initials et al. - YYYY - Title.pdf"  (standard format)
    # Also matches "De Greef KP et al. - YYYY - Title.pdf"
    r"^(?P<surname>" + _SURNAME_GROUP + r")"
    r"(?:\s+[A-Z]{1,5})?"
    r"\s+et al\.\s*-\s*(?P<year>\d{4})\s*-\s*(?P<title>.+?)(?:_\d+)?\.pdf$"
)
FILENAME_HYPH_RE = re.compile(
    # "Surname-YYYY-Title.pdf"  (hyphenated, no et al.)
    # Surname must be a single word here (compound prefixes like "De Greef"
    # are written with a space and use the SPACED form below).
    r"^(?P<surname>" + _SURNAME_WORD + r")"
    r"\s*-\s*(?P<year>\d{4})\s*-\s*(?P<title>.+?)(?:_\d+)?\.pdf$"
)
FILENAME_HYPH_SPACED_RE = re.compile(
    # "De Greef-2010-Title.pdf" / "van Keulen-2021-Title.pdf"
    # Compound surname with space, hyphen separator to the year.
    r"^(?P<surname>" + _SURNAME_PREFIX + r"\s+" + _SURNAME_WORD + r")"
    r"\s*-\s*(?P<year>\d{4})\s*-\s*(?P<title>.+?)(?:_\d+)?\.pdf$"
)
FILENAME_Y_RE = re.compile(
    # Spanish: "Surname Initials y Surname2 Initials2 - YYYY - Title.pdf"
    r"^(?P<surname>" + _SURNAME_GROUP + r")"
    r"(?:\s+[A-Z]{1,5})?"
    r"\s+y\s+" + _SURNAME_GROUP
    + r"(?:\s+[A-Z]{1,5})?"
    r"\s*-\s*(?P<year>\d{4})\s*-\s*(?P<title>.+?)(?:_\d+)?\.pdf$"
)
FILENAME_ROBIN_RE = re.compile(
    # "Robin et al. 2019 (JAPA) (2)_218.pdf" - no hyphen between author and year,
    # title is just journal abbreviation or empty.
    r"^(?P<surname>" + _SURNAME_GROUP + r")"
    r"\s+et al\.\s+(?P<year>\d{4})\s*(?P<title>.*?)(?:_\d+)?\.pdf$"
)
FILENAME_SURNAME_ONLY_RE = re.compile(
    # Fallback: "Scott-IntegratedMotivationalInterviewing(AM)_229.pdf" -
    # surname followed by hyphen and arbitrary blob (no year present).
    # Year defaults to 0 so the matcher routes through the surname-only branch.
    r"^(?P<surname>" + _SURNAME_WORD + r")-(?!\d{4})(?P<title>.+?)(?:_\d+)?\.pdf$"
)


# ---------------------------------------------------------------------------
# Pure helper functions (importable from tests)
# ---------------------------------------------------------------------------
class ParsedStudyRef(NamedTuple):
    """Parsed representation of a `study_ref` row."""

    surname: str
    year: int
    ref_numbers: tuple[int, ...]
    label: str  # e.g. "Keyserling et al. (2002)" - the original study_ref minus [N].


class ParsedFilename(NamedTuple):
    """Parsed representation of an xlsx filename.

    `year` may be 0 when the filename has no year (e.g. surname-only
    fallback). The matcher treats year==0 as "year unknown".
    """

    surname: str
    year: int
    title: str


def _normalize_str(value: str) -> str:
    """Lowercase + strip accents + collapse whitespace for fuzzy comparison."""
    if value is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accents).strip().casefold()


def expand_multirefs(refs_token: str) -> tuple[int, ...]:
    """Expand a ref-token from the study_ref bracket into individual ints.

    Accepts ranges with either ASCII hyphen `-` or unicode en-dash `–`,
    and comma-separated lists. Examples:
        "1"         -> (1,)
        "12,13"     -> (12, 13)
        "3-5"       -> (3, 4, 5)
        "3–5"  -> (3, 4, 5)
        "46,47"     -> (46, 47)
    """
    token = refs_token.replace("–", "-").replace("—", "-").strip()
    out: list[int] = []
    for chunk in token.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start_n, end_n = int(start_s.strip()), int(end_s.strip())
            if end_n < start_n:
                start_n, end_n = end_n, start_n
            out.extend(range(start_n, end_n + 1))
        else:
            out.append(int(chunk))
    return tuple(out)


def parse_study_ref(study_ref: str) -> Optional[ParsedStudyRef]:
    """Parse a single `study_ref` string from gold_rob.csv."""
    if study_ref is None:
        return None
    text = str(study_ref).strip()
    # Normalize en-dash inside brackets to ASCII for the regex.
    text_norm = text.replace("–", "-")
    match = STUDY_REF_RE.match(text_norm)
    if not match:
        return None
    refs = expand_multirefs(match.group("refs"))
    # Build the human-readable label by stripping the bracket from the original.
    label = re.sub(r"\s*\[[^\]]+\]\s*$", "", text).strip()
    return ParsedStudyRef(
        surname=match.group("surname"),
        year=int(match.group("year")),
        ref_numbers=refs,
        label=label,
    )


def parse_filename(filename: str) -> Optional[ParsedFilename]:
    """Parse an xlsx `Nombre del Archivo Descargado` string.

    Returns None for unparseable filenames (e.g. "APPENDIX 1_23.pdf",
    "PROTOCOLO ...", "INTERVENCION CARDIACTION_3.pdf"). The caller treats
    those as non-candidates - they are auxiliary files attached to the same
    covidence_id and we rely on the other (parseable) filename(s) of the
    same study to register the candidate.
    """
    if filename is None:
        return None
    text = str(filename).strip()
    # Order matters: Y (Spanish "y" conjunction) must be tried before STD
    # because "Costanzo C y Walker SN" contains "C" which STD would consume
    # as initials, leaving " y Walker SN - YYYY - ..." which then fails.
    # HYPH_SPACED before HYPH so "De Greef-2010" wins over a hypothetical
    # single-word match. ROBIN_RE last because it is permissive on title.
    for pattern in (
        FILENAME_Y_RE,
        FILENAME_STD_RE,
        FILENAME_HYPH_SPACED_RE,
        FILENAME_HYPH_RE,
        FILENAME_ROBIN_RE,
    ):
        m = pattern.match(text)
        if m:
            return ParsedFilename(
                surname=m.group("surname"),
                year=int(m.group("year")),
                title=m.group("title").strip(),
            )
    # Surname-only fallback (e.g. "Scott-IntegratedMotivationalInterviewing(AM)_229.pdf").
    # year=0 signals "year unknown" to the matcher.
    m = FILENAME_SURNAME_ONLY_RE.match(text)
    if m:
        return ParsedFilename(
            surname=m.group("surname"),
            year=0,
            title=m.group("title").strip(),
        )
    return None


def build_xlsx_candidates(
    xlsx_df: pd.DataFrame,
) -> tuple[dict[str, list[ParsedFilename]], list[str]]:
    """Group parsed filenames by covidence_id for the `Incluido` subset.

    Returns: (candidates, fully_unparseable_covs)
      - candidates: dict[covidence_id, list[ParsedFilename]] - only covs
        that have at least one parseable filename.
      - fully_unparseable_covs: list of cov ids whose every filename failed
        to parse (auxiliary-only filenames). These are invisible to the
        matcher and must be flagged.
    """
    incluido = xlsx_df[xlsx_df["Clasificación"] == "Incluido"].copy()
    groups: dict[str, list[ParsedFilename]] = defaultdict(list)
    unparseable_per_cov: dict[str, int] = defaultdict(int)
    for _, row in incluido.iterrows():
        cov = row["ID del Estudio (Covidence)"]
        filename = row["Nombre del Archivo Descargado"]
        parsed = parse_filename(filename)
        if parsed is None:
            unparseable_per_cov[str(cov)] += 1
            continue
        groups[str(cov)].append(parsed)
    fully_unparseable = sorted(
        cov for cov in unparseable_per_cov if cov not in groups
    )
    if fully_unparseable:
        print(
            "[ADVERTENCIA] covidence_ids sin ningun filename parseable: "
            + ", ".join(fully_unparseable)
        )
    return dict(groups), fully_unparseable


def _label_tokens(label: str) -> set[str]:
    """Tokenize a study label or title for tiebreaker comparison."""
    norm = _normalize_str(label)
    return {tok for tok in re.split(r"[^a-z0-9]+", norm) if len(tok) >= 3}


def find_match(
    parsed_ref: ParsedStudyRef,
    rob_outcome_hint: str,
    xlsx_candidates: dict[str, list[ParsedFilename]],
) -> tuple[Optional[str], str, str, Optional[str]]:
    """Match a parsed gold_rob study to a covidence_id.

    Returns: (covidence_id, confidence, match_method, notes).
    confidence in {"high", "medium", "low", "unresolved"}.
    """
    surname_key = _normalize_str(parsed_ref.surname)
    # Bucket candidates by (surname_key, year) and by surname_key alone.
    exact_hits: dict[str, list[ParsedFilename]] = defaultdict(list)
    pm1_hits: dict[str, list[ParsedFilename]] = defaultdict(list)
    surname_hits: dict[str, list[ParsedFilename]] = defaultdict(list)
    for cov, files in xlsx_candidates.items():
        for pf in files:
            if _normalize_str(pf.surname) != surname_key:
                continue
            surname_hits[cov].append(pf)
            # year==0 means "year unknown" (surname-only fallback) - it never
            # qualifies as exact or +/-1 match.
            if pf.year == 0:
                continue
            if pf.year == parsed_ref.year:
                exact_hits[cov].append(pf)
            elif abs(pf.year - parsed_ref.year) == 1:
                pm1_hits[cov].append(pf)

    # ---- exact author+year ----
    if exact_hits:
        if len(exact_hits) == 1:
            cov = next(iter(exact_hits))
            return cov, "high", "exact_author_year", None
        # Multiple covidence_ids share (surname, year) -> tiebreaker by title.
        # We compare gold_rob "label + outcome hint" tokens vs xlsx title tokens.
        ref_tokens = _label_tokens(parsed_ref.label) | _label_tokens(rob_outcome_hint)
        scored: list[tuple[str, int]] = []
        for cov, files in exact_hits.items():
            best = 0
            for pf in files:
                overlap = len(ref_tokens & _label_tokens(pf.title))
                if overlap > best:
                    best = overlap
            scored.append((cov, best))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        if scored[0][1] > 0 and (len(scored) == 1 or scored[0][1] > scored[1][1]):
            return (
                scored[0][0],
                "high",
                "exact_author_year_title_tiebreaker",
                "tiebreaker by title token overlap",
            )
        # Tiebreaker inconclusive -> low confidence pick of the first.
        cov_list = sorted(exact_hits.keys())
        return (
            cov_list[0],
            "low",
            "exact_author_year_ambiguous",
            f"ambiguous tiebreaker among {cov_list}",
        )

    # ---- author + year +/- 1 ----
    if pm1_hits:
        if len(pm1_hits) == 1:
            cov = next(iter(pm1_hits))
            return cov, "medium", "author_year_pm1", "year mismatch within +/-1"
        # Ambiguous medium -> low.
        cov_list = sorted(pm1_hits.keys())
        return (
            cov_list[0],
            "low",
            "author_year_pm1_ambiguous",
            f"ambiguous +/-1 match among {cov_list}",
        )

    # ---- surname-only ----
    if surname_hits:
        cov_list = sorted(surname_hits.keys())
        return (
            cov_list[0] if len(cov_list) == 1 else cov_list[0],
            "low",
            "surname_only",
            f"surname-only match, year mismatch (candidates: {cov_list})",
        )

    return None, "unresolved", "no_match", "no surname match in xlsx Incluido subset"


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------
def load_gold_rob(path: Path = GOLD_ROB_CSV) -> pd.DataFrame:
    """Load gold_rob.csv with the columns needed for the mapping."""
    df = pd.read_csv(path)
    needed = {"study_ref", "ref_number", "outcome"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"gold_rob.csv le faltan columnas: {sorted(missing)}")
    return df


def load_xlsx(path: Path = XLSX_PATH) -> pd.DataFrame:
    """Load the deporte xlsx (read-only)."""
    return pd.read_excel(path)


def build_mapping(
    rob_df: pd.DataFrame, xlsx_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the ref_number -> covidence_id mapping DataFrame.

    Returns: (mapping_df, stats) where stats holds counts and unresolved
    entries for downstream reporting.
    """
    # 1. Deduplicate gold_rob rows by (study_ref, ref_number).
    rob_unique = rob_df.drop_duplicates(subset=["study_ref", "ref_number"]).copy()
    # 2. Build xlsx candidates grouped by covidence_id.
    xlsx_candidates, fully_unparseable_covs = build_xlsx_candidates(xlsx_df)

    # 3. Iterate over the dedup'd gold_rob rows, parse, and match.
    rows: list[dict[str, object]] = []
    seen_refs: set[int] = set()
    parse_failures: list[str] = []
    for _, r in rob_unique.iterrows():
        parsed = parse_study_ref(r["study_ref"])
        if parsed is None:
            parse_failures.append(str(r["study_ref"]))
            continue
        outcome_hint = "" if pd.isna(r.get("outcome")) else str(r.get("outcome", ""))
        cov, conf, method, note = find_match(parsed, outcome_hint, xlsx_candidates)
        for rn in parsed.ref_numbers:
            rows.append(
                {
                    "ref_number": int(rn),
                    "covidence_id": cov,
                    "confidence": conf,
                    "study_label": parsed.label,
                    "match_method": method,
                    "notes": note,
                }
            )
            seen_refs.add(int(rn))

    # 4. Fill in missing refs in the domain [1..73] as unresolved.
    for rn in range(REF_DOMAIN_MIN, REF_DOMAIN_MAX + 1):
        if rn in seen_refs:
            continue
        rows.append(
            {
                "ref_number": rn,
                "covidence_id": None,
                "confidence": "unresolved",
                "study_label": "",
                "match_method": "missing_in_gold_rob",
                "notes": f"ref={rn}: ausente en gold_rob",
            }
        )

    mapping_df = (
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        .sort_values("ref_number")
        .reset_index(drop=True)
    )

    # 5. Stats for the report.
    confidence_counts = mapping_df["confidence"].value_counts().to_dict()
    method_counts = mapping_df["match_method"].value_counts().to_dict()
    unresolved = mapping_df[mapping_df["confidence"] == "unresolved"][
        ["ref_number", "study_label", "match_method", "notes"]
    ].to_dict("records")
    # Coverage: count distinct covidence_ids actually matched (high|medium).
    matched_high_med = (
        mapping_df.loc[
            mapping_df["confidence"].isin(["high", "medium"]),
            "covidence_id",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    all_includes = sorted(
        xlsx_df.loc[
            xlsx_df["Clasificación"] == "Incluido",
            "ID del Estudio (Covidence)",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    missing_includes = sorted(set(all_includes) - set(matched_high_med))

    # Set of covs matched only with low confidence (informational).
    low_confidence_covs = set(
        mapping_df.loc[mapping_df["confidence"] == "low", "covidence_id"]
        .dropna()
        .unique()
        .tolist()
    )

    stats: dict[str, object] = {
        "confidence_counts": confidence_counts,
        "method_counts": method_counts,
        "unresolved": unresolved,
        "parse_failures": parse_failures,
        "n_includes_total": len(all_includes),
        "n_includes_matched": len(set(matched_high_med)),
        "missing_includes": missing_includes,
        "low_confidence_covs": low_confidence_covs,
        "unparseable_covs": set(fully_unparseable_covs),
    }
    return mapping_df, stats


# ---------------------------------------------------------------------------
# Report (Spanish)
# ---------------------------------------------------------------------------
# Arm-level labels found in gold_metrics.csv that are NOT real covidence_ids.
# They represent treatment arms inside multi-arm studies (per spec).
GOLD_METRICS_ARM_LABELS = [
    "#B10657",
    "#1354RCT 2",
    "GC (Implementation intentions)",
    "E2",
    "E3",
    "E4",
    "E2 print+call",
    "E2 (center)",
    "E3 (mixed)",
    "E2 (print)",
]


def render_report(
    mapping_df: pd.DataFrame, stats: dict[str, object], xlsx_df: pd.DataFrame
) -> str:
    """Render the markdown report in Spanish."""
    lines: list[str] = []
    a = lines.append

    a("# Mapping `ref_number` -> `covidence_id` (deporte)")
    a("")
    a("Sprint T16.D.4.A. Generado por `scripts/build_ref_to_covidence_mapping.py`.")
    a("")
    # --- 1. Propósito
    a("## 1. Proposito")
    a("")
    a(
        "Este mapping resuelve la deuda de Bloque C **exclusivamente** para "
        "`gold_rob` (que indexa por `ref_number`); `gold_metrics` ya tiene "
        "`covidence_id` nativo y **no requiere** mapping."
    )
    a("")
    a(
        "- Hero 3 (A5 / extraction) usa `gold_metrics.csv` -> covidence_id "
        "nativo. **No depende de este mapping**."
    )
    a(
        "- Hero 4 (A6 / RoB) usa `gold_rob.csv` -> indexa por `ref_number`. "
        "**Depende de este mapping** para identificar el estudio."
    )
    a("")
    # --- 2. Resumen numerico
    a("## 2. Resumen numerico")
    a("")
    cc = stats["confidence_counts"]
    a("| Confidence | Count |")
    a("|------------|------:|")
    for level in ("high", "medium", "low", "unresolved"):
        a(f"| {level} | {int(cc.get(level, 0))} |")
    a(f"| **total** | **{len(mapping_df)}** |")
    a("")
    n_total = int(stats["n_includes_total"])
    n_matched = int(stats["n_includes_matched"])
    pct = (n_matched / n_total * 100) if n_total else 0.0
    a(
        f"Cobertura de incluidos: **{n_matched} / {n_total}** "
        f"({pct:.1f}%) covidence_ids con clasificacion `Incluido` en el "
        "xlsx aparecen al menos una vez en el mapping con confidence "
        "high o medium."
    )
    a("")
    if stats["missing_includes"]:
        a(
            "covidence_ids `Incluido` que **NO** aparecen en el mapping con "
            "confidence high/medium (a revisar por reviewer):"
        )
        a("")
        for cov in stats["missing_includes"]:
            note = ""
            if cov in stats.get("low_confidence_covs", set()):
                note = "  (matched con confidence=`low`, surname-only)"
            elif cov in stats.get("unparseable_covs", set()):
                note = "  (xlsx solo tiene filenames no parseables; sin contraparte en gold_rob)"
            a(f"- `{cov}`{note}")
        a("")
    else:
        a("Todos los covidence_ids `Incluido` quedan cubiertos.")
        a("")

    # --- 3. Tabla completa
    a("## 3. Tabla completa (73 filas)")
    a("")
    a("| ref_number | covidence_id | confidence | study_label | match_method |")
    a("|-----------:|--------------|------------|-------------|--------------|")
    for _, r in mapping_df.iterrows():
        cov = r["covidence_id"] if r["covidence_id"] else "—"
        label = r["study_label"] if r["study_label"] else "—"
        a(
            f"| {int(r['ref_number'])} | {cov} | {r['confidence']} | "
            f"{label} | {r['match_method']} |"
        )
    a("")

    # --- 4. Anomalias en gold_metrics
    a("## 4. Anomalias estructurales en `gold_metrics` (informativo)")
    a("")
    a(
        "El dataset `gold_metrics.csv` contiene 76 valores unicos en la "
        "columna `covidence_id`: 66 son IDs validos con formato `#NNN` "
        "(coinciden con los `Incluido` del xlsx) y **10 son etiquetas "
        "de brazo** dentro de estudios multi-arm, no covidence_ids "
        "adicionales:"
    )
    a("")
    for lbl in GOLD_METRICS_ARM_LABELS:
        a(f"- `{lbl}`")
    a("")
    a(
        "Estas etiquetas son **brazos** o **controles** dentro de estudios "
        "multi-arm. El evaluador de Hero 3 debera colapsarlas a nivel de "
        "estudio (T16.D.5). NO afectan al mapping de este informe."
    )
    a("")

    # --- 5. Discrepancias
    a("## 5. Discrepancias entre fuentes (informativo)")
    a("")
    a(
        "- **65** study_refs unicos en `gold_rob.csv` (tras deduplicar por "
        "(study_ref, ref_number))."
    )
    a(
        "- **72** ref_numbers individuales tras expandir multi-refs (`[3-5]`, "
        "`[12,13]`, etc.) - falta **1** en el rango `1..73` "
        "(ref=52, ausente en gold_rob)."
    )
    a(
        "- **66** covidence_ids unicos con clasificacion `Incluido` en el "
        "xlsx (== los 66 IDs validos `#NNN` de `gold_metrics`)."
    )
    a(
        "- **67** estudios reportados en el paper publicado de Paula."
    )
    a("")
    a(
        "Esta cadena `65 -> 72 -> 66 -> 67` es **informativa**, no afecta a "
        "la correccion del mapping. Las diferencias reflejan: (a) multi-refs "
        "comprimidos en gold_rob, (b) ref=52 ausente en gold_rob, "
        "(c) un estudio adicional en el paper publicado respecto al xlsx."
    )
    a("")

    # --- 6. Unresolved con razon
    a("## 6. Unresolved (con razon)")
    a("")
    if stats["unresolved"]:
        a("| ref_number | study_label | match_method | notes |")
        a("|-----------:|-------------|--------------|-------|")
        for u in stats["unresolved"]:
            lbl = u["study_label"] if u["study_label"] else "—"
            notes = u["notes"] if u["notes"] else "—"
            a(
                f"| {int(u['ref_number'])} | {lbl} | "
                f"{u['match_method']} | {notes} |"
            )
    else:
        a("Sin filas unresolved.")
    a("")

    # --- 7. Distribucion de match_method
    a("## 7. Distribucion de `match_method`")
    a("")
    a("| match_method | count |")
    a("|--------------|------:|")
    for method, count in sorted(
        stats["method_counts"].items(), key=lambda kv: (-int(kv[1]), kv[0])
    ):
        a(f"| {method} | {int(count)} |")
    a("")

    # --- 8. Diagnostico (si aplica)
    if stats["parse_failures"]:
        a("## 8. Filas de `gold_rob` cuyo `study_ref` no se pudo parsear")
        a("")
        for sr in stats["parse_failures"]:
            a(f"- `{sr}`")
        a("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye el mapping ref_number -> covidence_id para gold_rob "
            "(deporte). Salida: ref_to_covidence_mapping.csv + ref_mapping_report.md"
        )
    )
    parser.add_argument(
        "--gold-rob",
        default=str(GOLD_ROB_CSV),
        help="Ruta al gold_rob.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--xlsx",
        default=str(XLSX_PATH),
        help="Ruta al Resumen_Total_PDFs.xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "--out-csv",
        default=str(OUTPUT_CSV),
        help="Ruta del CSV de salida (default: %(default)s)",
    )
    parser.add_argument(
        "--out-report",
        default=str(OUTPUT_REPORT),
        help="Ruta del report markdown (default: %(default)s)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    gold_rob_path = Path(args.gold_rob)
    xlsx_path = Path(args.xlsx)
    out_csv_path = Path(args.out_csv)
    out_report_path = Path(args.out_report)

    print(f"[INFO] Cargando gold_rob: {gold_rob_path}")
    rob_df = load_gold_rob(gold_rob_path)
    print(f"[INFO]   gold_rob shape: {rob_df.shape}")

    print(f"[INFO] Cargando xlsx: {xlsx_path}")
    xlsx_df = load_xlsx(xlsx_path)
    print(f"[INFO]   xlsx shape:    {xlsx_df.shape}")

    print("[INFO] Construyendo mapping...")
    mapping_df, stats = build_mapping(rob_df, xlsx_df)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(out_csv_path, index=False)
    print(f"[INFO] CSV escrito: {out_csv_path}  ({len(mapping_df)} filas)")

    report_text = render_report(mapping_df, stats, xlsx_df)
    out_report_path.write_text(report_text, encoding="utf-8")
    print(f"[INFO] Report escrito: {out_report_path}")

    # Final summary print.
    cc = stats["confidence_counts"]
    print()
    print("=== RESUMEN ===")
    print(f"  high       : {int(cc.get('high', 0))}")
    print(f"  medium     : {int(cc.get('medium', 0))}")
    print(f"  low        : {int(cc.get('low', 0))}")
    print(f"  unresolved : {int(cc.get('unresolved', 0))}")
    print(f"  total      : {len(mapping_df)}")
    print()
    print(
        f"  Incluidos cubiertos (high|medium): "
        f"{int(stats['n_includes_matched'])} / {int(stats['n_includes_total'])}"
    )
    if stats["missing_includes"]:
        print("  Incluidos NO cubiertos:")
        for cov in stats["missing_includes"]:
            print(f"    - {cov}")
    if stats["unresolved"]:
        print()
        print("  Unresolved:")
        for u in stats["unresolved"]:
            print(
                f"    - ref={int(u['ref_number'])}: "
                f"{u['match_method']} ({u['notes']})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
