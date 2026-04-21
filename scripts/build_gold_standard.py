#!/usr/bin/env python3
"""
build_gold_standard.py — Genera data/gold_standard.json desde el CSV de la investigadora.

Lee el CSV del registro maestro (por defecto data/gold_standard/deporte/registro_maestro.csv)
y produce un JSON compatible con experiments/exp8_gold_standard_eval.py.

Schema JSON esperado (ver experiments/exp8_gold_standard_eval.py):
    {
        "query": "...",
        "protocol": { ... },
        "papers": [
            {
                "pmid": "12345678",
                "title": "...",
                "abstract": "...",
                "authors": [...],
                "year": ...,
                "journal": "...",
                "gold_decision": "INCLUDE" | "EXCLUDE",
                "gold_reason": "...",
                "stage_decided": "title_abstract" | "full_text"
            },
            ...
        ]
    }

Zona sagrada: NO sobrescribimos data/gold_standard.json. Escribimos a
data/gold_standard.json.new y dejamos que el usuario lo revise.

Formatos de CSV soportados:
  1. Registro maestro (Covidence) - columnas: "Covidence ID", "Cita (Autores)",
     "Archivos Físicos Extrapolados", "Aviso". Todas las filas se interpretan
     como INCLUDE (son los papers extraídos tras screening).
  2. Gold standard extendido - columnas mínimas: "id" (o "pmid"/"doi"), "title",
     "decision" (INCLUDE/EXCLUDE). Columnas opcionales: "abstract", "authors",
     "year", "journal", "reason", "stage".
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Identificadores en inglés, mensajes de logging en español (regla #12 del proyecto)
logger = logging.getLogger("build_gold_standard")

DEFAULT_CSV = Path("data/gold_standard/deporte/registro_maestro.csv")
DEFAULT_OUTPUT = Path("data/gold_standard.json.new")
SACRED_OUTPUT = Path("data/gold_standard.json")

# Columnas del registro maestro (Covidence)
REGISTRO_COLS = {
    "Covidence ID",
    "Cita (Autores)",
    "Archivos Físicos Extrapolados",
}

# Columnas de un gold standard extendido (mínimas obligatorias)
GOLD_REQUIRED_COLS = {"title", "decision"}
GOLD_ID_COLS = {"id", "pmid", "doi"}


# ══════════════════════════════════════════════════════════════════
# Parseo de formato "registro maestro" (Covidence)
# ══════════════════════════════════════════════════════════════════

def _parse_authors_year(citation: str) -> Tuple[List[str], Optional[int]]:
    """Extrae authors y year de una cita 'Hartman et al. (2016)'."""
    authors: List[str] = []
    year: Optional[int] = None

    citation = (citation or "").strip()
    year_match = re.search(r"\((\d{4})\)", citation)
    if year_match:
        try:
            year = int(year_match.group(1))
        except ValueError:
            year = None
        citation = citation[: year_match.start()].strip()

    if citation:
        # "Hartman et al." -> ["Hartman et al."]; "Smith & Jones" -> ["Smith", "Jones"]
        citation = citation.rstrip(",.;")
        if " & " in citation:
            authors = [part.strip() for part in citation.split(" & ") if part.strip()]
        else:
            authors = [citation]

    return authors, year


def _build_entry_from_registro(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Convierte una fila del CSV registro-maestro a la entrada JSON.

    Todas las filas del registro maestro son papers INCLUIDOS en el meta-análisis
    (ya fueron extraídos y tienen PDFs). Por eso gold_decision = INCLUDE y
    stage_decided = full_text.
    """
    covidence_id = (row.get("Covidence ID") or "").strip().lstrip("#")
    citation = (row.get("Cita (Autores)") or "").strip()
    files = (row.get("Archivos Físicos Extrapolados") or "").strip()

    if not covidence_id and not citation:
        return None

    authors, year = _parse_authors_year(citation)

    entry = {
        "pmid": covidence_id or "",
        "covidence_id": covidence_id,
        "title": citation,  # No tenemos título real en el registro, usamos cita
        "abstract": "",
        "authors": authors,
        "year": year,
        "journal": "",
        "files": [f.strip() for f in files.split("|") if f.strip()],
        "gold_decision": "INCLUDE",
        "gold_reason": "",
        "stage_decided": "full_text",
    }
    return entry


def _is_registro_maestro(fieldnames: List[str]) -> bool:
    """Detecta si el CSV es el formato registro-maestro de Covidence."""
    if not fieldnames:
        return False
    normalized = {f.strip() for f in fieldnames}
    return REGISTRO_COLS.issubset(normalized)


# ══════════════════════════════════════════════════════════════════
# Parseo de formato "gold standard extendido"
# ══════════════════════════════════════════════════════════════════

def _parse_authors_field(raw: str) -> List[str]:
    """Divide 'Smith J; Jones K' o 'Smith J, Jones K' en lista."""
    if not raw:
        return []
    # Separadores comunes en campos de autores
    for sep in (";", "|"):
        if sep in raw:
            return [a.strip() for a in raw.split(sep) if a.strip()]
    return [raw.strip()] if raw.strip() else []


def _build_entry_from_gold(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Convierte una fila del CSV gold-standard extendido a la entrada JSON."""
    # ID: admite id, pmid o doi
    paper_id = ""
    for key in ("id", "pmid", "doi"):
        val = (row.get(key) or "").strip()
        if val:
            paper_id = val
            break

    title = (row.get("title") or "").strip()
    decision_raw = (row.get("decision") or "").strip().upper()

    if not title and not paper_id:
        return None

    if decision_raw not in ("INCLUDE", "EXCLUDE"):
        raise ValueError(
            f"decision invalida '{decision_raw}' en fila con id='{paper_id}' / title='{title[:40]}'"
            " (esperado INCLUDE o EXCLUDE)"
        )

    year_raw = (row.get("year") or "").strip()
    try:
        year = int(year_raw) if year_raw else None
    except ValueError:
        year = None

    entry = {
        "pmid": paper_id,
        "title": title,
        "abstract": (row.get("abstract") or "").strip(),
        "authors": _parse_authors_field(row.get("authors") or ""),
        "year": year,
        "journal": (row.get("journal") or "").strip(),
        "gold_decision": decision_raw,
        "gold_reason": (row.get("reason") or "").strip(),
        "stage_decided": (row.get("stage") or "title_abstract").strip(),
    }
    return entry


def _detect_format(fieldnames: List[str]) -> str:
    """Devuelve 'registro' | 'gold' | 'unknown'."""
    if _is_registro_maestro(fieldnames):
        return "registro"
    normalized = {f.strip().lower() for f in (fieldnames or [])}
    has_id = bool(GOLD_ID_COLS & normalized)
    has_required = GOLD_REQUIRED_COLS.issubset(normalized)
    if has_id and has_required:
        return "gold"
    return "unknown"


# ══════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════

def _detect_delimiter(sample: str) -> str:
    """Detecta delimitador via csv.Sniffer, fallback a ';'."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ";"


def _missing_columns(fieldnames: List[str], fmt: str) -> List[str]:
    """Devuelve las columnas faltantes para un formato dado."""
    normalized = {f.strip() for f in (fieldnames or [])}
    normalized_lower = {f.strip().lower() for f in (fieldnames or [])}
    if fmt == "registro":
        return sorted(REGISTRO_COLS - normalized)
    if fmt == "gold":
        missing = list(GOLD_REQUIRED_COLS - normalized_lower)
        if not (GOLD_ID_COLS & normalized_lower):
            missing.append("id|pmid|doi")
        return sorted(missing)
    return []


def load_csv(csv_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Carga el CSV, detecta formato y devuelve (formato, entries)."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = _detect_delimiter(sample)
        logger.debug("Delimitador detectado: %r", delimiter)

        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        fmt = _detect_format(fieldnames)
        logger.info("Formato detectado: %s (columnas=%s)", fmt, fieldnames)

        if fmt == "unknown":
            missing_registro = _missing_columns(fieldnames, "registro")
            missing_gold = _missing_columns(fieldnames, "gold")
            logger.error(
                "Formato CSV no reconocido. Columnas encontradas: %s", fieldnames
            )
            logger.error(
                "Si es un registro maestro Covidence, faltan: %s", missing_registro
            )
            logger.error(
                "Si es un gold standard extendido, faltan: %s", missing_gold
            )
            sys.exit(2)

        entries: List[Dict[str, Any]] = []
        for idx, row in enumerate(reader, start=2):  # header = linea 1
            try:
                if fmt == "registro":
                    entry = _build_entry_from_registro(row)
                else:
                    entry = _build_entry_from_gold(row)
            except ValueError as exc:
                logger.error("Error en linea %d: %s", idx, exc)
                sys.exit(3)
            if entry is None:
                logger.debug("Linea %d vacia, saltada", idx)
                continue
            entries.append(entry)

    return fmt, entries


def build_gold_standard(
    csv_path: Path,
    output_path: Path,
    query: str = "",
    protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Genera el dict gold standard y lo escribe a output_path."""
    fmt, entries = load_csv(csv_path)

    if not entries:
        logger.error("El CSV no contiene filas utilizables.")
        sys.exit(4)

    n_include = sum(1 for e in entries if e.get("gold_decision") == "INCLUDE")
    n_exclude = sum(1 for e in entries if e.get("gold_decision") == "EXCLUDE")
    logger.info(
        "Entradas generadas: %d total (INCLUDE=%d, EXCLUDE=%d)",
        len(entries),
        n_include,
        n_exclude,
    )

    gold = {
        "query": query,
        "protocol": protocol or {},
        "source_csv": str(csv_path),
        "source_format": fmt,
        "papers": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    logger.info("JSON escrito en %s", output_path)
    return gold


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera data/gold_standard.json.new desde el CSV de la investigadora."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Ruta al CSV de entrada (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta del JSON de salida (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Query clinica asociada al gold standard (opcional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar logs de debug",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    csv_path: Path = args.csv
    output_path: Path = args.output

    logger.info("Construyendo gold standard desde %s", csv_path)

    if not csv_path.exists():
        logger.error(
            "No se encuentra el CSV de gold standard en %s", csv_path
        )
        logger.error(
            "Crea el fichero con la investigadora o pasa --csv <ruta> explicitamente."
        )
        sys.exit(1)

    # Protegemos la zona sagrada: nunca sobrescribimos data/gold_standard.json
    if output_path.resolve() == SACRED_OUTPUT.resolve():
        logger.error(
            "El output '%s' coincide con la zona sagrada. Usa otro destino"
            " (por defecto data/gold_standard.json.new).",
            output_path,
        )
        sys.exit(5)

    build_gold_standard(csv_path, output_path, query=args.query)

    print()
    print(f"[OK] Gold standard generado en: {output_path}")
    print(
        "Revisa data/gold_standard.json.new y renombra a data/gold_standard.json si te convence."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
