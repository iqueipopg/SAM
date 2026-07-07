# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""T16.D.8.C — Auditoría de study protocols en los 66 includes del gold deporte.

Reutiliza la heurística de detección descrita en T16.D.7.E (script
`scripts/diagnose_a5_zerooutcomes.py`):
  - Patrones literales sobre los primeros 2000 caracteres del PDF
    (`study protocol`, `protocol for a random*`, `protocol paper`).
  - Detección de sección Results vía heading regex multiline.
  - PDFs: tomamos como fuente de verdad el campo `pdf_used` de
    `data/gold_standard/deporte/unified/extracted_metadata.csv`,
    que es el PDF que A5 ve realmente. Buscamos primero en
    `pdfs_incluidos/`, fallback `pdfs_todos/`.
  - Cruce con outcomes en `data/gold_standard_normalized/gold_metrics.csv`.

Salidas:
  - outputs/eval/gold_audit/protocols_audit.json
  - outputs/eval/gold_audit/protocols_audit.md
  - data/gold_standard/deporte/unified/gold_screening_unified_with_protocols.csv

RESTRICCIONES.
  - No modifica `gold_screening_unified.csv` original.
  - No toca `agents/`.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GOLD_CSV = REPO / "data/gold_standard/deporte/unified/gold_screening_unified.csv"
META_CSV = REPO / "data/gold_standard/deporte/unified/extracted_metadata.csv"
METRICS_CSV = REPO / "data/gold_standard_normalized/gold_metrics.csv"
PDFS_INCL = REPO / "data/gold_standard/deporte/pdfs_incluidos"
PDFS_TODOS = REPO / "data/gold_standard/deporte/pdfs_todos"

OUT_DIR = REPO / "outputs/eval/gold_audit"
OUT_JSON = OUT_DIR / "protocols_audit.json"
OUT_MD = OUT_DIR / "protocols_audit.md"
OUT_CSV = REPO / "data/gold_standard/deporte/unified/gold_screening_unified_with_protocols.csv"

# Heuristic constants (sourced from scripts/diagnose_a5_zerooutcomes.py)
PROTOCOL_PATTERNS = [
    r"study\s*protocol",
    r"s\s+t\s+u\s+d\s+y\s+p\s+r\s+o\s+t\s+o\s+c\s+o\s+l",
    r"protocol\s+for\s+a\s+random",
    r"protocol\s+paper",
]
RESULTS_HEADING = re.compile(
    r"^\s*(results|findings|outcomes|main results)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _pdf_text(pdf_path: Path) -> str:
    """Extract text using pypdf -> pdfminer fallback."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages)
        if len(text) > 500:
            return text
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(pdf_path)) or ""
    except Exception:
        return ""


def _detect_protocol(text_head: str) -> dict:
    head_norm = re.sub(r"\s+", " ", text_head[:2000]).lower()
    hits = [pat for pat in PROTOCOL_PATTERNS if re.search(pat, head_norm)]
    return {"is_protocol": bool(hits), "patterns_matched": hits}


def _detect_results_section(text: str) -> dict:
    matches = list(RESULTS_HEADING.finditer(text))
    return {
        "found": bool(matches),
        "n_headings": len(matches),
        "first_offset": matches[0].start() if matches else None,
    }


def _locate_pdf(pdf_name: str) -> Path | None:
    if not pdf_name or not isinstance(pdf_name, str):
        return None
    for d in (PDFS_INCL, PDFS_TODOS):
        candidate = d / pdf_name
        if candidate.exists():
            return candidate
    return None


def _outcomes_by_covidence(metrics: pd.DataFrame) -> dict[str, int]:
    return metrics.groupby("covidence_id").size().to_dict()


def _classify_confidence(patterns_matched: list[str], results_present: bool) -> str:
    """Confidence in the is_protocol label.

    high   : >=2 patterns matched AND no results section
    medium : 1 pattern matched AND no results section
    low    : 1+ patterns matched BUT results section present (likely results paper
             that mentions 'study protocol' in references / methods)
    none   : no patterns matched
    """
    if not patterns_matched:
        return "none"
    if not results_present and len(patterns_matched) >= 2:
        return "high"
    if not results_present and len(patterns_matched) == 1:
        return "medium"
    return "low"


def _hypothesis(is_protocol: bool, n_outcomes_in_gold: int) -> str:
    if not is_protocol:
        return "no_protocol"
    if n_outcomes_in_gold > 0:
        return (
            "companion_paper_or_dataset_incoherence: protocol detectado pero el "
            "gold tiene outcomes para este covidence_id, probablemente derivados "
            "de un companion paper (results) no listado en pdf_used"
        )
    return (
        "true_protocol_no_outcomes: protocol detectado y sin outcomes en gold; "
        "candidato a exclusión del numerador/denominador de Hero 3"
    )


def main() -> int:
    if not GOLD_CSV.exists():
        print(f"ERROR: missing {GOLD_CSV}", file=sys.stderr)
        return 1
    if not META_CSV.exists():
        print(f"ERROR: missing {META_CSV}", file=sys.stderr)
        return 1
    if not METRICS_CSV.exists():
        print(f"ERROR: missing {METRICS_CSV}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold = pd.read_csv(GOLD_CSV)
    meta = pd.read_csv(META_CSV)
    metrics = pd.read_csv(METRICS_CSV)

    includes = gold[gold["human_label_ft"] == "include"].copy()
    n_includes = len(includes)

    pdf_map = dict(zip(meta["covidence_id"], meta["pdf_used"]))
    outcomes_map = _outcomes_by_covidence(metrics)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    per_paper: list[dict] = []

    for _, row in includes.iterrows():
        cid = row["covidence_id"]
        title = row.get("title") or ""
        authors = row.get("authors") or ""
        first_author = (str(authors).split(";")[0] if authors else "").strip()
        year = row.get("year")
        try:
            year_int = int(float(year)) if pd.notna(year) else None
        except Exception:
            year_int = None

        pdf_name = pdf_map.get(cid)
        if not isinstance(pdf_name, str):
            pdf_name = ""
        pdf_path = _locate_pdf(pdf_name)

        record: dict = {
            "covidence_id": cid,
            "title": title,
            "first_author": first_author,
            "year": year_int,
            "pdf_filename": pdf_name,
            "pdf_found": bool(pdf_path),
            "pdf_char_length": 0,
            "is_protocol": False,
            "patterns_matched": [],
            "results_section_present": None,
            "n_results_headings": 0,
            "detection_confidence": "no_pdf",
            "n_outcomes_in_gold": int(outcomes_map.get(cid, 0)),
            "hypothesis": "",
            "head_excerpt": "",
        }

        if pdf_path is not None:
            text = _pdf_text(pdf_path)
            record["pdf_char_length"] = len(text)
            proto = _detect_protocol(text[:2000])
            results = _detect_results_section(text)
            record["is_protocol"] = proto["is_protocol"]
            record["patterns_matched"] = proto["patterns_matched"]
            record["results_section_present"] = results["found"]
            record["n_results_headings"] = results["n_headings"]
            record["detection_confidence"] = _classify_confidence(
                proto["patterns_matched"], results["found"]
            )
            record["head_excerpt"] = text[:400]

        record["hypothesis"] = _hypothesis(record["is_protocol"], record["n_outcomes_in_gold"])
        per_paper.append(record)

    # Secondary scan: orphan protocol PDFs sitting in pdfs_incluidos that
    # are NOT referenced by pdf_used. These signal companion-paper structure.
    canonical_pdfs = {p for p in pdf_map.values() if isinstance(p, str)}
    orphan_protocols: list[dict] = []
    if PDFS_INCL.exists():
        for f in sorted(PDFS_INCL.iterdir()):
            if not f.is_file() or f.suffix.lower() != ".pdf":
                continue
            if f.name in canonical_pdfs:
                continue
            text = _pdf_text(f)
            head = text[:2000]
            proto = _detect_protocol(head)
            if not proto["is_protocol"]:
                continue
            results = _detect_results_section(text)
            orphan_protocols.append({
                "pdf_filename": f.name,
                "patterns_matched": proto["patterns_matched"],
                "results_section_present": results["found"],
                "n_results_headings": results["n_headings"],
                "head_excerpt": text[:400],
            })

    # Aggregates
    n_with_pdf = sum(1 for r in per_paper if r["pdf_found"])
    n_protocols = sum(1 for r in per_paper if r["is_protocol"])
    n_protocols_high = sum(1 for r in per_paper if r["is_protocol"] and r["detection_confidence"] == "high")
    n_protocols_medium = sum(1 for r in per_paper if r["is_protocol"] and r["detection_confidence"] == "medium")
    n_protocols_low = sum(1 for r in per_paper if r["is_protocol"] and r["detection_confidence"] == "low")
    protocols = [r for r in per_paper if r["is_protocol"]]
    protocols_with_outcomes = [r for r in protocols if r["n_outcomes_in_gold"] > 0]
    protocols_without_outcomes = [r for r in protocols if r["n_outcomes_in_gold"] == 0]

    summary = {
        "timestamp_utc": timestamp,
        "n_includes_total": n_includes,
        "n_with_pdf": n_with_pdf,
        "n_without_pdf": n_includes - n_with_pdf,
        "n_protocols_detected": n_protocols,
        "pct_protocols": round(100.0 * n_protocols / n_includes, 1) if n_includes else 0.0,
        "n_protocols_high_confidence": n_protocols_high,
        "n_protocols_medium_confidence": n_protocols_medium,
        "n_protocols_low_confidence": n_protocols_low,
        "n_protocols_with_outcomes_in_gold": len(protocols_with_outcomes),
        "n_protocols_without_outcomes_in_gold": len(protocols_without_outcomes),
        "heuristic": {
            "patterns": PROTOCOL_PATTERNS,
            "results_heading_regex": RESULTS_HEADING.pattern,
            "head_chars": 2000,
            "source": "scripts/diagnose_a5_zerooutcomes.py (T16.D.7.E)",
        },
        "pdf_resolution": {
            "primary": str(PDFS_INCL.relative_to(REPO)),
            "fallback": str(PDFS_TODOS.relative_to(REPO)),
            "mapping_source": str(META_CSV.relative_to(REPO)) + " column pdf_used",
        },
        "n_orphan_protocols_in_pdfs_incluidos": len(orphan_protocols),
    }

    audit = {
        "summary": summary,
        "protocols": protocols,
        "orphan_protocols_in_pdfs_incluidos": orphan_protocols,
        "all_papers": per_paper,
    }

    # Write JSON
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    # Write Markdown
    md = _render_md(summary, protocols, per_paper, orphan_protocols)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT_MD}")

    # Write proposed CSV (copy + is_protocol + protocol_patterns_matched)
    enriched = gold.copy()
    proto_lookup = {r["covidence_id"]: r for r in per_paper}
    enriched["is_protocol"] = enriched["covidence_id"].map(
        lambda c: proto_lookup[c]["is_protocol"] if c in proto_lookup else None
    )
    enriched["protocol_patterns_matched"] = enriched["covidence_id"].map(
        lambda c: ",".join(proto_lookup[c]["patterns_matched"]) if c in proto_lookup else ""
    )
    enriched["protocol_detection_confidence"] = enriched["covidence_id"].map(
        lambda c: proto_lookup[c]["detection_confidence"] if c in proto_lookup else ""
    )
    enriched.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # Stdout summary
    print("\n=== SUMMARY ===")
    print(f"Total includes: {n_includes}")
    print(f"With PDF located: {n_with_pdf}")
    print(f"Protocols detected: {n_protocols} ({summary['pct_protocols']}%)")
    print(f"  high confidence:   {n_protocols_high}")
    print(f"  medium confidence: {n_protocols_medium}")
    print(f"  low confidence:    {n_protocols_low}")
    print(f"Protocols WITH outcomes in gold: {len(protocols_with_outcomes)}")
    print(f"Protocols WITHOUT outcomes in gold: {len(protocols_without_outcomes)}")
    print(f"Orphan protocol PDFs in pdfs_incluidos (not pdf_used): {len(orphan_protocols)}")
    for o in orphan_protocols:
        print(f"  - {o['pdf_filename']}  patterns={len(o['patterns_matched'])}")
    return 0


def _render_md(summary: dict, protocols: list[dict], all_papers: list[dict],
               orphan_protocols: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# T16.D.8.C — Auditoría de study protocols en gold deporte\n")
    lines.append(f"Timestamp: `{summary['timestamp_utc']}`\n")

    lines.append("## Resumen ejecutivo\n")
    n_inc = summary["n_includes_total"]
    n_prot = summary["n_protocols_detected"]
    pct = summary["pct_protocols"]
    lines.append(
        "Se aplicó la heurística de detección de study protocols definida en "
        "`scripts/diagnose_a5_zerooutcomes.py` (T16.D.7.E) sobre los "
        f"{n_inc} estudios incluidos del gold standard deporte. La heurística "
        "busca patrones literales (`study protocol`, `protocol for a random*`, "
        "`protocol paper`) en los primeros 2000 caracteres del PDF y verifica "
        "la presencia de sección Results mediante un regex multiline. La fuente "
        f"de PDFs es la columna `pdf_used` de `extracted_metadata.csv`. Se "
        f"localizaron {summary['n_with_pdf']}/{n_inc} PDFs. Se detectaron "
        f"**{n_prot} protocols** ({pct}% del total), de los cuales "
        f"{summary['n_protocols_high_confidence']} de alta confianza, "
        f"{summary['n_protocols_medium_confidence']} de confianza media y "
        f"{summary['n_protocols_low_confidence']} de baja confianza. "
        f"{summary['n_protocols_with_outcomes_in_gold']} protocols tienen "
        f"outcomes en `gold_metrics.csv` (potencial incoherencia: companion "
        f"paper no listado) y {summary['n_protocols_without_outcomes_in_gold']} "
        "no los tienen.\n"
    )
    lines.append(
        "**Nota metodológica importante**: la fuente de verdad para el PDF "
        "asociado a cada include es la columna `pdf_used` de "
        "`extracted_metadata.csv`, que es exactamente lo que A5 procesa en el "
        "pipeline. T16.D.7.E identificó protocols en #13950 y #5298 utilizando "
        "una fuente distinta (`outputs/eval/micro/pdf_extractability_check.json`), "
        "que para esos dos casos apuntaba a ficheros `PROTOCOLO INTERVENCION_*.pdf` "
        "en `pdfs_incluidos/`. Bajo el mapping canónico actual, esos covidence_id "
        "apuntan a sus results papers correspondientes (van Keulen 2011 y "
        "Peacock 2020), por lo que la heurística no los marca como protocol. La "
        "sección 'Protocols huérfanos' más abajo recoge los PDFs de tipo "
        "protocol que existen físicamente en `pdfs_incluidos/` pero no son "
        "apuntados por ningún `pdf_used`; son evidencia del companion-paper "
        "pattern en el dataset.\n"
    )

    lines.append("## Tabla de protocols detectados\n")
    if not protocols:
        lines.append("(ningún protocol detectado)\n")
    else:
        lines.append(
            "| covidence_id | year | first_author | confidence | patterns | results_section | n_outcomes_gold | hypothesis |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|"
        )
        for p in protocols:
            pat = ", ".join(p["patterns_matched"]) or "-"
            res = "True" if p["results_section_present"] else "False"
            hyp_short = "companion_paper" if p["n_outcomes_in_gold"] > 0 else "true_protocol"
            lines.append(
                f"| {p['covidence_id']} | {p['year'] or ''} | {p['first_author'][:30]} | "
                f"{p['detection_confidence']} | {pat} | {res} | "
                f"{p['n_outcomes_in_gold']} | {hyp_short} |"
            )
        lines.append("")

    lines.append("## Detalle por protocol detectado\n")
    for p in protocols:
        lines.append(f"### {p['covidence_id']} — {p['first_author']} ({p['year']})\n")
        lines.append(f"- Título: {p['title'][:200]}")
        lines.append(f"- PDF: `{p['pdf_filename']}` (chars: {p['pdf_char_length']})")
        lines.append(f"- Patrones disparados: {p['patterns_matched']}")
        lines.append(f"- Sección Results presente: {p['results_section_present']} "
                     f"(n_headings: {p['n_results_headings']})")
        lines.append(f"- Confianza detección: **{p['detection_confidence']}**")
        lines.append(f"- Outcomes en gold_metrics.csv: **{p['n_outcomes_in_gold']}**")
        lines.append(f"- Hipótesis: {p['hypothesis']}")
        lines.append("- Excerpt (primeros 400 chars):")
        lines.append("```")
        lines.append(p["head_excerpt"])
        lines.append("```\n")

    lines.append("## Protocols huérfanos en pdfs_incluidos/ (no apuntados por pdf_used)\n")
    if not orphan_protocols:
        lines.append(
            "No se detectaron PDFs de tipo protocol en `pdfs_incluidos/` fuera "
            "de los listados en `pdf_used`.\n"
        )
    else:
        lines.append(
            f"Se detectaron {len(orphan_protocols)} PDFs de tipo protocol en "
            "`pdfs_incluidos/` que NO están referenciados por la columna "
            "`pdf_used` de `extracted_metadata.csv`. Estos son companion "
            "protocols del mismo trial cuyo results paper sí está mapeado. "
            "No afectan a A5 directamente (no se procesan), pero son evidencia "
            "de que el dataset gold contiene material protocol que podría usarse "
            "para validar la heurística o para reasignaciones manuales.\n"
        )
        lines.append("| pdf_filename | patterns_matched | results_section_present |")
        lines.append("|---|---|---|")
        for o in orphan_protocols:
            pat = ", ".join(o["patterns_matched"]) or "-"
            res = "True" if o["results_section_present"] else "False"
            lines.append(f"| `{o['pdf_filename']}` | {pat} | {res} |")
        lines.append("")

    lines.append("## Análisis e implicaciones\n")
    if summary["n_protocols_detected"] == 0:
        lines.append(
            "Bajo el mapping canónico (`pdf_used` de `extracted_metadata.csv`), "
            "la heurística no detectó ningún protocol entre los 66 includes. "
            "Esto significa que A5 está procesando results papers, no protocols, "
            "para el conjunto evaluado. Sin embargo, el análisis secundario revela "
            f"{summary['n_orphan_protocols_in_pdfs_incluidos']} PDFs de tipo "
            "protocol presentes físicamente en `pdfs_incluidos/` pero no "
            "referenciados por `pdf_used`. La existencia de estos protocols "
            "huérfanos indica que el dataset gold sigue un companion-paper "
            "pattern: para algunos trials hay tanto el protocolo como el results "
            "paper, y el roster apunta correctamente al results. La discrepancia "
            "histórica observada en T16.D.7.E (#13950, #5298 mapeados a protocols) "
            "se debió a usar `pdf_extractability_check.json` del sample micro como "
            "fuente de mapping, que para esos casos resolvía a los PROTOCOLO "
            "PDFs huérfanos en lugar de a los results papers canónicos. La "
            "implicación para Hero 3 es que, al usar `pdf_used` como fuente, el "
            "fenómeno protocol-no-extraction es inexistente en el sample medio "
            "salvo que se altere el mapping. Se recomienda asegurar que la "
            "evaluación Hero 3 lea `pdf_used` y no rutas hardcoded a "
            "`pdfs_incluidos/`.\n"
        )
    if summary["n_protocols_with_outcomes_in_gold"] > 0:
        lines.append(
            f"- **{summary['n_protocols_with_outcomes_in_gold']} protocols con outcomes "
            "en gold**: Paula tiene métricas extraídas para estos covidence_ids, pero el "
            "PDF apuntado por `pdf_used` es el protocolo del trial. Esto indica que las "
            "métricas provienen de un *companion paper* (results paper publicado "
            "posteriormente) que no figura en `pdf_used`. Implicación para Hero 3: "
            "A5 sobre el protocolo no podrá producir outcomes; el numerador caerá "
            "indebidamente al cruzar con el gold. Mitigación: reasignar `pdf_used` "
            "al companion paper, o si no existe en `pdfs_todos/`, marcar el caso "
            "como `not_extractable` y excluirlo del denominador.\n"
        )
    if summary["n_protocols_without_outcomes_in_gold"] > 0:
        lines.append(
            f"- **{summary['n_protocols_without_outcomes_in_gold']} protocols sin "
            "outcomes en gold**: Paula incluyó el estudio pero no extrajo métricas. "
            "Coherente con la naturaleza del documento (sin Results). Implicación "
            "para Hero 3: no debería contar como fallo de A5 (la fuente no contiene "
            "outcomes extraíbles). Mitigación: filtrar antes de calcular `pct_complete` "
            "o reportar métrica separada condicional a `is_protocol=False`.\n"
        )
    if summary["n_protocols_low_confidence"] > 0:
        lines.append(
            f"- **{summary['n_protocols_low_confidence']} casos con confianza baja**: "
            "patrón disparado pero con sección Results presente. Probablemente son "
            "results papers que mencionan al protocolo del trial en su introducción "
            "o referencias. Recomendable inspección manual antes de aplicar filtro.\n"
        )

    lines.append("## Recomendación para Hero 3 en sample medio\n")
    lines.append(
        "Se propone aplicar el siguiente orden de tratamiento, en cascada:\n\n"
        "1. **Filtrar protocols antes de calcular `pct_complete`**: definir el "
        "denominador de Hero 3 como `{incluidos} \\ {is_protocol=True ∧ "
        "detection_confidence ∈ {high, medium}}`. Esto evita penalizar a A5 por "
        "documentos donde estructuralmente no existen outcomes que extraer.\n"
        "2. **Reportar métrica separada `pct_complete_conditional`**: además del "
        "valor filtrado, ofrecer una segunda métrica condicional a "
        "`results_section_present=True`, más estricta y comparable entre runs.\n"
        "3. **Investigar companion papers (future work)**: para los protocols con "
        "outcomes en gold, revisar si existe un results paper en `pdfs_todos/` y, "
        "si procede, reasignar `pdf_used` en una nueva versión del roster. Esto "
        "requiere contactar a Paula para confirmar la asignación.\n"
    )

    lines.append("## Schema propuesto para columna nueva\n")
    lines.append(
        "Se ha materializado una copia enriquecida del CSV en "
        "`data/gold_standard/deporte/unified/gold_screening_unified_with_protocols.csv` "
        "con las siguientes columnas adicionales (el original no se toca):\n\n"
        "- `is_protocol` (bool|None): True si la heurística detectó protocolo en "
        "el PDF asociado; None si no se localizó PDF.\n"
        "- `protocol_patterns_matched` (string): lista comma-separated de patrones "
        "regex que dispararon la detección. Vacío si no hubo detección.\n"
        "- `protocol_detection_confidence` (string): uno de "
        "`{high, medium, low, none, no_pdf}`. Solo `high` y `medium` deberían "
        "usarse como filtro automático; `low` requiere revisión manual.\n"
    )

    lines.append("## Ficheros consultados\n")
    lines.append(
        "- `data/gold_standard/deporte/unified/gold_screening_unified.csv` "
        "(66 includes humanos)\n"
        "- `data/gold_standard/deporte/unified/extracted_metadata.csv` "
        "(mapping covidence_id -> pdf_used)\n"
        "- `data/gold_standard_normalized/gold_metrics.csv` "
        "(outcomes por covidence_id)\n"
        "- `data/gold_standard/deporte/pdfs_incluidos/` y `pdfs_todos/` "
        "(PDFs físicos)\n"
        "- `scripts/diagnose_a5_zerooutcomes.py` (heurística origen, T16.D.7.E)\n"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
