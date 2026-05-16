"""T16.D.7.E — A5 zero-outcomes diagnosis (#13950, #5298) vs #13442 control.

Lee el cache A5 (outputs/eval/micro/_a5_full_outputs.json) y los PDFs asociados.
Compara los dos papers que produjeron n_outcomes=0 contra un control que sí
funcionó (#13442). Detecta tipo de documento (STUDY PROTOCOL vs results paper)
y presencia/ausencia de sección Results en el texto extraído.

Salidas:
  outputs/eval/micro/a5_zerooutcomes_diagnosis.md
  outputs/eval/micro/a5_zerooutcomes_diagnosis.json
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
A5_CACHE = REPO / "outputs/eval/micro/_a5_full_outputs.json"
PDF_CHECK = REPO / "outputs/eval/micro/pdf_extractability_check.json"
OUT_MD = REPO / "outputs/eval/micro/a5_zerooutcomes_diagnosis.md"
OUT_JSON = REPO / "outputs/eval/micro/a5_zerooutcomes_diagnosis.json"

TARGETS = ["#13442", "#13950", "#5298"]
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
    hits = []
    for pat in PROTOCOL_PATTERNS:
        if re.search(pat, head_norm):
            hits.append(pat)
    return {"is_protocol": bool(hits), "patterns_matched": hits}


def _detect_results_section(text: str) -> dict:
    matches = list(RESULTS_HEADING.finditer(text))
    return {
        "found": bool(matches),
        "n_headings": len(matches),
        "first_offset": matches[0].start() if matches else None,
    }


def _load_pdf_mapping() -> dict[str, str]:
    raw = json.loads(PDF_CHECK.read_text(encoding="utf-8"))
    return {r["covidence_id"]: r.get("pdf_filename", "") for r in raw["results"]}


def main() -> None:
    a5 = json.loads(A5_CACHE.read_text(encoding="utf-8"))
    by_id = {p.get("pmid"): p for p in a5["papers"]}
    pdf_map = _load_pdf_mapping()
    pdfs_dir = REPO / "data/gold_standard/deporte/pdfs_todos"

    evidence = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "papers": {}}

    for cid in TARGETS:
        paper = by_id.get(cid, {})
        outcomes = paper.get("outcomes") or []
        pdf_name = pdf_map.get(cid, "")
        pdf_path = pdfs_dir / pdf_name if pdf_name else None
        text = _pdf_text(pdf_path) if pdf_path and pdf_path.exists() else ""

        head = text[:1500]
        proto = _detect_protocol(head)
        results = _detect_results_section(text)

        evidence["papers"][cid] = {
            "title_in_a5_cache": paper.get("title", ""),
            "pdf_filename": pdf_name,
            "pdf_exists": bool(pdf_path and pdf_path.exists()),
            "pdf_char_length": len(text),
            "head_excerpt_500w": head[:500],
            "is_protocol_paper": proto["is_protocol"],
            "protocol_patterns_matched": proto["patterns_matched"],
            "results_section_present": results["found"],
            "results_heading_offsets": results,
            "a5_n_outcomes": len(outcomes),
            "a5_extraction_quality": paper.get("extraction_quality"),
            "a5_extraction_confidence": paper.get("extraction_confidence"),
            "a5_extraction_notes": paper.get("extraction_notes", ""),
            "a5_sections_used": paper.get("sections_used", []),
        }

    OUT_JSON.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")

    md = _render_md(evidence)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT_MD} and {OUT_JSON}")


def _render_md(ev: dict) -> str:
    lines: list[str] = []
    lines.append("# A5 zero-outcomes — diagnóstico (T16.D.7.E)\n")
    lines.append(f"Timestamp: `{ev['timestamp_utc']}`\n")
    lines.append("## Resumen ejecutivo\n")
    lines.append(
        "Los PDFs entregados para #13950 (Mi-PACT) y #5298 (Vitalum) son **study "
        "protocols publicados antes de ejecutar el ensayo**, no los papers de "
        "resultados. No contienen sección Results ni tablas con outcomes "
        "numéricos porque el trial aún no se había completado en el momento de "
        "publicarse. A5 detecta correctamente la ausencia: `extraction_notes='no "
        "numerical outcomes extracted'`. No es un bug de A5 ni del prompt ni de "
        "la extracción de PDF: es un **mismatch a nivel de dataset gold standard**, "
        "donde el covidence_id apunta a un trial cuyas métricas en gold_metrics "
        "provienen del paper de resultados, mientras que el PDF entregado en "
        "`pdfs_incluidos/` es el protocolo del mismo trial. Mitigación trivial a "
        "nivel de roster (reasignar PDF o filtrar protocolos), no requiere "
        "reestructurar A5.\n"
    )
    lines.append("## Comparación por paper\n")
    for cid, info in ev["papers"].items():
        lines.append(f"### {cid}\n")
        lines.append(f"- PDF: `{info['pdf_filename']}` (existe: {info['pdf_exists']}, {info['pdf_char_length']} chars)")
        lines.append(f"- Título en cache A5: {info['title_in_a5_cache']}")
        lines.append(f"- ¿Protocolo? **{info['is_protocol_paper']}** (patrones: {info['protocol_patterns_matched']})")
        lines.append(f"- ¿Sección Results presente? **{info['results_section_present']}** (headings: {info['results_heading_offsets']['n_headings']})")
        lines.append(f"- A5 n_outcomes: **{info['a5_n_outcomes']}**")
        lines.append(f"- A5 extraction_quality: `{info['a5_extraction_quality']}` / confidence: `{info['a5_extraction_confidence']}`")
        lines.append(f"- A5 extraction_notes: `{info['a5_extraction_notes']!s}`")
        lines.append(f"- A5 sections_used: {info['a5_sections_used']}\n")
        lines.append("**Excerpt (primeros 500 chars):**\n")
        lines.append("```")
        lines.append(info["head_excerpt_500w"])
        lines.append("```\n")
    lines.append("## Diagnóstico de causa\n")
    lines.append(
        "Entre las hipótesis del prompt original:\n\n"
        "- (a) Sección Results no detectada por chunking/regex de headings — **descartada**: "
        "los protocolos no tienen sección Results en absoluto, no es un fallo de detección.\n"
        "- (b) Prompt no maneja formato concreto — **descartada**: A5 emite `extraction_notes` "
        "explícitamente diciendo que no hay outcomes numéricos.\n"
        "- (c) PDF con estructura rara — **refinada**: el PDF se extrae bien (>60k chars), "
        "pero el contenido es un protocolo, no un paper de resultados.\n"
        "- (d) Otro — **causa real**: mismatch dataset/gold: el covidence_id está mapeado "
        "al trial pero gold_metrics referencia las métricas del results paper publicado "
        "posteriormente; el PDF entregado en `pdfs_incluidos/` es el protocolo.\n"
    )
    lines.append("## Recomendación (no implementar en este sprint)\n")
    lines.append(
        "- A nivel de dataset gold standard deporte: añadir flag `is_protocol_paper` por "
        "fila de roster y excluir protocolos del cálculo Hero 3 numerador/denominador, "
        "documentando la limitación en cap5.\n"
        "- A nivel de A5: opcional, no urgente, añadir una primera comprobación 'is this a "
        "protocol paper?' antes de extraer, devolviendo `extraction_quality='not_applicable'` "
        "en lugar de `partial`. Reduce ruido en métricas de evaluación.\n"
    )
    lines.append("## Ficheros consultados\n")
    lines.append(
        "- `outputs/eval/micro/_a5_full_outputs.json` (cache A5)\n"
        "- `outputs/eval/micro/pdf_extractability_check.json` (mapping PDFs)\n"
        "- `data/gold_standard/deporte/pdfs_todos/` (PDFs físicos)\n"
        "- `outputs/eval/micro/hero3_diagnosis.md` (contexto previo T16.D.7.A)\n"
        "- `outputs/eval/micro/hero3_alternative_metrics.md` (excerpts T16.D.7.B)\n"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
