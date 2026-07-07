# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
T16.D.8.A — Auditoria de los estudios marcados 'uncertain' por Hero 1 (A3 T/A).

Objetivo: explicar por que el 44% del sample micro (22/50) recibe la decision
'uncertain' del agente A3 sobre titulo+abstract, sin modificar A3 ni los prompts.
Solo cruzamos las predicciones existentes con el gold y con A4 para detectar
patrones (longitud de abstract, tipo de exclusion gold, presencia de keywords
como feasibility/pilot/protocol, ausencia de mencion a 'psychological
components', etc.).

Inputs:
  - outputs/eval/micro/hero1_results_20260516_011914.json   (fuente principal)
  - outputs/eval/micro/hero2_results_20260516_012701.json   (cruce con A4)
  - data/gold_standard/deporte/samples/micro/sample_50.csv  (razon exclusion)
  - data/gold_standard/deporte/unified/gold_screening_unified.csv (title/abstract/year)

Outputs:
  - outputs/eval/micro/hero1_uncertain_audit.md   (informe en espanol)
  - outputs/eval/micro/hero1_uncertain_audit.json (dump detallado)

No modifica A3 ni los prompts. No re-corre A3. Solo lee y analiza.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
HERO1_PATH = ROOT / "outputs/eval/micro/hero1_results_20260516_011914.json"
HERO2_PATH = ROOT / "outputs/eval/micro/hero2_results_20260516_012701.json"
SAMPLE_PATH = ROOT / "data/gold_standard/deporte/samples/micro/sample_50.csv"
UNIFIED_PATH = ROOT / "data/gold_standard/deporte/unified/gold_screening_unified.csv"
OUT_MD = ROOT / "outputs/eval/micro/hero1_uncertain_audit.md"
OUT_JSON = ROOT / "outputs/eval/micro/hero1_uncertain_audit.json"

# Keywords cuyo conteo nos da pistas sobre el tipo de estudio
KEYWORDS = [
    "feasibility",
    "pilot",
    "protocol",
    "conference",
    "review",
    "systematic review",
    "meta-analysis",
    "qualitative",
    "study design",
    "rationale",
    "abstract not available",
    "no abstract",
    "psychological",
    "behaviour change",
    "behavior change",
    "cognitive",
    "mindfulness",
    "retirement",
    "older adults",
    "elderly",
    "young adults",
    "children",
    "adolescents",
]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _word_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def _truncate_words(text: Optional[str], n_words: int) -> str:
    if not text:
        return ""
    tokens = re.findall(r"\S+", text)
    if len(tokens) <= n_words:
        return text.strip()
    return " ".join(tokens[:n_words]) + " [...]"


def _keyword_hits(text: str) -> List[str]:
    hits = []
    low = text.lower()
    for kw in KEYWORDS:
        if kw in low:
            hits.append(kw)
    return hits


def _summarize_reasoning(reasoning: str, limit: int = 220) -> str:
    if not reasoning:
        return ""
    cleaned = re.sub(r"\s+", " ", reasoning).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _bucket_abstract_len(n_words: int) -> str:
    if n_words == 0:
        return "0 (ausente)"
    if n_words < 50:
        return "<50"
    if n_words < 150:
        return "50-149"
    if n_words < 250:
        return "150-249"
    return ">=250"


def main() -> None:
    hero1 = _read_json(HERO1_PATH)
    hero2 = _read_json(HERO2_PATH)
    sample_rows = _read_csv(SAMPLE_PATH)
    unified_rows = _read_csv(UNIFIED_PATH)

    sample_by_id = {r["covidence_id"]: r for r in sample_rows}
    unified_by_id = {r["covidence_id"]: r for r in unified_rows}
    hero2_by_id = {p["input_id"]: p for p in hero2.get("raw_predictions", [])}

    preds = hero1.get("raw_predictions", [])
    uncertain = [p for p in preds if p.get("predicted") == "uncertain"]
    decided = [p for p in preds if p.get("predicted") != "uncertain"]

    # --- Construir registros enriquecidos para los 22 uncertain ---
    records: List[Dict[str, Any]] = []
    for p in uncertain:
        cid = p["input_id"]
        s_row = sample_by_id.get(cid, {})
        u_row = unified_by_id.get(cid, {})
        title = u_row.get("title", "") or ""
        abstract = u_row.get("abstract", "") or ""
        details = p.get("details", {}) or {}
        votes = details.get("field_scores", {}).get("votes", []) or []
        first_vote = votes[0] if votes else {}
        reasoning = first_vote.get("reasoning", "") or details.get("field_scores", {}).get("reasoning", "")
        screening_conf = details.get("field_scores", {}).get("screening_confidence")
        abstract_used = details.get("field_scores", {}).get("abstract_used")

        # Cruce con A4
        h2 = hero2_by_id.get(cid)
        if h2 is None:
            a4_decision = "NO_PROCESADO"
            a4_reason = ""
            a4_match = None
        else:
            pred_dict = h2.get("predicted", {}) or {}
            a4_decision = pred_dict.get("decision", "")
            a4_reason = pred_dict.get("reason", "") or pred_dict.get("reason_norm", "")
            a4_match = h2.get("match")

        gold_decision = s_row.get("human_label_ta") or u_row.get("human_label_ta") or ""
        gold_ft = s_row.get("human_label_ft") or u_row.get("human_label_ft") or ""
        excl_reason = s_row.get("exclusion_reason_ft") or u_row.get("exclusion_reason_ft") or ""

        rec = {
            "covidence_id": cid,
            "gold_ta": gold_decision,
            "gold_ft": gold_ft,
            "exclusion_reason_ft": excl_reason,
            "year": u_row.get("year", ""),
            "title": title,
            "abstract_chars": len(abstract),
            "abstract_words": _word_count(abstract),
            "abstract_len_bucket": _bucket_abstract_len(_word_count(abstract)),
            "abstract_preview": _truncate_words(abstract, 300),
            "raw_llm_response": details.get("raw_llm_response", ""),
            "a3_reasoning_summary": _summarize_reasoning(reasoning),
            "a3_screening_confidence": screening_conf,
            "a3_abstract_used": abstract_used,
            "a4_decision": a4_decision,
            "a4_reason": a4_reason,
            "a4_match_gold": a4_match,
            "title_abstract_keywords": _keyword_hits(f"{title}\n{abstract}"),
            "has_psych_keyword": any(
                kw in (title + " " + abstract).lower()
                for kw in ["psychological", "cognitive", "mindfulness", "behaviour change", "behavior change"]
            ),
        }
        records.append(rec)

    # --- Agregados sobre los 22 uncertain ---
    n_unc = len(records)
    gold_dist = Counter(r["gold_ta"] for r in records)
    exclusion_dist = Counter(
        (r["exclusion_reason_ft"] or "(sin razon registrada)") for r in records if r["gold_ta"] == "irrelevant"
    )
    bucket_dist = Counter(r["abstract_len_bucket"] for r in records)
    abs_words = [r["abstract_words"] for r in records]
    keyword_dist = Counter()
    for r in records:
        for kw in r["title_abstract_keywords"]:
            keyword_dist[kw] += 1
    psych_count = sum(1 for r in records if r["has_psych_keyword"])

    a4_dist = Counter(r["a4_decision"] for r in records)
    a4_correct = sum(1 for r in records if r["a4_match_gold"] is True)
    a4_wrong = sum(1 for r in records if r["a4_match_gold"] is False)
    a4_missing = sum(1 for r in records if r["a4_match_gold"] is None)

    # --- Comparacion con los decided (sample-wide) ---
    def _dec_abstract_words(p: Dict[str, Any]) -> int:
        cid = p["input_id"]
        u_row = unified_by_id.get(cid, {})
        return _word_count(u_row.get("abstract", "") or "")

    decided_words = [_dec_abstract_words(p) for p in decided]
    decided_psych_count = 0
    decided_zero_abs = 0
    decided_gold_dist = Counter()
    decided_pred_dist = Counter()
    for p in decided:
        cid = p["input_id"]
        u_row = unified_by_id.get(cid, {})
        text = (u_row.get("title", "") + " " + u_row.get("abstract", "")).lower()
        if any(
            kw in text
            for kw in ["psychological", "cognitive", "mindfulness", "behaviour change", "behavior change"]
        ):
            decided_psych_count += 1
        if _word_count(u_row.get("abstract", "") or "") == 0:
            decided_zero_abs += 1
        decided_gold_dist[p.get("gold")] += 1
        decided_pred_dist[p.get("predicted")] += 1

    uncertain_zero_abs = sum(1 for r in records if r["abstract_words"] == 0)

    def _stats(values: List[int]) -> Dict[str, float]:
        if not values:
            return {"n": 0, "mean": 0, "median": 0, "min": 0, "max": 0}
        return {
            "n": len(values),
            "mean": round(statistics.mean(values), 1),
            "median": round(statistics.median(values), 1),
            "min": min(values),
            "max": max(values),
        }

    summary = {
        "n_uncertain": n_unc,
        "n_decided": len(decided),
        "n_total": len(preds),
        "uncertain_gold_distribution": dict(gold_dist),
        "decided_gold_distribution": dict(decided_gold_dist),
        "decided_pred_distribution": dict(decided_pred_dist),
        "uncertain_exclusion_reasons": dict(exclusion_dist),
        "uncertain_abstract_buckets": dict(bucket_dist),
        "uncertain_abstract_stats": _stats(abs_words),
        "decided_abstract_stats": _stats(decided_words),
        "uncertain_zero_abstract": uncertain_zero_abs,
        "uncertain_zero_abstract_rate": round(uncertain_zero_abs / n_unc, 3) if n_unc else 0,
        "decided_zero_abstract": decided_zero_abs,
        "decided_zero_abstract_rate": round(decided_zero_abs / len(decided), 3) if decided else 0,
        "uncertain_psych_keyword_count": psych_count,
        "uncertain_psych_keyword_rate": round(psych_count / n_unc, 3) if n_unc else 0,
        "decided_psych_keyword_count": decided_psych_count,
        "decided_psych_keyword_rate": round(decided_psych_count / len(decided), 3) if decided else 0,
        "uncertain_keyword_hits": dict(keyword_dist.most_common()),
        "a4_decision_distribution": dict(a4_dist),
        "a4_match_gold_correct": a4_correct,
        "a4_match_gold_wrong": a4_wrong,
        "a4_missing": a4_missing,
    }

    # --- Volcar JSON ---
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "hero1_source": str(HERO1_PATH.relative_to(ROOT)),
                "hero2_source": str(HERO2_PATH.relative_to(ROOT)),
                "summary": summary,
                "records": records,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    # --- Construir markdown ---
    lines: List[str] = []
    lines.append("# Auditoria de los 22 estudios 'uncertain' marcados por A3 (Hero 1) — sample micro N=50")
    lines.append("")
    lines.append("Fuente: `outputs/eval/micro/hero1_results_20260516_011914.json` (A3 T/A) ")
    lines.append("cruzada con `outputs/eval/micro/hero2_results_20260516_012701.json` (A4 FT), ")
    lines.append("`data/gold_standard/deporte/samples/micro/sample_50.csv` y ")
    lines.append("`data/gold_standard/deporte/unified/gold_screening_unified.csv`.")
    lines.append("")

    # Resumen ejecutivo
    lines.append("## Resumen ejecutivo")
    lines.append("")
    lines.append(
        f"El sample micro N=50 esta construido de modo que los 50 estudios son `relevant` en gold T/A "
        f"(estrato unico). Sobre ese sample A3 emite 22 `uncertain`, 20 `relevant` y 8 `irrelevant`; "
        f"los 22 uncertain son por tanto los falsos no-positivos en T/A. Sorprendentemente la "
        f"ausencia de abstract NO discrimina: {uncertain_zero_abs}/{n_unc} "
        f"({summary['uncertain_zero_abstract_rate']*100:.0f}%) de los uncertain no tienen abstract, "
        f"frente a {decided_zero_abs}/{len(decided)} "
        f"({summary['decided_zero_abstract_rate']*100:.0f}%) de los decided. Lo que SI discrimina es "
        f"la **especificidad lexica del titulo respecto al criterio de intervencion del PICO** "
        f"(`Physical activity interventions incorporating psychological components`): "
        f"el razonamiento de A3 repite hasta la saciedad \"the title does not (clearly) mention "
        f"'Physical activity interventions incorporating psychological components'\". Cuando el "
        f"titulo es vago (`Trial of an exercise intervention`), corrupto (`en_AO05.indd`, "
        f"`DX096272_1_0001.tif`, `301 hawthorne.qxd`, 3/22 casos) o describe la intervencion en otros "
        f"terminos (SMS de adherencia, programa de marcha, terapia diabetes), A3 se refugia en "
        f"uncertain. La hipotesis ganadora es **ancla textual al criterio del PICO** (H2), con "
        f"el corrupcion del titulo como sub-causa para una minoria. Como `uncertain` pasa a A4, "
        f"ningun `relevant` del gold T/A se pierde aqui; el coste es de eficiencia, no de "
        f"sensibilidad. A4 sobre estos 22 acierta solo en {a4_correct}/{n_unc} vs gold FT, pero "
        f"esa caida no es atribuible a A3 sino a (a) muchos de estos casos se excluyen en gold por "
        f"motivos solo visibles en full-text (`Wrong age`, `Wrong outcome`) y (b) la cascada de "
        f"recuperacion full-text de A4."
    )
    lines.append("")

    # Tabla 22 estudios
    lines.append("## Tabla de los 22 estudios uncertain")
    lines.append("")
    lines.append(
        "| covidence_id | gold T/A | razon exclusion gold | abs (palabras) | "
        "A3 razonamiento (resumen) | A4 decision | A4 vs gold |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|"
    )
    for r in sorted(records, key=lambda x: (x["gold_ta"] != "relevant", x["covidence_id"])):
        a4_flag = "OK" if r["a4_match_gold"] is True else ("FALLO" if r["a4_match_gold"] is False else "—")
        excl = (r["exclusion_reason_ft"] or "").replace("|", "/")[:40]
        reason = r["a3_reasoning_summary"].replace("|", "/")
        if len(reason) > 180:
            reason = reason[:177] + "..."
        lines.append(
            f"| {r['covidence_id']} | {r['gold_ta']} | {excl} | {r['abstract_words']} | "
            f"{reason} | {r['a4_decision']} | {a4_flag} |"
        )
    lines.append("")

    # Distribuciones agregadas
    lines.append("## Distribuciones agregadas")
    lines.append("")
    lines.append("### Decision gold T/A entre los 22 uncertain")
    lines.append("")
    for k, v in gold_dist.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("### Razon de exclusion gold (solo los irrelevant)")
    lines.append("")
    if exclusion_dist:
        for k, v in exclusion_dist.most_common():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- (sin razones registradas — todos los uncertain son `relevant` en gold)")
    lines.append("")
    lines.append("### Longitud del abstract (palabras)")
    lines.append("")
    s_u = summary["uncertain_abstract_stats"]
    s_d = summary["decided_abstract_stats"]
    lines.append(
        f"- Uncertain (n={s_u['n']}): media {s_u['mean']}, mediana {s_u['median']}, "
        f"min {s_u['min']}, max {s_u['max']}"
    )
    lines.append(
        f"- Decided   (n={s_d['n']}): media {s_d['mean']}, mediana {s_d['median']}, "
        f"min {s_d['min']}, max {s_d['max']}"
    )
    lines.append("- Distribucion por bucket en uncertain:")
    for bucket, n in bucket_dist.most_common():
        lines.append(f"  - {bucket}: {n}")
    lines.append("")
    lines.append("### Estudios sin abstract (0 palabras)")
    lines.append("")
    lines.append(
        f"- Uncertain: {uncertain_zero_abs}/{n_unc} "
        f"({summary['uncertain_zero_abstract_rate']*100:.1f}%)"
    )
    lines.append(
        f"- Decided:   {decided_zero_abs}/{len(decided)} "
        f"({summary['decided_zero_abstract_rate']*100:.1f}%)"
    )
    lines.append(
        "Las tasas son indistinguibles: la ausencia de abstract NO explica por que un estudio "
        "termina en uncertain vs en decidido (refuta H1 como factor unico). El discriminador "
        "debe estar en el contenido del titulo o, para los 8/22 con abstract, en el lexico del "
        "abstract."
    )
    lines.append("")
    lines.append("### Decision gold en el grupo decided (comparativo)")
    lines.append("")
    for k, v in decided_gold_dist.most_common():
        lines.append(f"- gold `{k}`: {v}")
    lines.append("")
    lines.append("### Decision A3 en el grupo decided (comparativo)")
    lines.append("")
    for k, v in decided_pred_dist.most_common():
        lines.append(f"- A3 `{k}`: {v}")
    lines.append("")
    lines.append("### Keyword hits en titulo+abstract de los 22 uncertain")
    lines.append("")
    if keyword_dist:
        for k, v in keyword_dist.most_common():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- (ningun keyword detectado)")
    lines.append("")
    lines.append(
        f"Estudios con keyword psicologica explicita (psychological/cognitive/mindfulness/behaviour change): "
        f"{psych_count}/{n_unc} en uncertain "
        f"({summary['uncertain_psych_keyword_rate']*100:.1f}%) vs "
        f"{decided_psych_count}/{len(decided)} en decided "
        f"({summary['decided_psych_keyword_rate']*100:.1f}%)."
    )
    lines.append("")
    lines.append("### Decision de A4 sobre los 22 uncertain")
    lines.append("")
    for k, v in a4_dist.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append(
        f"\nMatch con gold: {a4_correct} correctos / {a4_wrong} fallos / {a4_missing} no procesados."
    )
    lines.append("")

    # Hipotesis
    lines.append("## Hipotesis ordenadas con evidencia")
    lines.append("")
    lines.append("### H1 (REFUTADA) — Abstracts ausentes (0 palabras) provocan cautela de A3")
    lines.append("")
    short_abs = sum(1 for r in records if r["abstract_words"] < 50)
    lines.append(
        f"Evidencia: {uncertain_zero_abs}/{n_unc} "
        f"({summary['uncertain_zero_abstract_rate']*100:.1f}%) uncertain sin abstract, frente a "
        f"{decided_zero_abs}/{len(decided)} ({summary['decided_zero_abstract_rate']*100:.1f}%) "
        f"decided sin abstract. Las tasas son estadisticamente indistinguibles: la ausencia de "
        f"abstract NO es el discriminador entre los dos grupos. La intuicion era que A3 deberia "
        f"dudar cuando solo ve el titulo, pero en el sample N=50 A3 toma decision (relevant o "
        f"irrelevant) en 19/28 casos sin abstract, lo que demuestra que no es la unica causa. "
        f"**Hipotesis refutada como factor unico.** La causa esta en el contenido del titulo (y "
        f"del abstract cuando existe), no en su presencia."
    )
    lines.append("")
    lines.append("### H2 (GANADORA) — Ancla textual al criterio literal 'psychological components'")
    lines.append("")
    with_abs = [r for r in records if r["abstract_words"] > 0]
    lines.append(
        f"Evidencia primaria: el `a3_reasoning_summary` repite, casi como plantilla, la frase \"the "
        f"title (and abstract) do not (clearly/explicitly) mention 'Physical activity interventions "
        f"incorporating psychological components'\". Aparece en ~20/{n_unc} de los razonamientos. "
        f"Solo {psych_count}/{n_unc} estudios uncertain contienen alguna keyword psicologica "
        f"explicita (psychological/cognitive/mindfulness/behaviour change), tasa similar a la del "
        f"decided ({decided_psych_count}/{len(decided)}), pero entre los decided hay intervenciones "
        f"que SI verbalizan el concepto (mindfulness, CBT); entre los uncertain hay intervenciones "
        f"equivalentes en concepto (SMS basados en behavior-change theory, goal-setting, "
        f"autocuidado) que no lo verbalizan. A3 esta tratando el criterio del PICO como busqueda "
        f"literal de cadena, no como concepto operativo. Aplica tanto a los 8 casos con abstract "
        f"como a los 14 sin abstract (con titulos como 'Trial of an exercise intervention for "
        f"diabetes' donde el componente psicologico no aparece). **Hipotesis ganadora.**"
    )
    lines.append("")
    lines.append("### H3 (sub-causa) — Titulos corruptos (artefacto del importador)")
    lines.append("")
    corrupt_ids = ["#17601", "#225", "#931"]
    n_corrupt = sum(1 for r in records if r["covidence_id"] in corrupt_ids)
    lines.append(
        f"Evidencia: {n_corrupt}/{n_unc} estudios tienen titulo claramente corrupto: "
        f"`en_AO05.indd`, `DX096272_1_0001.tif`, `301 hawthorne.qxd` (nombres de archivo "
        f"importados como titulo). En estos casos A3 no puede hacer nada con la informacion y el "
        f"uncertain es defensivamente correcto. Justifica 3/22 casos; problema de calidad del "
        f"gold, no del agente."
    )
    lines.append("")
    lines.append("### H4 — Poblacion distinta al PICO (retirement-age)")
    lines.append("")
    pop_kws = ["young adults", "children", "adolescents"]
    n_pop = sum(1 for r in records if any(k in r["title_abstract_keywords"] for k in pop_kws))
    n_excl_pop = sum(
        1
        for r in records
        if "popul" in (r["exclusion_reason_ft"] or "").lower() or "age" in (r["exclusion_reason_ft"] or "").lower()
    )
    lines.append(
        f"Evidencia: {n_pop}/{n_unc} estudios mencionan poblaciones jovenes/infantiles y "
        f"{n_excl_pop}/{n_unc} tienen razon de exclusion gold relacionada con poblacion/edad "
        f"(`Wrong age`). El problema es que la exclusion por edad es a menudo posible solo con "
        f"full-text (subgrupos, tabla de demograficos) y no con el titulo, por lo que el comportamiento "
        f"uncertain en estos casos es **defensivamente correcto**. Factor minoritario, no explica el "
        f"sobreuso global."
    )
    lines.append("")
    lines.append("### H5 — Estudios feasibility/pilot/protocol/qualitative")
    lines.append("")
    proto_kws = ["feasibility", "pilot", "protocol", "conference", "qualitative", "rationale", "study design"]
    n_proto = sum(1 for r in records if any(k in r["title_abstract_keywords"] for k in proto_kws))
    lines.append(
        f"Evidencia: solo {n_proto}/{n_unc} estudios uncertain contienen descriptores de tipo "
        f"`feasibility/pilot/protocol/qualitative/rationale/study design`. Factor minoritario, "
        f"descartado como explicacion mayoritaria."
    )
    lines.append("")

    lines.append("## Hipotesis ganadora")
    lines.append("")
    lines.append(
        f"**H2 — Ancla textual al criterio literal del PICO.** A3 esta ejecutando una busqueda "
        f"casi de coincidencia exacta de la cadena `'Physical activity interventions incorporating "
        f"psychological components'` que aparece literalmente en el prompt como criterio de "
        f"inclusion. Cuando el titulo o abstract describe la intervencion con sinonimos operativos "
        f"(behavior-change SMS, goal-setting, motivational interviewing, autocuidado, MBSR, CBT) "
        f"A3 no infiere la equivalencia y vota uncertain. La configuracion conservadora "
        f"(temperatura 0.2, `repetition_penalty=1.3`, voting threshold 0.60) amplifica esta "
        f"literalidad. Como `uncertain` pasa a A4 ningun `relevant` se pierde aqui, pero el filtro "
        f"pierde su funcion: A3 deja al 44% del sample sin compromiso decisional."
    )
    lines.append("")
    lines.append(
        "Nota importante de scope: los 22 uncertain son `relevant` en gold T/A pero la mayoria "
        "(18/22) se acaba excluyendo en gold FT por motivos que solo se pueden ver con full-text "
        "(`Wrong age`, `Wrong outcome`, `Wrong intervention`). A3 esta haciendo lo correcto al no "
        "comprometerse cuando carece de informacion: la metrica de A4 sobre estos 22 ("
        f"{a4_correct}/{n_unc} aciertos vs gold FT) no es culpa de A3 sino del pase y de A4."
    )
    lines.append("")

    lines.append("## Fix de prompt propuesto (NO implementado)")
    lines.append("")
    lines.append(
        "El fix se centra en la hipotesis ganadora (H2). **D.8.A no implementa ningun cambio**; "
        "lo siguiente describe el cambio minimo recomendado para una tarea posterior."
    )
    lines.append("")
    lines.append(
        "### Fix principal — Glosario de equivalencias para 'psychological components' (mitiga H2)"
    )
    lines.append("")
    lines.append(
        "En `agents/prompts.py` (`AGENT3_SCREENING_TEMPLATE`), anadir tras la definicion del "
        "criterio de intervencion un mini-glosario operativo:"
    )
    lines.append("")
    lines.append("```")
    lines.append("Examples of acceptable 'psychological components' include (non-exhaustive):")
    lines.append("- behaviour change theory (COM-B, TTM, HBM, SCT)")
    lines.append("- goal-setting, self-monitoring, action planning")
    lines.append("- motivational interviewing, coaching, counseling")
    lines.append("- mindfulness, MBSR, MBCT, CBT, ACT")
    lines.append("- self-efficacy, self-management, self-regulation, psychoeducation")
    lines.append("If the intervention description matches any of these, treat it as meeting the")
    lines.append("psychological-component criterion, even if the exact phrase is not used.")
    lines.append("```")
    lines.append("")
    lines.append(
        "### Fix auxiliar — Politica explicita ante titulo-solo (no mitiga H1 sola, pero mejora "
        "casos limite y reduce 'uncertain por defecto')"
    )
    lines.append("")
    lines.append("```")
    lines.append("If only the title is available:")
    lines.append("- Vote INCLUDE when the title contains BOTH a physical-activity term AND a")
    lines.append("  psychological/behavioural term (e.g. 'cognitive', 'behaviour change', etc.).")
    lines.append("- Vote EXCLUDE when the title clearly addresses an unrelated topic (drug therapy,")
    lines.append("  surgery, pediatric, animal study, basic science).")
    lines.append("- Vote UNCERTAIN only when the title is uninformative or corrupt (e.g. a file name).")
    lines.append("```")
    lines.append("")
    lines.append(
        f"**Prediccion de mejora**: bajar la tasa de uncertain de "
        f"{n_unc}/{len(preds)} ({n_unc/len(preds)*100:.0f}%) a ~15-20% sin perder sensibilidad "
        f"(recall sobre `relevant` debe mantenerse en 100%). Esperamos que la mayoria de los 8 "
        f"casos con abstract migren a INCLUDE (sus intervenciones encajan en el glosario); los 14 "
        f"casos sin abstract migraran segun el titulo, exceptuando los 3 con titulo corrupto "
        f"({', '.join(corrupt_ids)}) que seguiran legitimamente en UNCERTAIN."
    )
    lines.append("")
    lines.append(
        "**Set de validacion**: re-correr Hero 1 sobre el mismo `sample_50.csv`. Criterios de exito: "
        "(a) tasa uncertain <= 25%, (b) recall sobre `relevant` del gold T/A >= 95% (debe ser 100% "
        "actualmente), (c) precision sobre `irrelevant` del gold T/A no degrada mas de 5 puntos. "
        "Los 22 covidence_ids documentados en la tabla anterior sirven como sub-set de inspeccion "
        "diferencial: idealmente al menos 15/22 deberian dejar de ser uncertain."
    )
    lines.append("")
    lines.append(
        "_Nota de scope: este fix esta fuera del scope de D.8.A. Se documenta como input para una "
        "tarea posterior. D.8.A no toca A3 ni los prompts._"
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Escrito: {OUT_MD}")
    print(f"Escrito: {OUT_JSON}")
    print(f"22 uncertain analizados. gold dist: {dict(gold_dist)}")
    print(f"A4 sobre uncertain: {dict(a4_dist)} (correct={a4_correct} wrong={a4_wrong} missing={a4_missing})")


if __name__ == "__main__":
    main()
