#!/usr/bin/env python3
# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
build_arm_to_parent_mapping.py — Mapping `arm_label` -> `parent_covidence_id`.

Sprint T16.D.5.D.0: resolve 10 anomalies in `data/gold_standard_normalized/gold_metrics.csv`
where the `covidence_id` field contains a multi-arm treatment label (e.g. ``E2``,
``E3``, ``E2 (center)``, ``GC (Implementation intentions)``) instead of a real
Covidence ID (``#NNNNN``). These rows belong to multi-arm RCTs in which the
research group encoded extra intervention arms in separate rows of the
``estadisticos_deporte.xlsx`` worksheet, keeping the same per-study ``ID2``
sequence number as the parent study.

Strategy:
  1. Load ``estadisticos_deporte.xlsx`` sheet ``BASELINE`` (canonical source of
     the metrics — same rows that populate ``gold_metrics.csv``).
  2. Iterate every row whose ``CovidenceID`` does NOT match the canonical
     pattern ``^#\\d+$`` (we keep ``#B10657`` and similar valid sub-IDs out of
     the anomaly set — they already match ``startswith('#')`` in gold_metrics).
  3. For each anomalous row, identify the parent ``#NNNNN``:
       (a) Same ``ID2`` block contains a row with a canonical ``#NNNNN``  →
           the arm shares the study with that parent (HIGH confidence — the
           research group reused the same ID2 for sibling arms).
       (b) Otherwise, take the immediately preceding row whose ``CovidenceID``
           matches ``^#\\d+$`` and verify that ``N_GC`` matches (same control
           group → same study). HIGH confidence when N_GC matches; MEDIUM if
           the arm row has NaN N_GC but Outcome/Filtro are identical.
  4. Cross-check the inferred parent against ``promoting_PA_HPR_2025/sup2.pdf``
     (multi-arm descriptions for each included study) and against
     ``registro_maestro.csv`` (canonical author/year per covidence_id).
  5. Emit one CSV row per unique (arm_label, ID2) pair: this is the most
     granular unambiguous mapping, since labels like ``E2`` are reused across
     ~13 distinct studies.

Identifiers and comments in English. User-facing messages in Spanish (rule #12).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Paths (configurable via CLI; defaults match the repo layout)
# ---------------------------------------------------------------------------

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ESTADISTICOS = (
    DEFAULT_REPO_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "estadisticos_deporte.xlsx"
)
DEFAULT_GOLD_METRICS = (
    DEFAULT_REPO_ROOT / "data" / "gold_standard_normalized" / "gold_metrics.csv"
)
DEFAULT_OUTPUT = (
    DEFAULT_REPO_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "unified"
    / "arm_to_parent_mapping.csv"
)


# Regex for canonical Covidence IDs (#NNNNN). We also accept #BNNNNN-style
# sub-IDs (already represented as `#B10657`) which are NOT anomalies because
# gold_metrics keeps them with the `#` prefix.
CANONICAL_CID_RE = re.compile(r"^#[A-Za-z]?\d+$")

# Known per-study arm semantics derived from sup2.pdf cross-check.
# Maps (parent_covidence_id, arm_label) -> short Spanish description for notes.
ARM_DESCRIPTIONS = {
    # Chapman et al. (2018) #11482 — VHS vs self-generated implementation intentions
    ("#11482", "GC (Implementation intentions)"):
        "Brazo control activo: Volitional Help Sheet con implementation "
        "intentions auto-generadas (Chapman et al. 2018, sup2 p.8).",
    # Robin et al. (2019) #13036 — 2 IGs
    ("#13036", "E2"):
        "Segundo brazo experimental (IG2) de Robin et al. 2019 (sup2 p.9).",
    # Poppe et al. (2019) — RCT1 = #13154, RCT2 reported in same paper.
    # The label `#1354RCT 2` is itself a Covidence ID per registro_maestro.csv,
    # but does not match the `#NNNNN` filter. Parent assigned per task hint.
    ("#13154", "E2"):
        "Segundo brazo experimental (IG2) del RCT1 de Poppe et al. 2019 "
        "(sup2 p.9 indica IG1 y IG2 con mindfulness/prompts).",
    # Mailey et al. (2020) #13730 — 4 message-framing arms (sup2 p.11)
    ("#13730", "E2"):
        "Segundo brazo (mensaje SDT autonomia) de Mailey et al. 2020 (4 arms, sup2 p.11).",
    ("#13730", "E3"):
        "Tercer brazo (mensaje SDT intrinseca) de Mailey et al. 2020 (4 arms, sup2 p.11).",
    ("#13730", "E4"):
        "Cuarto brazo (mensaje verbal-persuasion / 'Easy') de Mailey et al. 2020 "
        "(4 arms, sup2 p.11).",
    # Lo et al. (2020) #13779 — multi-arm
    ("#13779", "E2"):
        "Segundo brazo experimental de Lo et al. 2020 (RCT individualized aerobic).",
    # Plotnikoff et al. (2013) #20047 — IG1 print, IG2 print+call (sup2 p.12)
    ("#20047", "E2 print+call"):
        "IG2 (pamphlets + counselling call) de Plotnikoff et al. 2013 ADAPT trial "
        "(sup2 p.12).",
    # Keyserling et al. (2002) #2098 — 3 arms (sup2 p.12)
    ("#2098", "E2"):
        "Segundo brazo (group counselling) de Keyserling et al. 2002 (sup2 p.12).",
    # Mouton et al. (2014) #21660 — web/center/mixed (sup2 p.13)
    ("#21660", "E2 (center)"):
        "IG2 center-based de Mouton et al. 2014 (web vs center vs combined, sup2 p.13).",
    ("#21660", "E3 (mixed)"):
        "IG3 mixed (web+center) de Mouton et al. 2014 (sup2 p.13).",
    # Kim & Kang (2006) #3123 — IG1 web, IG2 print, CG (sup2 p.16)
    ("#3123", "E2 (print)"):
        "IG2 (pamphlet impreso) de Kim & Kang 2006 (sup2 p.16).",
    # Costanzo et al. (2006) #4336 — second arm row with NaN values
    ("#4336", "E2"):
        "Segundo brazo de Costanzo et al. 2006 (fila con N/X NaN — datos "
        "presentes en estadisticos solo como placeholder estructural).",
    # Van Keulen et al. (2011) #5298 — IG1 print, IG2 call, IG3 call+print (sup2 p.18)
    ("#5298", "E2"):
        "IG2 (motivational interviewing calls) de Van Keulen et al. 2011 (sup2 p.18).",
    ("#5298", "E3"):
        "IG3 (tailored letters + calls) de Van Keulen et al. 2011 (sup2 p.18).",
    # Buman et al. (2011) #5723 — 2 arms IG/CG (sup2 p.19)
    ("#5723", "E2"):
        "Segundo brazo de Buman et al. 2011 (fila E2 con N NaN; peer-support vs CG, sup2 p.19).",
    # Strath et al. (2011) #5724 — 4 arms IG1 print, IG2 print, IG3 call+print (sup2 p.19)
    ("#5724", "E2"):
        "IG2 (pedometer + individualized education) de Strath et al. 2011 (sup2 p.19).",
    ("#5724", "E3"):
        "IG3 (IG2 + telephone feedback calls) de Strath et al. 2011 (sup2 p.19).",
    # Ruiz et al. (2012) #6100 — 2 IGs (avatar resembles vs differs)
    ("#6100", "E2"):
        "IG2 (avatar diferente al participante) de Ruiz et al. 2012 (sup2 p.20).",
    # Lin et al. (2016) #9254 — 3 arms (sup2 p.22)
    ("#9254", "E2"):
        "IG2 (face-to-face + print) de Lin et al. 2016 (sup2 p.22).",
    # Almeida et al. (2015) #9350 — 4 arms (sup2 p.23)
    ("#9350", "E2"):
        "IG2 (goal-setting + action planning) de Almeida et al. 2015 (sup2 p.23).",
    ("#9350", "E3"):
        "IG3 (interactive computer + ecological model) de Almeida et al. 2015 (sup2 p.23).",
    # Special: #1354RCT 2 is its own canonical Covidence ID (self-parent).
    # Poppe RCT 1 (#13154) and Poppe RCT 2 (#1354RCT 2) are epistemologically
    # distinct studies; we accept the unusual ID format rather than collapse.
    ("#1354RCT 2", "#1354RCT 2"):
        "Self-parent: formato ID anomalo pero estudio legitimo (Poppe et al. "
        "2019 RCT 2). Distinto de #13154 que es RCT 1 del mismo articulo.",
    ("#1354RCT 2", "E2"):
        "Segundo brazo experimental (IG2) de Poppe et al. 2019 RCT 2 "
        "(comparte ID2=46 con la fila canonica '#1354RCT 2').",
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def is_canonical_cid(value: str) -> bool:
    """True when value matches the canonical ``#NNNNN`` (or ``#B10657``) form."""
    return bool(CANONICAL_CID_RE.match(str(value).strip()))


def is_gold_metrics_anomaly(value: str) -> bool:
    """Return True when the value would be flagged anomalous in gold_metrics.

    The gold_metrics check is ``~startswith('#')``. The special case
    ``#1354RCT 2`` DOES start with ``#`` but contains a space and the suffix
    ``RCT 2``, which the task explicitly lists as the 9th anomaly.
    """
    s = str(value).strip()
    if not s.startswith("#"):
        return True
    if not CANONICAL_CID_RE.match(s):
        # starts with `#` but has spaces / RCT suffix
        return True
    return False


def find_parent_for_arm(
    df: pd.DataFrame,
    row_idx: int,
) -> tuple[Optional[str], str, str]:
    """Resolve the parent covidence_id for the anomalous row at ``row_idx``.

    Returns a tuple ``(parent_covidence_id, confidence, evidence)``.

    The lookup proceeds in two stages:
      1. Same-ID2 stage: scan all rows sharing this row's ID2 for a canonical
         ``#NNNNN`` peer. If found, confidence is HIGH (same study).
      2. Adjacency + N_GC stage: walk backwards from ``row_idx`` until we hit
         a canonical-cid row, then compare N_GC to confirm same control group.
    """
    arm_row = df.iloc[row_idx]
    id2 = arm_row["ID2"]

    # Stage 1: same-ID2 peer
    peers = df[df["ID2"] == id2]
    canonical_peers = peers[peers["CovidenceID"].astype(str).map(is_canonical_cid)]
    if not canonical_peers.empty:
        parent = canonical_peers["CovidenceID"].iloc[0]
        return (
            parent,
            "high",
            (
                "comparte ID2={id2} con fila canonica {p}; el grupo de "
                "investigacion reusa el mismo ID2 para todos los brazos del "
                "mismo estudio."
            ).format(id2=id2, p=parent),
        )

    # Stage 1b: same-ID2 peer is the special '#1354RCT 2' label (self-parent).
    # #1354RCT 2 is its own canonical Covidence ID (Poppe RCT 2), distinct
    # from #13154 (Poppe RCT 1). Arms sharing its ID2 belong to this study.
    rct2_peers = peers[peers["CovidenceID"].astype(str).str.strip() == "#1354RCT 2"]
    if not rct2_peers.empty:
        return (
            "#1354RCT 2",
            "high",
            (
                "comparte ID2={id2} con la fila canonica '#1354RCT 2' "
                "(Poppe et al. 2019 RCT 2)."
            ).format(id2=id2),
        )

    # Stage 2: nearest preceding canonical row
    prev_idx = row_idx - 1
    while prev_idx >= 0:
        cid = str(df.iloc[prev_idx]["CovidenceID"]).strip()
        if is_canonical_cid(cid):
            parent_row = df.iloc[prev_idx]
            n_gc_match = (
                pd.notna(arm_row["N_GC"])
                and pd.notna(parent_row["N_GC"])
                and float(arm_row["N_GC"]) == float(parent_row["N_GC"])
            )
            outcome_match = str(arm_row["Outcome"]).strip() == str(
                parent_row["Outcome"]
            ).strip()
            if n_gc_match:
                conf = "high"
                evidence = (
                    "N_GC={n} coincide con la fila precedente canonica "
                    "{p} (ID2={pid2}, Outcome={out!r}); brazo adicional "
                    "del mismo estudio."
                ).format(
                    n=arm_row["N_GC"],
                    p=cid,
                    pid2=parent_row["ID2"],
                    out=arm_row["Outcome"],
                )
            elif outcome_match and pd.isna(arm_row["N_GC"]):
                conf = "high"
                evidence = (
                    "Outcome={out!r} y Filtro coinciden con la fila "
                    "precedente {p} (ID2={pid2}); N_GC en blanco "
                    "(placeholder estructural)."
                ).format(out=arm_row["Outcome"], p=cid, pid2=parent_row["ID2"])
            elif outcome_match:
                conf = "medium"
                evidence = (
                    "Outcome={out!r} coincide con {p} (ID2={pid2}) pero "
                    "N_GC difiere ({nrow} vs {npar}); revisar manualmente."
                ).format(
                    out=arm_row["Outcome"],
                    p=cid,
                    pid2=parent_row["ID2"],
                    nrow=arm_row["N_GC"],
                    npar=parent_row["N_GC"],
                )
            else:
                conf = "medium"
                evidence = (
                    "Fila canonica precedente {p} (ID2={pid2}) pero "
                    "Outcome/N_GC no coinciden exactamente; asociacion "
                    "por adyacencia."
                ).format(p=cid, pid2=parent_row["ID2"])
            return cid, conf, evidence
        prev_idx -= 1

    return None, "unresolved", "no se encontro fila canonica precedente con #NNNNN"


def build_mapping(
    estadisticos_path: Path,
    gold_metrics_path: Path,
) -> list[dict[str, str]]:
    """Build the arm -> parent mapping.

    Returns a list of row dicts ready to be written to CSV.
    """
    df = pd.read_excel(estadisticos_path, sheet_name="BASELINE")

    # Verify the universe of anomalies in gold_metrics matches what we expect
    gm = pd.read_csv(gold_metrics_path)
    gm_anomalies = sorted(
        {s for s in gm["covidence_id"].astype(str).unique() if not is_canonical_cid(s)}
    )
    print(f"[INFO] gold_metrics anomalias unicas: {len(gm_anomalies)}")
    for a in gm_anomalies:
        print(f"  - {a!r}")

    # Collect anomalous rows from estadisticos (source of truth for ID2)
    rows: "OrderedDict[tuple[str, int], dict[str, str]]" = OrderedDict()
    for idx in range(len(df)):
        cid = str(df.iloc[idx]["CovidenceID"]).strip()
        if is_canonical_cid(cid):
            continue
        if not is_gold_metrics_anomaly(cid):
            continue
        id2 = df.iloc[idx]["ID2"]
        key = (cid, int(id2))
        if key in rows:
            continue
        parent, conf, evidence = find_parent_for_arm(df, idx)

        # Special case: #1354RCT 2 is its own canonical id (self-parent).
        if cid == "#1354RCT 2":
            parent = "#1354RCT 2"
            conf = "high"
            evidence = (
                "Self-parent: '#1354RCT 2' es el covidence_id legitimo "
                "de Poppe et al. 2019 RCT 2 en registro_maestro. Formato "
                "ID anomalo pero estudio resoluble; NO se colapsa con "
                "#13154 (Poppe RCT 1) porque son intervenciones distintas."
            )

        descr = ARM_DESCRIPTIONS.get((parent or "", cid), "")
        notes_parts = [evidence]
        if descr:
            notes_parts.append(descr)
        rows[key] = {
            "arm_label": cid,
            "parent_covidence_id": parent or "",
            "confidence": conf,
            "notes": " | ".join(notes_parts),
        }

    return list(rows.values())


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["arm_label", "parent_covidence_id", "confidence", "notes"]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize(rows: list[dict[str, str]]) -> None:
    counts = {"high": 0, "medium": 0, "unresolved": 0}
    for r in rows:
        counts[r["confidence"]] = counts.get(r["confidence"], 0) + 1
    print()
    print("[RESUMEN]")
    print(f"  Total filas:    {len(rows)}")
    print(f"  Confidence=high:       {counts.get('high', 0)}")
    print(f"  Confidence=medium:     {counts.get('medium', 0)}")
    print(f"  Confidence=unresolved: {counts.get('unresolved', 0)}")
    # Unique arm_label coverage
    unique_labels = {r["arm_label"] for r in rows}
    print(f"  arm_labels unicos:     {len(unique_labels)}")
    for r in rows:
        if r["confidence"] != "high":
            print(
                f"    [!] {r['arm_label']!r} -> "
                f"{r['parent_covidence_id'] or '(unresolved)'} ({r['confidence']})"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--estadisticos",
        type=Path,
        default=DEFAULT_ESTADISTICOS,
        help="Path to estadisticos_deporte.xlsx",
    )
    p.add_argument(
        "--gold-metrics",
        type=Path,
        default=DEFAULT_GOLD_METRICS,
        help="Path to gold_metrics.csv (for verification only)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.estadisticos.exists():
        print(f"[ERROR] no existe {args.estadisticos}", file=sys.stderr)
        return 1
    if not args.gold_metrics.exists():
        print(f"[ERROR] no existe {args.gold_metrics}", file=sys.stderr)
        return 1
    rows = build_mapping(args.estadisticos, args.gold_metrics)
    write_csv(rows, args.output)
    summarize(rows)
    print(f"\n[OK] CSV escrito en {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
