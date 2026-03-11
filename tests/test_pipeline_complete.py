"""
Pipeline completo: Agent 1 → Agent 2 → Agent 3 → Agent 4
Ejecutar desde raíz: python test_pipeline_complete.py
"""

import json, torch, gc, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent1_protocol import Agent1Protocol
from agents.agent2_search import Agent2Search
from agents.agent3_screening import Agent3Screening
from agents.agent4_fulltext import Agent4FullText

# ─────────────────────────────────────────────
#  CONFIGURACIÓN  (solo cambia USER_QUERY)
# ─────────────────────────────────────────────
USER_QUERY = "¿TCC versus medicación para depresión en adultos?"
MAX_RESULTS = 30
DEDUP_THR = 0.90
ENSEMBLE = 3
VOTE_THR = 0.60
A4_ENSEMBLE = 1  # Full texts are long → 1 vote for speed


def free_vram(agent):
    if hasattr(agent, "model") and agent.model is not None:
        del agent.model
        agent.model = None
    if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
        del agent.tokenizer
        agent.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    print(f"   🧹 VRAM liberada → {torch.cuda.memory_allocated()/1e9:.1f} GB en uso")


def run_pipeline(user_query: str):

    print("\n" + "=" * 65)
    print("  PIPELINE META-ANÁLISIS: A1 → A2 → A3 → A4")
    print("=" * 65)
    print(f"  Query: {user_query}\n")

    # ── AGENT 1 ────────────────────────────────────────────────────
    print("▶ AGENT 1 - Generando protocolo...")
    a1 = Agent1Protocol()
    protocol = a1.generate_protocol(user_query)

    print(f"   Framework   : {protocol.get('framework')}")
    print(f"   Intervention: {protocol.get('intervention', '')}")
    print(f"   Comparator  : {protocol.get('comparator', '')}")
    print(f"   Outcome     : {protocol.get('outcome', '')}")
    print(f"   Inclusión   : {len(protocol.get('inclusion_criteria',[]))} criterios")
    print(f"   Exclusión   : {len(protocol.get('exclusion_criteria',[]))} criterios")

    # Query PubMed generada con el modelo AÚN en memoria
    pubmed_query = a1.generate_pubmed_query(user_query, protocol)
    print(f"   PubMed query: '{pubmed_query}'")

    free_vram(a1)  # liberar DESPUÉS de generate_pubmed_query

    # ── AGENT 2 ────────────────────────────────────────────────────
    print("\n▶ AGENT 2 - Búsqueda PubMed...")
    a2 = Agent2Search()
    papers = a2.search_pubmed(pubmed_query, max_results=MAX_RESULTS)
    papers = a2.deduplicate(papers, threshold=DEDUP_THR)
    print(f"   ✅ {len(papers)} papers únicos")

    if not papers:
        print("   ⚠️  0 papers. Revisa la query PubMed de arriba.")
        return None

    # ── AGENT 3 ────────────────────────────────────────────────────
    print(f"\n▶ AGENT 3 - Screening título/abstract...")
    est = round(len(papers) * ENSEMBLE * 15 / 60, 1)
    print(f"   Estimado: ~{est} min ({len(papers)} papers × {ENSEMBLE} votos × ~15s)")

    a3 = Agent3Screening(ensemble_size=ENSEMBLE, voting_threshold=VOTE_THR)
    results = a3.screen_batch(papers, protocol)

    s3 = results["stats"]
    print("\n" + "-" * 50)
    print(f"  A3 Screening: {s3['included']} INCLUDE, "
          f"{s3['excluded']} EXCLUDE, {s3['uncertain']} UNCERTAIN")

    free_vram(a3)  # liberar antes de A4

    # ── AGENT 4 ────────────────────────────────────────────────────
    included_papers = results["include"]

    if not included_papers:
        print("\n  ⚠️  No papers passed A3 screening. Skipping A4.")
        a4_result = None
    else:
        print(f"\n▶ AGENT 4 - Full-text retrieval + PRISMA screening...")
        print(f"   Papers pasados A3: {len(included_papers)}")

        a4 = Agent4FullText(ensemble_size=A4_ENSEMBLE, voting_threshold=VOTE_THR)

        # Phase A: retrieval (no GPU)
        # Phase B: full-text screening (loads GPU)
        a4_result = a4.run(included_papers, papers, protocol)

        free_vram(a4)

    # ── PRISMA FLOW ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  📊 PRISMA FLOW (Agents 1-4)")
    print("=" * 65)
    print(f"  Identificados (A2)           : {s3['total']}")
    print(f"  Excl. título/abstract (A3)   : {s3['excluded']}")
    print(f"  Inciertos (A3)               : {s3['uncertain']}  ← revisión humana")
    print(f"  Pasados a full-text (A3→A4)  : {s3['included']}")

    if a4_result:
        s4 = a4_result["stats"]
        print(f"  Excl. full-text (A4)         : {s4['fulltext_excluded']}")
        print(f"  ✅ INCLUIDOS FINAL (A4)      : {s4['fulltext_included']}")
        print(f"  Retrieval: PMC={s4['pmc_fulltext']}, "
              f"Unpaywall={s4['unpaywall_pdf']}, "
              f"Abstract={s4['abstract_only']}")
    else:
        print(f"  ✅ INCLUIDOS FINAL           : 0")

    print(f"  ⏱️  Tiempo A3                 : {s3['elapsed_s']}s")
    if a4_result:
        print(f"  ⏱️  Tiempo A4                 : {s4.get('elapsed_s', 0)}s")

    # ── GUARDAR ────────────────────────────────────────────────────
    os.makedirs("./outputs", exist_ok=True)
    output = {
        "query": user_query,
        "protocol": protocol,
        "pubmed_query": pubmed_query,
        "prisma": {
            "identified": s3["total"],
            "screened_title_abstract": s3["total"],
            "excluded_title_abstract": s3["excluded"],
            "uncertain_title_abstract": s3["uncertain"],
            "assessed_fulltext": s3["included"],
            "excluded_fulltext": a4_result["stats"]["fulltext_excluded"] if a4_result else 0,
            "included_final": a4_result["stats"]["fulltext_included"] if a4_result else 0,
        },
        "screening": results,
        "fulltext_retrieval": a4_result,
    }
    out_path = "./outputs/pipeline_a1_a2_a3_a4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Guardado: {out_path}")
    print("  🏁 Pipeline completado.")
    return output


if __name__ == "__main__":
    run_pipeline(USER_QUERY)
