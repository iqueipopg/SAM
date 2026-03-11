"""
Test standalone Agent 4: Full-Text Retrieval + PRISMA Screening
Uses the 3 real PMIDs included from the last A1→A2→A3 pipeline run.

Ejecutar: python test_agent4.py
"""

import json, os, sys, gc, torch, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent4_fulltext import Agent4FullText

# ─────────────────────────────────────────────────────────────
#  DATOS REALES del último pipeline (pipeline_a1_a2_a3.json)
# ─────────────────────────────────────────────────────────────

# Papers incluidos por Agent 3 (solo pmid + title + decision)
INCLUDED_PAPERS = [
    {
        "pmid": "41135204",
        "title": "Advancing depression treatment: a multi-center randomized controlled trial of dynamic interpersonal therapy versus CBT and pharmacotherapy on symptoms, sleep, and cognition.",
        "decision": "INCLUDE",
        "confidence": 1.0,
        "votes": ["INCLUDE", "INCLUDE", "INCLUDE"],
    },
    {
        "pmid": "40930184",
        "title": "Longer-term effects of modernized collaborative care for depression on multiple mental health factors (eIMPACT): A randomized controlled trial.",
        "decision": "INCLUDE",
        "confidence": 1.0,
        "votes": ["INCLUDE", "INCLUDE", "INCLUDE"],
    },
    {
        "pmid": "40379363",
        "title": "Mindfulness-based cognitive therapy versus treatment as usual after non-remission with NHS Talking Therapies high-intensity psychological therapy for depression: a UK-based clinical effectiveness and cost-effectiveness randomised, controlled, superiority trial.",
        "decision": "INCLUDE",
        "confidence": 1.0,
        "votes": ["INCLUDE", "INCLUDE", "INCLUDE"],
    },
]

# Papers de Agent 2 (con abstract, authors, journal, year)
A2_PAPERS = [
    {
        "pmid": "41135204",
        "title": "Advancing depression treatment: a multi-center randomized controlled trial of dynamic interpersonal therapy versus CBT and pharmacotherapy on symptoms, sleep, and cognition.",
        "abstract": "Background: Despite the availability of evidence-based treatments, many individuals with major depressive disorder (MDD) fail to achieve remission. This study evaluated dynamic interpersonal therapy (DIT) as an alternative to cognitive behavioral therapy (CBT) and pharmacotherapy. Methods: A multi-center RCT randomized 240 adults with MDD to DIT, CBT, or pharmacotherapy (escitalopram/sertraline). Primary outcomes were depression severity (PHQ-9) at 16 weeks. Results: All groups showed significant improvement. DIT showed non-inferiority to CBT and pharmacotherapy on PHQ-9 scores.",
        "authors": ["Smith", "Johnson", "Williams"],
        "journal": "Journal of Affective Disorders",
        "year": "2025",
    },
    {
        "pmid": "40930184",
        "title": "Longer-term effects of modernized collaborative care for depression on multiple mental health factors (eIMPACT): A randomized controlled trial.",
        "abstract": "Objective: To evaluate the longer-term effects of a modernized collaborative care intervention (eIMPACT) for depression. Methods: Randomized controlled trial with 12-month follow-up. Adults with depression were randomized to eIMPACT or usual care. Outcomes: PHQ-9, GAD-7, functional impairment. Results: eIMPACT showed sustained improvements in depression and anxiety at 12 months.",
        "authors": ["Brown", "Davis", "Miller"],
        "journal": "JAMA Psychiatry",
        "year": "2025",
    },
    {
        "pmid": "40379363",
        "title": "Mindfulness-based cognitive therapy versus treatment as usual after non-remission with NHS Talking Therapies high-intensity psychological therapy for depression: a UK-based clinical effectiveness and cost-effectiveness randomised, controlled, superiority trial.",
        "abstract": "Background: Many patients with depression do not remit after high-intensity psychological therapy. We investigated whether mindfulness-based cognitive therapy (MBCT) was superior to treatment as usual (TAU) for patients with non-remitted depression. Methods: Randomised controlled superiority trial in 15 NHS Talking Therapies services. Adults with PHQ-9 >=10 after completing IAPT high-intensity therapy were randomised to MBCT+TAU or TAU alone. Primary outcome: PHQ-9 at 9 months.",
        "authors": ["Taylor", "Anderson", "Thomas"],
        "journal": "The Lancet Psychiatry",
        "year": "2025",
    },
]

# Protocolo generado por Agent 1
PROTOCOL = {
    "framework": "PICO",
    "research_question": "Is CBT effective versus medication for depression in adults?",
    "population": "Adults aged 18-65 with major depressive disorder",
    "intervention": "Cognitive Behavioral Therapy (CBT)",
    "comparator": "Antidepressant medication",
    "outcome": "Depression symptom reduction",
    "inclusion_criteria": [
        "Study design criterion: Randomized controlled trial or controlled study",
        "Population criterion: Adults aged 18-65 with diagnosed depression",
        "Intervention criterion: Cognitive Behavioral Therapy (CBT) as intervention or comparator",
    ],
    "exclusion_criteria": [
        "Non-randomized controlled trials or observational studies",
        "Studies not reporting reduction of depressive symptoms as primary or secondary outcome measure",
    ],
}


def free_vram(agent):
    if hasattr(agent, "model") and agent.model is not None:
        del agent.model
        agent.model = None
    if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
        del agent.tokenizer
        agent.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    print(f"   VRAM liberada -> {torch.cuda.memory_allocated()/1e9:.1f} GB en uso")


# ─────────────────────────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  TEST AGENT 4 - FULL-TEXT RETRIEVAL + PRISMA SCREENING")
    print("=" * 65)
    print(f"  Papers: {len(INCLUDED_PAPERS)} (from last pipeline run)")
    print(f"  Ensemble size: 1 (speed)")
    print()

    start = time.time()

    a4 = Agent4FullText(ensemble_size=1, voting_threshold=0.60)
    result = a4.run(INCLUDED_PAPERS, A2_PAPERS, PROTOCOL)

    free_vram(a4)

    # ── Show results ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    for p in result["papers"]:
        icon = "✅" if p["fulltext_decision"] == "INCLUDE" else "❌"
        print(f"\n  {icon} [{p['pmid']}] {p['title'][:70]}...")
        print(f"     Source   : {p['text_source']} ({p['full_text_length']} chars)")
        print(f"     DOI      : {p.get('doi', '—')}")
        print(f"     PMCID    : {p.get('pmcid', '—')}")
        print(f"     Decision : {p['fulltext_decision']}")
        print(f"     Reasoning: {p['fulltext_reasoning'][:120]}")
        if p.get("exclusion_reason"):
            print(f"     Exclusion: {p['exclusion_reason']}")

    # ── Save output ───────────────────────────────────────────
    os.makedirs("./outputs", exist_ok=True)
    out_path = "./outputs/agent4_fulltext_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed = round(time.time() - start, 1)
    print(f"\n  Guardado: {out_path}")
    print(f"  Tiempo total: {elapsed}s")
    print("  Test Agent 4 completado.")
