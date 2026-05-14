"""Dedup threshold sweep for A2 semantic deduplication.

v2 — Extended sweep with larger pool (min_per_db=50) plus 5 synthetic
identifiable pairs injected to validate the algorithm under controlled
conditions.

Sweeps the semantic dedup threshold of Agent2Search across a fixed grid
{0.85, 0.88, 0.90, 0.92, 0.95, 0.98} on a shared raw cross-source pool
(PubMed + Europe PMC) AUGMENTED with 5 synthetic paper pairs covering
different similarity targets (0.99 down to 0.80 negative control). The
synthetic pairs have empty PMIDs and unique DOIs so they always reach
Tier 3 (semantic similarity) of the dedup.

The script reuses the reference run's user query
("mindfulness for anxiety adolescents") to keep the experiment aligned
with `outputs/run_20260507_081359/`. Semantic Scholar and Google Scholar
are disabled to avoid CAPTCHA / 429 / non-determinism. The raw pool is
persisted to `outputs/exp_bloque_b/a2_raw_predudup_n50.json` before the
sweep so the experiment is reproducible without re-hitting the network.

Backward compatibility:
  - v1 cache path `a2_raw_predudup.json` is still read as a fallback.
  - JSON output keeps `raw_total`, `sweep[*].threshold/unique/collapsed_pairs`.
  - CSV keeps `threshold,unique,collapsed_pairs` as the first 3 columns.

Run from the repository root:
    ~/meta-analisis-automatizado/venv/bin/python scripts/exp_dedup_threshold.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# Ensure the repository root is importable when running the script directly
# (e.g. `python scripts/exp_dedup_threshold.py`).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Hardcoded paths relative to the repository root (cwd at execution time).
REFERENCE_RUN_DIR = Path("outputs/run_20260507_081359")
A1_CHECKPOINT_PATH = REFERENCE_RUN_DIR / "checkpoint_a1_protocol.json"

OUTPUT_DIR = Path("outputs/exp_bloque_b")
# v2 raw cache (n=50 per DB). The v1 cache (n=15 per DB) is kept on disk
# for archival but is no longer consulted by this script.
RAW_CACHE_PATH = OUTPUT_DIR / "a2_raw_predudup_n50.json"
CSV_PATH = OUTPUT_DIR / "dedup_threshold.csv"
JSON_PATH = OUTPUT_DIR / "dedup_threshold.json"
PNG_PATH = OUTPUT_DIR / "dedup_threshold.png"

# Threshold grid for the sweep (ascending order).
THRESHOLD_GRID = [0.85, 0.88, 0.90, 0.92, 0.95, 0.98]

# Floor of papers per database for the cross-source pool. v2 uses 50 to
# enlarge the natural pool; v1 used 15.
MIN_PER_DB = 50

# Reference query (matches outputs/run_20260507_081359 user query). The
# A1 checkpoint stores it under the "query" key.
FALLBACK_QUERY = "Does mindfulness reduce anxiety in adolescents?"


# ---------------------------------------------------------------------------
# Synthetic pairs design — 5 pairs covering similarity targets 0.99 -> 0.80.
# Each pair contributes 2 papers to the combined pool. PMIDs are left empty
# so Tier 1 (PMID match) never fires; DOIs are unique so Tier 2 (DOI match)
# never fires. This forces every collapse to go through Tier 3 (semantic
# similarity), which is what we want to characterize.
# ---------------------------------------------------------------------------

SYNTHETIC_PAIR_DESIGN = [
    {"id": 1, "target_similarity": 0.99, "description": "casing+whitespace"},
    {"id": 2, "target_similarity": 0.95, "description": "subtitle added"},
    {"id": 3, "target_similarity": 0.93, "description": "diacritics in authors"},
    {"id": 4, "target_similarity": 0.90, "description": "word reorder in title"},
    {"id": 5, "target_similarity": 0.80, "description": "negative control - different papers"},
]


# Abstract bodies (~150-200 words each, plain ASCII to avoid surprises).
_ABSTRACT_1 = (
    "Background: Adolescent anxiety disorders are highly prevalent and "
    "associated with substantial functional impairment. Mindfulness-Based "
    "Stress Reduction has emerged as a promising intervention, but evidence "
    "in this age group remains heterogeneous. Methods: We conducted a "
    "randomized controlled trial in 184 adolescents aged 13 to 17 with "
    "moderate to severe anxiety symptoms. Participants were assigned to an "
    "eight week MBSR program or to a waitlist control condition. The primary "
    "outcome was change in the Generalized Anxiety Disorder seven item scale "
    "from baseline to post intervention. Secondary outcomes included "
    "perceived stress, sleep quality, and self reported mindfulness. "
    "Results: MBSR was associated with a clinically meaningful reduction in "
    "anxiety symptoms compared to the control arm. Improvements were "
    "maintained at the three month follow up assessment. Engagement was "
    "high and no serious adverse events were reported. Conclusions: An "
    "eight week MBSR program produces durable reductions in anxiety in "
    "adolescents and constitutes a viable scalable intervention for school "
    "and community settings. Implementation considerations and dosing "
    "trade offs are discussed."
)

_ABSTRACT_2 = (
    "Background: Cognitive Behavioral Therapy is widely recommended for "
    "depressive disorders in young people, yet effect sizes vary across "
    "trials. We synthesized contemporary evidence on CBT for youth "
    "depression to clarify magnitude and moderators of effect. Methods: "
    "Eligible randomized trials enrolled participants aged eight to "
    "eighteen with a primary diagnosis of unipolar depression. We pooled "
    "effects on standardized depression symptom scales using random "
    "effects models and explored heterogeneity by format, duration, and "
    "comparator type. Results: Twenty four trials with 3120 participants "
    "met inclusion criteria. CBT showed a moderate reduction in depressive "
    "symptoms compared to waitlist controls and a smaller reduction "
    "compared to active controls. Effects were larger in adolescent "
    "samples than in children and were sustained at six month follow up. "
    "Heterogeneity was substantial and partly explained by therapist "
    "training. Conclusions: CBT remains an effective first line "
    "psychological treatment for youth depression. Future trials should "
    "report fidelity and dosing in detail to support implementation."
)

_ABSTRACT_3 = (
    "Background: Test anxiety affects a substantial proportion of "
    "secondary school students and undermines academic performance and "
    "wellbeing. Brief school delivered mindfulness programs may offer "
    "scalable relief. Methods: This cluster randomized trial enrolled 312 "
    "students across eight schools. Schools were randomized to a six "
    "session mindfulness program embedded in tutorial time or to usual "
    "tutorial activities. The primary outcome was the Westside Test "
    "Anxiety Scale, measured before and after the intervention period. "
    "Secondary outcomes included academic self efficacy and emotion "
    "regulation. Results: Students in the mindfulness arm reported "
    "significant reductions in test anxiety relative to the control arm "
    "with a small to moderate effect size. Improvements in emotion "
    "regulation accompanied the anxiety reductions. Teacher fidelity was "
    "high and student engagement was rated positively. Conclusions: "
    "Short school based mindfulness programs reduce test anxiety in "
    "secondary students and can be feasibly delivered within existing "
    "tutorial structures."
)

_ABSTRACT_4 = (
    "Background: Acceptance and Commitment Therapy is increasingly used "
    "for adolescent anxiety, yet most evidence comes from adult samples. "
    "We conducted a pilot trial to evaluate feasibility, acceptability, "
    "and preliminary efficacy of ACT for teen anxiety. Methods: Forty "
    "two adolescents aged 14 to 18 with elevated anxiety symptoms were "
    "randomized to ten sessions of group ACT or to a psychoeducation "
    "control. Feasibility outcomes included recruitment, retention, and "
    "session attendance. Clinical outcomes included anxiety symptoms, "
    "psychological flexibility, and quality of life. Results: "
    "Recruitment and retention exceeded prespecified thresholds. "
    "Adolescents reported high acceptability and engagement. The ACT arm "
    "showed larger reductions in anxiety and larger gains in "
    "psychological flexibility than the control arm, with effect sizes "
    "in the moderate range. Conclusions: ACT is feasible and acceptable "
    "as a group intervention for adolescent anxiety. A fully powered "
    "trial is warranted to confirm efficacy and identify mediators."
)

_ABSTRACT_5A = (
    "Background: Yoga is increasingly proposed as an adjunctive "
    "intervention for generalized anxiety in adolescents, yet rigorous "
    "evidence is limited. Methods: We randomized 96 adolescents with "
    "generalized anxiety to twelve weeks of supervised hatha yoga or to "
    "an active stretching control. Outcomes included anxiety severity, "
    "physiological stress markers, and sleep quality. Results: The yoga "
    "arm showed greater reductions in self reported anxiety and morning "
    "cortisol than the control arm. Adherence was high and no adverse "
    "events were reported. Conclusions: Supervised yoga is a feasible "
    "adjunctive intervention for adolescent generalized anxiety and "
    "produces measurable physiological and clinical benefits over a "
    "structured stretching comparator. Larger trials are needed to "
    "examine durability of effects and cost effectiveness in routine "
    "care settings, particularly for school based and community "
    "delivery models that emphasize scalability and equitable access."
)

_ABSTRACT_5B = (
    "Background: Pharmacological treatment of pediatric obsessive "
    "compulsive disorder relies heavily on selective serotonin reuptake "
    "inhibitors, but real world prescribing patterns are poorly "
    "characterized. Methods: We conducted a retrospective cohort study "
    "of 4218 children and adolescents aged six to seventeen with newly "
    "diagnosed OCD using electronic health records from a regional "
    "network. We extracted index SSRI, dose trajectory, augmentation "
    "patterns, and discontinuation reasons. Results: Sertraline and "
    "fluoxetine accounted for most index prescriptions. Median time to "
    "target dose was longer than guidelines recommend, and "
    "augmentation with antipsychotics occurred in a non trivial "
    "minority of cases despite limited supporting evidence. "
    "Conclusions: Pediatric OCD prescribing diverges from guideline "
    "recommendations in dose escalation and augmentation. System level "
    "decision support and clinician education may reduce variability "
    "and improve outcomes for children and adolescents with OCD."
)


def build_synthetic_pairs() -> list[dict]:
    """Return the 10 synthetic papers (5 pairs * 2 roles).

    Design notes per pair:
      - Pair 1 (~0.99): identical title content but different casing and
        trailing whitespace; same authors with different casing. Should
        ALWAYS collapse, even at threshold 0.85.
      - Pair 2 (~0.95): _b adds a subtitle ("A Systematic Review and Meta
        Analysis") to _a's title; abstracts and authors identical.
      - Pair 3 (~0.93): identical titles and abstracts; only difference is
        author rendering (diacritics vs ASCII, commas vs spaces).
      - Pair 4 (~0.90): word order in title rearranged; abstract identical.
      - Pair 5 (~0.80): negative control. Different titles and different
        abstracts. Should NEVER collapse.

    PMIDs are left empty so Tier 1 (exact PMID) never fires. DOIs are
    unique per paper (10.9999/synthetic.<id><role>) so Tier 2 (exact DOI)
    never fires. Every collapse must therefore be driven by Tier 3
    (semantic similarity), which is what we want to characterize.
    """
    pairs: list[dict] = []

    # ----- Pair 1: casing + whitespace (target ~0.99) -----
    p1_authors_a = ["Smith J", "Lee K"]
    p1_authors_b = ["SMITH j", "LEE k"]
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.1a",
            "title": "Mindfulness-Based Stress Reduction for Adolescent Anxiety: A Randomized Trial",
            "abstract": _ABSTRACT_1,
            "authors": p1_authors_a,
            "journal": "JAMA Pediatrics",
            "year": "2024",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 1,
            "_synthetic_role": "a",
        }
    )
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.1b",
            "title": "MINDFULNESS-BASED STRESS REDUCTION for ADOLESCENT ANXIETY: A Randomized Trial   ",
            "abstract": _ABSTRACT_1,
            "authors": p1_authors_b,
            "journal": "JAMA Pediatrics",
            "year": "2024",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 1,
            "_synthetic_role": "b",
        }
    )

    # ----- Pair 2: subtitle added (target ~0.95) -----
    p2_authors = ["Brown A", "Davis R", "Wilson T"]
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.2a",
            "title": "Cognitive Behavioral Therapy in Youth Depression",
            "abstract": _ABSTRACT_2,
            "authors": p2_authors,
            "journal": "Journal of Child Psychology",
            "year": "2023",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 2,
            "_synthetic_role": "a",
        }
    )
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.2b",
            "title": "Cognitive Behavioral Therapy in Youth Depression: A Systematic Review and Meta-Analysis",
            "abstract": _ABSTRACT_2,
            "authors": p2_authors,
            "journal": "Journal of Child Psychology",
            "year": "2023",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 2,
            "_synthetic_role": "b",
        }
    )

    # ----- Pair 3: diacritics in authors only (target ~0.93) -----
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.3a",
            "title": "School-Based Mindfulness Interventions Reduce Test Anxiety",
            "abstract": _ABSTRACT_3,
            "authors": ["Garcia-Lopez, M.", "Nunez, P.", "Sanchez, R."],
            "journal": "School Mental Health",
            "year": "2025",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 3,
            "_synthetic_role": "a",
        }
    )
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.3b",
            "title": "School-Based Mindfulness Interventions Reduce Test Anxiety",
            "abstract": _ABSTRACT_3,
            "authors": ["Garcia Lopez M", "Nunez P", "Sanchez R"],
            "journal": "School Mental Health",
            "year": "2025",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 3,
            "_synthetic_role": "b",
        }
    )

    # ----- Pair 4: word reorder in title (target ~0.90) -----
    p4_authors = ["Martin H", "Clark E"]
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.4a",
            "title": "A Pilot Study of Acceptance and Commitment Therapy for Teen Anxiety",
            "abstract": _ABSTRACT_4,
            "authors": p4_authors,
            "journal": "Behaviour Research and Therapy",
            "year": "2022",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 4,
            "_synthetic_role": "a",
        }
    )
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.4b",
            "title": "Acceptance and Commitment Therapy for Teen Anxiety: A Pilot Study",
            "abstract": _ABSTRACT_4,
            "authors": p4_authors,
            "journal": "Behaviour Research and Therapy",
            "year": "2022",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 4,
            "_synthetic_role": "b",
        }
    )

    # ----- Pair 5: negative control, genuinely different papers (target ~0.80) -----
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.5a",
            "title": "Yoga as Adjunct Treatment for Generalized Anxiety in Adolescents",
            "abstract": _ABSTRACT_5A,
            "authors": ["Patel S", "Ng D"],
            "journal": "Complementary Therapies in Medicine",
            "year": "2023",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 5,
            "_synthetic_role": "a",
        }
    )
    pairs.append(
        {
            "pmid": "",
            "doi": "10.9999/synthetic.5b",
            "title": "Pharmacological Treatment Patterns in Pediatric OCD",
            "abstract": _ABSTRACT_5B,
            "authors": ["O'Connor M", "Hughes B"],
            "journal": "Journal of Clinical Psychiatry",
            "year": "2024",
            "sources": ["synthetic"],
            "_synthetic": True,
            "_synthetic_pair_id": 5,
            "_synthetic_role": "b",
        }
    )

    return pairs


# ---------------------------------------------------------------------------
# Raw pool: load / fetch / cache.
# ---------------------------------------------------------------------------

def load_reference_query() -> tuple[str, str, dict]:
    """Return (user_query, pubmed_query, protocol) from the reference A1 checkpoint.

    Falls back to the documented experiment query if the checkpoint is
    missing or malformed.
    """
    try:
        with A1_CHECKPOINT_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        user_query = data.get("query") or FALLBACK_QUERY
        pubmed_query = data.get("pubmed_query") or user_query
        protocol = data.get("protocol") or {}
        return user_query, pubmed_query, protocol
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[exp_dedup] No se pudo leer A1 checkpoint ({exc}); usando query fallback")
        return FALLBACK_QUERY, FALLBACK_QUERY, {}


def fetch_raw_pool(agent, pubmed_query: str, search_terms: str, max_per_db: int) -> list[dict]:
    """Fetch the raw cross-source pool from PubMed + Europe PMC only.

    S2 and Google Scholar are intentionally skipped to keep the experiment
    deterministic and avoid CAPTCHA / 429 throttling.
    """
    raw = []

    print(f"[exp_dedup] PubMed query: '{pubmed_query[:120]}'")
    try:
        pubmed_papers = agent._search_pubmed(pubmed_query, max_results=max_per_db)
    except Exception as exc:
        print(f"[exp_dedup] PubMed search failed: {exc}")
        pubmed_papers = []
    print(f"[exp_dedup] PubMed: {len(pubmed_papers)} papers")
    raw.extend(pubmed_papers)

    print(f"[exp_dedup] EPMC terms: '{search_terms[:120]}'")
    try:
        epmc_papers = agent._search_europepmc(search_terms, max_results=max_per_db)
    except Exception as exc:
        print(f"[exp_dedup] Europe PMC search failed: {exc}")
        epmc_papers = []
    print(f"[exp_dedup] Europe PMC: {len(epmc_papers)} papers")
    raw.extend(epmc_papers)

    return raw


def load_or_fetch_raw_pool() -> list[dict]:
    """Load cached raw pool if it exists, otherwise fetch and persist it.

    Only the v2 cache `a2_raw_predudup_n50.json` is consulted. The v1
    cache (n=15 per DB) is intentionally not used as a fallback because
    v2 requires the larger natural pool. When the v2 cache is missing
    we run A2 live with MIN_PER_DB=50.
    """
    if RAW_CACHE_PATH.exists():
        with RAW_CACHE_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        raw = payload.get("papers") or []
        print(f"[exp_dedup] Cache encontrado: {RAW_CACHE_PATH} ({len(raw)} papers)")
        return raw

    print(f"[exp_dedup] Cache no encontrado, ejecutando A2 (PubMed + EPMC) con min_per_db={MIN_PER_DB} ...")
    # Late import to avoid pulling sentence-transformers when only reading
    # the cache.
    from agents.agent2_search import Agent2Search

    agent = Agent2Search(min_per_db=MIN_PER_DB)

    user_query, pubmed_query, protocol = load_reference_query()
    if protocol:
        search_terms = agent._build_search_terms(protocol)
    else:
        search_terms = user_query

    raw = fetch_raw_pool(
        agent,
        pubmed_query=pubmed_query,
        search_terms=search_terms,
        max_per_db=MIN_PER_DB,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_run": str(REFERENCE_RUN_DIR),
        "user_query": user_query,
        "pubmed_query": pubmed_query,
        "search_terms": search_terms,
        "min_per_db": MIN_PER_DB,
        "databases": ["pubmed", "europepmc"],
        "n_raw": len(raw),
        "papers": raw,
    }
    with RAW_CACHE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[exp_dedup] Cache guardado: {RAW_CACHE_PATH} ({len(raw)} papers)")
    return raw


# ---------------------------------------------------------------------------
# Sweep.
# ---------------------------------------------------------------------------

def _count_pair_survivors(survivors: list[dict]) -> Counter:
    """Count how many surviving entries each synthetic pair has."""
    counts: Counter = Counter()
    for p in survivors:
        if p.get("_synthetic"):
            pid = p.get("_synthetic_pair_id")
            if pid is not None:
                counts[pid] += 1
    return counts


def run_threshold_sweep(raw_papers: list[dict]) -> list[dict]:
    """Run dedup() at each threshold and return per-threshold metrics.

    Backward-compatible fields (v1):
      - threshold
      - unique
      - collapsed_pairs   (raw_total - unique)

    New fields (v2, only meaningful when synthetic pairs are in the pool):
      - unique_natural          number of non-synthetic survivors
      - collapsed_natural       (n_natural_raw) - unique_natural
      - synthetic_pairs_collapsed  list[int] of pair_ids with <2 survivors
      - synthetic_pairs_surviving  list[int] of pair_ids with exactly 2 survivors
    """
    # Late import to keep CLI startup cheap if the cache exists.
    from agents.agent2_search import Agent2Search

    raw_total = len(raw_papers)
    if raw_total == 0:
        print("[exp_dedup] Pool raw vacio; no se puede ejecutar el sweep")
        return []

    # Pre-compute pool composition so we can derive natural-only metrics.
    n_synth_raw = sum(1 for p in raw_papers if p.get("_synthetic"))
    n_natural_raw = raw_total - n_synth_raw

    agent = Agent2Search(min_per_db=MIN_PER_DB)

    sweep_rows = []
    for threshold in THRESHOLD_GRID:
        print(f"[exp_dedup] Sweep threshold={threshold:.2f} ...")
        # Deep-ish copy of the raw list so each call sees pristine inputs;
        # `_merge_paper` mutates the kept paper in-place, which would
        # otherwise contaminate subsequent threshold iterations.
        papers_copy = [dict(p) for p in raw_papers]
        unique = agent.deduplicate(papers_copy, threshold=threshold)
        n_unique = len(unique)
        collapsed = raw_total - n_unique

        # v2 metrics.
        unique_natural = sum(1 for p in unique if not p.get("_synthetic"))
        collapsed_natural = n_natural_raw - unique_natural

        pair_counts = _count_pair_survivors(unique)
        pair_ids_in_pool = sorted({p.get("_synthetic_pair_id") for p in raw_papers if p.get("_synthetic")})
        synthetic_pairs_collapsed = [pid for pid in pair_ids_in_pool if pair_counts.get(pid, 0) < 2]
        synthetic_pairs_surviving = [pid for pid in pair_ids_in_pool if pair_counts.get(pid, 0) >= 2]

        sweep_rows.append(
            {
                "threshold": threshold,
                "unique": n_unique,
                "collapsed_pairs": collapsed,
                "unique_natural": unique_natural,
                "collapsed_natural": collapsed_natural,
                "synthetic_pairs_collapsed": synthetic_pairs_collapsed,
                "synthetic_pairs_surviving": synthetic_pairs_surviving,
            }
        )
        print(
            f"[exp_dedup]   -> unique={n_unique} collapsed_pairs={collapsed} "
            f"unique_natural={unique_natural} collapsed_natural={collapsed_natural} "
            f"synth_collapsed={synthetic_pairs_collapsed}"
        )
    return sweep_rows


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    """Write the sweep table as CSV.

    v1 columns (preserved first): threshold, unique, collapsed_pairs.
    v2 columns appended:          unique_natural, collapsed_natural,
                                  synthetic_pairs_collapsed (pipe-joined).
    """
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "threshold",
                "unique",
                "collapsed_pairs",
                "unique_natural",
                "collapsed_natural",
                "synthetic_pairs_collapsed",
            ]
        )
        for row in rows:
            synth_str = "|".join(str(pid) for pid in row.get("synthetic_pairs_collapsed", []))
            writer.writerow(
                [
                    f"{row['threshold']:.2f}",
                    row["unique"],
                    row["collapsed_pairs"],
                    row.get("unique_natural", ""),
                    row.get("collapsed_natural", ""),
                    synth_str,
                ]
            )


def write_json(
    raw_total: int,
    rows: list[dict],
    path: Path,
    *,
    raw_natural: int | None = None,
    raw_synthetic: int | None = None,
) -> None:
    """Write the sweep table as JSON.

    v1 keys preserved: raw_total, sweep[*].threshold/unique/collapsed_pairs.
    v2 keys added when synthetic pairs are used: raw_natural, raw_synthetic,
    sweep[*].unique_natural/collapsed_natural/synthetic_pairs_collapsed/
    synthetic_pairs_surviving, and synthetic_pair_design.
    """
    sweep_payload: list[dict] = []
    for row in rows:
        entry = {
            "threshold": row["threshold"],
            "unique": row["unique"],
            "collapsed_pairs": row["collapsed_pairs"],
        }
        # Add v2 fields only when present (preserves v1-only outputs in tests).
        if "unique_natural" in row:
            entry["unique_natural"] = row["unique_natural"]
        if "collapsed_natural" in row:
            entry["collapsed_natural"] = row["collapsed_natural"]
        if "synthetic_pairs_collapsed" in row:
            entry["synthetic_pairs_collapsed"] = row["synthetic_pairs_collapsed"]
        if "synthetic_pairs_surviving" in row:
            entry["synthetic_pairs_surviving"] = row["synthetic_pairs_surviving"]
        sweep_payload.append(entry)

    payload: dict = {
        "raw_total": raw_total,
        "sweep": sweep_payload,
    }
    if raw_natural is not None:
        payload["raw_natural"] = raw_natural
    if raw_synthetic is not None:
        payload["raw_synthetic"] = raw_synthetic
        # Only include the design block when synthetic pairs were used.
        if raw_synthetic > 0:
            payload["synthetic_pair_design"] = SYNTHETIC_PAIR_DESIGN

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def plot_sweep(rows: list[dict], raw_total: int, path: Path) -> None:
    """Render two-curve plot of collapsed counts vs threshold.

    Curve 1 (tab:blue):   collapsed_natural per threshold.
    Curve 2 (tab:orange): #synthetic_pairs_collapsed per threshold.

    Falls back to a single 'unique vs threshold' curve when v2 metrics are
    absent (legacy v1 path, preserved for backward compatibility).
    """
    thresholds = [row["threshold"] for row in rows]

    has_v2 = rows and "collapsed_natural" in rows[0]

    if has_v2:
        collapsed_nat = [row["collapsed_natural"] for row in rows]
        synth_collapsed = [len(row.get("synthetic_pairs_collapsed", [])) for row in rows]
        n_natural_raw = rows[0].get("unique_natural", 0) + rows[0].get("collapsed_natural", 0)
        n_synth_raw = raw_total - n_natural_raw

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            thresholds,
            collapsed_nat,
            color="tab:blue",
            marker="o",
            linewidth=2,
            markersize=8,
            label="Duplicados naturales colapsados",
        )
        ax.plot(
            thresholds,
            synth_collapsed,
            color="tab:orange",
            marker="s",
            linewidth=2,
            markersize=8,
            label="Pares sinteticos colapsados",
        )

        # Numeric labels per marker.
        for x, y in zip(thresholds, collapsed_nat):
            ax.annotate(
                str(y), xy=(x, y), xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=9, color="tab:blue",
            )
        for x, y in zip(thresholds, synth_collapsed):
            ax.annotate(
                str(y), xy=(x, y), xytext=(0, -14),
                textcoords="offset points", ha="center", va="top",
                fontsize=9, color="tab:orange",
            )

        ax.set_xlabel("Threshold de similitud semantica")
        ax.set_ylabel("Numero colapsado")
        ax.set_title(
            f"Sweep de threshold de dedup - pool ampliado (n_real={n_natural_raw}, n_synth={n_synth_raw})"
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        if thresholds:
            x_pad = (max(thresholds) - min(thresholds)) * 0.06 or 0.01
            ax.set_xlim(min(thresholds) - x_pad, max(thresholds) + x_pad)

        all_y = collapsed_nat + synth_collapsed
        if all_y:
            y_min = min(all_y)
            y_max = max(all_y)
            if y_min == y_max:
                ax.set_ylim(y_min - 1, y_max + 2)
            else:
                span = y_max - y_min
                ax.set_ylim(max(0, y_min - span * 0.20), y_max + span * 0.30)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    # ----- Legacy v1 path (single curve), kept for backward compat. -----
    uniques = [row["unique"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        thresholds,
        uniques,
        color="#0D9488",
        marker="o",
        linewidth=2,
        markersize=8,
    )
    if uniques:
        y_range = max(uniques) - min(uniques) if max(uniques) != min(uniques) else 1
        y_offset = max(0.3, y_range * 0.04)
        for x, y in zip(thresholds, uniques):
            ax.annotate(
                str(y),
                xy=(x, y),
                xytext=(0, y_offset * 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.set_xlabel("Threshold de similitud semantica")
    ax.set_ylabel("Numero de papers unicos")
    ax.set_title(
        f"Sweep de threshold de dedup semantica - A2 raw cross-source (n_raw={raw_total})"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    if thresholds:
        x_pad = (max(thresholds) - min(thresholds)) * 0.06 or 0.01
        ax.set_xlim(min(thresholds) - x_pad, max(thresholds) + x_pad)
    if uniques:
        y_min = min(uniques)
        y_max = max(uniques)
        if y_min == y_max:
            ax.set_ylim(y_min - 1, y_max + 2)
        else:
            span = y_max - y_min
            ax.set_ylim(y_min - span * 0.15, y_max + span * 0.30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_natural = load_or_fetch_raw_pool()
    n_natural = len(raw_natural)
    if n_natural == 0:
        print("[exp_dedup] Pool raw natural vacio; abortando sweep")
        return

    # Inject the 10 synthetic papers.
    synthetic = build_synthetic_pairs()
    n_synth = len(synthetic)
    pool_combined = list(raw_natural) + synthetic
    raw_total = len(pool_combined)
    print(
        f"[exp_dedup] Pool combinado: {n_natural} naturales + {n_synth} sinteticos "
        f"= {raw_total} papers"
    )

    sweep_rows = run_threshold_sweep(pool_combined)
    if not sweep_rows:
        print("[exp_dedup] Sweep vacio; abortando exportacion")
        return

    write_csv(sweep_rows, CSV_PATH)
    write_json(
        raw_total,
        sweep_rows,
        JSON_PATH,
        raw_natural=n_natural,
        raw_synthetic=n_synth,
    )
    plot_sweep(sweep_rows, raw_total, PNG_PATH)

    # Console summary (Spanish for user-facing output).
    print()
    print(f"Pool raw total: {raw_total} (naturales={n_natural}, sinteticos={n_synth})")
    print(
        f"{'threshold':>10} {'unique':>7} {'collapsed':>10} "
        f"{'u_natural':>10} {'c_natural':>10} {'synth_collapsed':>18}"
    )
    for row in sweep_rows:
        synth_str = ",".join(str(pid) for pid in row.get("synthetic_pairs_collapsed", []))
        print(
            f"{row['threshold']:>10.2f} {row['unique']:>7d} "
            f"{row['collapsed_pairs']:>10d} "
            f"{row.get('unique_natural', 0):>10d} "
            f"{row.get('collapsed_natural', 0):>10d} "
            f"{synth_str:>18}"
        )

    # Sanity checks (warnings only, surface algorithm bugs without crashing).
    print()
    print("Verificaciones:")
    pair1_collapses = [bool(1 in row.get("synthetic_pairs_collapsed", [])) for row in sweep_rows]
    pair5_collapses = [bool(5 in row.get("synthetic_pairs_collapsed", [])) for row in sweep_rows]

    if all(pair1_collapses):
        print("  [OK] Par 1 (sim~0.99) colapsa en TODOS los thresholds.")
    else:
        offending = [
            f"{row['threshold']:.2f}"
            for row, c in zip(sweep_rows, pair1_collapses)
            if not c
        ]
        print(f"  [BUG] Par 1 NO colapsa en thresholds: {offending}")

    if not any(pair5_collapses):
        print("  [OK] Par 5 (control negativo) NO colapsa en ningun threshold.")
    else:
        offending = [
            f"{row['threshold']:.2f}"
            for row, c in zip(sweep_rows, pair5_collapses)
            if c
        ]
        print(f"  [BUG] Par 5 (control negativo) colapsa en thresholds: {offending}")

    uniques_natural = [row.get("unique_natural", 0) for row in sweep_rows]
    is_monotonic = all(
        uniques_natural[i] >= uniques_natural[i - 1] for i in range(1, len(uniques_natural))
    )
    if is_monotonic:
        print("  [OK] unique_natural es monotonicamente no-decreciente.")
    else:
        print(f"  [BUG] unique_natural NO es monotonico: {uniques_natural}")

    print()
    print(f"CSV:  {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")
    print(f"PNG:  {PNG_PATH}")


if __name__ == "__main__":
    main()
