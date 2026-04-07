"""Agent 3: Title/Abstract Screening with Ensemble Voting (Qwen2.5-7B)"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
import json, torch, re, time
from typing import List, Dict


class Agent3Screening(BaseAgent):
    """Title/abstract screening with ensemble voting.

    Input: list of papers (from A2) and protocol (from A1).
    Output: dict with 'include', 'exclude', 'uncertain' lists and 'stats'.
    GPU: yes -- runs N LLM votes per paper for majority-rule decisions.
    """

    def __init__(
        self,
        ensemble_size: int = 3,
        voting_threshold: float = 0.60,
        aggregation_method: str = "majority",
    ):
        super().__init__("agent3_screening")
        self.ensemble_size = ensemble_size  # Nº votaciones por paper
        self.voting_threshold = voting_threshold  # % mínimo para decisión firme
        self.aggregation_method = aggregation_method  # "majority" | "weighted"

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def screen_batch(self, papers: List[Dict], protocol: Dict) -> Dict:
        """
        Screening completo de un batch.
        Input : lista papers (A2) + protocolo (A1)
        Output: {include, exclude, uncertain, stats}
        """
        if self.model is None:
            self.load_model()

        results = {"include": [], "exclude": [], "uncertain": [], "stats": {}}

        print(f"\n{'='*60}")
        print("AGENT 3 - SCREENING TÍTULO/ABSTRACT")
        print(f"{'='*60}")
        print(f"📋 Papers a evaluar : {len(papers)}")
        print(f"⚙️  Ensemble size    : {self.ensemble_size}")
        if len(self.model_paths) > 1:
            print(f"⚙️  Multi-model      : {len(self.model_paths)} models")
        print(f"⚙️  Voting threshold : {int(self.voting_threshold*100)}%")
        print(
            f"⚙️  Uncertain zone   : {int((1-self.voting_threshold)*100)}-{int(self.voting_threshold*100)}%"
        )

        start = time.time()

        for i, paper in enumerate(papers):
            title_short = paper.get("title", "")[:75]
            print(f"\n[{i+1}/{len(papers)}] {title_short}...")

            try:
                result = self._screen_single(paper, protocol)
            except Exception as e:
                print(f"  [!] Error screening paper: {e} — conservative EXCLUDE")
                result = {
                    "pmid": paper.get("pmid", ""),
                    "title": paper.get("title", ""),
                    "decision": "EXCLUDE",
                    "confidence": 0.0,
                    "votes": [],
                    "reasoning": f"Screening error: {e}",
                    "flags": ["screening_error"],
                    "screening_confidence": 0.0,
                    "borderline": False,
                }

            icon = {"INCLUDE": "\u2705", "EXCLUDE": "\u274c", "UNCERTAIN": "\u26a0\ufe0f "}[
                result["decision"]
            ]
            pct = int(result["confidence"] * 100)
            vote_summary = [v["decision"] for v in result["votes"]]
            print(
                f"  \u2192 {icon} {result['decision']} | conf: {pct}% | votos: {vote_summary}"
                f" | screen_conf: {result['screening_confidence']}"
                f"{' [BORDERLINE]' if result['borderline'] else ''}"
            )
            if result["flags"]:
                print(f"     \U0001f6a9 flags: {result['flags']}")

            results[result["decision"].lower()].append(result)

        elapsed = round(time.time() - start, 1)
        results["stats"] = {
            "total": len(papers),
            "included": len(results["include"]),
            "excluded": len(results["exclude"]),
            "uncertain": len(results["uncertain"]),
            "elapsed_s": elapsed,
            "s_per_paper": round(elapsed / max(len(papers), 1), 1),
            "ensemble_size": self.ensemble_size,
            "voting_threshold": self.voting_threshold,
        }

        self._print_summary(results["stats"])
        return results

    # ------------------------------------------------------------------ #
    #  CORE: SINGLE PAPER SCREENING                                        #
    # ------------------------------------------------------------------ #

    def _screen_single(self, paper: Dict, protocol: Dict) -> Dict:
        """Evalúa un paper con ensemble voting. Retorna decisión + metadatos."""
        vote_details = []  # list of dicts with full vote info
        multi_model = len(self.model_paths) > 1

        for i in range(self.ensemble_size):
            model_name = "default"
            if multi_model:
                model_path = self.model_paths[i % len(self.model_paths)]
                self.load_model(model_path)
                model_name = os.path.basename(model_path)
            vote_result = self._single_vote(paper, protocol)
            vote_details.append({
                "decision": vote_result["vote"],
                "reasoning": vote_result["reasoning"],
                "model": model_name,
            })

        raw_votes = [v["decision"] for v in vote_details]
        decision, confidence = self._aggregate_votes(raw_votes)
        flags = self._detect_flags(paper)

        # Screening confidence: 0 = maximally uncertain, 1 = unanimous
        n = len(raw_votes)
        include_frac = raw_votes.count("INCLUDE") / n if n > 0 else 0.5
        screening_confidence = round(abs(include_frac - 0.5) * 2, 2)
        is_borderline = screening_confidence < 0.3

        return {
            "pmid": paper.get("pmid", ""),
            "title": paper.get("title", ""),
            "decision": decision,
            "confidence": round(confidence, 2),
            "votes": vote_details,
            "reasoning": vote_details[0]["reasoning"] if vote_details else "",
            "flags": flags,
            "screening_confidence": screening_confidence,
            "borderline": is_borderline,
        }

    def _single_vote(self, paper: Dict, protocol: Dict) -> Dict:
        """Una inferencia LLM → voto INCLUDE / EXCLUDE / UNCERTAIN."""
        # Clean criteria: strip markdown bold markers that may confuse the LLM
        incl = "\n".join(
            f"  - {c.replace('**', '')}" for c in protocol.get("inclusion_criteria", [])
        )
        excl = "\n".join(
            f"  - {c.replace('**', '')}" for c in protocol.get("exclusion_criteria", [])
        )
        title = paper.get("title", "")
        _abs_max = self.config.get("screening", {}).get("abstract_max_chars", 800)
        abstract = str(paper.get("abstract", ""))[:_abs_max]

        # Build framework-aware screening rule
        framework = protocol.get("framework", "PICO")
        intv = protocol.get("intervention", protocol.get("exposure", ""))
        pop = protocol.get("population", "")
        outcome = protocol.get("outcome", "")
        comparator = protocol.get("comparator", "")

        if framework == "PECO":
            # Observational: exposure must be primary, not a covariate
            focus_rule = (
                f'CRITICAL SCREENING RULE:\n'
                f'The paper must study "{intv}" as its PRIMARY exposure — not merely mention it as a covariate, '
                f'confounder, or baseline characteristic in a table. If "{intv}" only appears as a demographic '
                f'variable or adjustment factor, the paper does NOT meet inclusion criteria.\n'
                f'The PRIMARY outcome must be related to "{outcome}" in "{pop}".'
            )
        elif framework == "PICO":
            # PICO: check intervention AND comparator AND outcome relevance
            comp_rule = ""
            if comparator:
                comp_rule = (
                    f'Comparator: ideally compared to "{comparator}" or similar. '
                    f'Papers comparing "{intv}" to other treatments are still relevant at this stage.\n'
                )
            focus_rule = (
                f'SCREENING FOCUS:\n'
                f'Population: "{pop}"\n'
                f'Intervention: "{intv}" must be one of the interventions studied, tested, or compared.\n'
                f'{comp_rule}'
                f'Outcome: should be related to "{outcome}".\n'
                f'INCLUDE if "{intv}" is tested, compared, or evaluated in any way relevant to the research question.\n'
                f'EXCLUDE only if the paper clearly has nothing to do with "{intv}", targets an unrelated '
                f'condition, or is a review/editorial/guideline.'
            )
        elif framework == "SPIDER":
            sample = protocol.get("sample", pop)
            pi = protocol.get("phenomenon_of_interest", intv)
            design = protocol.get("design", "")
            evaluation = protocol.get("evaluation", outcome)
            research_type = protocol.get("research_type", "")
            focus_rule = (
                f'SCREENING FOCUS (SPIDER):\n'
                f'Sample: "{sample}"\n'
                f'Phenomenon of Interest: "{pi}"\n'
                f'Design: "{design}"\n'
                f'Evaluation: "{evaluation}"\n'
                f'Research type: "{research_type}"\n'
                f'INCLUDE if the paper examines "{pi}" in "{sample}" using a relevant design.\n'
                f'EXCLUDE if the topic, sample, or methodology clearly does not match.'
            )
        else:
            # PCC and other frameworks: generic but still structured
            concept = protocol.get("concept", intv)
            context = protocol.get("context", "")
            focus_rule = (
                f'SCREENING FOCUS:\n'
                f'Population: "{pop}"\n'
                f'Concept/Intervention: "{concept}"\n'
                f'Context: "{context}"\n'
                f'Outcome: "{outcome}"\n'
                f'INCLUDE if the paper is relevant to "{concept}" in "{pop}".\n'
                f'EXCLUDE if clearly irrelevant to the topic, population, or context.'
            )

        prompt = f"""You are a Cochrane screening expert. Evaluate if this paper meets the review criteria.

INCLUSION CRITERIA:
{incl}

EXCLUSION CRITERIA:
{excl}

PAPER:
Title: {title}
Abstract: {abstract}

{focus_rule}

DECISION RULES:
- INCLUDE  : the paper involves "{intv}" as one of the interventions studied, compared, or analyzed AND the topic is relevant to the inclusion criteria. Include even if "{intv}" is studied alongside other methods (e.g., prediction models, secondary analyses) — what matters is that "{intv}" data is present.
- EXCLUDE  : "{intv}" is NOT mentioned or studied at all, OR the paper targets a completely different condition/population with no relevance, OR it is an editorial/letter/case report, OR it clearly triggers an exclusion criterion
- UNCERTAIN: the paper mentions "{intv}" but there is not enough information to confirm it meets all criteria

IMPORTANT: Be INCLUSIVE at this stage — title/abstract screening should cast a wide net. If "{intv}" appears in the title or abstract as something being tested, compared, or evaluated, lean towards INCLUDE. Only EXCLUDE when the paper is clearly irrelevant. Reviews and guidelines should be EXCLUDED.

Respond ONLY with valid JSON:
{{"vote": "INCLUDE or EXCLUDE or UNCERTAIN", "reasoning": "one sentence"}}"""

        response = self._simple_generate(prompt, max_tokens=120)

        try:
            cleaned = self._clean_json(response)
            parsed = json.loads(cleaned)
            vote = parsed.get("vote", "EXCLUDE").upper().strip()
            if vote not in ("INCLUDE", "EXCLUDE", "UNCERTAIN"):
                vote = "EXCLUDE"
            return {"vote": vote, "reasoning": parsed.get("reasoning", "")}
        except Exception:
            return {
                "vote": "EXCLUDE",
                "reasoning": "JSON parse error – conservative EXCLUDE",
            }

    # ------------------------------------------------------------------ #
    #  VOTING AGGREGATION                                                  #
    # ------------------------------------------------------------------ #

    def _aggregate_votes(self, votes: List[str]):
        """
        Agrega votos y determina decisión final.

        Lógica:
          - Si INCLUDE  >= voting_threshold  → INCLUDE
          - Si EXCLUDE  >= voting_threshold  → EXCLUDE
          - Resto (empate o mayoría UNCERTAIN) → UNCERTAIN
        """
        n = len(votes)
        include_n = votes.count("INCLUDE")
        exclude_n = votes.count("EXCLUDE")
        confidence = include_n / n

        if include_n / n >= self.voting_threshold:
            return "INCLUDE", confidence
        elif exclude_n / n >= self.voting_threshold:
            return "EXCLUDE", confidence
        else:
            return "UNCERTAIN", confidence

    # ------------------------------------------------------------------ #
    #  FLAG DETECTION (reglas Python, sin LLM)                            #
    # ------------------------------------------------------------------ #

    def _detect_flags(self, paper: Dict) -> List[str]:
        """Detecta señales de alerta metodológicas del texto título+abstract."""
        flags = []
        text = (paper.get("title", "") + " " + str(paper.get("abstract", ""))).lower()

        # Muestra pequeña  (<30 participantes)
        if re.search(
            r"\b(n\s*[=<]\s*[12]\d|[12]\d\s+participants?|[12]\d\s+patients?)\b", text
        ):
            flags.append("small_sample_size")

        # Financiación industria
        if any(
            t in text
            for t in [
                "funded by",
                "sponsored by",
                "grant from",
                "industry-funded",
                "pharmaceutical company",
            ]
        ):
            flags.append("industry_funded")

        # Sin grupo control
        if any(
            t in text
            for t in [
                "no control",
                "uncontrolled",
                "without control group",
                "open-label single arm",
            ]
        ):
            flags.append("no_control_group")

        # Reporte de caso / serie
        if any(t in text for t in ["case report", "case series", "case study"]):
            flags.append("case_report_series")

        # Revisión secundaria (no estudio primario)
        if re.search(
            r"^(systematic review|meta-analysis|narrative review|scoping review)",
            text[:60],
        ):
            flags.append("secondary_study")

        return flags

    # ------------------------------------------------------------------ #
    #  CALIBRATION & SENSITIVITY ANALYSIS                                  #
    # ------------------------------------------------------------------ #

    def analyze_voting_sensitivity(self, results: dict) -> dict:
        """Post-hoc analysis of how different thresholds would change results."""
        # Collect all papers from include/exclude/uncertain
        papers = (
            results.get("include", [])
            + results.get("exclude", [])
            + results.get("uncertain", [])
        )
        thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        analysis = {}
        for t in thresholds:
            included = 0
            excluded = 0
            uncertain = 0
            for p in papers:
                votes = p.get("votes", [])
                if not votes:
                    continue
                # votes may be list of dicts or list of strings (backward compat)
                decisions = []
                for v in votes:
                    if isinstance(v, dict):
                        decisions.append(v.get("decision", "EXCLUDE"))
                    else:
                        decisions.append(v)
                n = len(decisions)
                include_frac = sum(1 for d in decisions if d == "INCLUDE") / n
                exclude_frac = sum(1 for d in decisions if d == "EXCLUDE") / n
                if include_frac >= t:
                    included += 1
                elif exclude_frac >= t:
                    excluded += 1
                else:
                    uncertain += 1
            analysis[str(t)] = {
                "included": included,
                "excluded": excluded,
                "uncertain": uncertain,
                "threshold": t,
            }

        # Calculate inter-rater agreement (Fleiss' kappa)
        kappa = self._calculate_fleiss_kappa(papers)

        # Borderline papers needing human review
        borderline_papers = [
            {"pmid": p.get("pmid", ""), "title": p.get("title", ""), "screening_confidence": p.get("screening_confidence", 0)}
            for p in papers if p.get("borderline", False)
        ]

        return {
            "threshold_sensitivity": analysis,
            "current_threshold": self.voting_threshold,
            "fleiss_kappa": round(kappa, 3),
            "interpretation": self._interpret_kappa(kappa),
            "total_papers": len(papers),
            "borderline_count": len(borderline_papers),
            "borderline_papers": borderline_papers,
        }

    def _calculate_fleiss_kappa(self, papers: list) -> float:
        """Calculate Fleiss' kappa for inter-rater agreement.

        Fleiss' kappa formula: kappa = (P_bar - P_bar_e) / (1 - P_bar_e)
        where:
          P_bar   = mean of per-subject agreement proportions
          P_bar_e = expected agreement by chance
        """
        categories = ["INCLUDE", "EXCLUDE", "UNCERTAIN"]
        n_subjects = 0  # papers with votes
        n_raters = self.ensemble_size

        if n_raters < 2:
            return 0.0  # kappa undefined for single rater

        # Count votes per category for each paper
        vote_matrix = []  # each entry: [n_include, n_exclude, n_uncertain]
        for p in papers:
            votes = p.get("votes", [])
            if not votes:
                continue
            counts = [0, 0, 0]
            for v in votes:
                decision = v.get("decision", v) if isinstance(v, dict) else v
                if decision == "INCLUDE":
                    counts[0] += 1
                elif decision == "EXCLUDE":
                    counts[1] += 1
                else:  # UNCERTAIN
                    counts[2] += 1
            vote_matrix.append(counts)
            n_subjects += 1

        if n_subjects == 0:
            return 0.0

        # P_i for each subject: proportion of agreeing pairs
        # P_i = (1 / (n*(n-1))) * (sum(n_ij^2) - n)
        p_values = []
        for counts in vote_matrix:
            n = sum(counts)
            if n < 2:
                continue
            sum_sq = sum(c * c for c in counts)
            p_i = (sum_sq - n) / (n * (n - 1))
            p_values.append(p_i)

        if not p_values:
            return 0.0

        P_bar = sum(p_values) / len(p_values)

        # P_bar_e: expected agreement by chance
        # p_j = proportion of all assignments to category j
        total_votes = sum(sum(counts) for counts in vote_matrix)
        if total_votes == 0:
            return 0.0

        P_bar_e = 0.0
        for j in range(len(categories)):
            p_j = sum(counts[j] for counts in vote_matrix) / total_votes
            P_bar_e += p_j * p_j

        if P_bar_e >= 1.0:
            return 1.0  # perfect agreement by definition

        kappa = (P_bar - P_bar_e) / (1.0 - P_bar_e)
        return kappa

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Fleiss' kappa value using Landis & Koch scale."""
        if kappa < 0:
            return "Poor"
        if kappa < 0.20:
            return "Slight"
        if kappa < 0.40:
            return "Fair"
        if kappa < 0.60:
            return "Moderate"
        if kappa < 0.80:
            return "Substantial"
        return "Almost perfect"

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _simple_generate(self, prompt: str, max_tokens: int = 150) -> str:
        if self._is_groq():
            return self.generate(prompt, max_new_tokens=max_tokens, temperature=0.2)
        inputs = self._tokenize_prompt(prompt)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,
                do_sample=self.do_sample,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self._decode_response(outputs, input_length)

    def _clean_json(self, text: str) -> str:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        cleaned = re.sub(r"[^\x00-\x7F]+", "", cleaned)
        start = cleaned.find("{")
        if start < 0:
            return "{}"
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1]
        return cleaned[start:] + "}"

    def _print_summary(self, stats: Dict):
        print(f"\n{'='*60}")
        print("📊 RESUMEN SCREENING")
        print(f"{'='*60}")
        print(f"   Total evaluados : {stats['total']}")
        print(f"   ✅ INCLUDE       : {stats['included']}")
        print(f"   ❌ EXCLUDE       : {stats['excluded']}")
        print(f"   ⚠️  UNCERTAIN     : {stats['uncertain']}")
        print(
            f"   ⏱️  Tiempo total  : {stats['elapsed_s']}s ({stats['s_per_paper']}s/paper)"
        )
        if stats["total"] > 0:
            pct_include = round(stats["included"] / stats["total"] * 100, 1)
            print(f"   📈 Tasa inclusión: {pct_include}%")


# ====================================================================== #
#  TEST INTEGRACIÓN                                                        #
# ====================================================================== #
if __name__ == "__main__":

    # ── Protocolo mock (output Agent 1 PICO) ──────────────────────────────
    MOCK_PROTOCOL = {
        "framework": "PICO",
        "research_question": "Does CBT improve depression outcomes versus antidepressants in adults?",
        "population": "Adults (18-65) with major depressive disorder",
        "intervention": "Cognitive Behavioral Therapy (CBT)",
        "comparator": "Antidepressant medication",
        "outcome": "Depression symptom reduction (BDI, PHQ-9, HAM-D)",
        "inclusion_criteria": [
            "Adults aged 18-65 with diagnosis of major depressive disorder",
            "RCT or controlled trial design",
            "Follow-up >= 8 weeks",
            "Quantitative depression scale used as outcome",
        ],
        "exclusion_criteria": [
            "Psychosis, bipolar disorder, or personality disorder",
            "Case reports, case series, or reviews",
            "Follow-up < 8 weeks",
            "No control/comparator group",
        ],
    }

    # ── Papers mock (output Agent 2) ──────────────────────────────────────
    MOCK_PAPERS = [
        {  # Esperado: INCLUDE ✅ (RCT, CBT vs antidepressants, MDD adults)
            "pmid": "11111111",
            "title": "CBT versus sertraline for major depression: a 16-week randomized controlled trial",
            "abstract": "Background: 120 adults with MDD randomized to CBT or sertraline 50mg for 16 weeks. "
            "Primary outcome: BDI-II at endpoint. Results: CBT group BDI-II -14.2 (SD 6.1), "
            "medication group -13.8 (SD 5.9). Response rates 68% vs 72% (p=0.41). "
            "Both treatments comparably effective. Limitations: no blinding.",
            "year": "2024",
            "journal": "JAMA Psychiatry",
        },
        {  # Esperado: EXCLUDE ❌ (pain/elderly, wrong population & outcome)
            "pmid": "22222222",
            "title": "Mindfulness-based stress reduction for chronic back pain in elderly: a pilot study",
            "abstract": "Objective: Evaluate MBSR in 18 elderly patients (>65 years) with chronic back pain. "
            "No control group. Results: VAS pain scores improved 35% after 8 weeks. "
            "Conclusion: MBSR may benefit elderly pain patients.",
            "year": "2022",
            "journal": "Pain Medicine",
        },
        {  # Esperado: INCLUDE ✅ (iCBT vs escitalopram, MDD, RCT)
            "pmid": "33333333",
            "title": "Internet-delivered CBT vs escitalopram for depression: 12-week randomized trial",
            "abstract": "Participants: 89 adults with moderate MDD (PHQ-9 >= 10). "
            "Randomized to 12-week online CBT program or escitalopram 10mg daily. "
            "Primary outcome: PHQ-9 at 12 weeks. iCBT group: -8.4 (SD 5.0), "
            "medication: -8.1 (SD 4.8). Difference -0.3 (95%CI -1.8 to 1.2). Equivalence confirmed.",
            "year": "2024",
            "journal": "Behaviour Research and Therapy",
        },
        {  # Esperado: EXCLUDE ❌ (systematic review, secondary study)
            "pmid": "44444444",
            "title": "Systematic review: prevalence of depression in medical students worldwide",
            "abstract": "Systematic review of 45 studies (N=52,000). Pooled prevalence 27.2% (95%CI 24.7-29.9%). "
            "High heterogeneity I²=92%. Burnout and academic stress as main predictors.",
            "year": "2023",
            "journal": "Academic Medicine",
        },
        {  # Esperado: UNCERTAIN ⚠️ (CBT para depresión pero info insuficiente)
            "pmid": "55555555",
            "title": "Psychological interventions for depression: updated review",
            "abstract": "This article reviews recent evidence on psychological treatments for depression. "
            "Various approaches including CBT are discussed in the context of clinical practice. "
            "Implications for treatment guidelines are presented.",
            "year": "2023",
            "journal": "Current Psychiatry Reports",
        },
    ]

    print("=" * 60)
    print("TEST AGENT 3 - ENSEMBLE SCREENING")
    print("=" * 60)
    print(
        f"Protocol : {MOCK_PROTOCOL['framework']} – {MOCK_PROTOCOL['research_question']}"
    )
    print(
        f"Papers   : {len(MOCK_PAPERS)} (2 include, 2 exclude, 1 uncertain esperados)"
    )

    agent = Agent3Screening(
        ensemble_size=3,
        voting_threshold=0.60,
        aggregation_method="majority",
    )

    results = agent.screen_batch(MOCK_PAPERS, MOCK_PROTOCOL)

    # ── Mostrar detalle ───────────────────────────────────────────────────
    for section, icon in [("include", "✅"), ("uncertain", "⚠️ "), ("exclude", "❌")]:
        papers = results[section]
        if papers:
            print(f"\n{icon} {section.upper()} ({len(papers)}):")
            for p in papers:
                print(f"  [{p['pmid']}] {p['title'][:65]}...")
                vote_summary = [v["decision"] for v in p["votes"]]
                print(
                    f"         conf={int(p['confidence']*100)}% | votos={vote_summary}"
                    f" | screen_conf={p.get('screening_confidence', '?')}"
                    f"{'  [BORDERLINE]' if p.get('borderline') else ''}"
                    + (f" | 🚩{p['flags']}" if p["flags"] else "")
                )
                if p["reasoning"]:
                    print(f"         → {p['reasoning'][:100]}")

    # ── Calibration analysis ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CALIBRATION & SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    sensitivity = agent.analyze_voting_sensitivity(results)
    print(f"  Fleiss' kappa: {sensitivity['fleiss_kappa']} ({sensitivity['interpretation']})")
    print(f"  Borderline papers: {sensitivity['borderline_count']}")
    print("\n  Threshold sensitivity:")
    for t_str, info in sensitivity["threshold_sensitivity"].items():
        marker = " <-- current" if float(t_str) == sensitivity["current_threshold"] else ""
        print(f"    {t_str}: INCLUDE={info['included']}, EXCLUDE={info['excluded']}, UNCERTAIN={info['uncertain']}{marker}")

    # ── Guardar JSON ──────────────────────────────────────────────────────
    os.makedirs("./outputs", exist_ok=True)
    out_path = "./outputs/agent3_screening_test.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n💾 Output guardado: {out_path}")
    print("\n🏁 Test Agent 3 completado.")
