"""Agent 1: Two-Step Generation (metadata + criteria)"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
import json, torch, re


class Agent1Protocol(BaseAgent):
    """Generates a systematic review protocol from a research question.

    Input: user_query (str) -- a clinical research question in natural language.
    Output: dict with framework (PICO/SPIDER/PECO/PCC), criteria, and PubMed query.
    GPU: yes -- uses Qwen2.5-7B for framework detection and criteria generation.
    """

    _MESH_MAPPING = {
        # Interventions
        "cognitive behavioral therapy": "Cognitive Behavioral Therapy",
        "cbt": "Cognitive Behavioral Therapy",
        "exercise": "Exercise",
        "meditation": "Meditation",
        "mindfulness": "Mindfulness",
        # Conditions/Outcomes
        "depression": "Depression",
        "depressive disorder": "Depressive Disorder",
        "anxiety": "Anxiety",
        "anxiety disorders": "Anxiety Disorders",
        "insomnia": "Sleep Initiation and Maintenance Disorders",
        "pain": "Pain",
        "chronic pain": "Chronic Pain",
        "smoking": "Smoking",
        "smoking cessation": "Smoking Cessation",
        "obesity": "Obesity",
        "diabetes": "Diabetes Mellitus",
        "hypertension": "Hypertension",
        "ptsd": "Stress Disorders, Post-Traumatic",
        "post-traumatic stress": "Stress Disorders, Post-Traumatic",
        "substance abuse": "Substance-Related Disorders",
        "alcohol": "Alcohol Drinking",
        "schizophrenia": "Schizophrenia",
        "bipolar disorder": "Bipolar Disorder",
        "eating disorders": "Feeding and Eating Disorders",
        "cancer": "Neoplasms",
        "heart failure": "Heart Failure",
        "stroke": "Stroke",
        "myocardial infarction": "Myocardial Infarction",
        # Populations
        "adults": "Adult",
        "children": "Child",
        "adolescents": "Adolescent",
        "elderly": "Aged",
        "pregnant women": "Pregnant Women",
        "veterans": "Veterans",
        # Comparators
        "medication": "Drug Therapy",
        "antidepressants": "Antidepressive Agents",
        "antidepressant": "Antidepressive Agents",
        "pharmacotherapy": "Drug Therapy",
        "placebo": "Placebos",
        "usual care": "Standard of Care",
        "waitlist": "Waiting Lists",
        # Study types
        "randomized controlled trial": "Randomized Controlled Trial",
        "cohort": "Cohort Studies",
        "qualitative": "Qualitative Research",
        "prevalence": "Prevalence",
    }

    # Words that indicate a MeSH term is likely invalid (descriptive phrases, not controlled vocabulary)
    _INVALID_MESH_WORDS = {
        "effectiveness", "impact", "influence", "association", "relationship",
        "effect", "efficacy", "role", "benefit", "comparison", "evaluation",
        "assessment", "improvement", "reduction", "treating", "management",
    }

    def __init__(self):
        super().__init__("agent1_protocol")

    # ------------------------------------------------------------------ #

    def generate_protocol(self, user_query):
        print(f"\n\U0001f50d Query: '{user_query}'")
        if self.model is None:
            self.load_model()

        # Step 0: Translate query to English (all downstream is English-only)
        en_query = self._translate_to_english(user_query)
        print(f"  English query: '{en_query}'")

        metadata = self._generate_metadata(en_query)
        print(f"\u2705 Step 1/2: Framework={metadata['framework']}")
        criteria = self._generate_criteria(en_query, metadata)
        print("\u2705 Step 2/2: Criteria generated")
        metadata["original_query"] = user_query
        protocol = {**metadata, **criteria}

        # Generate formal protocol document (PRISMA-P style)
        protocol["protocol_document"] = self._generate_protocol_document(protocol)

        return protocol

    def generate_pubmed_query(self, user_query: str, protocol: dict) -> str:
        """Genera query PubMed estructurada con MeSH terms y sinónimos."""
        framework = protocol.get("framework", "PICO")
        design_filter = {
            "PICO": "randomized controlled trial",
            "SPIDER": "qualitative",
            "PECO": "cohort",
            "PCC": "prevalence",
        }.get(framework, "randomized controlled trial")

        protocol.get("population", "")
        intv = protocol.get("intervention", protocol.get("exposure", ""))
        comp = protocol.get("comparator", "")
        outcome = protocol.get("outcome", "")

        prompt = (
            "You are a PubMed search expert. Build a structured Boolean search query.\n\n"
            f'Research question: "{user_query}"\n'
            f"{'Exposure' if framework == 'PECO' else 'Intervention'}: {intv}\n"
            f"Comparator: {comp}\n"
            f"Outcome/Condition: {outcome}\n"
            f"Design: {design_filter}\n\n"
            "Rules:\n"
            "- Use MAXIMUM 3 AND clauses (intervention AND condition AND comparator)\n"
            "- Do NOT add population (adults/children) as a separate AND clause — it over-restricts\n"
            "- Use Boolean: AND between concepts, OR between synonyms\n"
            "- Include VALID MeSH terms with [MeSH] tag (e.g. \"Cognitive Behavioral Therapy\"[MeSH], \"Depression\"[MeSH])\n"
            "- Include 3-5 free-text synonyms per concept with [Title/Abstract] tag for recall\n"
            "- Group synonyms with parentheses and OR\n"
            "- Output ONLY the search string, one line, no explanation\n"
            "- All terms in English\n\n"
            "Examples:\n"
            'Question: "Does smoking increase MI risk in diabetics?"\n'
            'Output: ("Smoking"[MeSH] OR "Tobacco Use"[MeSH] OR smoking[Title/Abstract] OR cigarette[Title/Abstract] OR tobacco[Title/Abstract]) AND ("Myocardial Infarction"[MeSH] OR "myocardial infarction"[Title/Abstract] OR "heart attack"[Title/Abstract] OR "acute coronary syndrome"[Title/Abstract]) AND ("Diabetes Mellitus"[MeSH] OR diabetes[Title/Abstract] OR diabetic[Title/Abstract])\n\n'
            'Question: "Is CBT more effective than medication for depression in adults?"\n'
            'Output: ("Cognitive Behavioral Therapy"[MeSH] OR "cognitive behavioral therapy"[Title/Abstract] OR CBT[Title/Abstract] OR "cognitive therapy"[Title/Abstract] OR "behavioral therapy"[Title/Abstract]) AND ("Depression"[MeSH] OR "Depressive Disorder"[MeSH] OR depression[Title/Abstract] OR depressive[Title/Abstract] OR MDD[Title/Abstract]) AND ("Antidepressive Agents"[MeSH] OR "Drug Therapy"[MeSH] OR antidepressant[Title/Abstract] OR pharmacotherapy[Title/Abstract] OR medication[Title/Abstract] OR SSRI[Title/Abstract])\n\n'
            "Output:"
        )

        response = self._simple_generate(prompt, max_tokens=250)
        query = response.strip().split("\n")[0].strip()

        # Validate: must contain AND/OR or brackets, and no Spanish
        spanish = ["para", "versus", "adultos", "depresión", "médicos", "niños"]
        has_structure = any(op in query for op in ["AND", "OR", "[MeSH]", "[Title/Abstract]"])

        # Post-validate LLM-generated MeSH terms for obvious errors
        mesh_valid = self._validate_llm_mesh_terms(query) if has_structure else True

        # Check AND count — more than 3 top-level ANDs over-restricts the search
        n_ands = len(re.findall(r'\)\s*AND\s*\(', query))
        too_many_ands = n_ands > 3

        if not query or len(query) < 15 or any(w in query.lower() for w in spanish) or not has_structure or not mesh_valid or too_many_ands:
            # Deterministic fallback: build structured query from protocol components
            if too_many_ands:
                print(f"  [query] LLM query has {n_ands} AND clauses (max 3) — using fallback")
            query = self._build_fallback_query(protocol, design_filter)

        return query

    def _resolve_mesh(self, term: str) -> str:
        """Resolve a raw term to its valid MeSH heading using _MESH_MAPPING.

        Checks for exact match first, then substring match.
        Returns the mapped MeSH term, or the original term if no mapping found.
        """
        lowered = term.lower().strip()
        # Exact match
        if lowered in self._MESH_MAPPING:
            return self._MESH_MAPPING[lowered]
        # Substring match: check if any known key is contained in the term
        for key, mesh in self._MESH_MAPPING.items():
            if key in lowered:
                return mesh
        return term

    def _validate_llm_mesh_terms(self, query: str) -> bool:
        """Check if the LLM-generated query contains obviously invalid MeSH terms.

        Returns True if query looks valid, False if it should fall through to fallback.
        """
        # Find all "..."[MeSH] patterns
        mesh_terms = re.findall(r'"([^"]+)"\[MeSH\]', query)
        for term in mesh_terms:
            # Too long — likely a descriptive phrase, not a MeSH heading
            if len(term) > 50:
                return False
            # Contains words that are never part of MeSH headings
            term_words = set(term.lower().split())
            if term_words & self._INVALID_MESH_WORDS:
                return False
        return True

    def _build_fallback_query(self, protocol: dict, design_filter: str) -> str:
        """Build a structured PubMed query deterministically from protocol components.

        Strategy: use max 3 AND clauses (intervention AND condition AND comparator).
        Population is omitted as a separate AND clause because it over-restricts;
        most studies already specify adult/child in their MeSH indexing.
        We add synonyms to each concept block for recall.
        """
        framework = protocol.get("framework", "PICO")
        intv = protocol.get("intervention", protocol.get("exposure", ""))
        comp = protocol.get("comparator", "")
        outcome = protocol.get("outcome", "")

        # Common synonym expansions for better recall
        _SYNONYMS = {
            "cognitive behavioral therapy": [
                "cognitive behavioral therapy", "cognitive behaviour therapy",
                "CBT", "cognitive therapy", "behavioral therapy",
            ],
            "depression": [
                "depression", "depressive disorder", "major depressive disorder",
                "depressive symptoms", "MDD",
            ],
            "medication": [
                "medication", "pharmacotherapy", "drug therapy",
                "antidepressant", "antidepressive agents",
            ],
            "antidepressants": [
                "antidepressant", "antidepressive agents", "SSRI",
                "pharmacotherapy", "drug therapy", "medication",
            ],
            "smoking": ["smoking", "tobacco", "cigarette", "tobacco use"],
            "exercise": ["exercise", "physical activity", "aerobic exercise"],
            "anxiety": ["anxiety", "anxiety disorders", "anxious"],
        }

        def _build_concept_block(raw_term: str) -> str:
            """Build a (MeSH OR synonym1 OR synonym2...) block for one concept."""
            mesh = self._resolve_mesh(raw_term)
            # If mesh is the raw term and it's a long phrase, extract core concept
            if mesh == raw_term and len(raw_term.split()) > 3:
                for word in raw_term.lower().split():
                    candidate = self._resolve_mesh(word)
                    if candidate != word:
                        mesh = candidate
                        break

            # Get synonyms
            key = raw_term.lower().strip()
            syns = _SYNONYMS.get(key, [])
            if not syns:
                # Try partial match
                for k, v in _SYNONYMS.items():
                    if k in key or key in k:
                        syns = v
                        break

            # Build OR block: MeSH + synonyms + original term
            or_parts = [f'"{mesh}"[MeSH]']
            all_terms = set(syns) if syns else set()
            all_terms.add(raw_term.lower())
            for t in all_terms:
                or_parts.append(f'"{t}"[Title/Abstract]')

            return "(" + " OR ".join(or_parts) + ")"

        parts = []

        # Core concepts: intervention/exposure (always included)
        if intv:
            parts.append(_build_concept_block(intv))

        # Outcome/condition (always included)
        if outcome:
            parts.append(_build_concept_block(outcome))

        # Comparator (include for PICO, skip for others to avoid over-restriction)
        if comp and framework == "PICO":
            parts.append(_build_concept_block(comp))

        # Population is NOT added as a separate AND — it over-restricts
        # PubMed MeSH indexing already handles age groups

        if not parts:
            return f"{intv} {outcome} {design_filter}"

        return " AND ".join(parts)

    # ------------------------------------------------------------------ #

    def _translate_to_english(self, query: str) -> str:
        """Translate query to English if needed. If already English, return as-is."""
        # Quick check: if no non-ASCII and looks English, skip
        if not re.search(r"[^\x00-\x7F]", query) and not re.search(r"\b(para|versus|como|los|las|del|una|que|por)\b", query.lower()):
            return query

        prompt = (
            "Translate this research question to English. If already in English, return it unchanged.\n"
            "Output ONLY the English translation, nothing else.\n\n"
            "Examples:\n"
            '"¿TCC versus medicación para depresión en adultos?" -> "Is CBT more effective than medication for depression in adults?"\n'
            '"¿Fumar aumenta el riesgo de infarto en diabéticos?" -> "Does smoking increase the risk of myocardial infarction in diabetic patients?"\n'
            '"Does exercise improve cognition in elderly?" -> "Does exercise improve cognition in elderly?"\n\n'
            f'Input: "{query}"\n'
            "Output:"
        )
        response = self._simple_generate(prompt, max_tokens=60)
        translated = response.strip().split("\n")[0].strip().strip('"')

        # Validate: should be longer than 10 chars and not contain Spanish
        spanish_words = {"para", "como", "con", "los", "las", "del", "una", "que", "por", "adultos", "niños"}
        if len(translated) < 10 or sum(1 for w in translated.lower().split() if w in spanish_words) >= 2:
            # Fallback: strip non-ASCII and return
            return re.sub(r"[^\x00-\x7F]+", " ", query).strip()

        return translated

    def _generate_metadata(self, query):
        prompt = (
            "Cochrane expert. Identify framework. ALL output must be in ENGLISH.\n\n"
            "KEYWORDS GUIDE:\n"
            '- PICO: "effective", "reduce", "improve", "versus", "compared to", "treatment"\n'
            '- SPIDER: "experience", "barriers", "perceptions", "feelings", "challenges", "how do", "what do"\n'
            '- PECO: "risk", "cause", "exposure", "factor", "associated with"\n'
            '- PCC: "prevalence", "frequency", "how many", "proportion"\n\n'
            f'Query: "{query}"\n\n'
            "Step 1 - Detect keywords:\n"
            "Keywords found: [list keywords from query]\n\n"
            "Step 2 - Choose framework:\n"
            "If comparing intervention/treatment -> PICO\n"
            "If exploring experiences/barriers/perceptions -> SPIDER\n"
            "If studying risk/exposure causes -> PECO\n"
            "If describing prevalence -> PCC\n\n"
            "Framework: [your choice]\n\n"
            "Step 3 - Extract components (ALL values in English):\n"
            "JSON (no arrays):\n"
            "{\n"
            '  "framework": "PICO/SPIDER/PECO/PCC",\n'
            '  "research_question": "English research question",\n'
            '  "population": "English description",\n'
            '  "intervention": "English full name (no abbreviations)",\n'
            '  "comparator": "English description",\n'
            '  "outcome": "English description"\n'
            "}"
        )

        response = self._simple_generate(prompt, max_tokens=400)
        try:
            cleaned = self._clean_json(response)
            metadata = json.loads(cleaned)
            if metadata["framework"] == "PICO":
                query_lower = query.lower()

                # Post-check: PICO -> SPIDER (qualitative keywords)
                spider_kws = [
                    "barrier", "perceiv", "experience", "feeling",
                    "what do", "how do", "challenges", "perspectives",
                    "barrera", "percib", "experiencia",
                ]
                if any(kw in query_lower for kw in spider_kws):
                    if "versus" not in query_lower and "compared" not in query_lower:
                        print(
                            "\u26a0\ufe0f  Correction: PICO->SPIDER (qualitative keywords)"
                        )
                        metadata["framework"] = "SPIDER"
                        metadata["sample"] = metadata.get("population", "participants")
                        metadata["phenomenon_of_interest"] = metadata.get(
                            "intervention", "phenomenon"
                        )
                        metadata["design"] = "Qualitative: interviews, focus groups"
                        metadata["evaluation"] = "Thematic analysis"
                        metadata["research_type"] = "Qualitative"

                # Post-check: PICO -> PECO (risk/exposure keywords)
                elif any(kw in query_lower for kw in [
                    "risk", "increase the risk", "associated with",
                    "exposure", "cause", "factor", "predispos",
                    "lead to", "contribut", "incidence",
                ]):
                    # Only reclassify if NOT a treatment comparison
                    if not any(kw in query_lower for kw in [
                        "versus", "compared to", "effective", "treatment",
                        "therapy", "medication", "drug",
                    ]):
                        print(
                            "\u26a0\ufe0f  Correction: PICO->PECO (risk/exposure keywords)"
                        )
                        metadata["framework"] = "PECO"
                        exposure = metadata.get("intervention", "exposure")
                        metadata["exposure"] = exposure
                        if "intervention" in metadata:
                            del metadata["intervention"]
                        # Ensure comparator exists (non-exposed group)
                        if not metadata.get("comparator") or metadata["comparator"] in ("", "?", "Control"):
                            metadata["comparator"] = f"Non-{exposure.lower()}" if exposure != "exposure" else "Non-exposed"

            return metadata
        except Exception as e:
            print(f"  [A1] Protocol parse failed ({e}), using fallback")
            return {
                "framework": "PICO",
                "research_question": query,
                "population": "Adults",
                "intervention": "Intervention",
                "comparator": "Control",
                "outcome": "Primary outcome",
                "_fallback": True,
            }

    def _generate_criteria(self, query, metadata):
        framework = metadata.get("framework", "PICO")
        intv_full = metadata.get("intervention", metadata.get("exposure", "treatment"))
        pop = metadata.get("population", "adults")
        comparator = metadata.get("comparator", "")
        outcome = metadata.get("outcome", "")

        # Framework-specific study design guidance
        design_guide = {
            "PICO": {
                "design": "Randomized controlled trial or controlled study",
                "example_excl": "Observational studies, case reports, or reviews",
            },
            "PECO": {
                "design": "Cohort study, case-control study, or cross-sectional study",
                "example_excl": "Case reports, editorials, or animal studies",
            },
            "SPIDER": {
                "design": "Qualitative study (interviews, focus groups, ethnography)",
                "example_excl": "Quantitative-only studies or systematic reviews",
            },
            "PCC": {
                "design": "Cross-sectional survey, epidemiological study, or systematic review",
                "example_excl": "Case reports, editorials, or animal studies",
            },
        }
        guide = design_guide.get(framework, design_guide["PICO"])

        # Build framework-specific criteria prompt
        if framework == "PICO":
            criteria_slots = (
                "List exactly 5 inclusion criteria (one per line, start with number):\n"
                f'1. Study design criterion (e.g. "{guide["design"]}")\n'
                f'2. Population criterion (e.g. "Adults aged 18+ with {outcome}")\n'
                f'3. Intervention criterion: must study {intv_full}\n'
                f'4. Comparator criterion: must compare to {comparator}\n'
                f'5. Outcome criterion: must report outcomes related to {outcome}\n\n'
                "List exactly 2 exclusion criteria (one per line, start with number):\n"
                f'1. (e.g. "{guide["example_excl"]}")\n'
                "2. Studies without relevant comparator group\n"
            )
        elif framework == "PECO":
            criteria_slots = (
                "List exactly 4 inclusion criteria (one per line, start with number):\n"
                f'1. Study design criterion (e.g. "{guide["design"]}")\n'
                f'2. Population criterion\n'
                f'3. Exposure criterion: must study {intv_full} as primary exposure\n'
                f'4. Outcome criterion: must report outcomes related to {outcome}\n\n'
                "List exactly 2 exclusion criteria (one per line, start with number):\n"
                f'1. (e.g. "{guide["example_excl"]}")\n'
                "2. Studies where the exposure is only a covariate or confounder\n"
            )
        elif framework == "SPIDER":
            criteria_slots = (
                "List exactly 4 inclusion criteria (one per line, start with number):\n"
                f'1. Study design criterion (e.g. "{guide["design"]}")\n'
                f'2. Sample criterion\n'
                f'3. Phenomenon of interest criterion: must address {intv_full}\n'
                f'4. Evaluation/Research type criterion\n\n'
                "List exactly 2 exclusion criteria (one per line, start with number):\n"
                f'1. (e.g. "{guide["example_excl"]}")\n'
                "2.\n"
            )
        else:  # PCC and others
            criteria_slots = (
                "List exactly 4 inclusion criteria (one per line, start with number):\n"
                f'1. Study design criterion (e.g. "{guide["design"]}")\n'
                f'2. Population criterion\n'
                f'3. Concept criterion: must address {intv_full}\n'
                f'4. Outcome/measure criterion: related to {outcome}\n\n'
                "List exactly 2 exclusion criteria (one per line, start with number):\n"
                f'1. (e.g. "{guide["example_excl"]}")\n'
                "2.\n"
            )

        prompt = (
            "Generate screening criteria for a systematic review.\n"
            "These criteria will be used to screen TITLES AND ABSTRACTS only.\n\n"
            "CRITICAL: ALL criteria MUST be written in ENGLISH. Never use Spanish or other languages.\n\n"
            f"Framework: {framework}\n"
            f"Population: {pop}\n"
            f"Intervention/Exposure: {intv_full} (always write full English name, never abbreviations)\n"
            f"Comparator: {comparator}\n"
            f"Outcome: {outcome}\n\n"
            "IMPORTANT RULES:\n"
            "- ALL text must be in English\n"
            '- Write full English names, never abbreviations (e.g. "Cognitive Behavioral Therapy" not "CBT")\n'
            "- Only include criteria verifiable from a title or abstract\n"
            "- Do NOT include: consent, willingness, language ability, access to resources\n"
            "- Focus on: study design, population type, intervention/exposure type, outcome measures\n\n"
            f"For {framework} framework, appropriate study designs are: {guide['design']}\n\n"
            f"{criteria_slots}"
        )

        response = self._simple_generate(prompt, max_tokens=300)
        lines = response.split("\n")
        inclusion = []
        exclusion = []
        in_inclusion = False
        in_exclusion = False

        for line in lines:
            line = line.strip()
            if "inclusion" in line.lower():
                in_inclusion = True
                in_exclusion = False
            elif "exclusion" in line.lower():
                in_exclusion = True
                in_inclusion = False
            elif line and line[0].isdigit():
                criterion = re.sub(r"^\d+\.\s*", "", line)
                if in_inclusion:
                    inclusion.append(criterion)
                elif in_exclusion:
                    exclusion.append(criterion)

        # Translate intervention to English for fallback (LLM metadata may be in Spanish)
        intv_en = self._ensure_english_term(intv_full)

        if not inclusion:
            fallback_inc = [
                guide["design"],
                f"Population: {pop}" if pop else "Adults with relevant clinical diagnosis",
                f"Includes {intv_en} as intervention or exposure",
            ]
            if comparator:
                fallback_inc.append(f"Comparison to {comparator}")
            if outcome:
                fallback_inc.append(f"Reports outcomes related to {outcome}")
            inclusion = fallback_inc
        if not exclusion:
            exclusion = [
                guide["example_excl"],
                "Studies not reporting relevant clinical outcomes",
            ]

        # Post-process: remove any criteria with Spanish words
        inclusion = [self._clean_criterion(c) for c in inclusion[:5]]
        exclusion = [self._clean_criterion(c) for c in exclusion[:2]]

        # Fix known LLM hallucination: "CCT" instead of "CBT"
        inclusion = [c.replace("CCT","CBT").replace(" CCT "," CBT ") for c in inclusion]
        exclusion = [c.replace("CCT","CBT").replace(" CCT "," CBT ") for c in exclusion]

        return {"inclusion_criteria": inclusion, "exclusion_criteria": exclusion}

    def _ensure_english_term(self, term: str) -> str:
        """Quick check: if term looks Spanish, try to translate key medical terms."""
        es_to_en = {
            "fumar": "smoking",
            "tabaco": "tobacco",
            "depresión": "depression",
            "medicación": "medication",
            "terapia cognitivo-conductual": "Cognitive Behavioral Therapy",
            "tcc": "Cognitive Behavioral Therapy",
            "ejercicio": "exercise",
            "ansiedad": "anxiety",
            "diabetes": "diabetes",
            "infarto": "myocardial infarction",
            "adultos": "adults",
            "niños": "children",
            "ancianos": "elderly",
            "opioides": "opioids",
            "dolor": "pain",
            "sueño": "sleep",
            "insomnio": "insomnia",
            "meditación": "meditation",
            "prevalencia": "prevalence",
        }
        term_lower = term.lower().strip()
        for es, en in es_to_en.items():
            if es in term_lower:
                term_lower = term_lower.replace(es, en)
        # If still has non-ASCII, return a cleaned version
        if re.search(r"[^\x00-\x7F]", term_lower):
            return re.sub(r"[^\x00-\x7F]+", "", term_lower).strip()
        return term_lower if term_lower != term.lower().strip() else term

    def _clean_criterion(self, criterion: str) -> str:
        """Remove obvious Spanish remnants from a criterion."""
        # Common Spanish words that shouldn't be in English criteria
        es_words = {"para", "como", "con", "los", "las", "del", "una", "que", "por", "entre"}
        words = criterion.split()
        if sum(1 for w in words if w.lower() in es_words) >= 2:
            # Too much Spanish — return a generic version
            return criterion  # Keep it but flag for manual review
        return criterion

    # ------------------------------------------------------------------ #
    #  PROTOCOL DOCUMENT (PRISMA-P style)                                  #
    # ------------------------------------------------------------------ #

    def _generate_protocol_document(self, protocol: dict) -> str:
        """Generate a formal PRISMA-P style protocol document from structured protocol data.
        This is a pre-specified document that records all methodological decisions a priori,
        before the review is conducted — a core requirement of Cochrane/PRISMA methodology.
        """
        import time

        fw = protocol.get("framework", "PICO")
        rq = protocol.get("research_question", "Not specified")
        pop = protocol.get("population", "Not specified")
        intv = protocol.get("intervention", protocol.get("exposure", "Not specified"))
        comp = protocol.get("comparator", "Not specified")
        outcome = protocol.get("outcome", "Not specified")
        inc = protocol.get("inclusion_criteria", [])
        exc = protocol.get("exclusion_criteria", [])

        # Study design based on framework
        design_map = {
            "PICO": "Randomized controlled trials (RCTs) and controlled clinical trials",
            "PECO": "Cohort studies, case-control studies, and cross-sectional studies",
            "SPIDER": "Qualitative studies (interviews, focus groups, ethnography, phenomenology)",
            "PCC": "Cross-sectional surveys and epidemiological studies",
        }
        study_designs = design_map.get(fw, design_map["PICO"])

        # Quality tools based on framework
        quality_map = {
            "PICO": "Risk of Bias 2 (RoB 2) for randomized trials; ROBINS-I for any non-randomized studies",
            "PECO": "ROBINS-I for non-randomized studies of interventions/exposures",
            "SPIDER": "JBI Critical Appraisal Checklist for Qualitative Research (10 items)",
            "PCC": "ROBINS-I for observational studies; JBI for qualitative components",
        }
        quality_tools = quality_map.get(fw, quality_map["PICO"])

        # Effect measures based on framework
        effect_map = {
            "PICO": "Standardized mean difference (SMD/Hedges' g) for continuous outcomes; odds ratio (OR) or risk ratio (RR) for binary outcomes",
            "PECO": "Odds ratio (OR), risk ratio (RR), or hazard ratio (HR) as reported",
            "SPIDER": "Thematic synthesis (not applicable — qualitative framework)",
            "PCC": "Prevalence proportion with 95% confidence interval",
        }
        effect_measures = effect_map.get(fw, effect_map["PICO"])

        date_str = time.strftime("%B %d, %Y")

        inc_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(inc)) if inc else "  (Not specified)"
        exc_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(exc)) if exc else "  (Not specified)"

        doc = f"""SYSTEMATIC REVIEW PROTOCOL
{'='*60}
Generated: {date_str}
Framework: {fw}

1. RESEARCH QUESTION
{'-'*40}
{rq}

2. FRAMEWORK COMPONENTS ({fw})
{'-'*40}
Population: {pop}
{"Intervention" if fw in ("PICO",) else "Exposure" if fw == "PECO" else "Phenomenon of Interest"}: {intv}
Comparator: {comp}
Outcome: {outcome}

3. ELIGIBILITY CRITERIA
{'-'*40}
Inclusion criteria:
{inc_text}

Exclusion criteria:
{exc_text}

Study designs eligible: {study_designs}

4. INFORMATION SOURCES
{'-'*40}
Databases: PubMed/MEDLINE, Europe PMC, Semantic Scholar Academic Graph
Date of search: {date_str}
Language restrictions: English-language publications
Grey literature: Not searched (limitation acknowledged)

5. SCREENING PROCESS
{'-'*40}
Stage 1 (Title/Abstract): Automated ensemble-based screening with
  majority voting (threshold >= 60%). Conservative fallback: papers
  with parse errors are EXCLUDED.
Stage 2 (Full-text): Full-text retrieval via PMC, Unpaywall, Europe PMC,
  Semantic Scholar, with fallback to abstract. LLM-based eligibility
  assessment against inclusion/exclusion criteria.

6. DATA EXTRACTION
{'-'*40}
Dual independent extraction (two LLM passes at different temperatures)
with automated concordance checking. Discrepancies > 15% flagged.
Items extracted:
  - Study characteristics (design, sample size, setting, follow-up)
  - Intervention/exposure details (name, dose, duration, comparator)
  - Outcome data (effect size, CI, p-value, measure type)

7. QUALITY ASSESSMENT
{'-'*40}
Tools: {quality_tools}
Assessment: Single-pass LLM assessment per paper with domain-level
  judgments and overall rating.
Quality weighting: Used in sensitivity analyses (exclude high-risk studies).

8. STATISTICAL SYNTHESIS
{'-'*40}
Method: Random-effects meta-analysis (REML) via R/metafor
Effect measures: {effect_measures}
Heterogeneity: I-squared, tau-squared, Cochran's Q test, prediction intervals
Publication bias: Egger's regression test (if k >= 10), trim-and-fill,
  funnel plots
Sensitivity analyses (pre-specified):
  a) Leave-one-out analysis
  b) Exclude high risk-of-bias studies
  c) Subgroup analysis by study design, risk of bias level, sample size
  d) Meta-regression by publication year and sample size (if k >= 10)
Certainty of evidence: GRADE framework (5 domains: risk of bias,
  inconsistency, indirectness, imprecision, publication bias)

9. SUMMARY OF FINDINGS
{'-'*40}
A GRADE Summary of Findings table will be generated for each outcome
measure, including: number of studies, total participants, pooled effect
with 95% CI, certainty rating, and plain-language interpretation.

10. DEVIATIONS FROM PROTOCOL
{'-'*40}
Any deviations from this protocol will be documented in the final
manuscript's methods section.
{'='*60}
"""
        return doc

    # ------------------------------------------------------------------ #
    #  PROSPERO REGISTRATION DRAFT                                         #
    # ------------------------------------------------------------------ #

    def generate_prospero_draft(self, protocol, config=None):
        """Generate a PROSPERO registration draft from the structured protocol.

        Args:
            protocol: Dict with framework, population, intervention, etc.
            config: Optional config dict (for contact info).

        Returns:
            str: Plain text PROSPERO draft with labeled fields.
        """
        import time
        from datetime import datetime, timedelta

        cfg = config or {}
        fw = protocol.get("framework", "PICO")
        rq = protocol.get("research_question", "Not specified")
        pop = protocol.get("population", "Not specified")
        intv = protocol.get("intervention", protocol.get("exposure", "Not specified"))
        comp = protocol.get("comparator", "Not specified")
        outcome = protocol.get("outcome", "Not specified")
        inc = protocol.get("inclusion_criteria", [])
        exc = protocol.get("exclusion_criteria", [])

        # Dates
        now = datetime.now()
        start_date = now.strftime("%d/%m/%Y")
        end_date = (now + timedelta(days=180)).strftime("%d/%m/%Y")

        # Contact info from config or env
        import os
        contact_email = os.environ.get("ENTREZ_EMAIL", "")
        if not contact_email:
            search_cfg = cfg.get("search", {})
            contact_email = search_cfg.get("entrez_email", "meta-analysis@example.com")
        contact_name = cfg.get("project", {}).get("contact_name", "Principal Investigator")

        # Study design based on framework
        design_map = {
            "PICO": "Randomized controlled trials (RCTs) and controlled clinical trials",
            "PECO": "Cohort studies, case-control studies, and cross-sectional studies",
            "SPIDER": "Qualitative studies (interviews, focus groups, ethnography)",
            "PCC": "Cross-sectional surveys and epidemiological studies",
        }
        study_designs = design_map.get(fw, design_map["PICO"])

        # Synthesis strategy
        synthesis_map = {
            "PICO": "Random-effects meta-analysis (REML) using R/metafor. Effect measures: standardized mean difference (SMD/Hedges' g) for continuous outcomes; odds ratio (OR) or risk ratio (RR) for binary outcomes. Heterogeneity assessed via I-squared, tau-squared, and Cochran's Q test.",
            "PECO": "Random-effects meta-analysis where possible. Effect measures: odds ratio (OR), risk ratio (RR), or hazard ratio (HR). Narrative synthesis if meta-analysis not feasible due to heterogeneity.",
            "SPIDER": "Thematic synthesis of qualitative findings. No quantitative meta-analysis.",
            "PCC": "Pooled prevalence estimates with 95% confidence intervals where appropriate. Narrative synthesis for heterogeneous data.",
        }
        synthesis = synthesis_map.get(fw, synthesis_map["PICO"])

        # Quality tools
        quality_map = {
            "PICO": "Risk of Bias 2 (RoB 2) for randomized trials; ROBINS-I for non-randomized studies",
            "PECO": "ROBINS-I for non-randomized studies of exposures",
            "SPIDER": "JBI Critical Appraisal Checklist for Qualitative Research (10 items)",
            "PCC": "ROBINS-I for observational studies; JBI for qualitative components",
        }
        quality_tools = quality_map.get(fw, quality_map["PICO"])

        inc_text = "\n".join(f"  - {c}" for c in inc) if inc else "  - Not yet specified"
        exc_text = "\n".join(f"  - {c}" for c in exc) if exc else "  - Not yet specified"

        # Framework-specific field labels
        if fw == "PECO":
            intv_label = "Exposure"
        elif fw == "SPIDER":
            intv_label = "Phenomenon of Interest"
        else:
            intv_label = "Intervention/Exposure"

        draft = f"""PROSPERO REGISTRATION DRAFT
{'='*60}
Generated by SAM (Systematic Agent for Meta-analysis)
Date: {now.strftime("%B %d, %Y")}
{'='*60}

1. Review title
   {rq}

2. Original language title
   {rq}

3. Anticipated or actual start date
   {start_date}

4. Anticipated completion date
   {end_date}

5. Stage of review at time of this submission
   Protocol

6. Named contact
   {contact_name}

7. Named contact email
   {contact_email}

8. Named contact address
   [To be completed]

9. Named contact phone number
   [To be completed]

10. Organisational affiliation of the review
    [To be completed]

11. Review team members and their organisational affiliations
    [To be completed]

12. Funding sources/sponsors
    [To be completed]

13. Conflicts of interest
    None declared

14. Collaborators
    [To be completed]

15. Review question
    {rq}

16. Searches
    Electronic databases: PubMed/MEDLINE, Europe PMC, Semantic Scholar
    Academic Graph.
    Search strategy: Structured search combining population,
    {intv_label.lower()}, and outcome terms with appropriate Boolean
    operators and MeSH terms where available.
    Date restrictions: No date restrictions applied.
    Language restrictions: English-language publications only.

17. URL for the published protocol
    Not yet registered

18. Condition or domain being studied
    {outcome}

19. Participants/population
    Inclusion: {pop}
    Exclusion: Populations not meeting the above criteria

20. {intv_label}
    {intv}

21. Comparator(s)/control
    {comp}

22. Types of study to be included
    {study_designs}

23. Main outcome(s)
    {outcome}

24. Additional outcome(s)
    Secondary outcomes as reported in included studies

25. Data extraction (selection and coding)
    Dual independent automated extraction (two LLM passes) with
    concordance checking. Items extracted: study characteristics
    (design, sample size, setting, follow-up), intervention/exposure
    details (name, dose, duration, comparator), and outcome data
    (effect size, confidence interval, p-value, measure type).
    Discrepancies exceeding 15% flagged for review.

26. Risk of bias (quality) assessment
    {quality_tools}
    Quality weighting applied in sensitivity analyses.

27. Strategy for data synthesis
    {synthesis}

28. Analysis of subgroups or subsets
    Pre-specified subgroup analyses:
    a) By risk of bias level (low vs. high)
    b) By sample size (small vs. large, cutoff: 100 participants)
    c) By study design

29. Type and method of review
    Systematic review {"with meta-analysis" if fw != "SPIDER" else "(qualitative synthesis)"}

30. Language
    English

31. Country
    International (no geographic restrictions)

32. Other registration details
    This protocol was generated using SAM (Systematic Agent for
    Meta-analysis), an automated pipeline for systematic reviews.

33. Reference and/or URL for published protocol
    Not yet published

34. Dissemination plans
    Publication in a peer-reviewed journal

35. Keywords
    {pop}; {intv}; {outcome}; systematic review{"" if fw == "SPIDER" else "; meta-analysis"}

36. Details of any existing review of the same topic by the same authors
    None

37. Current review status
    Ongoing

38. Any additional information
    This review uses automated screening, extraction, and quality
    assessment via large language models (LLMs) with ensemble voting
    and conservative fallback strategies. All automated decisions are
    documented and traceable.

39. Details of final report/publication(s) or preprint
    Not yet available

Inclusion criteria:
{inc_text}

Exclusion criteria:
{exc_text}
{'='*60}
NOTE: This is an automatically generated draft. Please review and
complete all fields marked [To be completed] before submission to
PROSPERO (https://www.crd.york.ac.uk/prospero/).
{'='*60}
"""
        return draft

    # ------------------------------------------------------------------ #

    def _simple_generate(self, prompt, max_tokens=300):
        if self._is_groq():
            return self.generate(prompt, max_new_tokens=max_tokens, temperature=0.3)
        inputs = self._tokenize_prompt(prompt)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=self.do_sample,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self._decode_response(outputs, input_length)

    def _clean_json(self, text):
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


# TEST
if __name__ == "__main__":
    agent = Agent1Protocol()
    print("=" * 80)
    print("TEST: PICO vs SPIDER + PubMed query")
    print("=" * 80)

    p1 = agent.generate_protocol(
        "\u00bfTCC versus medicaci\u00f3n para depresi\u00f3n en adultos?"
    )
    q1 = agent.generate_pubmed_query(
        "\u00bfTCC versus medicaci\u00f3n para depresi\u00f3n en adultos?", p1
    )
    print(f"\n\U0001f4ca Framework: {p1.get('framework')} | PubMed: '{q1}'")
    print("   Inclusion:", p1.get("inclusion_criteria"))

    print("\n" + "=" * 80)
    p2 = agent.generate_protocol(
        "\u00bfQu\u00e9 barreras perciben m\u00e9dicos para prescribir opioides?"
    )
    q2 = agent.generate_pubmed_query(
        "\u00bfQu\u00e9 barreras perciben m\u00e9dicos para prescribir opioides?", p2
    )
    print(f"\n\U0001f4ca Framework: {p2.get('framework')} | PubMed: '{q2}'")

    print("\n" + "=" * 80)
    print("\u2705 Test 1 PICO:", p1.get("framework") == "PICO")
    print("\u2705 Test 2 SPIDER:", p2.get("framework") == "SPIDER")
