"""Prompts Agent 1 - ENGLISH VERSION (Qwen funciona mejor en inglés)"""

AGENT1_PROMPT_ENGLISH = """You are a Cochrane expert. Generate protocol JSON.

FRAMEWORKS:
- PICO: intervention/treatment comparison
- SPIDER: qualitative experiences
- PECO: risk factors/etiology
- PCC: prevalence

EXAMPLE:
Query: "Do statins prevent MI in diabetics?"
{
  "framework": "PICO",
  "research_question": "Do statins reduce MI incidence in diabetic patients?",
  "population": "Adults with type 2 diabetes without prior CVD",
  "intervention": "Statins moderate-high dose",
  "comparator": "Placebo or standard care",
  "outcome": "MI incidence at 1+ year",
  "inclusion_criteria": ["RCT", "Follow-up ≥1 year", "n≥100"],
  "exclusion_criteria": ["Secondary prevention", "Liver disease"]
}

YOUR TURN:
Query: "{query}"

Respond ONLY valid JSON (no extra text):
"""


def format_agent1_prompt_v2(user_query):
    return AGENT1_PROMPT_ENGLISH.replace("{query}", user_query)


TEMPERATURE_CONFIG = {
    "agent1_protocol": 0.2,  # MÁS bajo
    "agent3_screening": 0.2,
    "agent5_extraction": 0.1,
    "agent6_quality": 0.2,
    "agent8_manuscript": 0.7,
}

# Otros agentes (sin cambios)
AGENT3_SCREENING_TEMPLATE = """Cochrane screening expert. Evaluate if this paper meets criteria.

INCLUSION CRITERIA:
{inclusion_criteria}

EXCLUSION CRITERIA:
{exclusion_criteria}

PAPER:
Title: {title}
Abstract: {abstract}

DECISION RULES:
- INCLUDE   : clearly meets inclusion, no exclusion triggered
- EXCLUDE   : clearly fails inclusion OR triggers exclusion
- UNCERTAIN : insufficient info in title/abstract to decide

Respond ONLY with valid JSON:
{{"vote": "INCLUDE or EXCLUDE or UNCERTAIN", "reasoning": "one sentence"}}"""


AGENT5_EXTRACTION_TEMPLATE = """Extract data with source_page.
Results: {results_text}
JSON: {{"outcomes": [...], "flags": [...]}}
"""

AGENT5_OUTCOMES_TEMPLATE = """You are a systematic review data extractor specializing in NUMERIC effect sizes for meta-analysis. Extract numerical outcomes from this paper's results.

PAPER TITLE: {title}
OUTCOME OF INTEREST: {outcome_of_interest}

RESULTS TEXT:
{results_text}

INSTRUCTIONS:
1. Extract EXACT numeric values from the text. If a value is not reported, use null. Do NOT describe results in prose.
2. For effect_size_type, use ONLY one of: SMD, OR, RR, HR, MD, PROP
3. If the paper reports means and SDs per group, set effect_size_type="MD", effect_size_value = mean_intervention - mean_control, and fill in mean_intervention, sd_intervention, mean_control, sd_control.
4. If the paper reports OR/RR with 95% CI, extract all three numbers (value, ci_lower, ci_upper).
5. If only a p-value and sample sizes are given, still report them — downstream analysis can use them.

HANDLING TABLES: Data often appears in tables where columns are separated by spaces. Look for patterns like:
- Column headers: "Group   Mean   SD   N" followed by rows of numbers
- Parenthetical CIs: "-1.59 (-0.94, -2.24)" means value=-1.59, ci_lower=-2.24, ci_upper=-0.94
- Mean ± SD format: "10.8 ± 3.1" means mean=10.8, sd=3.1

MULTI-ARM TRIALS: If the study has 3+ groups, extract the PRIMARY intervention vs control comparison. If there are multiple intervention arms, create one outcome per arm vs control.

Extract ALL reported outcomes as a JSON array. For EACH outcome provide these EXACT fields:
- name: outcome measure name (e.g. "Weight change at 12 months - behavioral vs control")
- effect_size_type: one of "SMD", "OR", "RR", "HR", "MD", "PROP" (uppercase, required)
- effect_size_value: the main numeric effect size as a float (e.g. 0.45, 1.82, -0.33) or null
- ci_lower: lower bound of 95% CI as a float (e.g. 0.21) or null
- ci_upper: upper bound of 95% CI as a float (e.g. 0.69) or null
- p_value: p-value as a float (e.g. 0.003) or null
- se: standard error of the effect size as a float or null
- n_intervention: number of participants in intervention/exposed group (integer) or null
- n_control: number of participants in control/unexposed group (integer) or null
- n_total: total number of participants (integer) or null
- mean_intervention: mean value in intervention group (float) or null
- sd_intervention: SD in intervention group (float) or null
- mean_control: mean value in control group (float) or null
- sd_control: SD in control group (float) or null

EXAMPLE 1 (means and SDs):
[{{"name": "Weight change (kg)", "effect_size_type": "MD", "effect_size_value": -3.5, "ci_lower": -5.2, "ci_upper": -1.8, "p_value": 0.006, "se": null, "n_intervention": 45, "n_control": 43, "n_total": 88, "mean_intervention": -4.2, "sd_intervention": 6.1, "mean_control": -0.7, "sd_control": 7.2}}]

EXAMPLE 2 (multi-arm, table data):
[{{"name": "Weight change at 12m - clinic vs control", "effect_size_type": "MD", "effect_size_value": -2.42, "ci_lower": null, "ci_upper": null, "p_value": 0.01, "se": null, "n_intervention": 37, "n_control": 58, "n_total": 95, "mean_intervention": -1.21, "sd_intervention": null, "mean_control": 1.21, "sd_control": null}}, {{"name": "Weight change at 12m - behavioral vs control", "effect_size_type": "MD", "effect_size_value": -3.03, "ci_lower": null, "ci_upper": null, "p_value": 0.01, "se": null, "n_intervention": 38, "n_control": 58, "n_total": 96, "mean_intervention": -1.82, "sd_intervention": null, "mean_control": 1.21, "sd_control": null}}]

Respond ONLY with a valid JSON array:"""

AGENT5_NUMERIC_RECOVERY_TEMPLATE = """You are a numeric data extraction specialist. A previous extraction attempt missed numeric effect sizes from this paper. Focus ONLY on extracting numbers.

PAPER TITLE: {title}

TEXT:
{text}

Extract ONLY numeric values for meta-analysis. For each outcome you can find, provide:
- name: outcome name (include arm label if multi-arm, e.g. "Weight change - behavioral vs control")
- effect_size_type: one of "SMD", "OR", "RR", "HR", "MD", "PROP"
- effect_size_value: the numeric effect size (float). For means difference: intervention_mean - control_mean
- ci_lower: lower 95% CI bound (float or null)
- ci_upper: upper 95% CI bound (float or null)
- p_value: p-value (float or null)
- se: standard error (float or null)
- n_intervention: intervention group N (int or null)
- n_control: control group N (int or null)
- n_total: total N (int or null)
- mean_intervention: group mean (float or null)
- sd_intervention: group SD (float or null)
- mean_control: group mean (float or null)
- sd_control: group SD (float or null)

Look for these patterns in the text:
- "d = 0.45" or "Cohen's d" or "Hedges' g" → effect_size_type = "SMD"
- "OR = 2.1 (95% CI: 1.2-3.8)" → effect_size_type = "OR", value = 2.1, ci_lower = 1.2, ci_upper = 3.8
- "RR = 0.75, 95% CI [0.58, 0.97]" → effect_size_type = "RR"
- "HR = 1.45 (1.10-1.92)" → effect_size_type = "HR"
- "p < 0.001" → p_value = 0.001
- TABLE DATA: rows with numbers separated by spaces. Example:
  "Weight change  -1.59  (-0.94, -2.24)  -1.21  (+0.16, -2.58)"
  → mean=-1.59, ci_lower=-2.24, ci_upper=-0.94 for one group
- MEAN ± SD: "10.8 ± 3.1" → mean=10.8, sd=3.1. Compute MD = mean_intervention - mean_control.
- If 3+ groups exist, extract primary intervention vs control. Create separate entries for each arm vs control.

Respond ONLY with a valid JSON array (even if only one outcome):"""

AGENT6_ROB2_TEMPLATE = """You are a Cochrane methodologist performing Risk of Bias 2.0 (RoB 2) assessment for a randomized controlled trial.

STUDY TITLE: {title}

PROTOCOL CONTEXT:
- Population: {population}
- Intervention: {intervention}
- Outcome: {outcome}

STUDY TEXT (methods + results):
{study_text}

Assess the following 5 RoB 2 domains. For each domain, assign a judgment and provide a one-sentence justification.

DOMAINS:
1. randomization: Bias arising from the randomization process
   - "Low": Adequate sequence generation AND allocation concealment described
   - "Some concerns": Sequence generation or concealment not fully described
   - "High": No randomization or clearly inadequate

2. deviations: Bias due to deviations from intended interventions
   - "Low": Participants and personnel blinded, OR deviations unlikely to affect outcome
   - "Some concerns": Unclear blinding with potential for deviations
   - "High": No blinding and deviations likely affected outcome

3. missing_data: Bias due to missing outcome data
   - "Low": <5% missing data OR appropriate handling (ITT, imputation)
   - "Some concerns": 5-20% missing with unclear handling
   - "High": >20% missing OR differential attrition without imputation

4. measurement: Bias in measurement of the outcome
   - "Low": Outcome assessors blinded OR objective measure
   - "Some concerns": Self-report measure without assessor blinding
   - "High": Unblinded subjective assessment likely influenced by intervention

5. selection: Bias in selection of the reported result
   - "Low": Pre-registered protocol with matching reported outcomes
   - "Some concerns": No pre-registration but results appear complete
   - "High": Evidence of selective reporting or outcome switching

OVERALL JUDGMENT RULE: Use the LEAST favorable domain judgment.
- All "Low" → overall "Low"
- Any "Some concerns" (none "High") → overall "Some concerns"
- Any "High" → overall "High"

Respond ONLY with valid JSON:
{{"randomization": {{"judgment": "Low|Some concerns|High", "justification": "..."}}, "deviations": {{"judgment": "...", "justification": "..."}}, "missing_data": {{"judgment": "...", "justification": "..."}}, "measurement": {{"judgment": "...", "justification": "..."}}, "selection": {{"judgment": "...", "justification": "..."}}, "overall": "Low|Some concerns|High", "overall_justification": "one sentence summary"}}"""


AGENT6_ROBINS_I_TEMPLATE = """You are a Cochrane methodologist performing ROBINS-I (Risk Of Bias In Non-randomised Studies of Interventions) assessment for an observational study.

STUDY TITLE: {title}

PROTOCOL CONTEXT:
- Population: {population}
- Exposure/Intervention: {intervention}
- Outcome: {outcome}

STUDY TEXT (methods + results):
{study_text}

Assess the following 7 ROBINS-I domains. For each domain, assign a judgment and provide a one-sentence justification.

DOMAINS:
1. confounding: Bias due to confounding
   - "Low": All critical confounders measured and appropriately adjusted for
   - "Moderate": Some confounders not fully adjusted but unlikely to change conclusions
   - "Serious": Important confounders not controlled, likely to affect estimate
   - "Critical": Confounding inherently inseparable from intervention effect
   - "No information": Insufficient information to judge

2. selection_of_participants: Bias in selection of participants into the study
   - "Low": All participants eligible, start of follow-up and intervention well defined
   - "Moderate": Selection may be related to intervention and outcome, but bias unlikely to be important
   - "Serious": Selection strongly related to both intervention and outcome
   - "Critical": Selection virtually certain to produce substantial bias
   - "No information": Insufficient information to judge

3. classification_of_interventions: Bias in classification of interventions
   - "Low": Intervention status well defined and based solely on information at time of intervention
   - "Moderate": Some aspects of classification may be affected by knowledge of outcome
   - "Serious": Classification of intervention likely differential (affected by outcome knowledge)
   - "Critical": Classification so problematic that any estimate is unreliable
   - "No information": Insufficient information to judge

4. deviations_from_interventions: Bias due to deviations from intended interventions
   - "Low": No deviations or deviations balanced and unlikely to affect outcome
   - "Moderate": Some deviations occurred but unlikely to substantially affect outcome
   - "Serious": Deviations likely affected the outcome and were not appropriately adjusted for
   - "Critical": Deviations substantially altered the comparison being made
   - "No information": Insufficient information to judge

5. missing_data: Bias due to missing data
   - "Low": Data reasonably complete or adequate analysis methods used
   - "Moderate": Some missing data but proportions and reasons similar across groups
   - "Serious": Substantial missing data likely related to both intervention and outcome
   - "Critical": Critical data missing making any estimate unreliable
   - "No information": Insufficient information to judge

6. measurement_of_outcomes: Bias in measurement of outcomes
   - "Low": Outcome measures objective or assessors blinded to intervention
   - "Moderate": Outcome assessment may have been influenced by knowledge of intervention
   - "Serious": Assessment of outcome clearly influenced by intervention knowledge
   - "Critical": Measurement of outcome so problematic results are unreliable
   - "No information": Insufficient information to judge

7. selection_of_results: Bias in selection of the reported result
   - "Low": Pre-specified analysis plan and reported results consistent
   - "Moderate": Unclear pre-specification but results appear complete
   - "Serious": Outcome measurements or analyses are clearly selected to produce a favorable result
   - "Critical": Reported results almost certainly selected from multiple analyses
   - "No information": Insufficient information to judge

OVERALL JUDGMENT RULE: Use the WORST-CASE domain judgment.
- All "Low" → overall "Low"
- Worst is "Moderate" → overall "Moderate"
- Any "Serious" (none "Critical") → overall "Serious"
- Any "Critical" → overall "Critical"
- All "No information" → overall "No information"

Respond ONLY with valid JSON:
{{"confounding": {{"judgment": "Low|Moderate|Serious|Critical|No information", "justification": "..."}}, "selection_of_participants": {{"judgment": "...", "justification": "..."}}, "classification_of_interventions": {{"judgment": "...", "justification": "..."}}, "deviations_from_interventions": {{"judgment": "...", "justification": "..."}}, "missing_data": {{"judgment": "...", "justification": "..."}}, "measurement_of_outcomes": {{"judgment": "...", "justification": "..."}}, "selection_of_results": {{"judgment": "...", "justification": "..."}}, "overall": "Low|Moderate|Serious|Critical|No information", "overall_justification": "one sentence summary"}}"""


AGENT6_JBI_TEMPLATE = """You are a Cochrane methodologist performing JBI Critical Appraisal for a qualitative research study.

STUDY TITLE: {title}

PROTOCOL CONTEXT:
- Population: {population}
- Phenomenon of Interest: {intervention}
- Outcome: {outcome}

STUDY TEXT (methods + results):
{study_text}

Assess the following 10 JBI Critical Appraisal Checklist items for Qualitative Research. For each item, assign "Yes", "No", "Unclear", or "Not applicable" and provide a one-sentence justification.

ITEMS:
1. philosophical_perspective: Is there congruity between the stated philosophical perspective and the research methodology?
2. methodology_question: Is there congruity between the research methodology and the research question or objectives?
3. methodology_methods: Is there congruity between the research methodology and the methods used to collect data?
4. methodology_analysis: Is there congruity between the research methodology and the representation and analysis of data?
5. methodology_interpretation: Is there congruity between the research methodology and the interpretation of results?
6. researcher_positioning: Is there a statement locating the researcher culturally or theoretically?
7. researcher_influence: Is the influence of the researcher on the research, and vice-versa, addressed?
8. participant_voices: Are participants, and their voices, adequately represented?
9. ethics: Is the research ethical according to current criteria (evidence of ethical approval and informed consent)?
10. conclusions: Do the conclusions drawn in the research report flow from the analysis or interpretation of the data?

OVERALL JUDGMENT RULE:
- Count the number of "Yes" answers.
- 8-10 "Yes" → overall "Include"
- 5-7 "Yes" → overall "Include with caution"
- 0-4 "Yes" → overall "Exclude"

Respond ONLY with valid JSON:
{{"philosophical_perspective": {{"answer": "Yes|No|Unclear|Not applicable", "justification": "..."}}, "methodology_question": {{"answer": "...", "justification": "..."}}, "methodology_methods": {{"answer": "...", "justification": "..."}}, "methodology_analysis": {{"answer": "...", "justification": "..."}}, "methodology_interpretation": {{"answer": "...", "justification": "..."}}, "researcher_positioning": {{"answer": "...", "justification": "..."}}, "researcher_influence": {{"answer": "...", "justification": "..."}}, "participant_voices": {{"answer": "...", "justification": "..."}}, "ethics": {{"answer": "...", "justification": "..."}}, "conclusions": {{"answer": "...", "justification": "..."}}, "yes_count": 0, "overall": "Include|Include with caution|Exclude", "overall_justification": "one sentence summary"}}"""


AGENT6_DESIGN_DETECT_TEMPLATE = """You are a research methods expert. Determine the study design of this paper.

TITLE: {title}
TEXT: {study_text}

Classify as ONE of: "rct", "observational", "qualitative"

Rules:
- "rct" if: randomized controlled trial, randomized clinical trial, cluster-randomized trial, crossover RCT
- "observational" if: cohort, case-control, cross-sectional, retrospective, prospective observational, case series
- "qualitative" if: qualitative research, phenomenological study, grounded theory, ethnographic study, narrative inquiry, thematic analysis, focus groups, qualitative interviews, interpretive description, content analysis (qualitative)

Respond ONLY with valid JSON:
{{"study_design": "rct|observational|qualitative", "reasoning": "one sentence"}}"""


def format_agent6_rob2_prompt(paper, protocol):
    """Format RoB 2 prompt for an RCT paper."""
    study_text = _get_a6_study_text(paper)
    return AGENT6_ROB2_TEMPLATE.format(
        title=paper.get("title", ""),
        population=protocol.get("population", ""),
        intervention=protocol.get("intervention", protocol.get("exposure", "")),
        outcome=protocol.get("outcome", ""),
        study_text=study_text,
    )


def format_agent6_robins_i_prompt(paper, protocol):
    """Format ROBINS-I prompt for an observational study."""
    study_text = _get_a6_study_text(paper)
    return AGENT6_ROBINS_I_TEMPLATE.format(
        title=paper.get("title", ""),
        population=protocol.get("population", ""),
        intervention=protocol.get("intervention", protocol.get("exposure", "")),
        outcome=protocol.get("outcome", ""),
        study_text=study_text,
    )


def format_agent6_jbi_prompt(paper, protocol):
    """Format JBI Critical Appraisal prompt for a qualitative study."""
    study_text = _get_a6_study_text(paper)
    return AGENT6_JBI_TEMPLATE.format(
        title=paper.get("title", ""),
        population=protocol.get("population", ""),
        intervention=protocol.get("intervention", protocol.get("phenomenon_of_interest", protocol.get("exposure", ""))),
        outcome=protocol.get("outcome", ""),
        study_text=study_text,
    )


def format_agent6_design_prompt(paper):
    """Format study design detection prompt."""
    study_text = _get_a6_study_text(paper, max_chars=2000)
    return AGENT6_DESIGN_DETECT_TEMPLATE.format(
        title=paper.get("title", ""),
        study_text=study_text,
    )


def _get_a6_study_text(paper, max_chars=5000):
    """Extract methods+results text from a paper for A6 assessment."""
    parts = []
    # Try structured sections first
    sc = paper.get("study_characteristics") or {}
    if sc:
        parts.append(f"Study design: {sc.get('study_design', 'unknown')}")
        parts.append(f"Sample size: {sc.get('sample_size', 'unknown')}")
        parts.append(f"Follow-up: {sc.get('followup_duration', 'unknown')}")

    sections = paper.get("sections") or {}
    for sec_name in ("methods", "results"):
        if sec_name in sections:
            parts.append(sections[sec_name])

    if not parts or len(" ".join(parts)) < 100:
        ft = paper.get("full_text", "") or paper.get("abstract", "") or ""
        parts.append(ft)

    combined = "\n\n".join(str(p) for p in parts if p)
    return combined[:max_chars]


def format_agent3_prompt(paper, protocol):
    return AGENT3_SCREENING_TEMPLATE.format(
        inclusion_criteria=protocol.get("inclusion_criteria"),
        exclusion_criteria=protocol.get("exclusion_criteria"),
        title=paper["title"],
        abstract=paper["abstract"],
    )


def format_agent5_prompt(results_text):
    return AGENT5_EXTRACTION_TEMPLATE.format(results_text=results_text)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 8 — MANUSCRIPT GENERATION WITH RAG
# ══════════════════════════════════════════════════════════════════════════════

# These prompts are DATA-DRIVEN: the caller pre-injects actual numbers, study
# lists, and synthesis results so the LLM only needs to write connecting prose.
# Placeholders filled by get_a8_section_prompt():
#   {protocol_summary}, {synthesis_summary}, {quality_summary}, {n_studies},
#   {rag_context}, {prisma_text}, {study_list_text}, {methods_data_block}

A8_SECTION_PROMPTS = {
    "abstract": """Write a structured abstract. Use ONLY the data provided. Write exactly 4 paragraphs.

DATA:
{synthesis_summary}
{quality_summary}

BACKGROUND: Two sentences. Why does this clinical question matter? State the question: {protocol_summary}

METHODS: One sentence summarizing the search. One sentence on eligibility. One sentence on analysis approach.

RESULTS: {n_studies} studies were included. State the effect sizes and confidence intervals from DATA above. State quality concerns if any.

CONCLUSION: One sentence with the main finding. One sentence with a clinical implication.

RULES:
- Each sentence must be under 20 words. Use a period after each sentence.
- Do NOT repeat yourself. Write each fact only once.
- No citations. No LaTeX. Plain text only.
- If no pooled estimate exists, say "Quantitative pooling was not feasible."
- STOP after CONCLUSION. Do not write anything else.""",

    "introduction": """Write exactly 4 paragraphs for the Introduction. Each paragraph should have 3-4 sentences.

QUESTION: {protocol_summary}

STUDIES FOR CONTEXT:
{rag_context}

Paragraph 1: What is the condition? How common is it? Why does treatment matter?
Paragraph 2: What treatments exist? Mention studies from CONTEXT as (Author, Year). Do not invent names.
Paragraph 3: What is the gap? Why is a new review needed?
Paragraph 4: State the aim of this review in one clear sentence.

RULES:
- Maximum 20 words per sentence. Use simple sentence structure: subject, verb, object.
- Do NOT write labels like "Paragraph 1:" — just write the text.
- Do NOT use the phrase "previous research" — cite specific studies or write general statements.
- No LaTeX. Plain text only.
- STOP after paragraph 4.""",

    "methods": """Write the Methods section using ONLY the facts below. Do not add anything.

FACTS:
{methods_data_block}

Write 6 short paragraphs:
1. Search strategy: which databases, what search terms.
2. Eligibility criteria: what was included, what was excluded.
3. Screening: how papers were screened (automated, two stages).
4. Data extraction: what data was extracted from each study.
5. Quality assessment: which tools were used (RoB 2, ROBINS-I, JBI).
6. Statistical analysis: random-effects model, effect measures, heterogeneity tests.

RULES:
- Maximum 20 words per sentence. Each paragraph should have 2-4 sentences.
- Do NOT say "two reviewers" — this used automated screening.
- Do NOT add methods not listed in FACTS above.
- No citations. No LaTeX. Plain text only.
- STOP after the statistical analysis paragraph.""",

    "results": """Write the Results section using ONLY the data below. Do not invent any numbers.

STUDY SELECTION:
{prisma_text}

INCLUDED STUDIES:
{study_list_text}

QUALITY:
{quality_summary}

META-ANALYSIS:
{synthesis_summary}

Write 4-5 paragraphs:
1. Study selection: state the PRISMA numbers exactly as given above.
2. Study characteristics: briefly describe each included study. Cite as (Author, Year).
3. Quality assessment: summarize the risk of bias findings.
4. Meta-analysis results: report exact effect sizes and CIs from the data above.
5. If sensitivity/publication bias data is available, report it. Otherwise skip this paragraph.

RULES:
- Maximum 20 words per sentence.
- Use ONLY the numbers given above. If a value is missing, write "not reported."
- Do NOT fabricate sample sizes, durations, or effect sizes.
- No LaTeX. Plain text only.
- STOP after the last paragraph.""",

    "discussion": """Write exactly 5 paragraphs for the Discussion. Each paragraph should have 3-5 sentences.

QUESTION: {protocol_summary}

RESULTS:
{synthesis_summary}

QUALITY:
{quality_summary}

LITERATURE:
{rag_context}

Paragraph 1: Summarize the main findings using exact numbers from RESULTS above.
Paragraph 2: Compare with existing evidence. Cite studies from LITERATURE as (Author, Year).
Paragraph 3: Strengths — systematic search, quality assessment, pre-specified protocol.
Paragraph 4: Limitations — small number of studies, heterogeneity, abstract-only screening, missing data.
Paragraph 5: Clinical implications and future research directions.

RULES:
- Maximum 20 words per sentence. Write short, clear sentences.
- Do NOT write section labels like "SUMMARY:" or "LIMITATIONS:" — just write the text.
- Do NOT fabricate claims or statistics not in the data.
- No LaTeX. Plain text only.
- STOP after paragraph 5.""",

    "conclusion": """Write exactly 2 paragraphs for the Conclusion. Each paragraph should have 2-3 sentences.

QUESTION: {protocol_summary}

RESULTS:
{synthesis_summary}

Paragraph 1: State the main finding. Include the key numbers if available. State the certainty level.
Paragraph 2: One clinical implication. One direction for future research.

RULES:
- Maximum 20 words per sentence.
- No citations. No LaTeX. Plain text only.
- Do NOT invent numbers. If no pooled estimate exists, say so.
- STOP after paragraph 2. Do not write anything else.""",
}

# RAG query templates per section
_A8_RAG_QUERIES = {
    "abstract": "main findings pooled effect outcomes summary {topic}",
    "introduction": "background context clinical relevance previous evidence {topic}",
    "methods": "study design methodology inclusion criteria statistical analysis {topic}",
    "results": "study characteristics sample size intervention outcomes results {topic}",
    "discussion": "comparison implications limitations findings evidence {topic}",
    "conclusion": "main conclusion clinical implication future research {topic}",
}

# Variable k (number of RAG chunks) per section
A8_RAG_K = {
    "abstract": 4,
    "introduction": 6,
    "methods": 4,
    "results": 10,
    "discussion": 8,
    "conclusion": 4,
}


def get_a8_rag_query(section_name: str, protocol: dict,
                     outcome_names: list = None, synthesis_keys: list = None) -> str:
    """Build a RAG query for a manuscript section.

    For Results and Discussion, appends extracted outcome names and synthesis
    group keys so RAG retrieves outcome-specific chunks (not just generic text).
    """
    template = _A8_RAG_QUERIES.get(section_name, "")
    if not template:
        return ""
    topic = protocol.get("research_question", protocol.get("population", ""))
    query = template.format(topic=topic)

    # Enhance Results/Discussion queries with actual outcome names
    if section_name in ("results", "discussion"):
        extras = []
        if outcome_names:
            # Top 10 unique outcome names (truncated)
            unique = list(dict.fromkeys(outcome_names))[:10]
            extras.extend(n[:50] for n in unique)
        if synthesis_keys:
            extras.extend(synthesis_keys[:5])
        if extras:
            query += " " + " ".join(extras)

    return query


def get_a8_section_prompt(
    section_name: str,
    protocol: dict,
    rag_context: str,
    synthesis_summary: str,
    quality_summary: str,
    n_studies: int,
    prisma_text: str = "",
    study_list_text: str = "",
    methods_data_block: str = "",
    **kwargs,
) -> str:
    """Build the full prompt for an A8 manuscript section.

    The caller (Agent8) pre-builds data blocks (prisma_text, study_list_text,
    methods_data_block) so the LLM only writes connecting prose.
    """
    template = A8_SECTION_PROMPTS.get(section_name, "")
    if not template:
        return ""

    # Compact protocol summary — single line for clarity
    rq = protocol.get("research_question", "")
    pop = protocol.get("population", "?")
    intv = protocol.get("intervention", protocol.get("exposure", "?"))
    comp = protocol.get("comparator", "?")
    outc = protocol.get("outcome", "?")
    protocol_summary = rq if rq else f"{intv} vs {comp} for {outc} in {pop}"

    filled = template.format(
        protocol_summary=protocol_summary,
        rag_context=rag_context if rag_context else "(No additional context available)",
        synthesis_summary=synthesis_summary if synthesis_summary else "No quantitative synthesis was performed due to insufficient data.",
        quality_summary=quality_summary if quality_summary else "Quality assessment data not available.",
        n_studies=n_studies,
        prisma_text=prisma_text if prisma_text else "(PRISMA data not available)",
        study_list_text=study_list_text if study_list_text else "(Study list not available)",
        methods_data_block=methods_data_block if methods_data_block else "(Methods details not available)",
    )

    # Add language instruction if specified
    language = kwargs.get("language", "en")
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, "")
    if lang_instruction and language != "en":
        filled = lang_instruction + "\n\n" + filled

    return filled


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE-AWARE PROMPTS FOR MANUSCRIPT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# Language instructions prepended to section prompts
_LANGUAGE_INSTRUCTIONS = {
    "en": "Write in academic English.",
    "es": (
        "IMPORTANTE: Escribe TODA la sección en español académico. "
        "Usa terminología científica estándar en español. "
        "Los datos y números no cambian, solo la prosa debe estar en español."
    ),
    "pt": (
        "IMPORTANTE: Escreva TODA a seção em português acadêmico. "
        "Use terminologia científica padrão em português. "
        "Os dados e números não mudam, apenas a prosa deve estar em português."
    ),
    "fr": (
        "IMPORTANT: Rédigez TOUTE la section en français académique. "
        "Utilisez la terminologie scientifique standard en français. "
        "Les données et les chiffres ne changent pas, seule la prose doit être en français."
    ),
    "de": (
        "WICHTIG: Schreiben Sie den GESAMTEN Abschnitt in akademischem Deutsch. "
        "Verwenden Sie wissenschaftliche Standardterminologie auf Deutsch. "
        "Die Daten und Zahlen ändern sich nicht, nur die Prosa soll auf Deutsch sein."
    ),
}

# Section-specific Spanish prompts (override A8_SECTION_PROMPTS for "es")
# The data blocks stay in English because they are factual; only instructions change.
_A8_SECTION_PROMPTS_ES = {
    "abstract": """Escribe ÚNICAMENTE un resumen estructurado para esta revisión sistemática. Usa EXACTAMENTE los datos de abajo. NO inventes ningún número.

PREGUNTA DE INVESTIGACIÓN: {protocol_summary}

DATOS A USAR:
{synthesis_summary}

{quality_summary}

FORMATO — escribe exactamente estos 4 párrafos etiquetados:

ANTECEDENTES: Escribe 1-2 oraciones sobre por qué esta pregunta clínica es importante.

MÉTODOS: {methods_data_block}

RESULTADOS: Se incluyeron {n_studies} estudios. Reporta el efecto pooled, intervalo de confianza, I-cuadrado y certeza GRADE usando los números exactos de DATOS.

CONCLUSIÓN: Escribe 1-2 oraciones con el hallazgo principal y una implicación clínica basada en los datos anteriores.

REGLAS DE ESCRITURA:
- Oraciones cortas y claras (máximo 25 palabras cada una).
- Sin citas. Sin referencias. Sin notas. Sin LaTeX. Solo texto plano.
- NUNCA inventes números, duraciones, tamaños muestrales o porcentajes que no estén en los DATOS.
- Detente después de CONCLUSIÓN.""",

    "introduction": """Escribe una Introducción (3-4 párrafos cortos) para una revisión sistemática.

PREGUNTA DE INVESTIGACIÓN: {protocol_summary}

CONTEXTO DE LOS ESTUDIOS INCLUIDOS:
{rag_context}

Párrafo 1: Antecedentes sobre la condición y la población — por qué es clínicamente relevante.
Párrafo 2: Qué se sabe actualmente. Cita estudios como [Autor, Año] usando SOLO los nombres anteriores.
Párrafo 3: Por qué se necesita una nueva revisión sistemática — qué vacío existe.
Párrafo 4: Establece el objetivo: "{protocol_summary}"

REGLAS DE ESCRITURA:
- Oraciones cortas y claras (máximo 25 palabras cada una). Evita oraciones interminables.
- Sin comandos LaTeX. Sin sección de referencias. Solo texto plano.
- Solo cita estudios del CONTEXTO anterior — NUNCA inventes nombres de autores o años.
- NUNCA fabriques estadísticas, duraciones o tamaños muestrales no proporcionados arriba.
- Detente después del párrafo 4.""",

    "methods": """Escribe la sección de Métodos para esta revisión sistemática usando ÚNICAMENTE la información de abajo. NO añadas nada que no esté indicado aquí.

{methods_data_block}

Escribe como prosa académica fluida (sin viñetas, sin listas numeradas). Organiza en estos párrafos:
- Estrategia de búsqueda y bases de datos
- Criterios de elegibilidad
- Proceso de selección y cribado
- Extracción de datos
- Evaluación de la calidad
- Análisis estadístico

REGLAS DE ESCRITURA:
- Oraciones cortas y claras (máximo 25 palabras cada una). Evita oraciones interminables.
- Sin citas. Sin LaTeX. Sin referencias. Solo texto plano.
- No digas "dos revisores" — este fue un proceso automatizado.
- SOLO indica hechos del bloque de información anterior. NO añadas métodos, herramientas o detalles no listados.
- Detente después del último párrafo.""",

    "results": """Escribe la sección de Resultados usando ÚNICAMENTE los datos de abajo. NO inventes ningún número.

SELECCIÓN DE ESTUDIOS:
{prisma_text}

ESTUDIOS INCLUIDOS:
{study_list_text}

EVALUACIÓN DE CALIDAD:
{quality_summary}

RESULTADOS DEL META-ANÁLISIS:
{synthesis_summary}

Escribe prosa fluida cubriendo:
1. Selección de estudios (usa los números exactos del PRISMA arriba)
2. Características de los estudios (menciona estudios de la lista, cita como [Autor, Año])
3. Resumen de la evaluación de calidad
4. Resultados del meta-análisis (usa los números exactos anteriores)
5. Análisis de sensibilidad y sesgo de publicación (si los datos están disponibles)

REGLAS DE ESCRITURA:
- Oraciones cortas y claras (máximo 25 palabras cada una). Evita oraciones interminables.
- Usa SOLO números de los datos anteriores. Sin LaTeX. Sin sección de referencias. Solo texto plano.
- Cita estudios como [Autor, Año].
- NUNCA fabriques tamaños muestrales, duraciones de seguimiento o tamaños de efecto no listados arriba.
- Si un valor no está proporcionado, NO lo adivines — omítelo o escribe "no reportado".
- Detente después del último párrafo.""",

    "discussion": """Escribe una Discusión (4-5 párrafos) para esta revisión sistemática.

PREGUNTA DE INVESTIGACIÓN: {protocol_summary}

RESULTADOS PRINCIPALES:
{synthesis_summary}

CALIDAD:
{quality_summary}

CONTEXTO DE LA LITERATURA:
{rag_context}

Párrafo 1: Resume los hallazgos principales — reafirma los resultados clave con los números reales de RESULTADOS PRINCIPALES.
Párrafo 2: Compara con la evidencia existente. Cita como [Autor, Año] usando SOLO nombres del CONTEXTO anterior.
Párrafo 3: Fortalezas de esta revisión (búsqueda sistemática, evaluación de calidad, métodos estadísticos).
Párrafo 4: Limitaciones (número de estudios, heterogeneidad, problemas de calidad, generalización).
Párrafo 5: Implicaciones para la práctica clínica y direcciones para investigación futura.

REGLAS DE ESCRITURA:
- Oraciones cortas y claras (máximo 25 palabras cada una). Evita oraciones interminables.
- Sin LaTeX. Sin sección de referencias. Solo texto plano.
- NUNCA fabriques afirmaciones o estadísticas que no estén en los datos anteriores.
- Sé lógicamente consistente — no te contradigas (p.ej., no digas "desaconsejar la cesación" cuando el hallazgo apoya la cesación).
- Detente después del párrafo 5.""",

    "conclusion": """Escribe una Conclusión (2 párrafos cortos) para esta revisión sistemática.

PREGUNTA DE INVESTIGACIÓN: {protocol_summary}

RESULTADOS PRINCIPALES:
{synthesis_summary}

Párrafo 1: Indica el hallazgo principal con los números clave de arriba. Indica el nivel de certeza si está disponible.
Párrafo 2: Una implicación clínica y una dirección para investigación futura.

REGLAS DE ESCRITURA:
- Oraciones cortas y claras (máximo 25 palabras cada una).
- Sin citas. Sin LaTeX. Sin referencias. Solo texto plano.
- NUNCA inventes números que no estén en RESULTADOS PRINCIPALES.
- Detente después del párrafo 2.""",
}

# Language-specific section prompts registry
# Only "es" has full overrides; others use English prompts with language instruction prepended
_A8_LOCALIZED_PROMPTS = {
    "es": _A8_SECTION_PROMPTS_ES,
}

# Babel language packages for LaTeX
BABEL_LANGUAGES = {
    "en": "english",
    "es": "spanish",
    "pt": "brazilian",
    "fr": "french",
    "de": "ngerman",
}

# Human-readable language names for UI
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Español",
    "pt": "Português",
    "fr": "Français",
    "de": "Deutsch",
}


def get_a8_section_prompt_localized(
    section_name: str,
    protocol: dict,
    rag_context: str,
    synthesis_summary: str,
    quality_summary: str,
    n_studies: int,
    prisma_text: str = "",
    study_list_text: str = "",
    methods_data_block: str = "",
    language: str = "en",
) -> str:
    """Build a language-aware prompt for an A8 manuscript section.

    If the language has full localized prompts (e.g., "es"), use those.
    Otherwise, use the English prompts with a language instruction prepended.
    """
    # Check for full localized prompt set
    localized = _A8_LOCALIZED_PROMPTS.get(language, {})
    template = localized.get(section_name)

    if template:
        # Use localized template
        rq = protocol.get("research_question", "")
        pop = protocol.get("population", "?")
        intv = protocol.get("intervention", protocol.get("exposure", "?"))
        comp = protocol.get("comparator", "?")
        outc = protocol.get("outcome", "?")
        protocol_summary = rq if rq else f"{intv} vs {comp} for {outc} in {pop}"

        return template.format(
            protocol_summary=protocol_summary,
            rag_context=rag_context if rag_context else "(No hay contexto adicional disponible)",
            synthesis_summary=synthesis_summary if synthesis_summary else "No se realizó síntesis cuantitativa por datos insuficientes.",
            quality_summary=quality_summary if quality_summary else "Datos de evaluación de calidad no disponibles.",
            n_studies=n_studies,
            prisma_text=prisma_text if prisma_text else "(Datos PRISMA no disponibles)",
            study_list_text=study_list_text if study_list_text else "(Lista de estudios no disponible)",
            methods_data_block=methods_data_block if methods_data_block else "(Detalles de métodos no disponibles)",
        )
    else:
        # Fall back to English prompt with language instruction
        return get_a8_section_prompt(
            section_name=section_name,
            protocol=protocol,
            rag_context=rag_context,
            synthesis_summary=synthesis_summary,
            quality_summary=quality_summary,
            n_studies=n_studies,
            prisma_text=prisma_text,
            study_list_text=study_list_text,
            methods_data_block=methods_data_block,
            language=language,
        )
