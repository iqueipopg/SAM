# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Agent 6: Quality Assessment / Risk of Bias (Qwen2.5-7B)

Assesses study quality using:
  - RoB 2.0 for RCTs (5 domains)
  - ROBINS-I for observational studies (7 domains)
  - JBI Critical Appraisal for qualitative studies (10 items)

Produces quality_weight per paper for downstream use in A7 sensitivity analysis.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
from agents.prompts import (
    format_agent6_rob2_prompt,
    format_agent6_robins_i_prompt,
    format_agent6_jbi_prompt,
    format_agent6_design_prompt,
    TEMPERATURE_CONFIG,
)
import json, torch, re, time
from typing import List, Dict, Optional


class Agent6Quality(BaseAgent):
    """Quality and risk-of-bias assessment using RoB 2.0, ROBINS-I, or JBI.

    Input: list of extracted papers (from A5) and protocol (from A1).
    Output: dict with 'papers' (each annotated with quality_assessment and quality_weight) and 'stats'.
    GPU: yes -- uses LLM for design detection and domain-level assessment.
    """

    def __init__(self):
        super().__init__("agent6_quality")
        self.temperature = TEMPERATURE_CONFIG.get("agent6_quality", 0.2)
        _mic = self.get_agent_param("max_input_chars", {})
        self._assessment_max_chars = _mic.get("assessment", 5000) if isinstance(_mic, dict) else 5000
        self._design_max_chars = _mic.get("design_detection", 2000) if isinstance(_mic, dict) else 2000

    # ================================================================== #
    #  PUBLIC API                                                          #
    # ================================================================== #

    def assess_all(self, a5_papers: List[Dict], protocol: Dict) -> Dict:
        """Assess quality/RoB for all papers from A5 extraction output."""
        papers = a5_papers
        if isinstance(a5_papers, dict) and "papers" in a5_papers:
            papers = a5_papers["papers"]

        print(f"\n{'='*60}")
        print("AGENT 6 - QUALITY / RISK OF BIAS ASSESSMENT")
        print(f"{'='*60}")
        print(f"  Papers to assess: {len(papers)}")

        start = time.time()

        if self.model is None:
            self.load_model()

        assessed = []
        n_rob2 = 0
        n_robins_i = 0
        n_jbi = 0
        n_failed = 0

        for i, paper in enumerate(papers):
            title_short = paper.get("title", "")[:70]
            print(f"\n  [{i+1}/{len(papers)}] {title_short}...")

            try:
                result = self._assess_single(paper, protocol)
                assessed.append(result)

                tool = result["quality_assessment"]["tool"]
                if tool == "RoB2":
                    n_rob2 += 1
                    print(f"    -> RoB2: {result['quality_assessment']['overall']} (weight={result['quality_weight']})")
                elif tool == "ROBINS-I":
                    n_robins_i += 1
                    print(f"    -> ROBINS-I: {result['quality_assessment']['overall']} (weight={result['quality_weight']})")
                elif tool == "JBI":
                    n_jbi += 1
                    print(f"    -> JBI: {result['quality_assessment']['overall']} ({result['quality_assessment'].get('yes_count', '?')}/10 Yes, weight={result['quality_weight']})")

            except Exception as e:
                print(f"    -> FAILED: {e}")
                n_failed += 1
                assessed.append(self._fallback_assessment(paper))

        elapsed = round(time.time() - start, 1)

        result = {
            "papers": assessed,
            "stats": {
                "total": len(papers),
                "rob2_assessed": n_rob2,
                "robins_i_assessed": n_robins_i,
                "jbi_assessed": n_jbi,
                "failed": n_failed,
                "elapsed_s": elapsed,
            },
        }

        self._print_summary(result)
        return result

    # ================================================================== #
    #  SINGLE PAPER ASSESSMENT                                             #
    # ================================================================== #

    def _assess_single(self, paper: Dict, protocol: Dict) -> Dict:
        """Assess a single paper: detect design, run RoB2, ROBINS-I, or JBI."""
        design = self._detect_study_design(paper)

        if design == "rct":
            assessment = self._assess_rob2(paper, protocol)
        elif design == "qualitative":
            assessment = self._assess_jbi(paper, protocol)
        else:
            assessment = self._assess_robins_i(paper, protocol)

        weight = self._compute_quality_weight(assessment)

        return {
            "pmid": paper.get("pmid", ""),
            "title": paper.get("title", ""),
            "study_design_detected": design,
            "quality_assessment": assessment,
            "quality_weight": weight,
            "study_characteristics": paper.get("study_characteristics"),
            "outcomes": paper.get("outcomes", []),
        }

    # ================================================================== #
    #  STUDY DESIGN DETECTION                                              #
    # ================================================================== #

    # Deterministic RCT regex: catches common phrasings in title/abstract
    _RCT_REGEX = re.compile(
        r"\b(randomi[sz]ed controlled trial|RCT|randomi[sz]ed trial|"
        r"randomi[sz]ed\s+(?:clinical\s+)?trial|"
        r"double[- ]blind.{0,20}trial|controlled clinical trial)\b",
        re.IGNORECASE,
    )

    def _detect_study_design(self, paper: Dict) -> str:
        """Detect study design: 'rct', 'observational', or 'qualitative'.
        Order:
          1) Deterministic RCT regex on title+abstract (no LLM needed).
          2) Keyword match on A5's study_characteristics.study_design.
          3) LLM fallback.
        """
        # (1) Deterministic RCT regex on title + abstract — skip LLM if matched
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        haystack = f"{title}\n{abstract}"
        if self._RCT_REGEX.search(haystack):
            return "rct"

        sc = paper.get("study_characteristics") or {}
        design_str = (sc.get("study_design") or "").lower().strip()

        # (2) Keyword match — check qualitative first (some qualitative studies may
        # mention "interviews" alongside observational keywords, so order matters)
        qual_keywords = ["qualitative", "phenomenolog", "grounded theory", "ethnograph",
                         "narrative inquiry", "thematic analysis", "focus group",
                         "interpretive description"]
        rct_keywords = ["rct", "randomized", "randomised", "random", "crossover rct", "cluster-randomized"]
        obs_keywords = ["cohort", "case-control", "case control", "cross-sectional",
                        "cross sectional", "retrospective", "prospective", "observational",
                        "case series", "ecological"]

        for kw in qual_keywords:
            if kw in design_str:
                return "qualitative"
        for kw in rct_keywords:
            if kw in design_str:
                return "rct"
        for kw in obs_keywords:
            if kw in design_str:
                return "observational"

        # (3) LLM fallback
        prompt = format_agent6_design_prompt(paper, max_chars=self._design_max_chars)
        response = self._simple_generate(prompt, max_tokens=100)
        parsed = self._parse_json_response(response)

        if parsed and parsed.get("study_design") in ("rct", "observational", "qualitative"):
            return parsed["study_design"]

        # Conservative default: treat as observational (stricter assessment)
        return "observational"

    # ================================================================== #
    #  RoB 2.0 ASSESSMENT (RCTs)                                           #
    # ================================================================== #

    def _assess_rob2(self, paper: Dict, protocol: Dict) -> Dict:
        """Run RoB 2.0 assessment for an RCT."""
        prompt = format_agent6_rob2_prompt(paper, protocol, max_chars=self._assessment_max_chars)
        response = self._simple_generate(prompt, max_tokens=600)
        parsed = self._parse_json_response(response)

        if not parsed:
            return self._default_rob2()

        # Validate and structure
        domains = {}
        valid_judgments = {"low", "some concerns", "high"}

        for domain_name in ("randomization", "deviations", "missing_data", "measurement", "selection"):
            d = parsed.get(domain_name, {})
            if isinstance(d, dict):
                judgment = str(d.get("judgment", "Some concerns")).strip()
                # Normalize capitalization
                judgment_lower = judgment.lower()
                if judgment_lower not in valid_judgments:
                    judgment = "Some concerns"
                else:
                    judgment = {"low": "Low", "some concerns": "Some concerns", "high": "High"}[judgment_lower]
                domains[domain_name] = {
                    "judgment": judgment,
                    "justification": str(d.get("justification", "Not specified")),
                }
            else:
                domains[domain_name] = {"judgment": "Some concerns", "justification": "Could not assess"}

        # Overall: least favorable rule
        judgments = [d["judgment"] for d in domains.values()]
        if "High" in judgments:
            overall = "High"
        elif "Some concerns" in judgments:
            overall = "Some concerns"
        else:
            overall = "Low"

        overall_just = parsed.get("overall_justification", "Derived from domain judgments")

        return {
            "tool": "RoB2",
            "domains": domains,
            "overall": overall,
            "overall_justification": str(overall_just),
        }

    def _default_rob2(self) -> Dict:
        """Return conservative default RoB2 when parsing fails."""
        domains = {}
        for d in ("randomization", "deviations", "missing_data", "measurement", "selection"):
            domains[d] = {"judgment": "Some concerns", "justification": "Assessment could not be parsed"}
        return {
            "tool": "RoB2",
            "domains": domains,
            "overall": "Some concerns",
            "overall_justification": "Default: LLM response could not be parsed",
        }

    # ================================================================== #
    #  ROBINS-I (Observational)                                            #
    # ================================================================== #

    _ROBINS_I_DOMAINS = (
        "confounding", "selection_of_participants", "classification_of_interventions",
        "deviations_from_interventions", "missing_data", "measurement_of_outcomes",
        "selection_of_results",
    )
    _ROBINS_I_VALID = {"low", "moderate", "serious", "critical", "no information"}
    _ROBINS_I_NORM = {
        "low": "Low", "moderate": "Moderate", "serious": "Serious",
        "critical": "Critical", "no information": "No information",
    }
    # Severity order for worst-case rule (higher = worse)
    _ROBINS_I_SEVERITY = {"Low": 0, "Moderate": 1, "Serious": 2, "Critical": 3, "No information": 2}

    def _assess_robins_i(self, paper: Dict, protocol: Dict) -> Dict:
        """Run ROBINS-I assessment for an observational study."""
        prompt = format_agent6_robins_i_prompt(paper, protocol, max_chars=self._assessment_max_chars)
        response = self._simple_generate(prompt, max_tokens=800)
        parsed = self._parse_json_response(response)

        if not parsed:
            return self._default_robins_i()

        # Validate and structure
        domains = {}
        for domain_name in self._ROBINS_I_DOMAINS:
            d = parsed.get(domain_name, {})
            if isinstance(d, dict):
                judgment = str(d.get("judgment", "No information")).strip()
                judgment_lower = judgment.lower()
                if judgment_lower not in self._ROBINS_I_VALID:
                    judgment = "No information"
                else:
                    judgment = self._ROBINS_I_NORM[judgment_lower]
                domains[domain_name] = {
                    "judgment": judgment,
                    "justification": str(d.get("justification", "Not specified")),
                }
            else:
                domains[domain_name] = {"judgment": "No information", "justification": "Could not assess"}

        # Overall: worst-case rule
        judgments = [d["judgment"] for d in domains.values()]
        if "Critical" in judgments:
            overall = "Critical"
        elif "Serious" in judgments:
            overall = "Serious"
        elif "Moderate" in judgments:
            overall = "Moderate"
        elif all(j == "No information" for j in judgments):
            overall = "No information"
        else:
            overall = "Low"

        overall_just = parsed.get("overall_justification", "Derived from domain judgments")

        return {
            "tool": "ROBINS-I",
            "domains": domains,
            "overall": overall,
            "overall_justification": str(overall_just),
        }

    def _default_robins_i(self) -> Dict:
        """Return conservative default ROBINS-I when parsing fails."""
        domains = {}
        for d in self._ROBINS_I_DOMAINS:
            domains[d] = {"judgment": "No information", "justification": "Assessment could not be parsed"}
        return {
            "tool": "ROBINS-I",
            "domains": domains,
            "overall": "Serious",
            "overall_justification": "Default: LLM response could not be parsed",
        }

    # ================================================================== #
    #  JBI CRITICAL APPRAISAL (Qualitative)                                #
    # ================================================================== #

    _JBI_ITEMS = (
        "philosophical_perspective", "methodology_question", "methodology_methods",
        "methodology_analysis", "methodology_interpretation", "researcher_positioning",
        "researcher_influence", "participant_voices", "ethics", "conclusions",
    )
    _JBI_VALID_ANSWERS = {"yes", "no", "unclear", "not applicable"}

    def _assess_jbi(self, paper: Dict, protocol: Dict) -> Dict:
        """Run JBI Critical Appraisal for a qualitative study."""
        prompt = format_agent6_jbi_prompt(paper, protocol, max_chars=self._assessment_max_chars)
        response = self._simple_generate(prompt, max_tokens=800)
        parsed = self._parse_json_response(response)

        if not parsed:
            return self._default_jbi()

        # Validate and structure
        items = {}
        for item_name in self._JBI_ITEMS:
            d = parsed.get(item_name, {})
            if isinstance(d, dict):
                answer = str(d.get("answer", "Unclear")).strip()
                answer_lower = answer.lower()
                if answer_lower not in self._JBI_VALID_ANSWERS:
                    answer = "Unclear"
                else:
                    # Normalize capitalization
                    answer = {"yes": "Yes", "no": "No", "unclear": "Unclear",
                              "not applicable": "Not applicable"}[answer_lower]
                items[item_name] = {
                    "answer": answer,
                    "justification": str(d.get("justification", "Not specified")),
                }
            else:
                items[item_name] = {"answer": "Unclear", "justification": "Could not assess"}

        yes_count = sum(1 for item in items.values() if item["answer"] == "Yes")

        if yes_count >= 8:
            overall = "Include"
        elif yes_count >= 5:
            overall = "Include with caution"
        else:
            overall = "Exclude"

        overall_just = parsed.get("overall_justification", f"{yes_count}/10 Yes answers")

        return {
            "tool": "JBI",
            "items": items,
            "yes_count": yes_count,
            "overall": overall,
            "overall_justification": str(overall_just),
        }

    def _default_jbi(self) -> Dict:
        """Return conservative default JBI when parsing fails."""
        items = {}
        for item_name in self._JBI_ITEMS:
            items[item_name] = {"answer": "Unclear", "justification": "Assessment could not be parsed"}
        return {
            "tool": "JBI",
            "items": items,
            "yes_count": 0,
            "overall": "Exclude",
            "overall_justification": "Default: LLM response could not be parsed",
        }

    # ================================================================== #
    #  QUALITY WEIGHT                                                      #
    # ================================================================== #

    def _compute_quality_weight(self, assessment: Dict) -> float:
        """Compute numeric weight from quality assessment.
        RoB2: Low→1.0, Some concerns→0.6, High→0.3
        ROBINS-I: Low→1.0, Moderate→0.7, Serious→0.4, Critical→0.2, No information→0.3
        JBI: Include→1.0, Include with caution→0.6, Exclude→0.2
        """
        overall = assessment.get("overall", "").lower()

        weight_map = {
            # RoB2
            "low": 1.0,
            "some concerns": 0.6,
            "high": 0.3,
            # ROBINS-I
            "moderate": 0.7,
            "serious": 0.4,
            "critical": 0.2,
            "no information": 0.3,
            # JBI
            "include": 1.0,
            "include with caution": 0.6,
            "exclude": 0.2,
        }

        return weight_map.get(overall, 0.5)

    # ================================================================== #
    #  LLM + JSON HELPERS                                                  #
    # ================================================================== #

    def _simple_generate(self, prompt: str, max_tokens: int = 500) -> str:
        if self._is_groq():
            return self.generate(prompt, max_new_tokens=max_tokens, temperature=self.temperature)
        inputs = self._tokenize_prompt(prompt)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self._decode_response(outputs, input_length)

    def _clean_json(self, text: str) -> str:
        """Extract first valid JSON object from LLM output."""
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

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON object from LLM response, return None on failure."""
        try:
            cleaned = self._clean_json(response)
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return parsed
            return None
        except Exception:
            return None

    # ================================================================== #
    #  FALLBACK                                                            #
    # ================================================================== #

    def _fallback_assessment(self, paper: Dict) -> Dict:
        """Return a safe fallback when assessment completely fails."""
        default_domains = {}
        for d in self._ROBINS_I_DOMAINS:
            default_domains[d] = {"judgment": "No information", "justification": "Assessment failed"}
        return {
            "pmid": paper.get("pmid", ""),
            "title": paper.get("title", ""),
            "study_design_detected": "unknown",
            "quality_assessment": {
                "tool": "ROBINS-I",
                "domains": default_domains,
                "overall": "Serious",
                "overall_justification": "Assessment failed — conservative default",
            },
            "quality_weight": 0.4,
            "study_characteristics": paper.get("study_characteristics"),
            "outcomes": paper.get("outcomes", []),
        }

    # ================================================================== #
    #  SUMMARY                                                             #
    # ================================================================== #

    def _print_summary(self, result: Dict):
        s = result["stats"]
        print(f"\n{'='*60}")
        print("AGENT 6 - QUALITY ASSESSMENT SUMMARY")
        print(f"{'='*60}")
        print(f"  Total papers     : {s['total']}")
        print(f"  RoB 2 assessed   : {s['rob2_assessed']}")
        print(f"  ROBINS-I assessed: {s['robins_i_assessed']}")
        print(f"  JBI assessed     : {s['jbi_assessed']}")
        print(f"  Failed           : {s['failed']}")
        print(f"  Time             : {s['elapsed_s']}s")

        # Weight distribution
        weights = [p["quality_weight"] for p in result["papers"]]
        if weights:
            avg_w = sum(weights) / len(weights)
            print(f"  Avg weight       : {avg_w:.2f}")
            high = sum(1 for w in weights if w >= 0.9)
            med = sum(1 for w in weights if 0.4 <= w < 0.9)
            low = sum(1 for w in weights if w < 0.4)
            print(f"  High quality     : {high}")
            print(f"  Medium quality   : {med}")
            print(f"  Low quality      : {low}")


# ====================================================================== #
#  STANDALONE TEST                                                         #
# ====================================================================== #
if __name__ == "__main__":
    a6 = Agent6Quality()
    print("Agent6Quality created. Run test_agent6.py for full test.")
