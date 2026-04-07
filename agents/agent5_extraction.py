"""Agent 5: Structured Data Extraction (Qwen2.5-7B)

Extracts quantitative/qualitative data from included papers using 3 focused prompts:
  1. Study characteristics (from methods section)
  2. Intervention/exposure details (from methods section)
  3. Outcomes (from results section)
Plus an optional 4th "numeric recovery" pass for papers missing effect sizes.

Framework-adaptive: PICO/PECO/SPIDER/PCC use different extraction language.

Dual extraction mode: runs extraction twice at different temperatures, then
reconciles results to flag hallucinated values.  Controlled by config.yaml:
  extraction.dual_extraction: true/false
"""

import sys, os, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
import json, torch, re, time
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple


def _find_numeric_dense_region(text: str, max_chars: int) -> str:
    """Find the region of text with the highest density of numbers.

    This catches results sections even when headings aren't detected —
    results sections have many numbers (effect sizes, p-values, CIs, N=, etc.).
    """

    if len(text) < 2000:
        return ""

    window = max_chars
    step = max_chars // 4  # 25% overlap between windows
    best_start = 0
    best_score = 0

    # Score each window by density of numeric patterns
    num_pattern = re.compile(
        r'(?:'
        r'\d+\.\d+'               # decimals: 0.05, 2.34
        r'|p\s*[<>=]\s*[\d.]+'    # p-values: p < 0.05
        r'|CI\s*[:=]?\s*[\d.]+'   # confidence intervals
        r'|[−-]?\d+\.?\d*\s*%'    # percentages
        r'|[Nn]\s*=\s*\d+'        # sample sizes: N=50
        r'|OR\s*[:=]\s*[\d.]+'    # odds ratios
        r'|RR\s*[:=]\s*[\d.]+'    # risk ratios
        r'|HR\s*[:=]\s*[\d.]+'    # hazard ratios
        r'|SMD\s*[:=]\s*[−-]?[\d.]+'  # standardized mean diff
        r'|[Dd]\s*=\s*[−-]?[\d.]+'    # Cohen's d
        r'|[Gg]\s*=\s*[−-]?[\d.]+'    # Hedges' g
        r'|SD\s*[:=]\s*[\d.]+'    # standard deviations
        r'|SE\s*[:=]\s*[\d.]+'    # standard errors
        r'|\(\s*\d+\.?\d*\s*[,;]\s*\d+\.?\d*\s*\)'  # (lower, upper)
        r'|[tT]\(\d+\)\s*=\s*[−-]?\d+\.?\d*'         # t-test: t(14)=2.3
        r'|[Ff]\(\d+\s*,\s*\d+\)\s*=\s*\d+\.?\d*'    # F-test: F(1,48)=5.67
        r'|[Bb]eta\s*=\s*[−-]?\d+\.?\d*'              # beta coefficient: beta=0.34
        r'|\[\s*[−-]?\d+\.?\d*\s*,\s*[−-]?\d+\.?\d*\s*\]'  # bracket CI: [0.12, 0.78]
        r')',
        re.IGNORECASE,
    )

    for pos in range(0, len(text) - window, step):
        chunk = text[pos:pos + window]
        score = len(num_pattern.findall(chunk))
        if score > best_score:
            best_score = score
            best_start = pos

    # Only return if we found a meaningfully numeric region
    if best_score < 5:
        return ""

    return text[best_start:best_start + window]


class Agent5Extraction(BaseAgent):
    """Structured data extraction using 3 focused prompts per paper.

    Input: list of included papers (from A4) with full text, and protocol (from A1).
    Output: dict with 'papers' list, each containing study_characteristics, intervention, and outcomes.
    GPU: yes -- uses LLM for extraction prompts and optional numeric recovery pass.
    """

    def __init__(self):
        super().__init__("agent5_extraction")
        ext_cfg = self.config.get("extraction", {})
        self.dual_extraction = ext_cfg.get("dual_extraction", False)
        self.threshold_numeric = ext_cfg.get("concordance_threshold_numeric", 0.15)
        self.threshold_sample = ext_cfg.get("concordance_threshold_sample", 0.10)

    # ================================================================== #
    #  PUBLIC API                                                          #
    # ================================================================== #

    def extract_all(self, included_papers: List[Dict], protocol: Dict) -> Dict:
        """Extract structured data from all included papers."""
        print(f"\n{'='*60}")
        print("AGENT 5 - STRUCTURED DATA EXTRACTION")
        print(f"{'='*60}")
        print(f"  Papers to extract: {len(included_papers)}")
        print(f"  Framework: {protocol.get('framework', '?')}")
        print(f"  Dual extraction: {'ON' if self.dual_extraction else 'OFF'}")

        start = time.time()

        if self.model is None:
            self.load_model()

        papers_extracted = []
        n_complete = 0
        n_partial = 0
        n_failed = 0
        n_high = 0
        n_medium = 0
        n_low = 0
        n_numeric_complete = 0
        n_numeric_recovered = 0

        for i, paper in enumerate(included_papers):
            title_short = paper.get("title", "")[:70]
            print(f"\n  [{i+1}/{len(included_papers)}] {title_short}...")

            try:
                extracted = self._extract_single_paper(paper, protocol)

                # --- Numeric post-processing on all outcomes ---
                if extracted.get("outcomes"):
                    for j, outcome in enumerate(extracted["outcomes"]):
                        extracted["outcomes"][j] = self._extract_numeric_fields(outcome)

                # --- Count numeric completeness before recovery ---
                outcomes_before = extracted.get("outcomes", [])
                n_complete_before = sum(
                    1 for o in outcomes_before if o.get("numeric_complete")
                )
                n_partial_before = len(outcomes_before) - n_complete_before

                # --- 4th pass: numeric recovery ---
                # Trigger on: (a) no outcomes, (b) no numeric-complete, (c) has partial outcomes
                needs_recovery = (
                    not outcomes_before  # no outcomes at all
                    or n_complete_before == 0  # none are numeric-complete
                    or (n_partial_before > 0 and n_complete_before < len(outcomes_before))  # some partial
                )

                if needs_recovery:
                    reason = ("no outcomes" if not outcomes_before
                              else f"{n_complete_before}/{len(outcomes_before)} numeric-complete")
                    print(f"    [numeric recovery] {reason}, attempting recovery...")
                    recovered = self._numeric_recovery_pass(paper, protocol)
                    if recovered:
                        recovered = [self._extract_numeric_fields(o) for o in recovered]
                        any(
                            o.get("numeric_complete") for o in recovered
                        )
                        n_new_complete = sum(1 for o in recovered if o.get("numeric_complete"))

                        if not outcomes_before:
                            # No original outcomes — use recovered
                            if any(o.get("numeric_complete") or o.get("effect_size_value") is not None
                                   for o in recovered):
                                extracted["outcomes"] = recovered
                                n_numeric_recovered += 1
                                extracted["extraction_quality"] = "partial"
                                print(f"    [numeric recovery] SUCCESS — recovered {len(recovered)} outcomes")
                        elif n_new_complete > n_complete_before:
                            # Recovery found more complete outcomes — merge
                            extracted["outcomes"] = self._merge_recovered_outcomes(
                                extracted["outcomes"], recovered
                            )
                            n_numeric_recovered += 1
                            print(f"    [numeric recovery] SUCCESS — improved {n_complete_before}→{n_new_complete} complete outcomes")
                        else:
                            print("    [numeric recovery] no improvement")

                        if n_numeric_recovered > 0 and "sections_used" in extracted:
                            if "recovery" not in extracted["sections_used"]:
                                extracted["sections_used"].append("recovery")
                    else:
                        print("    [numeric recovery] no results")

                # --- Final numeric completeness count ---
                if any(o.get("numeric_complete") for o in extracted.get("outcomes", [])):
                    n_numeric_complete += 1

                papers_extracted.append(extracted)

                q = extracted["extraction_quality"]
                if q == "complete":
                    n_complete += 1
                    print("    -> complete extraction")
                elif q == "partial":
                    n_partial += 1
                    print("    -> partial extraction")
                else:
                    n_failed += 1
                    print("    -> failed extraction")

                conf = extracted.get("extraction_confidence", "")
                if conf == "high":
                    n_high += 1
                elif conf == "medium":
                    n_medium += 1
                elif conf == "low":
                    n_low += 1

            except Exception as e:
                print(f"    -> error: {e}")
                n_failed += 1
                papers_extracted.append({
                    "pmid": paper.get("pmid", ""),
                    "title": paper.get("title", ""),
                    "study_characteristics": None,
                    "intervention_exposure": None,
                    "outcomes": [],
                    "extraction_quality": "failed",
                    "extraction_confidence": "low",
                    "concordance": None,
                    "extraction_notes": str(e),
                    "text_source": paper.get("text_source", ""),
                    "sections_used": [],
                })

        elapsed = round(time.time() - start, 1)

        result = {
            "papers": papers_extracted,
            "stats": {
                "total": len(included_papers),
                "complete_extraction": n_complete,
                "partial_extraction": n_partial,
                "failed_extraction": n_failed,
                "high_confidence": n_high,
                "medium_confidence": n_medium,
                "low_confidence": n_low,
                "numeric_complete": n_numeric_complete,
                "numeric_recovered": n_numeric_recovered,
                "dual_extraction": self.dual_extraction,
                "elapsed_s": elapsed,
            },
        }

        self._print_summary(result)
        return result

    # ================================================================== #
    #  PER-PAPER EXTRACTION                                                #
    # ================================================================== #

    def _extract_single_paper(self, paper: Dict, protocol: Dict) -> Dict:
        """Run 3 focused prompts on a single paper.

        If dual_extraction is enabled, uses ADAPTIVE strategy: runs pass 1,
        evaluates confidence heuristically, and only runs pass 2 if confidence
        is low (saving ~50% GPU time on clear-cut papers).

        Confidence triggers for pass 2:
        - No outcomes extracted
        - No numeric-complete outcomes
        - abstract-only text source
        - Very short text (< 2000 chars)
        """
        if not self.dual_extraction:
            result = self._extract_once(paper, protocol, temperature=0.1)
            result["extraction_confidence"] = "high"  # single pass, no comparison
            result["concordance"] = None
            return result

        # --- Adaptive dual extraction ---
        print("    [pass 1] temperature=0.1")
        result_a = self._extract_once(paper, protocol, temperature=0.1)

        # Evaluate pass 1 confidence heuristically
        needs_pass2 = self._needs_second_pass(result_a, paper)

        if not needs_pass2:
            print("    [pass 2] SKIPPED — pass 1 confident enough")
            result_a["extraction_confidence"] = "high"
            result_a["concordance"] = {"adaptive_skip": True, "reason": "pass1_confident"}
            return result_a

        print("    [pass 2] temperature=0.3 (low confidence on pass 1)")
        result_b = self._extract_once(paper, protocol, temperature=0.3)

        final = self._reconcile(result_a, result_b, paper, protocol)
        return final

    def _needs_second_pass(self, result: Dict, paper: Dict) -> bool:
        """Decide if pass 1 results need verification via a second extraction.

        Returns True if confidence is too low to trust a single pass.
        """
        outcomes = result.get("outcomes") or []
        text_source = paper.get("text_source", "")
        full_text = paper.get("full_text", "") or ""

        # Always verify abstract-only papers (limited data)
        if text_source == "abstract_only":
            return True

        # Always verify very short texts (might miss things)
        if len(full_text) < 2000:
            return True

        # No outcomes at all → verify
        if not outcomes:
            return True

        # Check numeric completeness of outcomes
        n_outcomes = len(outcomes)
        n_numeric_complete = sum(1 for o in outcomes if o.get("numeric_complete"))

        # No numeric-complete outcomes → verify
        if n_numeric_complete == 0:
            return True

        # Less than half outcomes are numeric-complete → verify
        if n_outcomes > 1 and n_numeric_complete < n_outcomes * 0.5:
            return True

        # Study characteristics missing → verify
        if result.get("study_characteristics") is None:
            return True

        # Passed all checks → confident enough
        return False

    def _extract_once(self, paper: Dict, protocol: Dict, temperature: float = 0.1) -> Dict:
        """Single extraction pass at a given temperature (original logic)."""
        sections_used = []

        # Prompt 1: Study characteristics
        study_chars = self._extract_study_characteristics(paper, protocol, temperature)
        if study_chars:
            study_chars = self._normalize_nulls(study_chars)
            sections_used.append("methods")

        # Prompt 2: Intervention/exposure details
        intervention = self._extract_intervention_details(paper, protocol, temperature)
        if intervention:
            intervention = self._normalize_nulls(intervention)
            if "methods" not in sections_used:
                sections_used.append("methods")

        # Prompt 3: Outcomes
        outcomes = self._extract_outcomes(paper, protocol, temperature)
        if outcomes:
            sections_used.append("results")

        # Determine quality
        has_chars = study_chars is not None
        has_outcomes = outcomes is not None and len(outcomes) > 0

        if has_chars and has_outcomes:
            quality = "complete"
        elif has_chars or has_outcomes:
            quality = "partial"
        else:
            quality = "failed"

        notes = []
        if paper.get("text_source") == "abstract_only":
            notes.append("abstract-only: limited extraction expected")
        if not has_outcomes:
            notes.append("no numerical outcomes extracted")
        if not has_chars:
            notes.append("study characteristics not extracted")

        return {
            "pmid": paper.get("pmid", ""),
            "title": paper.get("title", ""),
            "study_characteristics": study_chars,
            "intervention_exposure": intervention,
            "outcomes": outcomes if outcomes else [],
            "extraction_quality": quality,
            "extraction_notes": "; ".join(notes) if notes else "",
            "text_source": paper.get("text_source", ""),
            "sections_used": sections_used,
        }

    # ================================================================== #
    #  RECONCILIATION (dual extraction)                                    #
    # ================================================================== #

    def _reconcile(self, result_a: Dict, result_b: Dict, paper: Dict, protocol: Dict) -> Dict:
        """Compare two extraction passes and reconcile into a single result."""
        flags: List[str] = []

        # --- Study characteristics ---
        chars_a = result_a.get("study_characteristics")
        chars_b = result_b.get("study_characteristics")
        sample_match = True

        if chars_a and chars_b:
            ss_a = self._to_num(chars_a.get("sample_size"))
            ss_b = self._to_num(chars_b.get("sample_size"))
            if ss_a is not None and ss_b is not None:
                if not self._numeric_agree(ss_a, ss_b, self.threshold_sample):
                    sample_match = False
                    flags.append(
                        f"sample_size discrepancy: {ss_a} vs {ss_b}"
                    )
            design_a = (chars_a.get("study_design") or "").strip().lower()
            design_b = (chars_b.get("study_design") or "").strip().lower()
            if design_a and design_b and design_a != design_b:
                flags.append(f"study_design mismatch: '{chars_a.get('study_design')}' vs '{chars_b.get('study_design')}'")

        # Always use pass 1 for study_characteristics (lower temp)
        final_chars = chars_a

        # --- Intervention/exposure: always pass 1 ---
        final_intervention = result_a.get("intervention_exposure")

        # --- Outcomes (most critical) ---
        outcomes_a = result_a.get("outcomes") or []
        outcomes_b = result_b.get("outcomes") or []
        matched, discrepant, pass1_only, pass2_only, outcome_flags, final_outcomes = \
            self._reconcile_outcomes(outcomes_a, outcomes_b)
        flags.extend(outcome_flags)

        # --- Confidence ---
        confidence = self._classify_confidence(
            sample_match, matched, discrepant, pass1_only, pass2_only, flags
        )

        # --- Assemble concordance dict ---
        concordance = {
            "sample_size_match": sample_match,
            "outcomes_matched": matched,
            "outcomes_discrepant": discrepant,
            "outcomes_pass1_only": pass1_only,
            "outcomes_pass2_only": pass2_only,
            "flags": flags,
        }

        print(f"    [reconcile] confidence={confidence}  matched={matched}  "
              f"discrepant={discrepant}  flags={len(flags)}")

        # Build final result, inheriting quality/notes from pass 1
        final = dict(result_a)
        final["study_characteristics"] = final_chars
        final["intervention_exposure"] = final_intervention
        final["outcomes"] = final_outcomes
        final["extraction_confidence"] = confidence
        final["concordance"] = concordance
        return final

    # ------------------------------------------------------------------ #
    #  Outcome reconciliation                                              #
    # ------------------------------------------------------------------ #

    def _reconcile_outcomes(
        self,
        outcomes_a: List[Dict],
        outcomes_b: List[Dict],
    ) -> Tuple[int, int, int, int, List[str], List[Dict]]:
        """Match and reconcile outcomes between two passes.

        Returns (matched, discrepant, pass1_only, pass2_only, flags, final_outcomes).
        """
        flags: List[str] = []
        final_outcomes: List[Dict] = []
        used_b_indices: set = set()
        n_matched = 0
        n_discrepant = 0

        for oa in outcomes_a:
            name_a = (oa.get("name") or "").strip()
            best_idx, best_sim = -1, 0.0
            for j, ob in enumerate(outcomes_b):
                if j in used_b_indices:
                    continue
                name_b = (ob.get("name") or "").strip()
                sim = SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j

            if best_idx >= 0 and best_sim >= 0.55:
                used_b_indices.add(best_idx)
                ob = outcomes_b[best_idx]
                merged, disc_flags = self._merge_single_outcome(oa, ob)
                if disc_flags:
                    n_discrepant += 1
                    flags.extend(disc_flags)
                else:
                    n_matched += 1
                final_outcomes.append(merged)
            else:
                # Outcome only in pass 1 — keep but flag
                oa_copy = dict(oa)
                oa_copy["_concordance_note"] = "pass1_only"
                final_outcomes.append(oa_copy)

        pass1_only = len(outcomes_a) - n_matched - n_discrepant
        pass2_only = len(outcomes_b) - len(used_b_indices)

        # Flag outcomes only in pass 2
        for j, ob in enumerate(outcomes_b):
            if j not in used_b_indices:
                name_b = (ob.get("name") or "").strip()
                flags.append(f"outcome only in pass 2: '{name_b}'")
                ob_copy = dict(ob)
                ob_copy["_concordance_note"] = "pass2_only (uncertain)"
                final_outcomes.append(ob_copy)

        return n_matched, n_discrepant, pass1_only, pass2_only, flags, final_outcomes

    def _merge_single_outcome(
        self, oa: Dict, ob: Dict
    ) -> Tuple[Dict, List[str]]:
        """Merge a single matched outcome pair. Returns (merged, flags)."""
        flags: List[str] = []
        name_a = oa.get("name", "")
        merged = dict(oa)  # start from pass 1

        # measure_type / effect_size_type must match
        mt_a = (oa.get("measure_type") or oa.get("effect_size_type") or "").strip().upper()
        mt_b = (ob.get("measure_type") or ob.get("effect_size_type") or "").strip().upper()
        if mt_a and mt_b and mt_a != mt_b:
            flags.append(f"effect_size_type mismatch in '{name_a}': {mt_a} vs {mt_b} — REVIEW NEEDED")

        # Numeric fields: effect_size (value / effect_size_value), ci_lower, ci_upper
        for field in ("value", "effect_size_value", "ci_lower", "ci_upper"):
            va = self._to_num(oa.get(field))
            vb = self._to_num(ob.get(field))
            if va is not None and vb is not None:
                if self._numeric_agree(va, vb, self.threshold_numeric):
                    # Agree: average them
                    merged[field] = round((va + vb) / 2, 6)
                else:
                    flags.append(
                        f"{field} discrepancy in '{name_a}': {va} vs {vb}"
                    )
                    merged[field] = va  # keep pass 1

        # p_value: same sign check
        pa = self._to_num(oa.get("p_value"))
        pb = self._to_num(ob.get("p_value"))
        if pa is not None and pb is not None:
            # "same sign" = both significant or both not at 0.05
            sig_a = pa < 0.05
            sig_b = pb < 0.05
            if sig_a != sig_b:
                flags.append(
                    f"p_value significance mismatch in '{name_a}': {pa} vs {pb}"
                )
            merged["p_value"] = pa  # keep pass 1

        # SE concordance
        se_a = self._to_num(oa.get("se"))
        se_b = self._to_num(ob.get("se"))
        if se_a is not None and se_b is not None:
            if self._numeric_agree(se_a, se_b, self.threshold_numeric):
                merged["se"] = round((se_a + se_b) / 2, 6)
            else:
                flags.append(f"se discrepancy in '{name_a}': {se_a} vs {se_b}")
                merged["se"] = se_a

        return merged, flags

    # ------------------------------------------------------------------ #
    #  Confidence classification                                           #
    # ------------------------------------------------------------------ #

    def _classify_confidence(
        self,
        sample_match: bool,
        outcomes_matched: int,
        outcomes_discrepant: int,
        pass1_only: int,
        pass2_only: int,
        flags: List[str],
    ) -> str:
        """Classify extraction confidence as high / medium / low."""
        total_outcomes = outcomes_matched + outcomes_discrepant + pass1_only + pass2_only
        if total_outcomes == 0:
            # No outcomes at all — cannot be high confidence
            return "low" if not sample_match else "medium"

        # high: everything agrees
        if (sample_match
                and outcomes_discrepant == 0
                and pass1_only == 0
                and pass2_only == 0
                and len(flags) == 0):
            return "high"

        # low: major discrepancies
        has_effect_disc = any("value discrepancy" in f or "ci_lower discrepancy" in f
                             or "ci_upper discrepancy" in f for f in flags)
        if not sample_match and has_effect_disc:
            return "low"
        if outcomes_discrepant > outcomes_matched and outcomes_discrepant > 0:
            return "low"
        # Majority unmatched
        unmatched = pass1_only + pass2_only
        if unmatched > outcomes_matched and unmatched > 0:
            return "low"

        return "medium"

    # ------------------------------------------------------------------ #
    #  Numeric helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_num(val) -> Optional[float]:
        """Coerce value to float, return None on failure."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _numeric_agree(a: float, b: float, threshold: float) -> bool:
        """Check if two numbers agree within a relative threshold."""
        if a == 0 and b == 0:
            return True
        denom = max(abs(a), abs(b))
        if denom == 0:
            return True
        return abs(a - b) / denom <= threshold

    # ================================================================== #
    #  PROMPT 1: STUDY CHARACTERISTICS                                     #
    # ================================================================== #

    def _extract_study_characteristics(self, paper: Dict, protocol: Dict,
                                       temperature: float = 0.1) -> Optional[Dict]:
        methods_text = self._get_section_text(paper, "methods", max_chars=4000)
        if not methods_text:
            methods_text = self._get_section_text(paper, "full_text", max_chars=4000)

        protocol.get("framework", "PICO")

        prompt = f"""You are a systematic review data extractor. Extract study characteristics from this paper's methods section.

PAPER TITLE: {paper.get('title', '')}

METHODS TEXT:
{methods_text}

Extract the following into valid JSON:
- study_design: e.g. "RCT", "Cross-sectional", "Cohort", "Case-control"
- sample_size: total number of participants (integer or null)
- n_groups: object with group names and sizes, e.g. {{"intervention": 120, "control": 120}} or null
- age_mean: mean age of participants (number or null)
- age_sd: standard deviation of age (number or null)
- female_pct: percentage of female participants (number or null)
- setting: clinical setting, e.g. "Hospital", "Primary care"
- country: country where the study was conducted
- followup_duration: e.g. "12 months", "Cross-sectional (no follow-up)"

Respond ONLY with valid JSON. Use null for values not reported in the text.
{{"study_design": "...", "sample_size": ..., "n_groups": ..., "age_mean": ..., "age_sd": ..., "female_pct": ..., "setting": "...", "country": "...", "followup_duration": "..."}}"""

        response = self._simple_generate(prompt, max_tokens=300, temperature=temperature)
        return self._parse_json_response(response)

    # ================================================================== #
    #  PROMPT 2: INTERVENTION / EXPOSURE DETAILS                           #
    # ================================================================== #

    def _extract_intervention_details(self, paper: Dict, protocol: Dict,
                                      temperature: float = 0.1) -> Optional[Dict]:
        methods_text = self._get_section_text(paper, "methods", max_chars=3000)
        if not methods_text:
            methods_text = self._get_section_text(paper, "full_text", max_chars=3000)

        framework = protocol.get("framework", "PICO")

        if framework == "PECO":
            ie_label = "EXPOSURE"
            ie_field = protocol.get("exposure", protocol.get("intervention", ""))
            ie_desc = "exposure being studied"
        elif framework == "SPIDER":
            ie_label = "PHENOMENON OF INTEREST"
            ie_field = protocol.get("phenomenon_of_interest", protocol.get("intervention", ""))
            ie_desc = "phenomenon of interest"
        else:
            ie_label = "INTERVENTION"
            ie_field = protocol.get("intervention", "")
            ie_desc = "intervention being tested"

        comparator = protocol.get("comparator", "")

        prompt = f"""You are a systematic review data extractor. Extract {ie_desc} details from this paper.

PAPER TITLE: {paper.get('title', '')}
PROTOCOL {ie_label}: {ie_field}
PROTOCOL COMPARATOR: {comparator}

METHODS TEXT:
{methods_text}

Extract the following into valid JSON:
- name: name of the {ie_desc}
- description: brief description of what it involves (1-2 sentences)
- comparator: what is used as the comparison group
- dose_duration: dosage, frequency, or duration of the {ie_desc} (or "Not specified")

Respond ONLY with valid JSON:
{{"name": "...", "description": "...", "comparator": "...", "dose_duration": "..."}}"""

        response = self._simple_generate(prompt, max_tokens=250, temperature=temperature)
        return self._parse_json_response(response)

    # ================================================================== #
    #  PROMPT 3: OUTCOMES                                                  #
    # ================================================================== #

    def _extract_outcomes(self, paper: Dict, protocol: Dict,
                          temperature: float = 0.1) -> Optional[List[Dict]]:
        from agents.prompts import AGENT5_OUTCOMES_TEMPLATE

        _max_results = self.config.get("extraction", {}).get("max_chars", {}).get("outcomes", 6000)
        results_text = self._get_section_text(paper, "results", max_chars=_max_results)
        if not results_text:
            results_text = self._get_section_text(paper, "full_text", max_chars=_max_results)

        outcome_of_interest = protocol.get("outcome", "")

        prompt = AGENT5_OUTCOMES_TEMPLATE.format(
            title=paper.get("title", ""),
            outcome_of_interest=outcome_of_interest,
            results_text=results_text,
        )

        response = self._simple_generate(prompt, max_tokens=600, temperature=temperature)

        # Try to parse as array
        cleaned = self._clean_json_array(response)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                # Normalize field names for A7 compatibility
                return [self._normalize_outcome_fields(o) for o in parsed]
            if isinstance(parsed, dict):
                return [self._normalize_outcome_fields(parsed)]
            return None
        except Exception:
            return None

    def _normalize_outcome_fields(self, outcome: Dict) -> Dict:
        """Map new prompt field names to what A7 expects, preserving all fields."""
        o = dict(outcome)

        # Map effect_size_type → measure_type (A7 reads measure_type)
        if "effect_size_type" in o and "measure_type" not in o:
            o["measure_type"] = o["effect_size_type"]
        # Ensure measure_type is uppercase canonical
        mt = str(o.get("measure_type", "")).strip().upper()
        if mt in ("SMD", "OR", "RR", "HR", "MD", "PROP"):
            o["measure_type"] = mt

        # Map effect_size_value → value (A7 reads "value")
        if "effect_size_value" in o and "value" not in o:
            o["value"] = o["effect_size_value"]
        elif "effect_size_value" in o and o.get("value") is None:
            o["value"] = o["effect_size_value"]

        # Build n_per_group from n_intervention/n_control for backward compat
        if "n_per_group" not in o or o.get("n_per_group") is None:
            n_int = o.get("n_intervention")
            n_ctrl = o.get("n_control")
            if n_int is not None or n_ctrl is not None:
                o["n_per_group"] = {}
                if n_int is not None:
                    o["n_per_group"]["intervention"] = n_int
                if n_ctrl is not None:
                    o["n_per_group"]["control"] = n_ctrl

        return o

    # ================================================================== #
    #  NUMERIC POST-PROCESSING                                             #
    # ================================================================== #

    @staticmethod
    def _normalize_nulls(o: Dict) -> Dict:
        """Normalize empty/null-like values consistently to None.

        Handles: "Not specified", "NR", "N/A", "NA", "n/a", "", "null", "None", "?", "-"
        """
        _null_strings = {
            "", "not specified", "not reported", "nr", "n/a", "na", "null",
            "none", "?", "-", "n.r.", "n.a.", "not available", "not stated",
            "not provided", "not applicable", "not mentioned",
        }
        for key, val in o.items():
            if isinstance(val, str) and val.strip().lower() in _null_strings:
                o[key] = None
        return o

    def _extract_numeric_fields(self, outcome: Dict) -> Dict:
        """Post-process an outcome dict to ensure numeric fields are floats,
        calculate SMD from means/SDs if missing, and flag completeness.

        Mutates and returns the outcome dict.
        """
        o = dict(outcome)

        # --- Normalize null-like strings to None ---
        o = self._normalize_nulls(o)

        # --- Parse core numeric fields to float ---
        for field in ("value", "effect_size_value", "ci_lower", "ci_upper",
                       "p_value", "se",
                       "mean_intervention", "sd_intervention",
                       "mean_control", "sd_control"):
            v = o.get(field)
            if v is not None:
                parsed = self._to_num(v)
                o[field] = parsed  # None if unparseable

        # Parse integer fields
        for field in ("n_intervention", "n_control", "n_total"):
            v = o.get(field)
            if v is not None:
                parsed = self._to_num(v)
                o[field] = int(parsed) if parsed is not None else None

        # Sync value / effect_size_value
        if o.get("value") is None and o.get("effect_size_value") is not None:
            o["value"] = o["effect_size_value"]
        elif o.get("effect_size_value") is None and o.get("value") is not None:
            o["effect_size_value"] = o["value"]

        # --- (a) Calculate SMD from means/SDs if effect_size is missing ---
        if o.get("value") is None:
            m1 = o.get("mean_intervention")
            m2 = o.get("mean_control")
            sd1 = o.get("sd_intervention")
            sd2 = o.get("sd_control")
            n1 = o.get("n_intervention")
            n2 = o.get("n_control")

            if (m1 is not None and m2 is not None
                    and sd1 is not None and sd2 is not None
                    and sd1 > 0 and sd2 > 0):
                sd_pooled = math.sqrt((sd1**2 + sd2**2) / 2)
                smd = (m1 - m2) / sd_pooled
                o["value"] = round(smd, 6)
                o["effect_size_value"] = o["value"]
                o["measure_type"] = "SMD"
                o["effect_size_type"] = "SMD"
                o["_computed_from"] = "means_sds"

                # Compute SE and CI if we have sample sizes
                if n1 is not None and n2 is not None and n1 > 0 and n2 > 0:
                    se = math.sqrt(1/n1 + 1/n2 + smd**2 / (2*(n1+n2)))
                    o["se"] = round(se, 6)
                    o["ci_lower"] = round(smd - 1.96 * se, 6)
                    o["ci_upper"] = round(smd + 1.96 * se, 6)

        # --- (b) Estimate CI from SE if CI is missing but SE is present ---
        if o.get("value") is not None and o.get("se") is not None:
            if o.get("ci_lower") is None or o.get("ci_upper") is None:
                se = o["se"]
                val = o["value"]
                if se > 0:
                    o["ci_lower"] = round(val - 1.96 * se, 6)
                    o["ci_upper"] = round(val + 1.96 * se, 6)

        # --- (c) Estimate CI from p-value and N if still missing ---
        if (o.get("value") is not None
                and o.get("ci_lower") is None
                and o.get("p_value") is not None
                and o["value"] != 0):
            p = o["p_value"]
            val = o["value"]
            if p is not None and 0 < p < 1:
                try:
                    from scipy.stats import norm
                    z = abs(norm.ppf(p / 2))
                    if z > 0:
                        se_est = abs(val) / z
                        o["se"] = round(se_est, 6)
                        o["ci_lower"] = round(val - 1.96 * se_est, 6)
                        o["ci_upper"] = round(val + 1.96 * se_est, 6)
                        o["_ci_estimated_from"] = "p_value"
                except Exception:
                    pass

        # --- (d) Flag numeric completeness ---
        has_value = o.get("value") is not None
        has_ci = o.get("ci_lower") is not None and o.get("ci_upper") is not None
        has_se = o.get("se") is not None
        has_measure = o.get("measure_type") in ("SMD", "OR", "RR", "HR", "MD", "PROP")

        # "numeric_complete" means A7 can use this outcome
        o["numeric_complete"] = bool(has_value and (has_ci or has_se) and has_measure)

        return o

    # ================================================================== #
    #  4TH PASS: NUMERIC RECOVERY                                         #
    # ================================================================== #

    def _numeric_recovery_pass(self, paper: Dict, protocol: Dict) -> Optional[List[Dict]]:
        """Focused extraction pass that asks ONLY for numeric effect sizes.

        Run on papers where the standard 3-pass extraction failed to produce
        numeric_complete outcomes. Uses a specialized prompt.
        """
        from agents.prompts import AGENT5_NUMERIC_RECOVERY_TEMPLATE

        # Use the best available text (generous limit — recovery is the last chance)
        _max_results = self.config.get("extraction", {}).get("max_chars", {}).get("outcomes", 6000)
        text = self._get_section_text(paper, "results", max_chars=_max_results)
        if not text or len(text) < 50:
            text = self._get_section_text(paper, "full_text", max_chars=_max_results + 1000)
        if not text or len(text) < 50:
            text = paper.get("abstract", "") or ""
        if not text:
            return None

        prompt = AGENT5_NUMERIC_RECOVERY_TEMPLATE.format(
            title=paper.get("title", ""),
            text=text,
        )

        response = self._simple_generate(prompt, max_tokens=600, temperature=0.1)

        cleaned = self._clean_json_array(response)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [self._normalize_outcome_fields(o) for o in parsed]
            if isinstance(parsed, dict):
                return [self._normalize_outcome_fields(parsed)]
            return None
        except Exception:
            return None

    def _merge_recovered_outcomes(
        self, original: List[Dict], recovered: List[Dict]
    ) -> List[Dict]:
        """Merge recovered outcomes into original list.

        For each recovered outcome, if it matches an original outcome by name,
        replace only if the recovered version has numeric_complete=True and the
        original does not.  Unmatched recovered outcomes are appended.
        """
        final = list(original)
        used_orig_indices = set()

        for rec in recovered:
            if not rec.get("numeric_complete"):
                continue
            rec_name = (rec.get("name") or "").strip().lower()
            best_idx, best_sim = -1, 0.0
            for j, orig in enumerate(final):
                if j in used_orig_indices:
                    continue
                orig_name = (orig.get("name") or "").strip().lower()
                sim = SequenceMatcher(None, rec_name, orig_name).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j

            if best_idx >= 0 and best_sim >= 0.5:
                used_orig_indices.add(best_idx)
                if not final[best_idx].get("numeric_complete"):
                    rec["_source"] = "numeric_recovery"
                    final[best_idx] = rec
            else:
                # New outcome from recovery — append
                rec["_source"] = "numeric_recovery"
                final.append(rec)

        return final

    # ================================================================== #
    #  TEXT HELPERS                                                         #
    # ================================================================== #

    def _get_section_text(self, paper: Dict, section_name: str, max_chars: int = 4000) -> str:
        """Get text from a specific section, with smart fallback.

        Priority:
        1. Parsed sections dict (from A4 section parsing)
        2. Smart search: find the section heading in full_text and extract around it
        3. Last resort: beginning of full_text (only for methods/introduction)
        """
        if section_name == "full_text":
            text = paper.get("full_text", "") or ""
            return text[:max_chars] if text else ""

        # 1. Try parsed sections
        sections = paper.get("sections") or {}
        if section_name in sections:
            text = sections[section_name]
            return text[:max_chars] if len(text) > max_chars else text

        # 2. Smart fallback: search for section heading in full_text
        full_text = paper.get("full_text", "") or ""
        if full_text and len(full_text) > 1000:
            found = self._find_section_in_text(full_text, section_name, max_chars)
            if found:
                return found

        # 3. Last resort: beginning of text (better than nothing)
        return full_text[:max_chars] if full_text else ""

    @staticmethod
    def _find_section_in_text(text: str, section_name: str, max_chars: int) -> str:
        """Search for a section heading in plain text and extract content after it.

        This handles papers where A4 couldn't parse sections (e.g., messy PDF extraction).
        Returns the text from the heading to the next heading, up to max_chars.
        """
        import re

        # Map section names to heading patterns (loose: handles PDF quirks like
        # missing newlines, ALL CAPS, "Results and Discussion", trailing punctuation)
        heading_patterns = {
            "results": [
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:RESULTS?\b|FINDINGS\b|RESULTS?\s+AND\s+DISCUSSION)',
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Results?\b|Findings\b|Results?\s+and\s+[Dd]iscussion)',
            ],
            "methods": [
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:METHODS?\b|MATERIALS?\s*(?:AND|&)\s*METHODS?\b|STUDY\s*DESIGN|PROCEDURE|RESEARCH\s+DESIGN)',
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Methods?\b|Materials?\s*(?:and|&)\s*Methods?\b|Study\s*[Dd]esign|Patients?\s+and\s+[Mm]ethods?)',
            ],
            "introduction": [
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:INTRODUCTION|BACKGROUND)',
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Introduction|Background)',
            ],
            "discussion": [
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:DISCUSSION)',
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Discussion)',
            ],
            "conclusion": [
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:CONCLUSIONS?|SUMMARY)',
                r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Conclusions?|Summary)',
            ],
        }

        patterns = heading_patterns.get(section_name, [])
        if not patterns:
            return ""

        # Try each pattern (strict first, then loose)
        match = None
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                break

        if not match:
            # Last resort for results: find numeric-dense regions
            if section_name == "results":
                return _find_numeric_dense_region(text, max_chars)
            return ""

        start = match.end()
        # Find the next major heading to delimit the section
        next_heading = re.search(
            r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Discussion|Conclusion|Results?|Methods?|References|Bibliography|Acknowledgements?|DISCUSSION|CONCLUSION|REFERENCES)\s*(?:\n|$)',
            text[start:], re.IGNORECASE | re.MULTILINE,
        )
        end = start + next_heading.start() if next_heading else start + max_chars
        content = text[start:end].strip()
        return content[:max_chars] if len(content) > max_chars else content


    # ================================================================== #
    #  LLM + JSON HELPERS                                                  #
    # ================================================================== #

    def _simple_generate(self, prompt: str, max_tokens: int = 300,
                         temperature: float = 0.1) -> str:
        if self._is_groq():
            return self.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
        inputs = self._tokenize_prompt(prompt)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
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

    def _clean_json_array(self, text: str) -> str:
        """Extract first valid JSON array from LLM output."""
        cleaned = text.replace("```json", "").replace("```", "").strip()
        cleaned = re.sub(r"[^\x00-\x7F]+", "", cleaned)

        # Try array first
        start = cleaned.find("[")
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "[":
                    depth += 1
                elif cleaned[i] == "]":
                    depth -= 1
                    if depth == 0:
                        return cleaned[start : i + 1]
            return cleaned[start:] + "]"

        # Fallback: try single object wrapped in array
        obj = self._clean_json(text)
        if obj and obj != "{}":
            return f"[{obj}]"
        return "[]"

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
    #  SUMMARY                                                             #
    # ================================================================== #

    def _print_summary(self, result: Dict):
        s = result["stats"]
        print(f"\n{'='*60}")
        print("AGENT 5 - EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Total papers     : {s['total']}")
        print(f"  Complete         : {s['complete_extraction']}")
        print(f"  Partial          : {s['partial_extraction']}")
        print(f"  Failed           : {s['failed_extraction']}")
        print(f"  Numeric complete : {s.get('numeric_complete', 0)}/{s['total']} papers have analyzable effect sizes")
        print(f"  Numeric recovered: {s.get('numeric_recovered', 0)} papers recovered via 4th pass")
        if s.get("dual_extraction"):
            print(f"  High confidence  : {s['high_confidence']}")
            print(f"  Medium confidence: {s['medium_confidence']}")
            print(f"  Low confidence   : {s['low_confidence']}")
        print(f"  Time             : {s['elapsed_s']}s")


# ====================================================================== #
#  STANDALONE TEST                                                         #
# ====================================================================== #
if __name__ == "__main__":
    a5 = Agent5Extraction()
    print("Agent5Extraction created. Run test_agent5.py for full test.")
