"""Borderline Review Report Generator

Analyzes full pipeline output and flags items needing optional human review.
Purely analytical — no LLM calls, no pipeline blocking.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any


class ReviewReportGenerator:
    """Generates a human review report flagging borderline/uncertain decisions.

    Input: full pipeline_data dict (output of A1-A7).
    Output: dict with flagged items per stage, priority levels, and recommendations.
    GPU: no -- purely analytical, no LLM calls.
    """

    # Thresholds
    SCREENING_CONF_HIGH = 0.2    # below this = high priority
    SCREENING_CONF_MED = 0.4     # below this = medium priority
    QUALITY_WEIGHT_LOW = 0.3     # below this = flagged
    I2_THRESHOLD = 75.0          # above this = substantial heterogeneity
    EGGER_P_THRESHOLD = 0.05     # below this = significant publication bias
    LOO_CHANGE_THRESHOLD = 0.3   # relative change in pooled estimate to flag

    # Estimated minutes per flag type for human review time
    _REVIEW_MINUTES = {
        "screening_borderline": 2,
        "fulltext_borderline": 4,
        "extraction_concerns": 5,
        "quality_concerns": 3,
        "synthesis_concerns": 5,
    }

    def generate(self, pipeline_data: dict) -> dict:
        """Analyze full pipeline output and flag items needing human review."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "screening_borderline": [],
            "fulltext_borderline": [],
            "extraction_concerns": [],
            "quality_concerns": [],
            "synthesis_concerns": [],
            "recommendations": [],
        }

        self._analyze_screening(pipeline_data, report)
        self._analyze_fulltext(pipeline_data, report)
        self._analyze_extraction(pipeline_data, report)
        self._analyze_quality(pipeline_data, report)
        self._analyze_synthesis(pipeline_data, report)
        self._generate_summary(report)
        self._generate_recommendations(report)

        return report

    # ================================================================== #
    #  SCREENING (A3)                                                      #
    # ================================================================== #

    def _analyze_screening(self, pipeline_data: dict, report: dict):
        """Flag borderline A3 screening decisions."""
        screening = pipeline_data.get("screening")
        if not screening:
            return

        # Collect all papers from all buckets
        all_papers = (
            screening.get("include", [])
            + screening.get("exclude", [])
            + screening.get("uncertain", [])
        )

        for paper in all_papers:
            flags_for_paper = []
            sc = paper.get("screening_confidence", 1.0)
            decision = paper.get("decision", "")
            votes = paper.get("votes", [])
            is_borderline = paper.get("borderline", False)

            # Flag UNCERTAIN decisions
            if decision == "UNCERTAIN":
                flags_for_paper.append("Decision was UNCERTAIN")

            # Flag low screening confidence
            if sc < self.SCREENING_CONF_MED:
                flags_for_paper.append(
                    f"Low screening confidence: {sc:.2f}"
                )

            # Flag split votes
            if votes:
                vote_decisions = []
                for v in votes:
                    if isinstance(v, dict):
                        vote_decisions.append(v.get("decision", "EXCLUDE"))
                    else:
                        vote_decisions.append(v)
                unique_votes = set(vote_decisions)
                if len(unique_votes) > 1:
                    counts = {d: vote_decisions.count(d) for d in unique_votes}
                    counts_str = ", ".join(f"{k}={v}" for k, v in counts.items())
                    flags_for_paper.append(f"Split votes: {counts_str}")

            # Flag explicit borderline marker
            if is_borderline and not flags_for_paper:
                flags_for_paper.append("Marked borderline by screening agent")

            if flags_for_paper:
                # Determine priority
                if sc < self.SCREENING_CONF_HIGH:
                    priority = "high"
                elif sc < self.SCREENING_CONF_MED:
                    priority = "medium"
                elif decision == "UNCERTAIN":
                    priority = "medium"
                else:
                    priority = "low"

                # Extract reasoning snippets from votes
                reasoning_snippets = []
                for v in votes:
                    if isinstance(v, dict) and v.get("reasoning"):
                        reasoning_snippets.append(v["reasoning"][:150])

                report["screening_borderline"].append({
                    "pmid": paper.get("pmid", ""),
                    "title": paper.get("title", ""),
                    "decision": decision,
                    "screening_confidence": sc,
                    "include_fraction": paper.get("confidence", 0),
                    "flags": flags_for_paper,
                    "priority": priority,
                    "reasoning_snippets": reasoning_snippets[:3],
                })

    # ================================================================== #
    #  FULL-TEXT SCREENING (A4)                                            #
    # ================================================================== #

    def _analyze_fulltext(self, pipeline_data: dict, report: dict):
        """Flag borderline A4 full-text screening decisions."""
        ft = pipeline_data.get("fulltext_retrieval")
        if not ft:
            return

        # All papers that went through A4
        all_papers = ft.get("papers", [])
        # If no 'papers' key, combine include + exclude
        if not all_papers:
            all_papers = ft.get("include", []) + ft.get("exclude", [])

        for paper in all_papers:
            flags_for_paper = []
            sc = paper.get("screening_confidence", 1.0)
            decision = paper.get("fulltext_decision", "")
            is_borderline = paper.get("borderline", False)
            text_source = paper.get("text_source", "")
            is_abstract_only = (text_source == "abstract_only")

            # Flag low screening confidence
            if sc < self.SCREENING_CONF_MED:
                flags_for_paper.append(
                    f"Low full-text screening confidence: {sc:.2f}"
                )

            # Flag abstract-only screening
            if is_abstract_only:
                flags_for_paper.append(
                    "Screened on abstract only (no full text retrieved) "
                    "- decision may be less reliable"
                )

            # Flag split votes
            ft_votes = paper.get("fulltext_votes", [])
            if ft_votes:
                vote_decisions = []
                for v in ft_votes:
                    if isinstance(v, dict):
                        vote_decisions.append(v.get("decision", "EXCLUDE"))
                    else:
                        vote_decisions.append(v)
                unique_votes = set(vote_decisions)
                if len(unique_votes) > 1:
                    counts = {d: vote_decisions.count(d) for d in unique_votes}
                    counts_str = ", ".join(f"{k}={v}" for k, v in counts.items())
                    flags_for_paper.append(f"Split votes: {counts_str}")

            # Flag explicit borderline
            if is_borderline and not flags_for_paper:
                flags_for_paper.append("Marked borderline by full-text screening agent")

            if flags_for_paper:
                if is_abstract_only and sc < self.SCREENING_CONF_MED:
                    priority = "high"
                elif sc < self.SCREENING_CONF_HIGH:
                    priority = "high"
                elif is_abstract_only or sc < self.SCREENING_CONF_MED:
                    priority = "medium"
                else:
                    priority = "low"

                report["fulltext_borderline"].append({
                    "pmid": paper.get("pmid", ""),
                    "title": paper.get("title", ""),
                    "decision": decision,
                    "text_source": text_source,
                    "screening_confidence": sc,
                    "flags": flags_for_paper,
                    "priority": priority,
                    "reasoning": paper.get("fulltext_reasoning", ""),
                })

    # ================================================================== #
    #  EXTRACTION (A5)                                                     #
    # ================================================================== #

    def _analyze_extraction(self, pipeline_data: dict, report: dict):
        """Flag extraction concerns from A5."""
        ext = pipeline_data.get("data_extraction")
        if not ext:
            return

        papers = ext.get("papers", [])

        for paper in papers:
            flags_for_paper = []
            pmid = paper.get("pmid", "")
            title = paper.get("title", "")

            # Check numeric completeness
            outcomes = paper.get("outcomes", [])
            any_numeric_complete = any(
                o.get("numeric_complete") for o in outcomes
            )
            if not any_numeric_complete:
                if outcomes:
                    flags_for_paper.append(
                        "No analyzable effect sizes found (numeric_complete=false for all outcomes)"
                    )
                else:
                    flags_for_paper.append("No outcomes extracted at all")

            # Check concordance (dual extraction)
            concordance = paper.get("concordance")
            if concordance:
                conc_flags = concordance.get("flags", [])
                if conc_flags:
                    # Check for effect_size_type mismatch
                    type_mismatches = [
                        f for f in conc_flags
                        if "effect_size_type mismatch" in f
                    ]
                    value_discs = [
                        f for f in conc_flags
                        if "discrepancy" in f
                    ]
                    if type_mismatches:
                        flags_for_paper.append(
                            f"Effect size type mismatch between extraction passes: "
                            f"{'; '.join(type_mismatches)}"
                        )
                    if value_discs:
                        flags_for_paper.append(
                            f"Numeric value discrepancy between extraction passes: "
                            f"{'; '.join(value_discs[:3])}"
                        )

                if not concordance.get("sample_size_match", True):
                    flags_for_paper.append("Sample size disagreement between extraction passes")

                disc = concordance.get("outcomes_discrepant", 0)
                matched = concordance.get("outcomes_matched", 0)
                if disc > 0 and disc >= matched:
                    flags_for_paper.append(
                        f"Most outcomes discrepant between passes ({disc} discrepant vs {matched} matched)"
                    )

            # Check extraction quality / confidence
            eq = paper.get("extraction_quality", "")
            ec = paper.get("extraction_confidence", "")
            if eq == "failed":
                flags_for_paper.append("Extraction completely failed")
            if ec == "low":
                flags_for_paper.append("Low extraction confidence (dual extraction disagreement)")

            if flags_for_paper:
                priority = "high" if (not any_numeric_complete or eq == "failed") else "medium"

                report["extraction_concerns"].append({
                    "pmid": pmid,
                    "title": title,
                    "extraction_quality": eq,
                    "extraction_confidence": ec,
                    "n_outcomes": len(outcomes),
                    "n_numeric_complete": sum(1 for o in outcomes if o.get("numeric_complete")),
                    "flags": flags_for_paper,
                    "priority": priority,
                    "concordance_summary": self._summarize_concordance(concordance),
                })

    def _summarize_concordance(self, concordance: Optional[Dict]) -> Optional[Dict]:
        """Create a brief concordance summary."""
        if not concordance:
            return None
        return {
            "sample_size_match": concordance.get("sample_size_match"),
            "outcomes_matched": concordance.get("outcomes_matched", 0),
            "outcomes_discrepant": concordance.get("outcomes_discrepant", 0),
            "outcomes_pass1_only": concordance.get("outcomes_pass1_only", 0),
            "outcomes_pass2_only": concordance.get("outcomes_pass2_only", 0),
            "n_flags": len(concordance.get("flags", [])),
        }

    # ================================================================== #
    #  QUALITY (A6)                                                        #
    # ================================================================== #

    def _analyze_quality(self, pipeline_data: dict, report: dict):
        """Flag quality assessment concerns from A6."""
        qa = pipeline_data.get("quality_assessment")
        if not qa:
            return

        papers = qa.get("papers", [])

        for paper in papers:
            flags_for_paper = []
            pmid = paper.get("pmid", "")
            title = paper.get("title", "")
            weight = paper.get("quality_weight", 1.0)
            assessment = paper.get("quality_assessment", {})
            tool = assessment.get("tool", "")
            overall = assessment.get("overall", "")
            design = paper.get("study_design_detected", "")

            # Flag low quality weight
            if weight < self.QUALITY_WEIGHT_LOW:
                flags_for_paper.append(
                    f"Very low quality weight: {weight:.2f}"
                )

            # Check domains for multiple High/Critical
            if tool in ("RoB2", "ROBINS-I"):
                domains = assessment.get("domains", {})
                high_domains = []
                for dname, dinfo in domains.items():
                    judgment = (dinfo.get("judgment", "") if isinstance(dinfo, dict)
                                else "").lower()
                    if judgment in ("high", "critical"):
                        high_domains.append(dname)

                if len(high_domains) >= 2:
                    flags_for_paper.append(
                        f"Multiple domains rated High/Critical: {', '.join(high_domains)}"
                    )

            # Check JBI with low yes count
            if tool == "JBI":
                yes_count = assessment.get("yes_count", 0)
                if yes_count < 5:
                    flags_for_paper.append(
                        f"JBI: only {yes_count}/10 Yes answers"
                    )

            # Check for overall High / Critical / Exclude
            if overall in ("High", "Critical", "Exclude"):
                if not any("quality weight" in f for f in flags_for_paper):
                    flags_for_paper.append(
                        f"Overall assessment: {overall}"
                    )

            # Check if design detection fell back to default
            if design == "unknown":
                flags_for_paper.append(
                    "Study design could not be detected (used fallback assessment)"
                )

            if flags_for_paper:
                priority = "high" if weight < self.QUALITY_WEIGHT_LOW else "medium"

                domain_details = None
                if tool in ("RoB2", "ROBINS-I"):
                    domains = assessment.get("domains", {})
                    domain_details = {
                        dname: dinfo.get("judgment", "?")
                        for dname, dinfo in domains.items()
                        if isinstance(dinfo, dict)
                    }

                report["quality_concerns"].append({
                    "pmid": pmid,
                    "title": title,
                    "study_design": design,
                    "tool": tool,
                    "overall": overall,
                    "quality_weight": weight,
                    "flags": flags_for_paper,
                    "priority": priority,
                    "domain_details": domain_details,
                })

    # ================================================================== #
    #  SYNTHESIS (A7)                                                      #
    # ================================================================== #

    def _analyze_synthesis(self, pipeline_data: dict, report: dict):
        """Flag synthesis concerns from A7."""
        synthesis = pipeline_data.get("synthesis")
        if not synthesis:
            return

        analyses = synthesis.get("analyses", {})

        for measure, analysis in analyses.items():
            flags_for_analysis = []

            model = analysis.get("model", {})
            heterogeneity = analysis.get("heterogeneity", {})
            pub_bias = analysis.get("publication_bias", {})
            grade = analysis.get("grade", {})
            analysis.get("sensitivity", {})
            loo = analysis.get("leave_one_out", [])

            pooled = model.get("pooled_estimate")
            ci_lower = model.get("ci_lower")
            ci_upper = model.get("ci_upper")
            i2 = heterogeneity.get("I2", model.get("I2", 0))
            pi_lower = model.get("prediction_interval_lower")
            pi_upper = model.get("prediction_interval_upper")

            # Flag substantial heterogeneity
            if i2 is not None and i2 > self.I2_THRESHOLD:
                flags_for_analysis.append(
                    f"Substantial heterogeneity: I^2 = {i2:.1f}% "
                    f"(threshold: {self.I2_THRESHOLD}%)"
                )

            # Flag Egger's test significance (publication bias)
            egger = pub_bias.get("egger_test", {})
            if not egger.get("skipped", False):
                egger_p = egger.get("p_value")
                if egger_p is not None and egger_p < self.EGGER_P_THRESHOLD:
                    flags_for_analysis.append(
                        f"Egger's test significant (p = {egger_p:.4f}) "
                        f"- publication bias detected"
                    )

            # Flag trim-and-fill adjustment
            tf = pub_bias.get("trim_and_fill", {})
            n_missing = tf.get("n_missing", 0)
            if n_missing and n_missing > 0:
                flags_for_analysis.append(
                    f"Trim-and-fill estimates {n_missing} missing studies"
                )

            # Flag GRADE "Very low"
            grade_level = grade.get("final_level", "")
            if grade_level.lower() in ("very low", "very_low"):
                flags_for_analysis.append(
                    "GRADE certainty: Very low"
                )

            # Flag prediction interval crossing null when CI doesn't
            if (pooled is not None and pi_lower is not None and pi_upper is not None
                    and ci_lower is not None and ci_upper is not None):
                ci_crosses_null = (ci_lower <= 0 <= ci_upper)
                pi_crosses_null = (pi_lower <= 0 <= pi_upper)
                if pi_crosses_null and not ci_crosses_null:
                    flags_for_analysis.append(
                        f"Prediction interval crosses null "
                        f"[{pi_lower:.3f}, {pi_upper:.3f}] "
                        f"even though CI does not [{ci_lower:.3f}, {ci_upper:.3f}] "
                        f"- individual study effects may vary widely"
                    )

            # Flag leave-one-out influential studies
            if loo and pooled is not None and pooled != 0:
                for entry in loo:
                    loo_est = entry.get("pooled_estimate") or entry.get("estimate")
                    study_label = entry.get("study", entry.get("label", "?"))
                    if loo_est is not None:
                        try:
                            relative_change = abs(float(loo_est) - float(pooled)) / max(abs(float(pooled)), 0.001)
                            if relative_change > self.LOO_CHANGE_THRESHOLD:
                                flags_for_analysis.append(
                                    f"Removing '{study_label}' changes pooled estimate by "
                                    f"{relative_change*100:.0f}% "
                                    f"(from {pooled} to {loo_est})"
                                )
                        except (ValueError, TypeError):
                            pass

            if flags_for_analysis:
                priority = "high" if (
                    any("Very low" in f for f in flags_for_analysis)
                    or any("publication bias" in f for f in flags_for_analysis)
                ) else "medium"

                report["synthesis_concerns"].append({
                    "measure": measure,
                    "pooled_estimate": pooled,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "I2": i2,
                    "grade_level": grade_level,
                    "k_studies": model.get("k", model.get("k_studies")),
                    "flags": flags_for_analysis,
                    "priority": priority,
                })

    # ================================================================== #
    #  SUMMARY                                                             #
    # ================================================================== #

    def _generate_summary(self, report: dict):
        """Generate summary statistics for the report."""
        sections = [
            "screening_borderline",
            "fulltext_borderline",
            "extraction_concerns",
            "quality_concerns",
            "synthesis_concerns",
        ]

        total_flags = 0
        high_priority = 0
        medium_priority = 0
        low_priority = 0
        per_stage = {}

        for section in sections:
            items = report.get(section, [])
            stage_high = sum(1 for i in items if i.get("priority") == "high")
            stage_med = sum(1 for i in items if i.get("priority") == "medium")
            stage_low = sum(1 for i in items if i.get("priority") == "low")
            per_stage[section] = {
                "total": len(items),
                "high": stage_high,
                "medium": stage_med,
                "low": stage_low,
            }
            total_flags += len(items)
            high_priority += stage_high
            medium_priority += stage_med
            low_priority += stage_low

        # Overall pipeline confidence
        if high_priority >= 5 or total_flags >= 15:
            confidence = "low"
        elif high_priority >= 2 or total_flags >= 8:
            confidence = "moderate"
        else:
            confidence = "high"

        # Estimated review time
        review_minutes = 0
        for section in sections:
            n_items = len(report.get(section, []))
            review_minutes += n_items * self._REVIEW_MINUTES.get(section, 3)

        report["summary"] = {
            "total_flags": total_flags,
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority,
            "per_stage": per_stage,
            "pipeline_confidence": confidence,
            "estimated_review_minutes": review_minutes,
        }

    # ================================================================== #
    #  RECOMMENDATIONS                                                     #
    # ================================================================== #

    def _generate_recommendations(self, report: dict):
        """Generate actionable recommendations based on all flags."""
        recs = []

        # Screening recommendations
        scr = report.get("screening_borderline", [])
        if scr:
            n_high = sum(1 for i in scr if i.get("priority") == "high")
            recs.append(
                f"Review {len(scr)} borderline screening decision(s) "
                f"({n_high} high priority). "
                f"These papers had split votes or low confidence in title/abstract screening."
            )

        # Full-text recommendations
        ft = report.get("fulltext_borderline", [])
        if ft:
            n_abstract = sum(1 for i in ft if i.get("text_source") == "abstract_only")
            n_high = sum(1 for i in ft if i.get("priority") == "high")
            rec = f"Review {len(ft)} borderline full-text decision(s) ({n_high} high priority)."
            if n_abstract:
                rec += (
                    f" {n_abstract} paper(s) were screened on abstract only "
                    f"(full text not available) - consider manual retrieval."
                )
            recs.append(rec)

        # Extraction recommendations
        ext = report.get("extraction_concerns", [])
        if ext:
            n_no_numeric = sum(
                1 for i in ext
                if any("numeric" in f.lower() or "no outcomes" in f.lower()
                       for f in i.get("flags", []))
            )
            if n_no_numeric:
                recs.append(
                    f"{n_no_numeric} paper(s) lack numeric effect sizes "
                    f"- consider manual extraction from original PDFs."
                )
            n_concordance = sum(
                1 for i in ext
                if any("discrepancy" in f.lower() or "mismatch" in f.lower()
                       for f in i.get("flags", []))
            )
            if n_concordance:
                recs.append(
                    f"{n_concordance} paper(s) have extraction concordance issues "
                    f"- verify effect sizes against original text."
                )

        # Quality recommendations
        qa = report.get("quality_concerns", [])
        if qa:
            n_low_quality = sum(1 for i in qa if i.get("quality_weight", 1.0) < 0.3)
            if n_low_quality:
                recs.append(
                    f"{n_low_quality} paper(s) have very low quality weights "
                    f"- consider whether they should be included in sensitivity analysis only."
                )
            n_multi_high = sum(
                1 for i in qa
                if any("Multiple domains" in f for f in i.get("flags", []))
            )
            if n_multi_high:
                recs.append(
                    f"{n_multi_high} paper(s) have multiple high-risk domains "
                    f"- results should be interpreted with caution."
                )

        # Synthesis recommendations
        syn = report.get("synthesis_concerns", [])
        if syn:
            for item in syn:
                measure = item.get("measure", "?")
                for flag in item.get("flags", []):
                    if "heterogeneity" in flag.lower():
                        recs.append(
                            f"Substantial heterogeneity in {measure} analysis "
                            f"- consider investigating sources (subgroup analysis, meta-regression)."
                        )
                    elif "publication bias" in flag.lower():
                        recs.append(
                            f"Publication bias detected in {measure} analysis "
                            f"- interpret results with caution."
                        )
                    elif "Very low" in flag:
                        recs.append(
                            f"GRADE certainty is Very low for {measure} "
                            f"- the true effect may be substantially different from the estimate."
                        )
                    elif "Prediction interval crosses null" in flag:
                        recs.append(
                            f"Prediction interval crosses null for {measure} "
                            f"- future studies may show different directions of effect."
                        )

        # General recommendation if no flags
        if not recs:
            recs.append(
                "No significant concerns flagged. Pipeline decisions appear consistent "
                "and well-supported. Standard peer review of the manuscript is recommended."
            )

        report["recommendations"] = recs

    # ================================================================== #
    #  TEXT OUTPUT                                                          #
    # ================================================================== #

    def generate_text_report(self, report: dict) -> str:
        """Generate a plain text version of the review report."""
        lines = []
        lines.append("=" * 72)
        lines.append("BORDERLINE REVIEW REPORT")
        lines.append(f"Generated: {report.get('generated_at', 'N/A')}")
        lines.append("=" * 72)

        summary = report.get("summary", {})
        lines.append("")
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Pipeline confidence : {summary.get('pipeline_confidence', 'N/A').upper()}")
        lines.append(f"  Total flags         : {summary.get('total_flags', 0)}")
        lines.append(f"    High priority     : {summary.get('high_priority', 0)}")
        lines.append(f"    Medium priority   : {summary.get('medium_priority', 0)}")
        lines.append(f"    Low priority      : {summary.get('low_priority', 0)}")
        lines.append(f"  Est. review time    : {summary.get('estimated_review_minutes', 0)} minutes")

        # Per-stage breakdown
        per_stage = summary.get("per_stage", {})
        if per_stage:
            lines.append("")
            lines.append("  Per-stage breakdown:")
            stage_labels = {
                "screening_borderline": "Screening (A3)",
                "fulltext_borderline": "Full-text (A4)",
                "extraction_concerns": "Extraction (A5)",
                "quality_concerns": "Quality (A6)",
                "synthesis_concerns": "Synthesis (A7)",
            }
            for key, label in stage_labels.items():
                info = per_stage.get(key, {})
                total = info.get("total", 0)
                if total > 0:
                    lines.append(
                        f"    {label:22s}: {total} flags "
                        f"({info.get('high', 0)}H / {info.get('medium', 0)}M / {info.get('low', 0)}L)"
                    )

        # Screening borderline
        scr = report.get("screening_borderline", [])
        if scr:
            lines.append("")
            lines.append("SCREENING BORDERLINE DECISIONS (A3)")
            lines.append("-" * 40)
            for item in scr:
                pri = item.get("priority", "?").upper()
                lines.append(
                    f"  [{pri}] PMID {item.get('pmid', '?')} - "
                    f"{item.get('title', '?')[:80]}"
                )
                lines.append(
                    f"         Decision: {item.get('decision', '?')} | "
                    f"Confidence: {item.get('screening_confidence', '?')}"
                )
                for flag in item.get("flags", []):
                    lines.append(f"         * {flag}")

        # Fulltext borderline
        ft = report.get("fulltext_borderline", [])
        if ft:
            lines.append("")
            lines.append("FULL-TEXT BORDERLINE DECISIONS (A4)")
            lines.append("-" * 40)
            for item in ft:
                pri = item.get("priority", "?").upper()
                lines.append(
                    f"  [{pri}] PMID {item.get('pmid', '?')} - "
                    f"{item.get('title', '?')[:80]}"
                )
                lines.append(
                    f"         Source: {item.get('text_source', '?')} | "
                    f"Decision: {item.get('decision', '?')} | "
                    f"Confidence: {item.get('screening_confidence', '?')}"
                )
                for flag in item.get("flags", []):
                    lines.append(f"         * {flag}")

        # Extraction concerns
        ext = report.get("extraction_concerns", [])
        if ext:
            lines.append("")
            lines.append("EXTRACTION CONCERNS (A5)")
            lines.append("-" * 40)
            for item in ext:
                pri = item.get("priority", "?").upper()
                lines.append(
                    f"  [{pri}] PMID {item.get('pmid', '?')} - "
                    f"{item.get('title', '?')[:80]}"
                )
                lines.append(
                    f"         Quality: {item.get('extraction_quality', '?')} | "
                    f"Confidence: {item.get('extraction_confidence', '?')} | "
                    f"Outcomes: {item.get('n_numeric_complete', 0)}/{item.get('n_outcomes', 0)} numeric complete"
                )
                for flag in item.get("flags", []):
                    lines.append(f"         * {flag}")

        # Quality concerns
        qa = report.get("quality_concerns", [])
        if qa:
            lines.append("")
            lines.append("QUALITY CONCERNS (A6)")
            lines.append("-" * 40)
            for item in qa:
                pri = item.get("priority", "?").upper()
                lines.append(
                    f"  [{pri}] PMID {item.get('pmid', '?')} - "
                    f"{item.get('title', '?')[:80]}"
                )
                lines.append(
                    f"         Design: {item.get('study_design', '?')} | "
                    f"Tool: {item.get('tool', '?')} | "
                    f"Overall: {item.get('overall', '?')} | "
                    f"Weight: {item.get('quality_weight', '?')}"
                )
                for flag in item.get("flags", []):
                    lines.append(f"         * {flag}")
                dd = item.get("domain_details")
                if dd:
                    lines.append(
                        f"         Domains: {', '.join(f'{k}={v}' for k, v in dd.items())}"
                    )

        # Synthesis concerns
        syn = report.get("synthesis_concerns", [])
        if syn:
            lines.append("")
            lines.append("SYNTHESIS CONCERNS (A7)")
            lines.append("-" * 40)
            for item in syn:
                pri = item.get("priority", "?").upper()
                lines.append(
                    f"  [{pri}] {item.get('measure', '?')} analysis"
                )
                lines.append(
                    f"         Pooled: {item.get('pooled_estimate', '?')} "
                    f"[{item.get('ci_lower', '?')}, {item.get('ci_upper', '?')}] | "
                    f"I^2: {item.get('I2', '?')}% | "
                    f"GRADE: {item.get('grade_level', '?')} | "
                    f"k={item.get('k_studies', '?')}"
                )
                for flag in item.get("flags", []):
                    lines.append(f"         * {flag}")

        # Recommendations
        recs = report.get("recommendations", [])
        if recs:
            lines.append("")
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 40)
            for i, rec in enumerate(recs, 1):
                lines.append(f"  {i}. {rec}")

        lines.append("")
        lines.append("=" * 72)
        lines.append("END OF REPORT")
        lines.append("=" * 72)

        return "\n".join(lines)
