"""Agent 7: Statistical Meta-Analysis (R/metafor via rpy2)

No LLM — pure computation. Uses R's metafor package for:
  - Random-effects pooling (REML) with prediction intervals
  - Heterogeneity analysis (I^2, tau^2, Q)
  - Publication bias (Egger's test with k>=10 check, trim-and-fill)
  - Sensitivity analysis (leave-one-out, exclude high-risk)
  - Subgroup analysis (by design, risk of bias, sample size, text source)
  - Meta-regression (year, sample size — if k>=10)
  - Improved GRADE certainty assessment (5 domains)
  - Summary of Findings (SoF) table
  - Forest + funnel plots

Requires: R with metafor installed, rpy2 Python package.
"""

import os, time, math, re
from typing import Dict, List, Optional, Tuple

from agents.effect_converters import (
    convert_md_to_smd_study,
    convert_rr_to_or_study,
    convert_or_to_rr_study,
)

# Attempt matplotlib import for fallback plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Default config values (overridden by config.yaml if present)
_DEFAULT_SYNTHESIS_CONFIG = {
    "min_studies_for_egger": 10,
    "min_studies_for_meta_regression": 10,
    "min_studies_per_subgroup": 2,
    "subgroup_analyses": {
        "by_risk_of_bias": True,
        "by_sample_size": True,
        "by_study_design": True,
    },
}


# Clinical outcome category keyword map (lowercase keywords)
# Expanded with synonyms for nutrition/dietary/anthropometric outcomes
OUTCOME_CATEGORIES = {
    "weight": [
        "weight", "bmi", "body mass", "adiposity", "fat mass", "body fat",
        "body composition", "lean mass", "lean body", "visceral fat",
        "visceral adipos", "body weight", "weight change", "weight loss",
        "weight gain", "kg", "percent body fat", "fat-free mass",
        "trunk fat", "android fat", "fat percentage", "dexa",
        "body density", "skinfold", "bioimpedance", "bia",
    ],
    "waist": [
        "waist", "circumference", "abdominal", "waist-to-hip",
        "waist hip ratio", "whr", "central obesity", "sagittal",
        "abdominal girth",
    ],
    "glycemic": [
        "hba1c", "a1c", "glucose", "glycemic", "glycated",
        "fasting blood sugar", "insulin", "homa", "insulin resistance",
        "glycosylated", "blood glucose",
    ],
    "lipids": [
        "cholesterol", "triglyceride", "lipid", "hdl", "ldl",
        "total cholesterol", "vldl", "lipoprotein",
    ],
    "blood_pressure": [
        "blood pressure", "systolic", "diastolic", "hypertension",
        "sbp", "dbp", "arterial pressure", "mmhg",
    ],
    "dietary_intake": [
        "fruit", "vegetable", "diet", "calori", "fat intake",
        "nutrition", "eating", "food", "kcal", "energy intake",
        "fiber", "fibre", "sodium", "salt", "sugar intake",
        "saturated fat", "protein intake", "carbohydrate",
        "dietary quality", "diet quality", "healthy eating",
        "diet score", "dietary pattern", "food frequency",
        "serving", "portion", "mediterranean", "dash",
        "nutritional", "meal", "snack", "alcohol",
        "micronutrient", "macronutrient", "dietary change",
        "food intake", "nutrient", "consumption",
        "dietary adherence", "dietary compliance",
    ],
    "physical_activity": [
        "physical activity", "exercise", "step", "walk", "mvpa",
        "met ", "fitness", "sedentary", "aerobic", "strength",
        "endurance", "cardiorespiratory", "vo2",
    ],
    "adherence": [
        "adherence", "compliance", "self-management",
        "self-efficacy", "attendance", "engagement",
        "motivation", "goal attainment", "behavior change",
        "behavioural change", "behavioral change",
    ],
    "quality_of_life": [
        "quality of life", "well-being", "wellbeing",
        "happiness", "satisfaction", "sf-36", "sf-12",
        "eq-5d", "mental health", "depression", "anxiety",
        "psychological", "mood", "stress",
    ],
    "psychological": [
        "self-efficacy", "self efficacy", "locus of control",
        "readiness to change", "stage of change", "motivation",
        "intention", "attitude", "knowledge", "awareness",
        "health literacy", "empowerment",
    ],
}


class Agent7Synthesis:
    """Statistical meta-analysis engine — no LLM, uses R/metafor."""

    def __init__(self, config: Optional[Dict] = None):
        self.r = None
        self.metafor = None
        self._r_initialized = False
        # Merge provided config with defaults
        self._config = dict(_DEFAULT_SYNTHESIS_CONFIG)
        if config and isinstance(config, dict):
            synth_cfg = config.get("synthesis", {})
            if synth_cfg:
                for k, v in synth_cfg.items():
                    self._config[k] = v

    # ================================================================== #
    #  PUBLIC API                                                          #
    # ================================================================== #

    def synthesize(self, a6_result: Dict, protocol: Dict, output_dir: str = "outputs") -> Dict:
        """Run full meta-analysis on A6-assessed papers."""
        self._output_dir = output_dir
        self._protocol = protocol
        papers = a6_result
        if isinstance(a6_result, dict) and "papers" in a6_result:
            papers = a6_result["papers"]

        print(f"\n{'='*60}")
        print("AGENT 7 - STATISTICAL META-ANALYSIS")
        print(f"{'='*60}")
        print(f"  Papers input: {len(papers)}")

        start = time.time()

        # Init R
        r_ok = self._init_r()
        if not r_ok:
            print("  WARNING: R/metafor not available. Returning empty synthesis.")
            return self._empty_result(papers)

        # Prepare data grouped by measure type
        groups = self._prepare_data(papers)
        if not groups:
            print("  WARNING: No usable outcome data found in any paper. Check A5 extraction results.")
            return self._empty_result(papers)

        print(f"  Outcome groups (before conversion): {list(groups.keys())}")

        # Attempt effect-size conversions to merge compatible groups
        groups = self._attempt_conversions(groups)
        print(f"  Outcome groups (after conversion):  {list(groups.keys())}")

        # Create plots directory
        plots_dir = os.path.join(self._output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        analyses = {}
        for group_key, data in groups.items():
            k = len(data["yi"])
            # The statistical measure type to pass to R/metafor (SMD, MD, OR, etc.)
            measure = data.get("_measure_type", group_key)
            print(f"\n  --- Analyzing '{group_key}' (measure={measure}, k={k}) ---")

            if k < 2:
                print(f"    Skipping '{group_key}': need >= 2 studies")
                continue

            analysis = {}

            # Fit random-effects model (includes prediction interval)
            rma_result = self._fit_random_effects(data, measure)
            if not rma_result:
                print(f"    Failed to fit model for '{group_key}'")
                continue
            # Store measure type on rma_result for downstream use (GRADE imprecision)
            rma_result["_measure_type"] = measure
            analysis["model"] = rma_result
            # Record which clinical category / group key this analysis covers
            analysis["outcome_category"] = group_key
            analysis["measure_type"] = measure

            # Total participants
            analysis["total_participants"] = self._compute_total_participants(data)

            # Heterogeneity
            analysis["heterogeneity"] = self._heterogeneity_analysis(rma_result)

            # Publication bias (need >= 3 studies for trim-and-fill)
            if k >= 3:
                analysis["publication_bias"] = self._publication_bias(data, measure)
            else:
                analysis["publication_bias"] = {"note": "Skipped: fewer than 3 studies"}

            # Sensitivity: leave-one-out + exclude-high-risk + quality-weighted
            if k >= 3:
                analysis["sensitivity"] = {
                    "leave_one_out": self._sensitivity_leave_one_out(data, measure),
                    "exclude_high_risk": self._sensitivity_exclude_high_risk(data, measure),
                    "quality_weighted": self._sensitivity_quality_weighted(data, measure),
                }
            else:
                analysis["sensitivity"] = {"note": "Skipped: fewer than 3 studies"}

            # Subgroup analyses
            analysis["subgroup_analyses"] = self._run_subgroup_analyses(data, measure)

            # Meta-regression (if enough studies)
            analysis["meta_regression"] = self._run_meta_regression(data, measure)

            # Plots — use group_key (category name) for filenames
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', group_key)
            forest_path = self._generate_forest_plot(
                data, measure,
                os.path.join(plots_dir, f"forest_{safe_key}.png")
            )
            funnel_path = self._generate_funnel_plot(
                data, measure,
                os.path.join(plots_dir, f"funnel_{safe_key}.png")
            )
            analysis["plots"] = {"forest": forest_path, "funnel": funnel_path}

            # Improved GRADE assessment
            analysis["grade"] = self._grade_assessment(
                rma_result,
                analysis["heterogeneity"],
                analysis.get("publication_bias", {}),
                data,
            )

            # Summary of Findings table (use category label as outcome label)
            analysis["summary_of_findings"] = self._generate_sof(
                group_key, rma_result, analysis, data
            )

            analyses[group_key] = analysis

        elapsed = round(time.time() - start, 1)

        # Count papers with usable data
        papers_with_data = set()
        for data in groups.values():
            papers_with_data.update(data.get("pmids", []))

        result = {
            "analyses": analyses,
            "stats": {
                "total_papers_input": len(papers),
                "papers_with_usable_data": len(papers_with_data),
                "analyses_performed": len(analyses),
                "elapsed_s": elapsed,
            },
        }

        self._print_summary(result)
        return result

    # ================================================================== #
    #  R INITIALIZATION                                                    #
    # ================================================================== #

    def _init_r(self) -> bool:
        """Initialize rpy2 and load metafor. Auto-install if missing."""
        if self._r_initialized:
            return True

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import FloatVector, StrVector, IntVector
            from rpy2.robjects.packages import importr

            self.r = ro
            self.FloatVector = FloatVector
            self.StrVector = StrVector
            self.IntVector = IntVector

            # Add user R library path
            user_r_lib = os.path.expanduser("~/R/library")
            if os.path.isdir(user_r_lib):
                ro.r(f'.libPaths(c("{user_r_lib}", .libPaths()))')

            # Try loading metafor
            try:
                self.metafor = importr("metafor")
            except Exception:
                print("  Installing metafor in R...")
                utils = importr("utils")
                utils.chooseCRANmirror(ind=1)
                utils.install_packages("metafor")
                self.metafor = importr("metafor")

            self.base = importr("base")
            self.stats = importr("stats")
            self.grdevices = importr("grDevices")

            self._r_initialized = True
            print("  R/metafor initialized successfully")
            return True

        except ImportError:
            print("  ERROR: rpy2 not installed. Install with: pip install rpy2")
            return False
        except Exception as e:
            print(f"  ERROR initializing R: {e}")
            return False

    # ================================================================== #
    #  DATA PREPARATION                                                    #
    # ================================================================== #

    def _prepare_data(self, papers: List[Dict]) -> Dict[str, Dict]:
        """Group outcomes by clinical category (with measure-type fallback), build yi/vi arrays.

        Primary grouping: clinical outcome category (weight, glycemic, etc.) derived
        from keyword matching on the outcome name field.  If an outcome has no name or
        no category matches, the group key falls back to the canonical measure type
        (SMD, MD, OR, RR, HR, PROP) — preserving backward compatibility.

        Each group dict contains a ``_measure_type`` field recording the dominant
        statistical measure used within that group (required by R/metafor and GRADE).

        Returns dict of { group_key: { yi, vi, labels, quality_weights, pmids,
        study_designs, rob_levels, sample_sizes, years, abstract_only,
        _measure_type, ... } }
        """
        # Canonical measure type mapping
        type_map = {
            "smd": "SMD", "hedges_g": "SMD", "cohen_d": "SMD",
            "standardized mean difference": "SMD",
            "or": "OR", "odds_ratio": "OR", "odds ratio": "OR",
            "rr": "RR", "risk_ratio": "RR", "risk ratio": "RR",
            "relative_risk": "RR", "relative risk": "RR",
            "hr": "HR", "hazard_ratio": "HR", "hazard ratio": "HR",
            "md": "MD", "mean_diff": "MD", "mean_difference": "MD",
            "mean difference": "MD",
            "proportion": "PROP", "prevalence": "PROP",
        }

        groups = {}
        # Track which measure types appear in each clinical category group
        # so we can record _measure_type after all outcomes are added.
        group_measure_counts: Dict[str, Dict[str, int]] = {}

        for paper in papers:
            pmid = paper.get("pmid", "unknown")
            weight = paper.get("quality_weight", 0.5)
            outcomes = paper.get("outcomes", [])

            if not outcomes:
                continue

            # Extract study-level metadata for subgroup / meta-regression
            design = paper.get("study_design_detected", "unknown")
            rob_level = self._classify_rob_level(paper)
            sc = paper.get("study_characteristics") or {}
            sample_size = self._extract_sample_size(sc)
            year = self._extract_year(sc, paper)
            text_src = paper.get("text_source", "")

            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue

                raw_type = str(outcome.get("measure_type", "")).lower().strip()
                measure = type_map.get(raw_type)
                if not measure:
                    # Try partial match
                    for key, val in type_map.items():
                        if key in raw_type:
                            measure = val
                            break
                if not measure:
                    continue

                yi = outcome.get("value")
                if yi is None:
                    continue
                try:
                    yi = float(yi)
                except (ValueError, TypeError):
                    continue

                # Get variance: from CI or direct
                vi = self._get_variance(outcome, yi)
                if vi is None or vi <= 0:
                    continue

                # --- Determine group key ---
                # Use clinical category if the outcome has a name that matches;
                # fall back to measure type for unnamed or unclassifiable outcomes.
                outcome_name = str(outcome.get("name", "") or "").strip()
                if outcome_name:
                    category = self._classify_outcome_category(outcome_name)
                    # "other" category with a name: use category key so related
                    # unnamed outcomes still cluster together under their measure.
                    group_key = category if category != "other" else measure
                else:
                    # No name → backward-compat: group by measure type
                    group_key = measure

                # Build label from outcome name (truncated) rather than paper title
                if outcome_name:
                    label = f"{pmid}: {outcome_name[:50]}"
                else:
                    title = paper.get("title", "")[:40]
                    label = f"{pmid}: {title}"

                if group_key not in groups:
                    groups[group_key] = {
                        "yi": [], "vi": [], "labels": [],
                        "quality_weights": [], "pmids": [],
                        "abstract_only": [],
                        "study_designs": [],
                        "rob_levels": [],
                        "sample_sizes": [],
                        "years": [],
                        "outcomes_raw": [],
                        "converted_flags": [],
                        "conversion_notes": [],
                        "original_effect_sizes": [],
                        "original_ci_lowers": [],
                        "original_ci_uppers": [],
                        "_measure_type": measure,  # Will be updated below
                    }
                    group_measure_counts[group_key] = {}

                groups[group_key]["yi"].append(yi)
                groups[group_key]["vi"].append(vi)
                groups[group_key]["labels"].append(label)
                groups[group_key]["quality_weights"].append(weight)
                groups[group_key]["pmids"].append(pmid)
                groups[group_key]["abstract_only"].append(text_src == "abstract_only")
                groups[group_key]["study_designs"].append(design)
                groups[group_key]["rob_levels"].append(rob_level)
                groups[group_key]["sample_sizes"].append(sample_size)
                groups[group_key]["years"].append(year)
                groups[group_key]["outcomes_raw"].append(outcome)
                groups[group_key]["converted_flags"].append(False)
                groups[group_key]["conversion_notes"].append("")
                groups[group_key]["original_effect_sizes"].append(yi)
                groups[group_key]["original_ci_lowers"].append(outcome.get("ci_lower"))
                groups[group_key]["original_ci_uppers"].append(outcome.get("ci_upper"))

                # Track measure frequency per group
                group_measure_counts[group_key][measure] = (
                    group_measure_counts[group_key].get(measure, 0) + 1
                )

        # Set _measure_type to the most-common measure within each group
        for gkey, mcounts in group_measure_counts.items():
            if mcounts:
                dominant = max(mcounts, key=lambda m: mcounts[m])
                groups[gkey]["_measure_type"] = dominant

        return groups

    # ================================================================== #
    #  EFFECT-SIZE CONVERSIONS (merge compatible groups)                    #
    # ================================================================== #

    def _attempt_conversions(self, groups: Dict[str, Dict]) -> Dict[str, Dict]:
        """Harmonise effect-size measures within each group and merge compatible
        measure-type groups (backward-compat path).

        Two levels of conversion:

        1. **Intra-group**: A clinical-category group may contain both MD and SMD
           outcomes (different studies reported different metrics).  These are
           harmonised to SMD so metafor can pool them.

        2. **Inter-group**: When the old measure-type keys (MD/SMD, OR/RR) exist as
           top-level group keys (backward-compat path — no outcome names), they are
           merged as before.

        Studies that cannot be converted remain with their original measure type.
        Conversion metadata is recorded per study.
        """
        groups = dict(groups)  # shallow copy to avoid mutating caller's dict

        # ---- 1. Intra-group harmonisation (clinical category groups) ---- #
        for gkey in list(groups.keys()):
            # Skip pure measure-type keys (backward-compat path)
            if gkey in ("SMD", "MD", "OR", "RR", "HR", "PROP"):
                continue
            group = groups[gkey]
            mcounts = {}
            for raw_outcome in group.get("outcomes_raw", []):
                if not isinstance(raw_outcome, dict):
                    continue
                raw_type = str(raw_outcome.get("measure_type", "")).lower().strip()
                # Use the same type_map logic
                _type_map = {
                    "smd": "SMD", "hedges_g": "SMD", "cohen_d": "SMD",
                    "standardized mean difference": "SMD",
                    "or": "OR", "odds_ratio": "OR", "odds ratio": "OR",
                    "rr": "RR", "risk_ratio": "RR", "risk ratio": "RR",
                    "relative_risk": "RR", "relative risk": "RR",
                    "hr": "HR", "hazard_ratio": "HR", "hazard ratio": "HR",
                    "md": "MD", "mean_diff": "MD", "mean_difference": "MD",
                    "mean difference": "MD",
                    "proportion": "PROP", "prevalence": "PROP",
                }
                m = _type_map.get(raw_type)
                if not m:
                    for k, v in _type_map.items():
                        if k in raw_type:
                            m = v
                            break
                if m:
                    mcounts[m] = mcounts.get(m, 0) + 1

            has_md = mcounts.get("MD", 0) > 0
            has_smd = mcounts.get("SMD", 0) > 0

            if has_md and has_smd:
                # Mixed MD + SMD within a clinical group → convert MD entries to SMD
                groups[gkey] = self._harmonise_md_to_smd_ingroup(group)
            elif has_md and not has_smd:
                # All MD — try converting all to SMD for cleaner pooling
                converted = self._try_convert_all_md(group)
                if converted is not None:
                    converted["_measure_type"] = "SMD"
                    groups[gkey] = converted
                    print(f"    [{gkey}] Converted all MD to SMD for pooling")

            # Similar logic for OR/RR within a group (convert RR→OR)
            has_or = mcounts.get("OR", 0) > 0
            has_rr = mcounts.get("RR", 0) > 0
            if has_rr and has_or:
                groups[gkey] = self._harmonise_rr_to_or_ingroup(groups[gkey])

        # ---- 2. Inter-group merges (backward-compat: measure-type keys) ---- #
        # ----- MD → SMD merge ----- #
        if "MD" in groups and "SMD" in groups:
            groups = self._merge_md_into_smd(groups)
        elif "MD" in groups and "SMD" not in groups:
            # Try converting all MD to SMD — only if every study can convert
            converted_group = self._try_convert_all_md(groups["MD"])
            if converted_group is not None:
                converted_group["_measure_type"] = "SMD"
                groups["SMD"] = converted_group
                del groups["MD"]
                print(f"    Converted all {len(converted_group['yi'])} MD studies to SMD")

        # ----- RR → OR merge ----- #
        if "RR" in groups and "OR" in groups:
            groups = self._merge_rr_into_or(groups)
        # (We don't convert a lone RR group to OR — only merge when both exist)

        return groups

    def _merge_md_into_smd(self, groups: Dict[str, Dict]) -> Dict[str, Dict]:
        """Convert MD studies to SMD and merge into the SMD group."""
        md_group = groups["MD"]
        smd_group = groups["SMD"]
        converted_count = 0
        unconverted_indices = []

        for i in range(len(md_group["yi"])):
            outcome = md_group["outcomes_raw"][i]
            result = convert_md_to_smd_study(
                md_group["yi"][i],
                md_group["vi"][i],
                outcome,
            )
            if result is not None and result["vi"] is not None and result["vi"] > 0:
                # Append converted study to SMD group
                self._append_study_to_group(
                    smd_group, md_group, i,
                    yi_new=result["yi"],
                    vi_new=result["vi"],
                    converted=True,
                    conversion_note=result["conversion_note"],
                    original_es=result["original_effect_size"],
                    original_ci_lower=result.get("original_ci_lower"),
                    original_ci_upper=result.get("original_ci_upper"),
                )
                converted_count += 1
            else:
                unconverted_indices.append(i)

        if converted_count > 0:
            print(f"    Converted {converted_count} MD studies to SMD for pooling")

        if unconverted_indices:
            # Keep unconverted MD studies in their own group
            groups["MD"] = self._subset_group(md_group, unconverted_indices)
            if not groups["MD"]["yi"]:
                del groups["MD"]
        else:
            # All MD converted — remove the MD group
            if "MD" in groups:
                del groups["MD"]

        groups["SMD"] = smd_group
        return groups

    def _merge_rr_into_or(self, groups: Dict[str, Dict]) -> Dict[str, Dict]:
        """Convert RR studies to OR and merge into the OR group."""
        rr_group = groups["RR"]
        or_group = groups["OR"]

        # Estimate baseline risk from OR group if possible
        baseline_risk = self._estimate_baseline_risk(or_group)
        converted_count = 0
        unconverted_indices = []

        for i in range(len(rr_group["yi"])):
            result = convert_rr_to_or_study(
                rr_group["yi"][i],
                rr_group["vi"][i],
                baseline_risk=baseline_risk,
            )
            if result is not None and result["vi"] is not None and result["vi"] > 0:
                self._append_study_to_group(
                    or_group, rr_group, i,
                    yi_new=result["yi"],
                    vi_new=result["vi"],
                    converted=True,
                    conversion_note=result["conversion_note"],
                    original_es=result["original_effect_size"],
                    original_ci_lower=result.get("original_ci_lower"),
                    original_ci_upper=result.get("original_ci_upper"),
                )
                converted_count += 1
            else:
                unconverted_indices.append(i)

        if converted_count > 0:
            print(f"    Converted {converted_count} RR studies to OR for pooling")

        if unconverted_indices:
            groups["RR"] = self._subset_group(rr_group, unconverted_indices)
            if not groups["RR"]["yi"]:
                del groups["RR"]
        else:
            if "RR" in groups:
                del groups["RR"]

        groups["OR"] = or_group
        return groups

    def _harmonise_md_to_smd_ingroup(self, group: Dict) -> Dict:
        """Within a single clinical-category group, convert MD entries to SMD
        so all entries share a common scale for pooling.

        Entries that cannot be converted keep their original yi/vi but are
        flagged.  The group's ``_measure_type`` is updated to "SMD".
        """
        _type_map = {
            "smd": "SMD", "hedges_g": "SMD", "cohen_d": "SMD",
            "standardized mean difference": "SMD",
            "md": "MD", "mean_diff": "MD", "mean_difference": "MD",
            "mean difference": "MD",
        }
        new_group = self._empty_group()
        converted_count = 0

        for i in range(len(group["yi"])):
            outcome = group["outcomes_raw"][i]
            raw_type = str(outcome.get("measure_type", "")).lower().strip()
            m = _type_map.get(raw_type)
            if not m:
                for k, v in _type_map.items():
                    if k in raw_type:
                        m = v
                        break

            if m == "MD":
                result = convert_md_to_smd_study(group["yi"][i], group["vi"][i], outcome)
                if result is not None and result["vi"] is not None and result["vi"] > 0:
                    self._append_study_to_group(
                        new_group, group, i,
                        yi_new=result["yi"],
                        vi_new=result["vi"],
                        converted=True,
                        conversion_note=result["conversion_note"],
                        original_es=result["original_effect_size"],
                        original_ci_lower=result.get("original_ci_lower"),
                        original_ci_upper=result.get("original_ci_upper"),
                    )
                    converted_count += 1
                    continue
                # Conversion failed — keep as-is
            # Keep entry unchanged (SMD or unconvertable MD)
            self._append_study_to_group(
                new_group, group, i,
                yi_new=group["yi"][i],
                vi_new=group["vi"][i],
                converted=group["converted_flags"][i],
                conversion_note=group["conversion_notes"][i],
                original_es=group["original_effect_sizes"][i],
                original_ci_lower=group["original_ci_lowers"][i],
                original_ci_upper=group["original_ci_uppers"][i],
            )

        if converted_count > 0:
            print(f"    Intra-group: converted {converted_count} MD entries to SMD")

        new_group["_measure_type"] = "SMD"
        return new_group

    def _harmonise_rr_to_or_ingroup(self, group: Dict) -> Dict:
        """Within a single clinical-category group, convert RR entries to OR
        so all entries share a common scale for pooling.

        The group's ``_measure_type`` is updated to "OR".
        """
        _type_map = {
            "or": "OR", "odds_ratio": "OR", "odds ratio": "OR",
            "rr": "RR", "risk_ratio": "RR", "risk ratio": "RR",
            "relative_risk": "RR", "relative risk": "RR",
        }
        baseline_risk = self._estimate_baseline_risk(group)
        new_group = self._empty_group()
        converted_count = 0

        for i in range(len(group["yi"])):
            outcome = group["outcomes_raw"][i]
            raw_type = str(outcome.get("measure_type", "")).lower().strip()
            m = _type_map.get(raw_type)
            if not m:
                for k, v in _type_map.items():
                    if k in raw_type:
                        m = v
                        break

            if m == "RR":
                result = convert_rr_to_or_study(
                    group["yi"][i], group["vi"][i], baseline_risk=baseline_risk
                )
                if result is not None and result["vi"] is not None and result["vi"] > 0:
                    self._append_study_to_group(
                        new_group, group, i,
                        yi_new=result["yi"],
                        vi_new=result["vi"],
                        converted=True,
                        conversion_note=result["conversion_note"],
                        original_es=result["original_effect_size"],
                        original_ci_lower=result.get("original_ci_lower"),
                        original_ci_upper=result.get("original_ci_upper"),
                    )
                    converted_count += 1
                    continue
            # Keep entry unchanged (OR or unconvertable RR)
            self._append_study_to_group(
                new_group, group, i,
                yi_new=group["yi"][i],
                vi_new=group["vi"][i],
                converted=group["converted_flags"][i],
                conversion_note=group["conversion_notes"][i],
                original_es=group["original_effect_sizes"][i],
                original_ci_lower=group["original_ci_lowers"][i],
                original_ci_upper=group["original_ci_uppers"][i],
            )

        if converted_count > 0:
            print(f"    Intra-group: converted {converted_count} RR entries to OR")

        new_group["_measure_type"] = "OR"
        return new_group

    def _try_convert_all_md(self, md_group: Dict) -> Optional[Dict]:
        """Try to convert every study in an MD-only group to SMD.

        Returns a new SMD group if all studies convert, else None.
        """
        new_group = self._empty_group()
        for i in range(len(md_group["yi"])):
            outcome = md_group["outcomes_raw"][i]
            result = convert_md_to_smd_study(
                md_group["yi"][i],
                md_group["vi"][i],
                outcome,
            )
            if result is None or result["vi"] is None or result["vi"] <= 0:
                return None  # Cannot convert all — abort
            self._append_study_to_group(
                new_group, md_group, i,
                yi_new=result["yi"],
                vi_new=result["vi"],
                converted=True,
                conversion_note=result["conversion_note"],
                original_es=result["original_effect_size"],
                original_ci_lower=result.get("original_ci_lower"),
                original_ci_upper=result.get("original_ci_upper"),
            )
        return new_group if new_group["yi"] else None

    def _estimate_baseline_risk(self, or_group: Dict) -> float:
        """Estimate a baseline risk from the OR group for RR↔OR conversion.

        Uses the median inverse-logit of log-OR as a rough proxy.
        Falls back to 0.1 (conservative low baseline) if estimation fails.
        """
        try:
            log_ors = or_group["yi"]
            if not log_ors:
                return 0.1
            # inverse logit of median log-OR
            median_lor = sorted(log_ors)[len(log_ors) // 2]
            p = 1.0 / (1.0 + math.exp(-median_lor))
            # Clamp to reasonable range
            return max(0.01, min(p, 0.99))
        except Exception:
            return 0.1

    @staticmethod
    def _empty_group() -> Dict:
        """Return an empty group dict with all required keys."""
        return {
            "yi": [], "vi": [], "labels": [],
            "quality_weights": [], "pmids": [],
            "abstract_only": [],
            "study_designs": [],
            "rob_levels": [],
            "sample_sizes": [],
            "years": [],
            "outcomes_raw": [],
            "converted_flags": [],
            "conversion_notes": [],
            "original_effect_sizes": [],
            "original_ci_lowers": [],
            "original_ci_uppers": [],
            "_measure_type": "",  # Set by callers after construction
        }

    @staticmethod
    def _append_study_to_group(
        target: Dict, source: Dict, idx: int,
        yi_new: float, vi_new: float,
        converted: bool, conversion_note: str,
        original_es: float,
        original_ci_lower=None, original_ci_upper=None,
    ):
        """Copy study at index *idx* from *source* into *target* with new yi/vi."""
        target["yi"].append(yi_new)
        target["vi"].append(vi_new)
        target["labels"].append(source["labels"][idx])
        target["quality_weights"].append(source["quality_weights"][idx])
        target["pmids"].append(source["pmids"][idx])
        target["abstract_only"].append(source["abstract_only"][idx])
        target["study_designs"].append(source["study_designs"][idx])
        target["rob_levels"].append(source["rob_levels"][idx])
        target["sample_sizes"].append(source["sample_sizes"][idx])
        target["years"].append(source["years"][idx])
        target["outcomes_raw"].append(source["outcomes_raw"][idx])
        target["converted_flags"].append(converted)
        target["conversion_notes"].append(conversion_note)
        target["original_effect_sizes"].append(original_es)
        target["original_ci_lowers"].append(original_ci_lower)
        target["original_ci_uppers"].append(original_ci_upper)

    @staticmethod
    def _subset_group(group: Dict, indices: List[int]) -> Dict:
        """Return a new group dict containing only the given indices."""
        new = {}
        for key, vals in group.items():
            if isinstance(vals, list):
                new[key] = [vals[i] for i in indices]
            else:
                new[key] = vals
        return new

    def _classify_rob_level(self, paper: Dict) -> str:
        """Classify a paper's risk-of-bias into 'low', 'some_concerns', or 'high'
        based on quality_assessment from A6."""
        qa = paper.get("quality_assessment", {})
        overall = (qa.get("overall", "") or "").lower().strip()
        # Map from various tools to a unified 3-level scale
        # RoB2: Low, Some concerns, High
        # ROBINS-I: Low, Moderate, Serious, Critical, No information
        # JBI: Include, Include with caution, Exclude
        low_terms = {"low", "include"}
        high_terms = {"high", "serious", "critical", "exclude", "no information"}
        if overall in low_terms:
            return "low"
        elif overall in high_terms:
            return "high"
        else:
            return "some_concerns"

    def _extract_sample_size(self, sc: Dict) -> Optional[int]:
        """Extract total sample size from study_characteristics dict."""
        for key in ("sample_size", "n_total", "total_sample", "n", "participants"):
            val = sc.get(key)
            if val is not None:
                try:
                    return int(float(str(val).replace(",", "")))
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_year(self, sc: Dict, paper: Dict) -> Optional[int]:
        """Extract publication year from study_characteristics or paper metadata."""
        for key in ("year", "publication_year", "pub_year"):
            val = sc.get(key) or paper.get(key)
            if val is not None:
                try:
                    return int(float(str(val)))
                except (ValueError, TypeError):
                    pass
        # Try extracting from title or other fields
        title = paper.get("title", "")
        m = re.search(r'\b(19|20)\d{2}\b', title)
        if m:
            return int(m.group())
        return None

    @staticmethod
    def _classify_outcome_category(outcome_name: str) -> str:
        """Map an outcome name string to a clinical category.

        Strategy:
        1. Exact keyword matching (fast path)
        2. Fuzzy token overlap (catches "visceral fat area" → weight, etc.)

        Returns one of the keys in OUTCOME_CATEGORIES, or 'other' if no match.
        """
        name_lower = outcome_name.lower()

        # Pass 1: exact substring matching (fast)
        for category, keywords in OUTCOME_CATEGORIES.items():
            for kw in keywords:
                if kw in name_lower:
                    return category

        # Pass 2: fuzzy token overlap — catches multi-word outcome names
        # where no single keyword matches but tokens overlap
        name_tokens = set(re.split(r'[\s\-_/,;()]+', name_lower))
        name_tokens.discard("")
        best_category = "other"
        best_score = 0.0

        for category, keywords in OUTCOME_CATEGORIES.items():
            # Build token set from all keywords in this category
            cat_tokens = set()
            for kw in keywords:
                for token in re.split(r'[\s\-_/,;()]+', kw):
                    if len(token) >= 3:  # skip short tokens like "kg"
                        cat_tokens.add(token)

            if not cat_tokens or not name_tokens:
                continue

            # Jaccard-like overlap score
            overlap = name_tokens & cat_tokens
            if overlap:
                score = len(overlap) / len(name_tokens)
                if score > best_score and score >= 0.3:  # at least 30% token overlap
                    best_score = score
                    best_category = category

        return best_category

    def _get_variance(self, outcome: Dict, yi: float) -> Optional[float]:
        """Get variance from CI, SE, or estimate from p-value."""
        # Try CI
        ci_lower = outcome.get("ci_lower")
        ci_upper = outcome.get("ci_upper")
        if ci_lower is not None and ci_upper is not None:
            try:
                ci_l = float(ci_lower)
                ci_u = float(ci_upper)
                vi = self._variance_from_ci(yi, ci_l, ci_u)
                if vi and vi > 0:
                    return vi
            except (ValueError, TypeError):
                pass

        # Try SE directly
        se = outcome.get("se") or outcome.get("standard_error")
        if se is not None:
            try:
                se = float(se)
                return se ** 2
            except (ValueError, TypeError):
                pass

        # Rough estimate from p-value and effect size (less reliable)
        p_val = outcome.get("p_value")
        if p_val is not None and yi != 0:
            try:
                p = float(p_val)
                if 0 < p < 1:
                    from scipy.stats import norm
                    z = abs(norm.ppf(p / 2))
                    if z > 0:
                        se_est = abs(yi) / z
                        return se_est ** 2
            except Exception:
                pass

        return None

    def _variance_from_ci(self, yi: float, ci_lower: float, ci_upper: float) -> Optional[float]:
        """Back-calculate variance from 95% CI: vi = ((ci_upper - ci_lower) / 3.92)^2"""
        width = ci_upper - ci_lower
        if width <= 0:
            return None
        se = width / 3.92  # 2 * 1.96
        return se ** 2

    # ================================================================== #
    #  TOTAL PARTICIPANTS                                                  #
    # ================================================================== #

    def _compute_total_participants(self, data: Dict) -> Dict:
        """Compute total N across all studies in this analysis group."""
        sizes = [s for s in data.get("sample_sizes", []) if s is not None]
        total = sum(sizes) if sizes else None
        return {
            "total_n": total,
            "studies_with_n": len(sizes),
            "studies_without_n": len(data["yi"]) - len(sizes),
        }

    # ================================================================== #
    #  RANDOM-EFFECTS MODEL                                                #
    # ================================================================== #

    def _fit_random_effects(self, data: Dict, measure: str) -> Optional[Dict]:
        """Fit random-effects model via metafor rma(), including prediction interval."""
        try:
            yi_r = self.FloatVector(data["yi"])
            vi_r = self.FloatVector(data["vi"])

            fit = self.metafor.rma(yi=yi_r, vi=vi_r, method="REML")

            # Extract results
            b = float(fit.rx2("beta")[0])
            se = float(fit.rx2("se")[0])
            ci_lb = float(fit.rx2("ci.lb")[0])
            ci_ub = float(fit.rx2("ci.ub")[0])
            zval = float(fit.rx2("zval")[0])
            pval = float(fit.rx2("pval")[0])
            k = int(fit.rx2("k")[0])

            # Heterogeneity stats
            tau2 = float(fit.rx2("tau2")[0])
            I2 = float(fit.rx2("I2")[0])
            QE = float(fit.rx2("QE")[0])
            QEp = float(fit.rx2("QEp")[0])

            # Prediction interval
            pi_lb = None
            pi_ub = None
            try:
                pred = self.r.r['predict'](fit)
                pi_lb = round(float(pred.rx2("pi.lb")[0]), 4)
                pi_ub = round(float(pred.rx2("pi.ub")[0]), 4)
            except Exception as e:
                print(f"    Prediction interval warning: {e}")

            return {
                "pooled_estimate": round(b, 4),
                "se": round(se, 4),
                "ci_lower": round(ci_lb, 4),
                "ci_upper": round(ci_ub, 4),
                "z_value": round(zval, 4),
                "p_value": round(pval, 6),
                "k_studies": k,
                "tau2": round(tau2, 4),
                "I2": round(I2, 2),
                "QE": round(QE, 4),
                "QEp": round(QEp, 6),
                "prediction_interval_lower": pi_lb,
                "prediction_interval_upper": pi_ub,
                "_fit": fit,  # Keep for downstream use
            }

        except Exception as e:
            print(f"    rma() error: {e}")
            return None

    # ================================================================== #
    #  HETEROGENEITY                                                       #
    # ================================================================== #

    def _heterogeneity_analysis(self, rma_result: Dict) -> Dict:
        """Extract and interpret heterogeneity statistics."""
        I2 = rma_result.get("I2", 0)
        tau2 = rma_result.get("tau2", 0)
        Q = rma_result.get("QE", 0)
        Q_p = rma_result.get("QEp", 1)

        if I2 < 25:
            interpretation = "Low heterogeneity"
        elif I2 < 50:
            interpretation = "Moderate heterogeneity"
        elif I2 < 75:
            interpretation = "Substantial heterogeneity"
        else:
            interpretation = "Considerable heterogeneity"

        return {
            "I2": round(I2, 2),
            "tau2": round(tau2, 4),
            "Q": round(Q, 4),
            "Q_p": round(Q_p, 6),
            "interpretation": interpretation,
        }

    # ================================================================== #
    #  PUBLICATION BIAS                                                    #
    # ================================================================== #

    def _publication_bias(self, data: Dict, measure: str) -> Dict:
        """Run Egger's test (if k>=10) and trim-and-fill analysis."""
        result = {}
        k = len(data["yi"])
        min_egger = self._config.get("min_studies_for_egger", 10)

        try:
            yi_r = self.FloatVector(data["yi"])
            vi_r = self.FloatVector(data["vi"])
            fit = self.metafor.rma(yi=yi_r, vi=vi_r, method="REML")

            # Egger's regression test — only if k >= threshold
            if k >= min_egger:
                try:
                    reg = self.metafor.regtest(fit)
                    result["egger_test"] = {
                        "z_value": round(float(reg.rx2("zval")[0]), 4),
                        "p_value": round(float(reg.rx2("pval")[0]), 6),
                        "significant": float(reg.rx2("pval")[0]) < 0.10,
                    }
                except Exception as e:
                    result["egger_test"] = {"error": str(e)}
            else:
                result["egger_test"] = {
                    "skipped": True,
                    "reason": f"Fewer than {min_egger} studies (k={k}); Egger's test unreliable",
                }

            # Trim-and-fill
            try:
                tf = self.metafor.trimfill(fit)
                tf_b = float(tf.rx2("beta")[0])
                tf_ci_lb = float(tf.rx2("ci.lb")[0])
                tf_ci_ub = float(tf.rx2("ci.ub")[0])
                tf_k0 = int(tf.rx2("k0")[0])
                result["trim_and_fill"] = {
                    "estimated_missing": tf_k0,
                    "adjusted_estimate": round(tf_b, 4),
                    "adjusted_ci_lower": round(tf_ci_lb, 4),
                    "adjusted_ci_upper": round(tf_ci_ub, 4),
                }
            except Exception as e:
                result["trim_and_fill"] = {"error": str(e)}

        except Exception as e:
            result["error"] = str(e)

        return result

    # ================================================================== #
    #  SENSITIVITY ANALYSIS                                                #
    # ================================================================== #

    def _sensitivity_leave_one_out(self, data: Dict, measure: str) -> List[Dict]:
        """Leave-one-out analysis: refit removing each study."""
        results = []
        try:
            yi_r = self.FloatVector(data["yi"])
            vi_r = self.FloatVector(data["vi"])
            fit = self.metafor.rma(yi=yi_r, vi=vi_r, method="REML")

            l1o = self.metafor.leave1out(fit)

            estimates = list(l1o.rx2("estimate"))
            ci_lbs = list(l1o.rx2("ci.lb"))
            ci_ubs = list(l1o.rx2("ci.ub"))

            for i in range(len(data["yi"])):
                results.append({
                    "study_removed": data["labels"][i] if i < len(data["labels"]) else f"Study {i+1}",
                    "pmid_removed": data["pmids"][i] if i < len(data["pmids"]) else "",
                    "estimate": round(float(estimates[i]), 4),
                    "ci_lower": round(float(ci_lbs[i]), 4),
                    "ci_upper": round(float(ci_ubs[i]), 4),
                })

        except Exception as e:
            print(f"    leave1out error: {e}")

        return results

    def _sensitivity_quality_weighted(self, data: Dict, measure: str) -> Dict:
        """Quality-weighted sensitivity analysis.

        Adjusts study variances by quality weight: vi_adj = vi / weight.
        Studies with lower quality get inflated variance (less influence).
        This is a common approach when quality scores are available.
        """
        try:
            weights = data.get("quality_weights", [])
            if not weights or len(weights) != len(data["yi"]):
                return {"note": "No quality weights available"}

            # Check if all weights are equal (no variation → no point)
            if len(set(weights)) <= 1:
                return {"note": "All quality weights identical — no adjustment needed"}

            # Adjust variance: vi_adj = vi / quality_weight
            # Lower quality → higher variance → less influence
            yi_adj = []
            vi_adj = []
            for i in range(len(data["yi"])):
                w = max(weights[i], 0.1)  # floor to avoid division by near-zero
                yi_adj.append(data["yi"][i])
                vi_adj.append(data["vi"][i] / w)

            yi_r = self.FloatVector(yi_adj)
            vi_r = self.FloatVector(vi_adj)
            fit = self.metafor.rma(yi=yi_r, vi=vi_r, method="REML")

            pooled_adj = round(float(fit.rx2("beta")[0]), 4)
            ci_lb = round(float(fit.rx2("ci.lb")[0]), 4)
            ci_ub = round(float(fit.rx2("ci.ub")[0]), 4)
            pval = round(float(fit.rx2("pval")[0]), 6)
            i2 = round(float(fit.rx2("I2")[0]), 2)

            # Compare with unweighted
            orig_fit = self.metafor.rma(
                yi=self.FloatVector(data["yi"]),
                vi=self.FloatVector(data["vi"]),
                method="REML",
            )
            pooled_orig = round(float(orig_fit.rx2("beta")[0]), 4)
            change = round(pooled_adj - pooled_orig, 4)
            direction_changed = (pooled_adj > 0) != (pooled_orig > 0)

            return {
                "pooled_estimate_adjusted": pooled_adj,
                "ci_lower": ci_lb,
                "ci_upper": ci_ub,
                "p_value": pval,
                "I2": i2,
                "pooled_estimate_original": pooled_orig,
                "change_from_original": change,
                "direction_changed": direction_changed,
                "method": "variance inflation by 1/quality_weight",
            }
        except Exception as e:
            return {"error": str(e)}

    def _sensitivity_exclude_high_risk(self, data: Dict, measure: str) -> Dict:
        """Refit model excluding studies with quality_weight <= 0.3 (high risk)."""
        try:
            indices = [i for i, w in enumerate(data["quality_weights"]) if w > 0.3]

            if len(indices) < 2:
                return {"note": "Not enough low/medium-risk studies to refit", "k_remaining": len(indices)}

            if len(indices) == len(data["yi"]):
                return {"note": "No high-risk studies to exclude", "k_remaining": len(indices)}

            yi_filtered = [data["yi"][i] for i in indices]
            vi_filtered = [data["vi"][i] for i in indices]

            yi_r = self.FloatVector(yi_filtered)
            vi_r = self.FloatVector(vi_filtered)
            fit = self.metafor.rma(yi=yi_r, vi=vi_r, method="REML")

            return {
                "k_remaining": len(indices),
                "k_excluded": len(data["yi"]) - len(indices),
                "pooled_estimate": round(float(fit.rx2("beta")[0]), 4),
                "ci_lower": round(float(fit.rx2("ci.lb")[0]), 4),
                "ci_upper": round(float(fit.rx2("ci.ub")[0]), 4),
                "p_value": round(float(fit.rx2("pval")[0]), 6),
                "I2": round(float(fit.rx2("I2")[0]), 2),
            }

        except Exception as e:
            return {"error": str(e)}

    # ================================================================== #
    #  SUBGROUP ANALYSIS                                                   #
    # ================================================================== #

    def _run_subgroup_analyses(self, data: Dict, measure: str) -> Dict:
        """Run subgroup analyses by study design, risk of bias, sample size, and text source."""
        result = {}
        min_per = self._config.get("min_studies_per_subgroup", 2)
        sg_cfg = self._config.get("subgroup_analyses", {})

        # By study design (RCT vs observational)
        if sg_cfg.get("by_study_design", True):
            designs = data.get("study_designs", [])
            if designs:
                rct_idx = [i for i, d in enumerate(designs) if d == "rct"]
                obs_idx = [i for i, d in enumerate(designs) if d != "rct"]
                if len(rct_idx) >= min_per and len(obs_idx) >= min_per:
                    result["by_study_design"] = self._subgroup_two_groups(
                        data, measure, rct_idx, obs_idx, "rct", "observational",
                        "study_designs"
                    )
                else:
                    result["by_study_design"] = {
                        "skipped": True,
                        "reason": f"Not enough studies per subgroup (RCT={len(rct_idx)}, Obs={len(obs_idx)}, min={min_per})",
                    }

        # By risk of bias level
        if sg_cfg.get("by_risk_of_bias", True):
            robs = data.get("rob_levels", [])
            if robs:
                low_idx = [i for i, r in enumerate(robs) if r == "low"]
                concerns_idx = [i for i, r in enumerate(robs) if r == "some_concerns"]
                high_idx = [i for i, r in enumerate(robs) if r == "high"]

                rob_result = {}
                groups_available = []
                for name, idx_list in [("low_risk", low_idx), ("some_concerns", concerns_idx), ("high_risk", high_idx)]:
                    if len(idx_list) >= min_per:
                        sub = self._fit_subgroup(data, measure, idx_list)
                        if sub:
                            rob_result[name] = sub
                            groups_available.append((name, idx_list))

                if len(groups_available) >= 2:
                    # Test for difference using moderator
                    mod_test = self._test_moderator_categorical(data, measure, robs, "rob_levels")
                    if mod_test:
                        rob_result["test_for_difference"] = mod_test
                    result["by_risk_of_bias"] = rob_result
                else:
                    result["by_risk_of_bias"] = {
                        "skipped": True,
                        "reason": f"Fewer than 2 subgroups with >= {min_per} studies",
                    }

        # By sample size (small <100 vs large >=100)
        if sg_cfg.get("by_sample_size", True):
            sizes = data.get("sample_sizes", [])
            if sizes and any(s is not None for s in sizes):
                small_idx = [i for i, s in enumerate(sizes) if s is not None and s < 100]
                large_idx = [i for i, s in enumerate(sizes) if s is not None and s >= 100]
                if len(small_idx) >= min_per and len(large_idx) >= min_per:
                    result["by_sample_size"] = self._subgroup_two_groups(
                        data, measure, small_idx, large_idx, "small_lt100", "large_gte100",
                        "sample_sizes"
                    )
                else:
                    result["by_sample_size"] = {
                        "skipped": True,
                        "reason": f"Not enough studies per subgroup (small={len(small_idx)}, large={len(large_idx)}, min={min_per})",
                    }
            else:
                result["by_sample_size"] = {
                    "skipped": True,
                    "reason": "Sample size data not available",
                }

        # By text source (full-text vs abstract-only)
        abstract_flags = data.get("abstract_only", [])
        if abstract_flags:
            ft_idx = [i for i, a in enumerate(abstract_flags) if not a]
            ab_idx = [i for i, a in enumerate(abstract_flags) if a]
            if len(ft_idx) >= min_per and len(ab_idx) >= min_per:
                result["by_text_source"] = self._subgroup_two_groups(
                    data, measure, ft_idx, ab_idx, "full_text", "abstract_only",
                    "abstract_only"
                )
            else:
                result["by_text_source"] = {
                    "skipped": True,
                    "reason": f"Not enough studies per subgroup (full_text={len(ft_idx)}, abstract={len(ab_idx)}, min={min_per})",
                }

        return result

    def _subgroup_two_groups(self, data: Dict, measure: str,
                             idx_a: List[int], idx_b: List[int],
                             name_a: str, name_b: str,
                             moderator_key: str) -> Dict:
        """Fit two subgroup models and run a moderator test."""
        result = {}
        sub_a = self._fit_subgroup(data, measure, idx_a)
        sub_b = self._fit_subgroup(data, measure, idx_b)
        if sub_a:
            result[name_a] = sub_a
        if sub_b:
            result[name_b] = sub_b

        # Test for subgroup difference using moderator variable
        if sub_a and sub_b:
            # Build a binary moderator for the moderator test
            mod_values = []
            for i in range(len(data["yi"])):
                if i in idx_a:
                    mod_values.append(name_a)
                elif i in idx_b:
                    mod_values.append(name_b)
                else:
                    mod_values.append("other")
            # Only include studies in either group
            combined_idx = sorted(set(idx_a) | set(idx_b))
            if len(combined_idx) >= 3:
                try:
                    yi_sub = [data["yi"][i] for i in combined_idx]
                    vi_sub = [data["vi"][i] for i in combined_idx]
                    mod_sub = [1 if i in idx_a else 0 for i in combined_idx]

                    yi_r = self.FloatVector(yi_sub)
                    vi_r = self.FloatVector(vi_sub)
                    mod_r = self.FloatVector(mod_sub)

                    fit_mod = self.metafor.rma(yi=yi_r, vi=vi_r, mods=mod_r, method="REML")
                    QM = float(fit_mod.rx2("QM")[0])
                    QMp = float(fit_mod.rx2("QMp")[0])
                    result["test_for_difference"] = {
                        "Q_between": round(QM, 4),
                        "p_value": round(QMp, 6),
                    }
                except Exception as e:
                    result["test_for_difference"] = {"error": str(e)}

        return result

    def _fit_subgroup(self, data: Dict, measure: str, indices: List[int]) -> Optional[Dict]:
        """Fit a random-effects model on a subset of studies."""
        try:
            yi_sub = [data["yi"][i] for i in indices]
            vi_sub = [data["vi"][i] for i in indices]

            if len(yi_sub) < 2:
                return None

            yi_r = self.FloatVector(yi_sub)
            vi_r = self.FloatVector(vi_sub)
            fit = self.metafor.rma(yi=yi_r, vi=vi_r, method="REML")

            return {
                "k": len(yi_sub),
                "pooled": round(float(fit.rx2("beta")[0]), 4),
                "ci": [round(float(fit.rx2("ci.lb")[0]), 4), round(float(fit.rx2("ci.ub")[0]), 4)],
                "p_value": round(float(fit.rx2("pval")[0]), 6),
                "I2": round(float(fit.rx2("I2")[0]), 2),
            }
        except Exception as e:
            print(f"    Subgroup fit error: {e}")
            return None

    def _test_moderator_categorical(self, data: Dict, measure: str,
                                     categories: List[str], label: str) -> Optional[Dict]:
        """Test for subgroup differences with a categorical moderator using
        a moderator model (rma with mods). Uses dummy coding."""
        try:
            unique_cats = sorted(set(categories))
            if len(unique_cats) < 2:
                return None

            # Create dummy variables (drop first level as reference)
            k = len(data["yi"])
            cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
            # Build R formula via numeric coding
            mod_vals = [cat_to_idx[categories[i]] for i in range(k)]

            yi_r = self.FloatVector(data["yi"])
            vi_r = self.FloatVector(data["vi"])

            # Use factor in R
            self.r.globalenv["yi_tmp"] = yi_r
            self.r.globalenv["vi_tmp"] = vi_r
            self.r.globalenv["mod_tmp"] = self.r.StrVector(categories)

            result_r = self.r.r("""
                mod_factor <- as.factor(mod_tmp)
                fit <- metafor::rma(yi=yi_tmp, vi=vi_tmp, mods=~mod_factor, method="REML")
                list(QM=fit$QM, QMp=fit$QMp)
            """)

            QM = float(result_r.rx2("QM")[0])
            QMp = float(result_r.rx2("QMp")[0])
            return {
                "Q_between": round(QM, 4),
                "p_value": round(QMp, 6),
            }
        except Exception as e:
            print(f"    Moderator test ({label}) error: {e}")
            return None

    # ================================================================== #
    #  META-REGRESSION                                                     #
    # ================================================================== #

    def _run_meta_regression(self, data: Dict, measure: str) -> Dict:
        """Run meta-regression with year and sample size if k >= threshold."""
        result = {}
        k = len(data["yi"])
        min_k = self._config.get("min_studies_for_meta_regression", 10)

        if k < min_k:
            return {
                "skipped": True,
                "reason": f"Fewer than {min_k} studies (k={k}); meta-regression not reliable",
            }

        # By publication year
        years = data.get("years", [])
        valid_year_idx = [i for i, y in enumerate(years) if y is not None]
        if len(valid_year_idx) >= min_k:
            result["by_year"] = self._meta_regression_continuous(
                data, measure, valid_year_idx, [years[i] for i in valid_year_idx], "publication year"
            )
        else:
            result["by_year"] = {
                "skipped": True,
                "reason": f"Only {len(valid_year_idx)} studies with year data (need {min_k})",
            }

        # By sample size
        sizes = data.get("sample_sizes", [])
        valid_size_idx = [i for i, s in enumerate(sizes) if s is not None]
        if len(valid_size_idx) >= min_k:
            result["by_sample_size"] = self._meta_regression_continuous(
                data, measure, valid_size_idx, [sizes[i] for i in valid_size_idx], "sample size"
            )
        else:
            result["by_sample_size"] = {
                "skipped": True,
                "reason": f"Only {len(valid_size_idx)} studies with sample size data (need {min_k})",
            }

        return result

    def _meta_regression_continuous(self, data: Dict, measure: str,
                                     indices: List[int], mod_values: List,
                                     mod_name: str) -> Dict:
        """Run meta-regression with a continuous moderator."""
        try:
            yi_sub = [data["yi"][i] for i in indices]
            vi_sub = [data["vi"][i] for i in indices]
            mod_float = [float(v) for v in mod_values]

            yi_r = self.FloatVector(yi_sub)
            vi_r = self.FloatVector(vi_sub)
            mod_r = self.FloatVector(mod_float)

            fit = self.metafor.rma(yi=yi_r, vi=vi_r, mods=mod_r, method="REML")

            # Coefficient for the moderator (index 1, index 0 is intercept)
            beta = list(fit.rx2("beta"))
            se_vals = list(fit.rx2("se"))
            pvals = list(fit.rx2("pval"))

            # Index 1 = moderator coefficient (index 0 = intercept)
            coef = float(beta[1]) if len(beta) > 1 else float(beta[0])
            coef_se = float(se_vals[1]) if len(se_vals) > 1 else float(se_vals[0])
            coef_p = float(pvals[1]) if len(pvals) > 1 else float(pvals[0])
            QM = float(fit.rx2("QM")[0])

            significant = coef_p < 0.05

            # Generate interpretation
            if significant:
                direction = "positive" if coef > 0 else "negative"
                interpretation = (f"Significant {direction} association between {mod_name} "
                                  f"and effect size (p={coef_p:.4f})")
            else:
                interpretation = f"No significant association between {mod_name} and effect size (p={coef_p:.4f})"

            return {
                "coefficient": round(coef, 4),
                "se": round(coef_se, 4),
                "p_value": round(coef_p, 6),
                "QM": round(QM, 4),
                "significant": significant,
                "interpretation": interpretation,
            }

        except Exception as e:
            return {"error": str(e)}

    # ================================================================== #
    #  PLOTS                                                               #
    # ================================================================== #

    def _generate_forest_plot(self, data: Dict, measure: str, path: str) -> Optional[str]:
        """Generate forest plot via R's metafor, with matplotlib fallback."""
        # Try R first
        try:
            yi_r = self.FloatVector(data["yi"])
            vi_r = self.FloatVector(data["vi"])
            fit = self.metafor.rma(yi=yi_r, vi=vi_r, method="REML")

            labels = self.r.StrVector(data["labels"])

            self.grdevices.png(filename=path, width=900, height=max(300, 80 * len(data["yi"])), res=100)
            self.metafor.forest(fit, slab=labels, xlab=measure)
            self.grdevices.dev_off()

            if os.path.exists(path):
                print(f"    Forest plot: {path}")
                return path

        except Exception as e:
            print(f"    R forest plot failed: {e}")

        # Matplotlib fallback
        return self._matplotlib_forest(data, measure, path)

    def _generate_funnel_plot(self, data: Dict, measure: str, path: str) -> Optional[str]:
        """Generate funnel plot via R's metafor, with matplotlib fallback."""
        try:
            yi_r = self.FloatVector(data["yi"])
            vi_r = self.FloatVector(data["vi"])
            fit = self.metafor.rma(yi=yi_r, vi=vi_r, method="REML")

            self.grdevices.png(filename=path, width=700, height=600, res=100)
            self.metafor.funnel(fit, xlab=measure)
            self.grdevices.dev_off()

            if os.path.exists(path):
                print(f"    Funnel plot: {path}")
                return path

        except Exception as e:
            print(f"    R funnel plot failed: {e}")

        return self._matplotlib_funnel(data, measure, path)

    def _matplotlib_forest(self, data: Dict, measure: str, path: str) -> Optional[str]:
        """Matplotlib fallback forest plot."""
        if not HAS_MATPLOTLIB:
            print("    matplotlib not available for fallback")
            return None

        try:
            n = len(data["yi"])
            fig, ax = plt.subplots(figsize=(8, max(3, n * 0.6)))

            yi = data["yi"]
            se = [math.sqrt(v) for v in data["vi"]]
            labels = data["labels"]

            positions = list(range(n, 0, -1))

            for i in range(n):
                ci_l = yi[i] - 1.96 * se[i]
                ci_u = yi[i] + 1.96 * se[i]
                ax.plot([ci_l, ci_u], [positions[i], positions[i]], "b-", linewidth=1)
                ax.plot(yi[i], positions[i], "bs", markersize=6)

            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_yticks(positions)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel(measure)
            ax.set_title(f"Forest Plot ({measure})")

            plt.tight_layout()
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            if os.path.exists(path):
                print(f"    Forest plot (matplotlib): {path}")
                return path

        except Exception as e:
            print(f"    matplotlib forest failed: {e}")
            return None

    def _matplotlib_funnel(self, data: Dict, measure: str, path: str) -> Optional[str]:
        """Matplotlib fallback funnel plot."""
        if not HAS_MATPLOTLIB:
            return None

        try:
            yi = data["yi"]
            se = [math.sqrt(v) for v in data["vi"]]

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(yi, se, c="navy", s=40, zorder=3)
            ax.set_xlabel(f"Effect Size ({measure})")
            ax.set_ylabel("Standard Error")
            ax.set_title(f"Funnel Plot ({measure})")
            ax.invert_yaxis()

            # Reference line at mean
            mean_yi = sum(yi) / len(yi)
            ax.axvline(x=mean_yi, color="gray", linestyle="--", linewidth=0.8)

            plt.tight_layout()
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            if os.path.exists(path):
                print(f"    Funnel plot (matplotlib): {path}")
                return path

        except Exception as e:
            print(f"    matplotlib funnel failed: {e}")
            return None

    # ================================================================== #
    #  GRADE ASSESSMENT (Improved - 5 domains)                             #
    # ================================================================== #

    def _grade_assessment(self, rma_result: Dict, heterogeneity: Dict,
                          pub_bias: Dict, data: Dict) -> Dict:
        """Improved GRADE certainty of evidence assessment.
        Start High for RCTs, Low for observational. Downgrade per 5 domains:
        risk_of_bias, inconsistency, indirectness, imprecision, publication_bias.
        """
        # Determine starting level based on predominant study design
        designs = data.get("study_designs", [])
        rct_pct = sum(1 for d in designs if d == "rct") / max(len(designs), 1) if designs else 0
        weights = data.get("quality_weights", [])
        avg_weight = sum(weights) / len(weights) if weights else 0.5

        # If majority RCTs, start High; otherwise Low
        if rct_pct > 0.5:
            starting = "High"
        else:
            starting = "Low"

        domains = {}
        downgrades = 0

        # ---- 1. Risk of bias ----
        # Use per-study quality weights
        some_concerns_or_worse = sum(1 for w in weights if w < 1.0) / max(len(weights), 1)
        high_risk_pct = sum(1 for w in weights if w <= 0.3) / max(len(weights), 1)

        if high_risk_pct > 0.5:
            domains["risk_of_bias"] = {
                "level": "very serious", "downgrades": 2,
                "reason": f"{high_risk_pct:.0%} of studies at high risk of bias",
            }
            downgrades += 2
        elif some_concerns_or_worse > 0.5:
            domains["risk_of_bias"] = {
                "level": "serious", "downgrades": 1,
                "reason": f"{some_concerns_or_worse:.0%} of studies have some concerns or worse",
            }
            downgrades += 1
        else:
            domains["risk_of_bias"] = {
                "level": "not serious", "downgrades": 0,
                "reason": "Majority of studies at low risk of bias",
            }

        # ---- 2. Inconsistency ----
        I2 = heterogeneity.get("I2", 0)
        pi_lb = rma_result.get("prediction_interval_lower")
        pi_ub = rma_result.get("prediction_interval_upper")
        pi_crosses_null = (pi_lb is not None and pi_ub is not None and pi_lb <= 0 <= pi_ub)

        if I2 > 75:
            domains["inconsistency"] = {
                "level": "very serious", "downgrades": 2,
                "reason": f"I^2 = {I2:.1f}% (considerable heterogeneity)",
            }
            downgrades += 2
        elif I2 > 50 and pi_crosses_null:
            domains["inconsistency"] = {
                "level": "serious", "downgrades": 1,
                "reason": f"I^2 = {I2:.1f}% and prediction interval crosses null",
            }
            downgrades += 1
        elif I2 > 50:
            domains["inconsistency"] = {
                "level": "serious", "downgrades": 1,
                "reason": f"I^2 = {I2:.1f}% (substantial heterogeneity)",
            }
            downgrades += 1
        else:
            domains["inconsistency"] = {
                "level": "not serious", "downgrades": 0,
                "reason": f"I^2 = {I2:.1f}% (acceptable)",
            }

        # ---- 3. Indirectness ----
        # Check: (a) abstract-only %, (b) study design mismatch with protocol
        abstract_pct = sum(data.get("abstract_only", [])) / max(len(data.get("abstract_only", [1])), 1)
        # Check if protocol expected RCTs but observational studies included
        protocol_type = ""
        if self._protocol:
            protocol_type = str(self._protocol.get("framework", "")).lower()
        obs_in_rct_review = (protocol_type in ("pico", "") and rct_pct < 0.7 and rct_pct > 0)
        indirect_pct = abstract_pct  # abstract-only as proxy
        if obs_in_rct_review:
            indirect_pct = max(indirect_pct, 1.0 - rct_pct)

        if indirect_pct > 0.5:
            reasons = []
            if abstract_pct > 0.3:
                reasons.append(f"{abstract_pct:.0%} abstract-only studies")
            if obs_in_rct_review:
                reasons.append(f"observational studies included ({1.0 - rct_pct:.0%}) in review expecting RCTs")
            domains["indirectness"] = {
                "level": "serious", "downgrades": 1,
                "reason": "; ".join(reasons) if reasons else f"{indirect_pct:.0%} indirect evidence",
            }
            downgrades += 1
        elif indirect_pct > 0.3:
            reasons = []
            if abstract_pct > 0.2:
                reasons.append(f"{abstract_pct:.0%} abstract-only studies")
            if obs_in_rct_review:
                reasons.append(f"some observational studies included ({1.0 - rct_pct:.0%})")
            domains["indirectness"] = {
                "level": "serious", "downgrades": 1,
                "reason": "; ".join(reasons) if reasons else f"{indirect_pct:.0%} potentially indirect",
            }
            downgrades += 1
        else:
            domains["indirectness"] = {
                "level": "not serious", "downgrades": 0,
                "reason": "Adequate directness of evidence",
            }

        # ---- 4. Imprecision ----
        ci_lower = rma_result.get("ci_lower", 0)
        ci_upper = rma_result.get("ci_upper", 0)
        ci_width = ci_upper - ci_lower
        crosses_null = ci_lower <= 0 <= ci_upper

        # Optimal Information Size (OIS) check
        total_participants = self._compute_total_participants(data)
        total_n = total_participants.get("total_n")
        # OIS: continuous=800 total (400/group), binary varies
        measure_type = rma_result.get("_measure_type", "")
        ois = 800  # default for continuous outcomes
        if any(m in str(measure_type).upper() for m in ["OR", "RR", "HR"]):
            ois = 300  # lower OIS for binary outcomes (rough estimate)

        below_ois = (total_n is not None and total_n < ois)

        # CI includes both clinically meaningful benefit and harm?
        # For SMD: benefit > 0.2, harm < -0.2 (or vice versa depending on direction)
        includes_benefit_and_harm = (ci_lower < -0.2 and ci_upper > 0.2)

        if (crosses_null and below_ois) or includes_benefit_and_harm:
            ois_note = f", total N={total_n} below OIS={ois}" if below_ois else ""
            domains["imprecision"] = {
                "level": "serious", "downgrades": 1,
                "reason": f"CI [{ci_lower:.2f}, {ci_upper:.2f}] crosses null{ois_note}",
            }
            downgrades += 1
        elif crosses_null and ci_width > 1.0:
            domains["imprecision"] = {
                "level": "very serious", "downgrades": 2,
                "reason": f"Wide CI [{ci_lower:.2f}, {ci_upper:.2f}] crossing null with width {ci_width:.2f}",
            }
            downgrades += 2
        elif crosses_null:
            domains["imprecision"] = {
                "level": "serious", "downgrades": 1,
                "reason": f"CI [{ci_lower:.2f}, {ci_upper:.2f}] crosses null",
            }
            downgrades += 1
        else:
            domains["imprecision"] = {
                "level": "not serious", "downgrades": 0,
                "reason": f"Precise: CI [{ci_lower:.2f}, {ci_upper:.2f}]",
            }

        # ---- 5. Publication bias ----
        egger = pub_bias.get("egger_test", {})
        tf = pub_bias.get("trim_and_fill", {})
        egger_skipped = egger.get("skipped", False)
        egger_p = egger.get("p_value", 1.0) if not egger_skipped else None

        # Trim-and-fill: significant if missing > 0 and adjusted estimate differs meaningfully
        tf_missing = tf.get("estimated_missing", 0) if isinstance(tf, dict) else 0
        tf_concern = tf_missing >= 2

        if egger_skipped:
            # Cannot assess properly — note in reason
            if tf_concern:
                domains["publication_bias"] = {
                    "level": "serious", "downgrades": 1,
                    "reason": (f"Egger's test skipped (k<{self._config.get('min_studies_for_egger', 10)}), "
                               f"but trim-and-fill suggests {tf_missing} missing studies"),
                }
                downgrades += 1
            else:
                domains["publication_bias"] = {
                    "level": "not serious", "downgrades": 0,
                    "reason": (f"Cannot fully assess: Egger's test skipped "
                               f"(k<{self._config.get('min_studies_for_egger', 10)}); "
                               f"trim-and-fill suggests {tf_missing} missing studies"),
                }
        elif isinstance(egger_p, (int, float)) and egger_p < 0.05:
            domains["publication_bias"] = {
                "level": "serious", "downgrades": 1,
                "reason": f"Egger p = {egger_p:.4f} (significant asymmetry)",
            }
            downgrades += 1
        elif isinstance(egger_p, (int, float)) and egger_p < 0.10:
            domains["publication_bias"] = {
                "level": "serious", "downgrades": 1,
                "reason": f"Egger p = {egger_p:.4f} (borderline asymmetry)",
            }
            downgrades += 1
        elif tf_concern:
            domains["publication_bias"] = {
                "level": "serious", "downgrades": 1,
                "reason": f"Trim-and-fill suggests {tf_missing} missing studies",
            }
            downgrades += 1
        else:
            domains["publication_bias"] = {
                "level": "not serious", "downgrades": 0,
                "reason": "No evidence of publication bias",
            }

        # Final level
        levels = ["High", "Moderate", "Low", "Very low"]
        start_idx = levels.index(starting) if starting in levels else 0
        final_idx = min(start_idx + downgrades, 3)
        final_level = levels[final_idx]

        return {
            "starting_level": starting,
            "total_downgrades": downgrades,
            "final_level": final_level,
            "domains": domains,
        }

    # ================================================================== #
    #  SUMMARY OF FINDINGS (SoF) TABLE                                     #
    # ================================================================== #

    def _generate_sof(self, measure: str, rma_result: Dict,
                      analysis: Dict, data: Dict) -> Dict:
        """Generate a GRADE-compliant Summary of Findings entry."""
        pooled = rma_result.get("pooled_estimate", 0)
        ci_l = rma_result.get("ci_lower", 0)
        ci_u = rma_result.get("ci_upper", 0)
        k = rma_result.get("k_studies", 0)
        p_val = rma_result.get("p_value", 1)

        total_p = analysis.get("total_participants", {})
        n_participants = total_p.get("total_n")
        grade = analysis.get("grade", {})
        certainty = grade.get("final_level", "Unknown")

        # Format the relative effect string
        relative_effect = f"{measure} {pooled:.2f} (95% CI: {ci_l:.2f} to {ci_u:.2f})"

        # Anticipated absolute effects
        # Direction text
        if pooled < 0:
            direction = "lower"
            abs_val = abs(pooled)
            harm_or_benefit = "lower"
        else:
            direction = "higher"
            abs_val = abs(pooled)
            harm_or_benefit = "higher"

        ci_direction_l = "lower" if ci_l < 0 else "higher"
        ci_direction_u = "lower" if ci_u < 0 else "higher"

        intervention_effect = (
            f"{measure} {abs_val:.2f} {direction} "
            f"({abs(ci_l):.2f} {ci_direction_l} to {abs(ci_u):.2f} {ci_direction_u})"
        )

        # Generate "what happens" narrative
        what_happens = self._generate_sof_narrative(
            measure, pooled, ci_l, ci_u, p_val, certainty
        )

        n_str = str(n_participants) if n_participants else "Not reported"

        return {
            "outcome": f"{measure} outcome",
            "n_studies": k,
            "n_participants": n_str,
            "relative_effect": relative_effect,
            "anticipated_absolute_effects": {
                "control_risk": "See individual study baselines",
                "intervention_effect": intervention_effect,
            },
            "certainty": certainty,
            "what_happens": what_happens,
        }

    def _generate_sof_narrative(self, measure: str, pooled: float,
                                 ci_l: float, ci_u: float,
                                 p_val: float, certainty: str) -> str:
        """Generate algorithmic narrative for SoF 'what happens' based on
        effect direction, magnitude, significance, and GRADE level."""
        crosses_null = ci_l <= 0 <= ci_u
        significant = p_val < 0.05
        abs_effect = abs(pooled)

        # Determine magnitude
        if abs_effect < 0.2:
            magnitude = "little to no difference"
        elif abs_effect < 0.5:
            magnitude = "a small difference"
        elif abs_effect < 0.8:
            magnitude = "a moderate difference"
        else:
            magnitude = "a large difference"

        # Direction
        if pooled < 0:
            direction_text = "reduction"
        elif pooled > 0:
            direction_text = "increase"
        else:
            direction_text = "no change"

        # Certainty-based language
        cert_lower = certainty.lower()
        if cert_lower == "high":
            confidence_word = "results in"
        elif cert_lower == "moderate":
            confidence_word = "likely results in"
        elif cert_lower == "low":
            confidence_word = "may result in"
        else:  # very low
            confidence_word = "may result in"

        # Build narrative
        if not significant and crosses_null:
            if cert_lower == "very low":
                narrative = (
                    f"The intervention {confidence_word} {magnitude} "
                    f"in {measure} outcomes, but the evidence is very uncertain"
                )
            else:
                narrative = (
                    f"The intervention {confidence_word} {magnitude} "
                    f"in {measure} outcomes (not statistically significant)"
                )
        elif significant:
            narrative = (
                f"The intervention {confidence_word} {magnitude} "
                f"({direction_text}) in {measure} outcomes"
            )
            if cert_lower in ("low", "very low"):
                narrative += ", but the evidence is uncertain"
        else:
            narrative = (
                f"The intervention {confidence_word} {magnitude} "
                f"in {measure} outcomes"
            )

        return narrative

    # ================================================================== #
    #  HELPERS                                                             #
    # ================================================================== #

    def _empty_result(self, papers: List[Dict]) -> Dict:
        return {
            "analyses": {},
            "stats": {
                "total_papers_input": len(papers),
                "papers_with_usable_data": 0,
                "analyses_performed": 0,
                "elapsed_s": 0,
            },
        }

    def _print_summary(self, result: Dict):
        s = result["stats"]
        print(f"\n{'='*60}")
        print("AGENT 7 - SYNTHESIS SUMMARY")
        print(f"{'='*60}")
        print(f"  Papers input          : {s['total_papers_input']}")
        print(f"  Papers with data      : {s['papers_with_usable_data']}")
        print(f"  Analyses performed    : {s['analyses_performed']}")
        print(f"  Time                  : {s['elapsed_s']}s")

        for measure, analysis in result["analyses"].items():
            model = analysis.get("model", {})
            grade = analysis.get("grade", {})
            pi_lb = model.get("prediction_interval_lower")
            pi_ub = model.get("prediction_interval_upper")
            pi_str = f", PI [{pi_lb}, {pi_ub}]" if pi_lb is not None else ""
            print(f"\n  {measure}:")
            print(f"    Pooled = {model.get('pooled_estimate', '?')} "
                  f"[{model.get('ci_lower', '?')}, {model.get('ci_upper', '?')}]{pi_str}")
            print(f"    p = {model.get('p_value', '?')}, I^2 = {model.get('I2', '?')}%")
            print(f"    GRADE: {grade.get('final_level', '?')} (from {grade.get('starting_level', '?')}, "
                  f"-{grade.get('total_downgrades', '?')})")

            # Print subgroup summary
            subgroups = analysis.get("subgroup_analyses", {})
            for sg_name, sg_data in subgroups.items():
                if isinstance(sg_data, dict) and not sg_data.get("skipped"):
                    print(f"    Subgroup ({sg_name}): performed")
                elif isinstance(sg_data, dict) and sg_data.get("skipped"):
                    print(f"    Subgroup ({sg_name}): skipped - {sg_data.get('reason', '')}")

            # Print meta-regression summary
            mr = analysis.get("meta_regression", {})
            if mr.get("skipped"):
                print(f"    Meta-regression: skipped - {mr.get('reason', '')}")
            else:
                for mr_name, mr_data in mr.items():
                    if isinstance(mr_data, dict) and not mr_data.get("skipped") and not mr_data.get("error"):
                        print(f"    Meta-regression ({mr_name}): coef={mr_data.get('coefficient')}, p={mr_data.get('p_value')}")

            # Print SoF
            sof = analysis.get("summary_of_findings", {})
            if sof:
                print(f"    SoF: {sof.get('what_happens', '')}")


# ====================================================================== #
#  STANDALONE TEST                                                         #
# ====================================================================== #
if __name__ == "__main__":
    a7 = Agent7Synthesis()
    print("Agent7Synthesis created. Run test_agent7.py for full test.")
