"""Risk of Bias Visualization Plots.

Generates:
1. Traffic light plot (per-study, per-domain grid with colored circles)
2. Summary bar plot (horizontal stacked bars showing proportion at each risk level)

Supports RoB 2 (5 domains), ROBINS-I (7 domains), and JBI studies.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Color mappings ────────────────────────────────────────────────────────────

RISK_COLORS = {
    # RoB2
    "Low":               "#4CAF50",  # Green
    "low":               "#4CAF50",
    "Some concerns":     "#FFC107",  # Amber
    "some concerns":     "#FFC107",
    "High":              "#F44336",  # Red
    "high":              "#F44336",
    # ROBINS-I
    "Moderate":          "#FFC107",
    "moderate":          "#FFC107",
    "Serious":           "#FF9800",  # Orange
    "serious":           "#FF9800",
    "Critical":          "#D32F2F",  # Dark red
    "critical":          "#D32F2F",
    # General
    "No information":    "#BDBDBD",  # Gray
    "no information":    "#BDBDBD",
    "Unclear":           "#BDBDBD",
    "unclear":           "#BDBDBD",
    "Not applicable":    "#E0E0E0",
    "not applicable":    "#E0E0E0",
}

DEFAULT_COLOR = "#BDBDBD"

# Canonical domain labels
ROB2_DOMAINS = [
    "randomization_process",
    "deviations_from_intervention",
    "missing_outcome_data",
    "outcome_measurement",
    "selection_of_reported_result",
]

ROB2_DOMAIN_LABELS = {
    "randomization_process": "D1\nRandom.",
    "deviations_from_intervention": "D2\nDeviat.",
    "missing_outcome_data": "D3\nMissing",
    "outcome_measurement": "D4\nMeasure.",
    "selection_of_reported_result": "D5\nSelect.",
}

ROBINS_I_DOMAINS = [
    "confounding",
    "selection_of_participants",
    "classification_of_interventions",
    "deviations_from_intended_interventions",
    "missing_data",
    "measurement_of_outcomes",
    "selection_of_reported_result",
]

ROBINS_I_DOMAIN_LABELS = {
    "confounding": "D1\nConfound.",
    "selection_of_participants": "D2\nSelect.",
    "classification_of_interventions": "D3\nClassif.",
    "deviations_from_intended_interventions": "D4\nDeviat.",
    "missing_data": "D5\nMissing",
    "measurement_of_outcomes": "D6\nMeasure.",
    "selection_of_reported_result": "D7\nReport.",
}


def _get_risk_color(judgment: str) -> str:
    """Map a risk judgment to a color."""
    return RISK_COLORS.get(judgment, RISK_COLORS.get(judgment.lower() if isinstance(judgment, str) else "", DEFAULT_COLOR))


def _extract_study_data(quality_data: dict) -> Tuple[List[Dict], str]:
    """Extract per-study domain judgments from A6 quality data.

    Returns (studies_list, primary_tool) where each study has:
    - label: str (PMID: Author)
    - domains: dict of {domain_name: judgment_str}
    - tool: str
    """
    papers = quality_data.get("papers", [])
    if isinstance(quality_data, list):
        papers = quality_data

    studies = []
    tool_counts = {"RoB2": 0, "ROBINS-I": 0, "JBI": 0}

    for p in papers:
        qa = p.get("quality_assessment", {})
        tool = qa.get("tool", "")
        if tool in tool_counts:
            tool_counts[tool] += 1

        pmid = p.get("pmid", "?")
        # Try to extract first author from title or authors
        title = p.get("title", "")
        authors = p.get("authors", [])
        if authors:
            first_author = authors[0].split(",")[0].split()[-1] if authors[0] else ""
        else:
            first_author = ""
        year = p.get("year", "")

        label = f"{pmid}"
        if first_author:
            label = f"{first_author}"
            if year:
                label += f", {year}"

        domains = {}
        if tool in ("RoB2", "ROBINS-I"):
            raw_domains = qa.get("domains", {})
            for dname, dinfo in raw_domains.items():
                if isinstance(dinfo, dict):
                    domains[dname] = dinfo.get("judgment", "No information")
                else:
                    domains[dname] = str(dinfo)
        elif tool == "JBI":
            # JBI items → treat as domains
            items = qa.get("items", {})
            for iname, iinfo in items.items():
                if isinstance(iinfo, dict):
                    answer = iinfo.get("answer", "Unclear")
                    # Map JBI Yes/No to risk-like colors
                    if answer == "Yes":
                        domains[iname] = "Low"
                    elif answer == "No":
                        domains[iname] = "High"
                    else:
                        domains[iname] = "Unclear"
                else:
                    domains[iname] = str(iinfo)

        # Add overall
        domains["Overall"] = qa.get("overall", "No information")

        studies.append({
            "label": label,
            "tool": tool,
            "domains": domains,
        })

    # Determine primary tool
    primary = max(tool_counts, key=tool_counts.get) if tool_counts else "RoB2"
    return studies, primary


def generate_rob_traffic_light(quality_data: dict, output_path: str) -> str:
    """Generate RoB traffic light plot (per-study, per-domain grid).

    Each cell is a colored circle: green=Low, yellow=Some concerns, red=High, gray=No info.

    Returns PNG path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    studies, primary_tool = _extract_study_data(quality_data)
    if not studies:
        return ""

    # Filter to studies matching primary tool for a coherent domain grid
    # Include all studies but use primary tool's domains as columns
    if primary_tool == "RoB2":
        domain_keys = ROB2_DOMAINS + ["Overall"]
        domain_labels = {**ROB2_DOMAIN_LABELS, "Overall": "Overall"}
    elif primary_tool == "ROBINS-I":
        domain_keys = ROBINS_I_DOMAINS + ["Overall"]
        domain_labels = {**ROBINS_I_DOMAIN_LABELS, "Overall": "Overall"}
    else:
        # JBI or mixed: collect all domain names from studies
        all_domains = set()
        for s in studies:
            all_domains.update(s["domains"].keys())
        all_domains.discard("Overall")
        domain_keys = sorted(all_domains)[:10] + ["Overall"]  # Limit to 10 + overall
        domain_labels = {d: d[:8] for d in domain_keys}
        domain_labels["Overall"] = "Overall"

    n_studies = len(studies)
    n_domains = len(domain_keys)

    # Figure sizing
    cell_size = 0.6
    fig_w = max(6, 2.5 + n_domains * cell_size * 1.8)
    fig_h = max(3, 1.5 + n_studies * cell_size * 1.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, n_domains - 0.5)
    ax.set_ylim(-0.5, n_studies - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Draw circles
    radius = 0.32
    for i, study in enumerate(studies):
        for j, domain in enumerate(domain_keys):
            judgment = study["domains"].get(domain, "No information")
            color = _get_risk_color(judgment)

            circle = plt.Circle((j, i), radius, color=color, ec="#616161", linewidth=0.8)
            ax.add_patch(circle)

            # Add symbol inside circle
            if judgment.lower() in ("low",):
                ax.text(j, i, "+", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
            elif judgment.lower() in ("high", "serious", "critical"):
                ax.text(j, i, "-", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
            elif judgment.lower() in ("some concerns", "moderate"):
                ax.text(j, i, "~", ha="center", va="center", fontsize=10, fontweight="bold", color="#333")
            else:
                ax.text(j, i, "?", ha="center", va="center", fontsize=9, color="#555")

    # Labels
    ax.set_xticks(range(n_domains))
    ax.set_xticklabels(
        [domain_labels.get(d, d[:8]) for d in domain_keys],
        fontsize=8, ha="center",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_studies))
    ax.set_yticklabels([s["label"] for s in studies], fontsize=8)

    # Remove box around plot
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Title
    ax.set_title(
        f"Risk of Bias Assessment ({primary_tool})",
        fontsize=12, fontweight="bold", pad=25,
    )

    # Legend
    legend_items = [
        mpatches.Patch(color="#4CAF50", label="Low"),
        mpatches.Patch(color="#FFC107", label="Some concerns"),
        mpatches.Patch(color="#F44336", label="High"),
        mpatches.Patch(color="#BDBDBD", label="Unclear/NI"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if os.path.exists(output_path):
        print(f"  RoB traffic light plot saved: {output_path}")
        return output_path
    return ""


def generate_rob_summary(quality_data: dict, output_path: str) -> str:
    """Generate RoB summary bar chart (horizontal stacked bars per domain).

    Shows proportion of studies at each risk level per domain.

    Returns PNG path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    studies, primary_tool = _extract_study_data(quality_data)
    if not studies:
        return ""

    # Determine domain keys
    if primary_tool == "RoB2":
        domain_keys = ROB2_DOMAINS + ["Overall"]
        domain_labels = {**ROB2_DOMAIN_LABELS, "Overall": "Overall"}
    elif primary_tool == "ROBINS-I":
        domain_keys = ROBINS_I_DOMAINS + ["Overall"]
        domain_labels = {**ROBINS_I_DOMAIN_LABELS, "Overall": "Overall"}
    else:
        all_domains = set()
        for s in studies:
            all_domains.update(s["domains"].keys())
        all_domains.discard("Overall")
        domain_keys = sorted(all_domains)[:10] + ["Overall"]
        domain_labels = {d: d[:12] for d in domain_keys}
        domain_labels["Overall"] = "Overall"

    # Count judgments per domain
    categories = ["Low", "Some concerns", "High", "Unclear"]
    cat_colors = ["#4CAF50", "#FFC107", "#F44336", "#BDBDBD"]

    # Map all judgments to categories
    def categorize(judgment: str) -> str:
        j = judgment.lower() if isinstance(judgment, str) else ""
        if j in ("low",):
            return "Low"
        elif j in ("some concerns", "moderate"):
            return "Some concerns"
        elif j in ("high", "serious", "critical"):
            return "High"
        else:
            return "Unclear"

    n_domains = len(domain_keys)
    n_studies = len(studies)
    if n_studies == 0:
        return ""

    # Build proportion data
    proportions = {cat: [] for cat in categories}
    for domain in domain_keys:
        counts = {cat: 0 for cat in categories}
        for s in studies:
            judgment = s["domains"].get(domain, "No information")
            cat = categorize(judgment)
            counts[cat] += 1
        for cat in categories:
            proportions[cat].append(counts[cat] / n_studies * 100)

    # Plot
    fig_h = max(3, n_domains * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    y_pos = np.arange(n_domains)
    left = np.zeros(n_domains)

    for cat, color in zip(categories, cat_colors):
        widths = proportions[cat]
        ax.barh(y_pos, widths, left=left, height=0.6, color=color, label=cat, edgecolor="white", linewidth=0.5)
        # Add percentage text
        for i, (w, l) in enumerate(zip(widths, left)):
            if w > 8:  # Only show if segment is wide enough
                ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=7.5, fontweight="bold",
                       color="white" if cat != "Some concerns" else "#333")
        left += widths

    # Clean up domain labels (remove newlines from traffic light labels)
    clean_labels = []
    for d in domain_keys:
        lbl = domain_labels.get(d, d[:12])
        clean_labels.append(lbl.replace("\n", " ").strip())

    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_xlabel("Proportion of studies (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title(
        f"Risk of Bias Summary ({primary_tool})",
        fontsize=12, fontweight="bold", pad=12,
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        fontsize=9,
        frameon=False,
    )

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if os.path.exists(output_path):
        print(f"  RoB summary plot saved: {output_path}")
        return output_path
    return ""
