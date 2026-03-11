"""PRISMA 2020 Flow Diagram Generator.

Generates a professional PRISMA flow diagram as PNG using matplotlib,
showing the identification, screening, eligibility, and inclusion stages.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from typing import Dict, Optional


# ── Color Palette ─────────────────────────────────────────────────────────────

HEADER_BG = "#1E6F8E"      # Dark teal for stage headers
HEADER_FG = "#FFFFFF"       # White text on headers
BOX_BG = "#FFFFFF"          # White for detail boxes
BOX_BORDER = "#B0BEC5"     # Light gray border
EXCLUDED_BG = "#FFF3E0"    # Light orange for exclusion boxes
EXCLUDED_BORDER = "#E65100" # Orange border for exclusions
ARROW_COLOR = "#546E7A"    # Dark gray for arrows
INCLUDED_BG = "#E8F5E9"    # Light green for final inclusion
INCLUDED_BORDER = "#2E7D32" # Green border


def generate_prisma_flow(prisma_data: dict, output_path: str) -> str:
    """Generate PRISMA 2020 flow diagram as PNG.

    prisma_data should have keys:
    - identified: int (total from all databases)
    - identified_per_db: dict (e.g., {"PubMed": 500, "Europe PMC": 300, ...})
    - duplicates_removed: int
    - screened: int
    - excluded_screening: int
    - fulltext_assessed: int
    - excluded_fulltext: int
    - not_retrieved: int
    - included_review: int
    - included_synthesis: int (studies with usable quantitative data)
    - quality_assessed: int

    Returns path to generated PNG.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    # Extract data with defaults
    identified = prisma_data.get("identified", 0)
    identified_per_db = prisma_data.get("identified_per_db", {})
    duplicates_removed = prisma_data.get("duplicates_removed", 0)
    screened = prisma_data.get("screened", 0)
    excluded_screening = prisma_data.get("excluded_screening", 0)
    fulltext_assessed = prisma_data.get("fulltext_assessed", 0)
    excluded_fulltext = prisma_data.get("excluded_fulltext", 0)
    not_retrieved = prisma_data.get("not_retrieved", 0)
    failed_criteria = excluded_fulltext - not_retrieved if excluded_fulltext > not_retrieved else excluded_fulltext
    included_review = prisma_data.get("included_review", 0)
    included_synthesis = prisma_data.get("included_synthesis", 0)
    quality_assessed = prisma_data.get("quality_assessed", 0)

    # Build per-database string
    if identified_per_db:
        db_parts = [f"{db}: n={n}" for db, n in identified_per_db.items()]
        db_str = " | ".join(db_parts)
    else:
        db_str = "PubMed"

    # ── Helper: draw a rounded box with text ──────────────────────────────

    def draw_box(x, y, w, h, texts, bg=BOX_BG, border=BOX_BORDER, header=None,
                 header_bg=HEADER_BG, header_fg=HEADER_FG, text_size=9):
        """Draw a rounded rectangle with optional header bar and text lines."""
        # Main box
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            facecolor=bg,
            edgecolor=border,
            linewidth=1.5,
        )
        ax.add_patch(box)

        if header:
            # Header bar at top of box
            hh = 0.38
            header_box = FancyBboxPatch(
                (x, y + h - hh), w, hh,
                boxstyle="round,pad=0.05",
                facecolor=header_bg,
                edgecolor=header_bg,
                linewidth=0,
            )
            ax.add_patch(header_box)
            ax.text(
                x + w / 2, y + h - hh / 2,
                header,
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=header_fg,
            )
            # Content text below header
            text_y_start = y + h - hh - 0.15
        else:
            text_y_start = y + h - 0.2

        for i, line in enumerate(texts):
            ax.text(
                x + w / 2, text_y_start - i * 0.32,
                line,
                ha="center", va="top",
                fontsize=text_size,
                color="#37474F",
            )

        return (x + w / 2, y)  # Return bottom center for arrows

    def draw_arrow(x1, y1, x2, y2, style="-"):
        """Draw arrow between points."""
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ARROW_COLOR,
                lw=1.8,
                connectionstyle="arc3,rad=0",
            ),
        )

    def draw_side_arrow(x1, y1, x2, y2):
        """Draw an L-shaped arrow going right then down."""
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ARROW_COLOR,
                lw=1.5,
                connectionstyle="angle,angleA=0,angleB=90,rad=5",
            ),
        )

    # ── Stage 1: Identification ───────────────────────────────────────────

    # Main identification box (left)
    box1_x, box1_y, box1_w, box1_h = 0.8, 11.8, 5.4, 1.6
    draw_box(
        box1_x, box1_y, box1_w, box1_h,
        texts=[
            f"Records identified (n={identified})",
            db_str[:60],
        ],
        header="Identification",
    )

    # Duplicates box (right)
    dup_x, dup_y, dup_w, dup_h = 6.8, 12.0, 2.8, 0.9
    draw_box(
        dup_x, dup_y, dup_w, dup_h,
        texts=[
            f"Duplicates removed",
            f"(n={duplicates_removed})",
        ],
        bg=EXCLUDED_BG,
        border=EXCLUDED_BORDER,
    )

    # Arrow from identification to duplicates
    draw_side_arrow(box1_x + box1_w, box1_y + box1_h / 2, dup_x, dup_y + dup_h / 2)

    # ── Stage 2: Screening ────────────────────────────────────────────────

    box2_x, box2_y, box2_w, box2_h = 0.8, 9.2, 5.4, 1.8
    draw_box(
        box2_x, box2_y, box2_w, box2_h,
        texts=[
            f"Records screened (n={screened})",
            f"Title/abstract screening",
        ],
        header="Screening",
    )

    # Arrow from identification to screening
    draw_arrow(box1_x + box1_w / 2, box1_y, box2_x + box2_w / 2, box2_y + box2_h)

    # Excluded screening box (right)
    exc_s_x, exc_s_y, exc_s_w, exc_s_h = 6.8, 9.5, 2.8, 0.9
    draw_box(
        exc_s_x, exc_s_y, exc_s_w, exc_s_h,
        texts=[
            f"Records excluded",
            f"(n={excluded_screening})",
        ],
        bg=EXCLUDED_BG,
        border=EXCLUDED_BORDER,
    )

    # Arrow from screening to excluded
    draw_side_arrow(box2_x + box2_w, box2_y + box2_h / 2, exc_s_x, exc_s_y + exc_s_h / 2)

    # ── Stage 3: Eligibility ──────────────────────────────────────────────

    box3_x, box3_y, box3_w, box3_h = 0.8, 6.2, 5.4, 2.2
    draw_box(
        box3_x, box3_y, box3_w, box3_h,
        texts=[
            f"Full-text articles assessed",
            f"for eligibility (n={fulltext_assessed})",
        ],
        header="Eligibility",
    )

    # Arrow from screening to eligibility
    draw_arrow(box2_x + box2_w / 2, box2_y, box3_x + box3_w / 2, box3_y + box3_h)

    # Excluded full-text box (right)
    exc_f_x, exc_f_y, exc_f_w, exc_f_h = 6.8, 6.2, 2.8, 1.8
    exc_ft_texts = [
        f"Full-text excluded (n={excluded_fulltext})",
    ]
    if not_retrieved > 0:
        exc_ft_texts.append(f"  Not retrieved (n={not_retrieved})")
    if failed_criteria > 0:
        exc_ft_texts.append(f"  Failed criteria (n={failed_criteria})")
    draw_box(
        exc_f_x, exc_f_y, exc_f_w, exc_f_h,
        texts=exc_ft_texts,
        bg=EXCLUDED_BG,
        border=EXCLUDED_BORDER,
        text_size=8.5,
    )

    # Arrow from eligibility to excluded
    draw_side_arrow(box3_x + box3_w, box3_y + box3_h / 2, exc_f_x, exc_f_y + exc_f_h / 2)

    # ── Stage 4: Included ─────────────────────────────────────────────────

    box4_x, box4_y, box4_w, box4_h = 0.8, 3.2, 5.4, 2.2
    included_texts = [
        f"Studies in qualitative review (n={included_review})",
    ]
    if quality_assessed > 0:
        included_texts.append(f"Quality assessed (n={quality_assessed})")
    if included_synthesis > 0:
        included_texts.append(f"Studies in meta-analysis (n={included_synthesis})")
    else:
        included_texts.append(f"Studies in meta-analysis (n={included_review})")

    draw_box(
        box4_x, box4_y, box4_w, box4_h,
        texts=included_texts,
        header="Included",
        header_bg="#2E7D32",
        bg=INCLUDED_BG,
        border=INCLUDED_BORDER,
    )

    # Arrow from eligibility to included
    draw_arrow(box3_x + box3_w / 2, box3_y, box4_x + box4_w / 2, box4_y + box4_h)

    # ── Title ─────────────────────────────────────────────────────────────

    ax.text(
        5, 13.7,
        "PRISMA 2020 Flow Diagram",
        ha="center", va="center",
        fontsize=15, fontweight="bold",
        color="#263238",
    )

    # ── Footer ────────────────────────────────────────────────────────────

    ax.text(
        5, 2.8,
        "Generated by SAM (Systematic Agent for Meta-analysis)",
        ha="center", va="center",
        fontsize=7,
        color="#90A4AE",
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    if os.path.exists(output_path):
        print(f"  PRISMA flow diagram saved: {output_path}")
        return output_path

    return ""
