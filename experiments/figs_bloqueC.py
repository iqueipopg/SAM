# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Bloque C (Qwen-vs-Claude) figure generator.

Renders the three figures of the per-agent model-comparison section of the
thesis (cap4, sec:bloqueC) into memoria/figuras/. It does NOT run any agent or
recompute any metric: it reads the FROZEN evaluation JSONs where a clean machine
value exists (A6), and otherwise uses the frozen values transcribed verbatim
from the congealed iteration reports, each annotated with its source path:line
and guarded by an assert. No number here is invented.

Run with the venv absolute path:
  ~/meta-analisis-automatizado/venv/bin/python experiments/figs_bloqueC.py

Outputs (PNG, 200 dpi) into memoria/figuras/:
  bloquec_kappa_por_agente.png   - kappa Qwen vs Claude, per agent (A4, A3, A6)
  bloquec_a6_confusion_claude.png - A6 Claude 3x3 RoB2 confusion matrix
  bloquec_a3_gap_c2_c2b.png      - A3 adjusted-sensitivity gap Qwen-Claude (C.2 -> C.2b)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "memoria" / "figuras"

# Frozen JSONs (machine-readable, used directly).
A6_CLAUDE_JSON = ROOT / "outputs/eval/medio/hero4_claude_20260601_061638.json"
A6_QWEN_JSON = ROOT / "outputs/eval/medio/hero4_results_20260518_021939.json"

# Corporate palette (cf. .streamlit/config.toml): teal (Claude) / slate (Qwen).
C_CLAUDE = "#0D9488"
C_QWEN = "#64748B"

# --------------------------------------------------------------------------- #
# Frozen values transcribed verbatim from the congealed iteration reports.
# Each carries its source path:line. These are NOT recomputed here.
# --------------------------------------------------------------------------- #

# A4 mono - kappa. Source: docs/iterations/T17.C.1_log.md:600-601 (exploratory,
# n_runs=1, NOT official). Claude-mono vs Qwen-mono V2 (N=386 vs 387).
A4_KAPPA_CLAUDE = 0.6232
A4_KAPPA_QWEN = 0.0499

# A3 screening - Cohen's kappa, instruction-equalised run (C.2b, N=300).
# Source: docs/iterations/T17.C.2b_iteration_report.md:25.
A3_KAPPA_CLAUDE = 0.319
A3_KAPPA_QWEN = 0.449

# A3 adjusted-sensitivity gap (Qwen - Claude) before/after equalising the
# phase instruction. Sources:
#   C.2  (instruction NOT equalised, N=2000): docs/iterations/T17.C.2_iteration_report.md:11
#        Qwen sens-adj 0.749 ; Claude 0.023  -> gap 0.726
#   C.2b (instruction equalised, N=300):      docs/iterations/T17.C.2b_iteration_report.md:23
#        Qwen sens-adj 0.5625 ; Claude 0.348  -> gap 0.215
A3_SENSADJ = {
    "C.2": {"qwen": 0.749, "claude": 0.023},
    "C.2b": {"qwen": 0.5625, "claude": 0.348},
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def kappa_per_agent_figure(out_path: Path) -> None:
    """Grouped bar chart: kappa Qwen vs Claude for A4, A3, A6.

    A6 kappa (quadratic-weighted) is read live from the frozen JSONs; A4 and A3
    use the frozen iteration-report values, asserted for consistency.
    """
    a6_claude = _load_json(A6_CLAUDE_JSON)["metrics"]["primary"]["cohens_kappa_quadratic"]
    a6_qwen = _load_json(A6_QWEN_JSON)["metrics"]["primary"]["cohens_kappa_quadratic"]
    # Guard against silent drift of the frozen baselines.
    assert abs(a6_claude - 0.1802) < 1e-4, f"A6 Claude kappa drift: {a6_claude}"
    assert abs(a6_qwen - 0.0119) < 1e-4, f"A6 Qwen kappa drift: {a6_qwen}"

    agents = [
        "A4 FT\n($\\kappa$ binario)",
        "A3 cribado\n($\\kappa$, C.2b)",
        "A6 RoB\n($\\kappa_{quad}$)",
    ]
    qwen = [A4_KAPPA_QWEN, A3_KAPPA_QWEN, a6_qwen]
    claude = [A4_KAPPA_CLAUDE, A3_KAPPA_CLAUDE, a6_claude]

    x = np.arange(len(agents))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w / 2, qwen, w, label="Qwen 2.5-7B", color=C_QWEN)
    b2 = ax.bar(x + w / 2, claude, w, label="Claude Opus 4.8", color=C_CLAUDE)
    ax.set_ylabel("$\\kappa$ de Cohen")
    ax.set_title("Concordancia con el gold por agente: Qwen vs Claude")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylim(-0.05, 0.75)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.legend(frameon=False)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def a6_confusion_figure(out_path: Path) -> None:
    """A6 Claude 3x3 RoB2 confusion matrix (gold rows x predicted cols)."""
    cm = _load_json(A6_CLAUDE_JSON)["metrics"]["primary"]["confusion_matrix"]
    labels = ["L", "SC", "H"]
    mat = np.array([[cm[g][p] for p in labels] for g in labels], dtype=int)
    # Guard against drift of the frozen matrix.
    assert mat.tolist() == [[1, 1, 1], [2, 17, 18], [0, 9, 12]], f"A6 cm drift: {mat.tolist()}"

    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(mat, cmap="Greens")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"pred {l}" for l in labels])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"gold {l}" for l in labels])
    ax.set_xlabel("Predicción A6 (Claude)")
    ax.set_ylabel("Gold (Paula)")
    ax.set_title("A6 RoB2 — matriz de confusión (Claude, N=61)")
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, str(mat[i, j]), ha="center", va="center",
                color="white" if mat[i, j] > 9 else "black", fontsize=12,
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="nº estudios")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def a3_gap_figure(out_path: Path) -> None:
    """A3 adjusted-sensitivity (Qwen vs Claude) for C.2 and C.2b, with the gap."""
    stages = ["C.2\n(instr. no igualada)", "C.2b\n(instr. igualada)"]
    qwen = [A3_SENSADJ["C.2"]["qwen"], A3_SENSADJ["C.2b"]["qwen"]]
    claude = [A3_SENSADJ["C.2"]["claude"], A3_SENSADJ["C.2b"]["claude"]]
    gaps = [q - c for q, c in zip(qwen, claude)]
    assert abs(gaps[0] - 0.726) < 1e-3 and abs(gaps[1] - 0.2145) < 1e-3, f"A3 gap drift: {gaps}"

    x = np.arange(len(stages))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, qwen, w, label="Qwen 2.5-7B", color=C_QWEN)
    ax.bar(x + w / 2, claude, w, label="Claude Opus 4.8", color=C_CLAUDE)
    for i, g in enumerate(gaps):
        top = max(qwen[i], claude[i])
        ax.annotate(
            f"gap = {g:.3f}", xy=(x[i], top + 0.03), ha="center", fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Sensibilidad ajustada")
    ax.set_title("A3 cribado: igualar la instrucción cierra el gap (0.726 → 0.215)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    kappa_per_agent_figure(FIG_DIR / "bloquec_kappa_por_agente.png")
    a6_confusion_figure(FIG_DIR / "bloquec_a6_confusion_claude.png")
    a3_gap_figure(FIG_DIR / "bloquec_a3_gap_c2_c2b.png")
    print(f"Figuras Bloque C escritas en {FIG_DIR}")


if __name__ == "__main__":
    main()
