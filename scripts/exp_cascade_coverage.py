"""Cascade coverage analysis for A4 full-text retrieval.

Reads the A4 checkpoint from the reference run, groups papers by their
text_source value, and produces CSV / JSON / PNG outputs summarising the
distribution of full-text sources used along the cascade.

Run from the repository root:
    python scripts/exp_cascade_coverage.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Hardcoded paths relative to the repository root (cwd at execution time).
CHECKPOINT_PATH = Path("outputs/run_20260507_081359/checkpoint_a4_fulltext.json")
OUTPUT_DIR = Path("outputs/exp_bloque_b")
CSV_PATH = OUTPUT_DIR / "cascade_coverage.csv"
JSON_PATH = OUTPUT_DIR / "cascade_coverage.json"
PNG_PATH = OUTPUT_DIR / "cascade_coverage.png"

NONE_LABEL = "none"


def load_papers(checkpoint_path: Path) -> list[dict]:
    """Load include + exclude papers from the A4 checkpoint."""
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    include = data.get("include", []) or []
    exclude = data.get("exclude", []) or []
    return list(include) + list(exclude)


def normalize_source(value) -> str:
    """Map missing / empty text_source values to the 'none' bucket."""
    if value is None:
        return NONE_LABEL
    if isinstance(value, str) and value.strip() == "":
        return NONE_LABEL
    return str(value)


def compute_distribution(papers: list[dict]) -> tuple[int, list[dict]]:
    """Group papers by text_source and compute counts + percentages."""
    total = len(papers)
    counts = Counter(normalize_source(p.get("text_source")) for p in papers)
    # Sort by n desc, then alphabetically for stable ties.
    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    rows = []
    for source, n in sorted_items:
        pct = (n / total * 100.0) if total else 0.0
        rows.append({"source": source, "n": n, "pct": pct})
    return total, rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write the distribution as CSV with columns source,n,pct."""
    import csv

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "n", "pct"])
        for row in rows:
            writer.writerow([row["source"], row["n"], f"{row['pct']:.2f}"])


def write_json(total: int, rows: list[dict], path: Path) -> None:
    """Write the distribution as JSON with total_papers + sources list."""
    payload = {
        "total_papers": total,
        "sources": [
            {"source": r["source"], "n": r["n"], "pct": round(r["pct"], 2)}
            for r in rows
        ],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def plot_distribution(rows: list[dict], path: Path) -> None:
    """Render a horizontal bar chart of the source distribution."""
    # Already sorted n desc; for horizontal bars, plotting in reverse puts
    # the largest bar at the top.
    sources = [r["source"] for r in rows]
    counts = [r["n"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = list(range(len(sources)))
    ax.barh(y_positions, counts, color="#0D9488")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sources)
    ax.invert_yaxis()  # largest at top
    ax.set_xlabel("Numero de papers")
    ax.set_ylabel("Fuente full-text")
    ax.set_title(
        "Cobertura de fuentes full-text - A4 cascade "
        "(run referencia 20260507_081359)"
    )

    # Annotate counts at end of each bar.
    max_count = max(counts) if counts else 0
    x_offset = max(0.1, max_count * 0.01)
    for y, n in zip(y_positions, counts):
        ax.text(n + x_offset, y, str(n), va="center", ha="left", fontsize=10)

    # Leave headroom for the annotations.
    if max_count > 0:
        ax.set_xlim(0, max_count * 1.12)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    papers = load_papers(CHECKPOINT_PATH)
    total, rows = compute_distribution(papers)

    write_csv(rows, CSV_PATH)
    write_json(total, rows, JSON_PATH)
    plot_distribution(rows, PNG_PATH)

    # Console summary (Spanish for user-facing output).
    print(f"Total papers analizados: {total}")
    print(f"{'source':<20} {'n':>4} {'pct':>7}")
    pct_sum = 0.0
    for row in rows:
        print(f"{row['source']:<20} {row['n']:>4} {row['pct']:>6.2f}%")
        pct_sum += row["pct"]
    print(f"Suma pct: {pct_sum:.2f}")
    print(f"CSV:  {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")
    print(f"PNG:  {PNG_PATH}")


if __name__ == "__main__":
    main()
