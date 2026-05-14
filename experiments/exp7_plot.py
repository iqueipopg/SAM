"""
Exp 7 plot generator.

Reads experiments/results/exp7_results.json (per-agent time_seconds and
vram_peak_mb) and renders outputs/exp_bloque_b/vram_profile.png with two
horizontal-bar panels (time and VRAM peak) per agent A1..A8.

This script does not run any agent; it only plots existing results.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = Path("experiments/results/exp7_results.json")
OUTPUT_DIR = Path("outputs/exp_bloque_b")
OUTPUT_PNG = OUTPUT_DIR / "vram_profile.png"


def load_profiles(path: Path):
    """Load per-agent profiles from the exp7 results JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("profiles", [])


def _vram_is_realistic(vram_peaks):
    """Return True if at least one VRAM peak >= 1000 MB (real measurement)."""
    return any(v >= 1000.0 for v in vram_peaks)


def render_plot(profiles, output_path: Path):
    """Render plot to output_path.

    If vram_peak_mb values look realistic (>=1000 MB at least once), draw the
    two-panel plot (time + VRAM). Otherwise the VRAM instrumentation is
    obsolete and we fall back to a single-panel plot with only time_seconds
    plus a textual note flagging the obsolescence.
    """
    agents = [p["agent"] for p in profiles]
    times = [float(p.get("time_seconds", 0.0)) for p in profiles]
    vram_peaks = [float(p.get("vram_peak_mb", 0.0)) for p in profiles]

    # Bars are drawn bottom-up; reverse so A1 appears at the top.
    agents_plot = list(reversed(agents))
    times_plot = list(reversed(times))
    vram_plot = list(reversed(vram_peaks))

    if _vram_is_realistic(vram_peaks):
        fig, (ax_time, ax_vram) = plt.subplots(2, 1, figsize=(10, 8))

        # Panel 1: time per agent.
        bars_t = ax_time.barh(agents_plot, times_plot, color="tab:blue")
        ax_time.set_title("Tiempo por agente (s)")
        ax_time.set_xlabel("Segundos")
        max_t = max(times_plot) if times_plot else 1.0
        ax_time.set_xlim(0, max_t * 1.15 if max_t > 0 else 1.0)
        for bar, value in zip(bars_t, times_plot):
            ax_time.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:.1f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        # Panel 2: VRAM peak per agent.
        bars_v = ax_vram.barh(agents_plot, vram_plot, color="tab:orange")
        ax_vram.set_title("VRAM peak por agente (MB)")
        ax_vram.set_xlabel("MB")
        max_v = max(vram_plot) if vram_plot else 1.0
        ax_vram.set_xlim(0, max_v * 1.15 if max_v > 0 else 1.0)
        for bar, value in zip(bars_v, vram_plot):
            ax_vram.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:.1f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        fig.suptitle(
            "Exp 7 - VRAM y tiempo por agente (RTX 4060 8GB, Qwen2.5-7B 4-bit)"
        )
    else:
        # Fallback: VRAM instrumentation obsolete (all peaks < 1000 MB).
        # Single panel with time_seconds + textual note.
        fig, ax_time = plt.subplots(1, 1, figsize=(10, 5))
        bars_t = ax_time.barh(agents_plot, times_plot, color="tab:blue")
        ax_time.set_title("Tiempo por agente (s)")
        ax_time.set_xlabel("Segundos")
        max_t = max(times_plot) if times_plot else 1.0
        ax_time.set_xlim(0, max_t * 1.20 if max_t > 0 else 1.0)
        for bar, value in zip(bars_t, times_plot):
            ax_time.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:.1f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        fig.suptitle(
            "Exp 7 - Tiempo por agente (RTX 4060 8GB, Qwen2.5-7B 4-bit)"
        )
        fig.text(
            0.5,
            0.01,
            "Medicion VRAM obsoleta (instrumentacion corregida pendiente de re-run)",
            ha="center",
            fontsize=9,
            style="italic",
            color="dimgray",
        )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"No se encontro {RESULTS_PATH}. Ejecuta exp7_vram_profiling.py primero."
        )
    profiles = load_profiles(RESULTS_PATH)
    if not profiles:
        raise SystemExit("El JSON no contiene 'profiles'. Nada que graficar.")
    render_plot(profiles, OUTPUT_PNG)
    print(f"Plot guardado en {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
