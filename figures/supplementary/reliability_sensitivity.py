import os
"""Plot the human participant-by-situation aggregation sensitivity analysis."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / "config/paths.json").is_file())
ROOT = Path(os.environ.get("PERSONA_REFERENCE_ROOT", REPO / "data/reference"))
INPUT = (
    ROOT / "analysis" / "results" / "human_person_situation_reliability_sensitivity"
    / "human_reliability_original_vs_aggregated.csv"
)
OUTPUT = Path(os.environ.get("PERSONA_FIGURE_OUTPUT", REPO / "outputs/figures")) / "human_reliability_aggregation_sensitivity"

TRAITS = ["State_E", "State_A", "State_C", "State_N", "State_O"]
TRAIT_LABELS = ["E", "A", "C", "N", "O"]
CONTEXTS = ["pos", "neg", "soc", "dut", "int", "adv", "dec", "mat"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
})


def matrix(df: pd.DataFrame, value: str) -> np.ndarray:
    return (
        df.pivot(index="state", columns="context", values=value)
        .reindex(index=TRAITS, columns=CONTEXTS)
        .to_numpy(float)
    )


def draw_heatmap(ax, values, title, cmap, vmin=0, vmax=1):
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=7, fontweight="normal", pad=4)
    ax.set_xticks(np.arange(8), CONTEXTS, fontsize=6)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False,
                   labelbottom=False, length=0, pad=1)
    ax.set_yticks(np.arange(5), TRAIT_LABELS, fontsize=6.5)
    ax.tick_params(axis="y", length=0, pad=2)
    for x in np.arange(-0.5, 8, 1):
        ax.axvline(x, color="white", linewidth=0.9)
    for y in np.arange(-0.5, 5, 1):
        ax.axhline(y, color="white", linewidth=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.45)
    return image


def main():
    df = pd.read_csv(INPUT)
    old_col = "reliability_original_human"
    new_col = "reliability_aggregated_human"
    if old_col not in df.columns or new_col not in df.columns:
        raise ValueError(f"Expected columns {old_col!r} and {new_col!r}; got {df.columns.tolist()}")

    old = matrix(df, old_col)
    new = matrix(df, new_col)
    cmap = LinearSegmentedColormap.from_list("reliability", ["#FFFFFF", "#B44742"])

    fig = plt.figure(figsize=(7.2, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.12], wspace=0.42,
                          left=0.055, right=0.965, top=0.83, bottom=0.20)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    image = draw_heatmap(ax_a, old, "Repeated human records", cmap)
    draw_heatmap(ax_b, new, "Participant–situation means", cmap)

    x = old.ravel()
    y = new.ravel()
    ax_c.scatter(x, y, s=17, color="#2F5F8F", alpha=0.85, linewidth=0)
    lim = (0, 1)
    ax_c.plot(lim, lim, color="#B8B8B8", linewidth=0.7, linestyle="--", zorder=0)
    ax_c.set_xlim(lim)
    ax_c.set_ylim(lim)
    ax_c.set_xticks([0, 0.5, 1.0])
    ax_c.set_yticks([0, 0.5, 1.0])
    ax_c.set_xlabel("Repeated-record reliability", fontsize=6.5)
    ax_c.set_ylabel("Aggregated reliability", fontsize=6.5)
    ax_c.tick_params(labelsize=6, width=0.5, length=2.5)
    r = np.corrcoef(x, y)[0, 1]
    ax_c.text(0.04, 0.95, f"r = {r:.3f}", transform=ax_c.transAxes,
              ha="left", va="top", fontsize=6.5, color="#333333")
    ax_c.set_title("Cell-wise comparison", fontsize=7, fontweight="normal", pad=4)

    cax = fig.add_axes([0.355, 0.075, 0.31, 0.035])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cbar.set_label("Reliability", fontsize=6.2, labelpad=1)
    cbar.ax.tick_params(labelsize=5.8, length=2, width=0.4, pad=1)
    cbar.outline.set_linewidth(0.4)

    for label, ax in zip("abc", [ax_a, ax_b, ax_c]):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.026, 0.90, label, fontsize=8, fontweight="bold",
                 ha="left", va="top")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
