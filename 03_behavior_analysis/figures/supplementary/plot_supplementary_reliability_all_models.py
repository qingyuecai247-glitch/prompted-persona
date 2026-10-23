"""Plot complete reliability heatmaps for Human and all 11 LLM datasets."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis" / "results" / "reliability_all_models.csv"
OUTPUT = ROOT / "Figures_1" / "reliability_all"

MODELS = [
    ("Human", "Human"),
    ("Deepseek-3.2", "DeepSeek-3.2"),
    ("Grok-3", "Grok-3"),
    ("GPT-5.2", "GPT-5.2"),
    ("Qwen-2.5-32B", "Qwen2.5-32B"),
    ("Phi-medium", "Phi-3-medium"),
    ("GPT-4o-mini", "GPT-4o-mini"),
    ("Llama-3.1-8B", "Llama-3.1-8B"),
    ("Qwen-2.5-7B", "Qwen2.5-7B"),
    ("Qwen-2.5-14B", "Qwen2.5-14B"),
    ("Phi-mini", "Phi-3-mini"),
    ("Phi-small", "Phi-3-small"),
]
TRAITS = ["State_E", "State_A", "State_C", "State_N", "State_O"]
TRAIT_LABELS = ["E", "A", "C", "N", "O"]
CONTEXTS = ["pos", "neg", "soc", "dut", "int", "adv", "dec", "mat"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def main():
    data = pd.read_csv(INPUT)
    cmap = LinearSegmentedColormap.from_list("reliability", ["#FFFFFF", "#BE4A45"])
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.2), constrained_layout=False)
    image = None

    for ax, (raw_name, display_name) in zip(axes.flat, MODELS):
        sub = data.loc[data["Group"].eq(raw_name)]
        matrix = (
            sub.pivot(index="trait", columns="situation", values="reliability")
            .reindex(index=TRAITS, columns=CONTEXTS)
            .to_numpy(dtype=float)
        )
        if matrix.shape != (5, 8) or not np.isfinite(matrix).any():
            raise ValueError(f"No complete reliability matrix for {raw_name}")
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_title(display_name, fontsize=6.8, fontweight="bold", pad=3.5)
        ax.set_xticks(np.arange(8), CONTEXTS, fontsize=5.3)
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False,
                       labelbottom=False, length=0, pad=1)
        ax.set_yticks(np.arange(5), TRAIT_LABELS, fontsize=5.8, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=2)
        for x in np.arange(-0.5, 8, 1):
            ax.axvline(x, color="white", linewidth=0.85)
        for y in np.arange(-0.5, 5, 1):
            ax.axhline(y, color="white", linewidth=0.85)
        for spine in ax.spines.values():
            spine.set_color("#D7D7D7")
            spine.set_linewidth(0.4)

    fig.subplots_adjust(left=0.045, right=0.93, top=0.94, bottom=0.035,
                        wspace=0.22, hspace=0.34)
    cax = fig.add_axes([0.947, 0.20, 0.012, 0.60])
    cbar = fig.colorbar(image, cax=cax, ticks=[0, 0.5, 1])
    cbar.set_label("Reliability", fontsize=6.3, labelpad=4)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cbar.outline.set_linewidth(0.4)

    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
