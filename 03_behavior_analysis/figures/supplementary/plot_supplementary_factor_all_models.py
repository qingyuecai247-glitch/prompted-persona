"""Plot centred-SCSC five-factor structures for Human and all 11 LLMs."""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis" / "results" / "factor_modularity_centered_diamonds_phi_medium"
INPUT = RESULTS / "all_factor_loadings.csv"
OUTPUT = ROOT / "Figures_1" / "factor_all"

MODEL_ORDER = ["human", "deepseek", "grok3", "gpt52", "qwen32", "phimed",
               "gpt4omini", "llama", "qwen7", "qwen14", "phimini", "phismall"]
MODEL_LABELS = {
    "human": "Human", "deepseek": "DeepSeek-3.2", "grok3": "Grok-3",
    "gpt52": "GPT-5.2", "qwen32": "Qwen2.5-32B-Instruct",
    "phimed": "Phi-3-medium-4k-instruct", "gpt4omini": "GPT-4o-mini",
    "llama": "Llama-3.1-8B-Instruct", "qwen7": "Qwen2.5-7B-Instruct",
    "qwen14": "Qwen2.5-14B-Instruct", "phimini": "Phi-3-mini-4k-instruct",
    "phismall": "Phi-3-small-8k-instruct",
}
TRAIT_ORDER = ["N", "A", "E", "C", "O"]
TRAIT_COLORS = {"O": "#D1494E", "C": "#3478B7", "E": "#F08A18",
                "A": "#4FA64F", "N": "#8255A6"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
})


def ordered_rows(sub):
    return [item for trait in TRAIT_ORDER
            for item in sorted(sub.loc[sub.trait.eq(trait), "item"].unique())]


def draw_model(fig, spec, sub, title):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec,
                                              width_ratios=[0.57, 0.43], wspace=0.05)
    ax_h, ax_f = fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])
    rows, factors = ordered_rows(sub), [f"F{i}" for i in range(1, 6)]
    mat = (sub.pivot(index="item", columns="factor", values="loading")
           .reindex(index=rows, columns=factors).to_numpy(float))
    ax_h.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    ax_h.set_title(title, fontsize=6.4, fontweight="bold", pad=4)
    ax_h.set_xticks(range(5), factors, fontsize=5.1, fontweight="bold")
    ax_h.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=1)
    ax_h.set_yticks([])
    for y in [7.5, 15.5, 23.5, 31.5]:
        ax_h.axhline(y, color="white", lw=0.8)
    # Keep trait-group labels outside the heatmap rather than over the first
    # loading column.  The labels remain aligned with the five eight-item rows.
    for i, trait in enumerate(TRAIT_ORDER):
        ax_h.text(-0.68, i * 8 + 3.5, trait, ha="right", va="center",
                  color=TRAIT_COLORS[trait], fontsize=5.7, fontweight="bold", clip_on=False)
    for spine in ax_h.spines.values():
        spine.set_visible(False)

    factor_y, row_y = np.linspace(5, 35, 5), np.arange(40)
    item_trait = sub.drop_duplicates("item").set_index("item")["trait"]
    for row_index, item in enumerate(rows):
        trait = item_trait[item]
        for _, record in sub.loc[sub.item.eq(item) & sub.abs_loading.gt(0.30)].iterrows():
            factor_index = int(str(record.factor).replace("F", "")) - 1
            ax_f.plot([0.02, 0.96], [row_y[row_index], factor_y[factor_index]],
                      color=TRAIT_COLORS[trait], alpha=0.58,
                      lw=0.25 + 1.2 * float(record.abs_loading), solid_capstyle="round")
    for index, y in enumerate(factor_y, 1):
        ax_f.plot([0.96, 0.99], [y, y], color="#333333", lw=0.55)
        ax_f.text(1.01, y, f"F{index}", va="center", ha="left", fontsize=5.1)
    ax_f.set_xlim(0, 1.14); ax_f.set_ylim(40, -1); ax_f.axis("off")


def main():
    data = pd.read_csv(INPUT)
    missing = set(MODEL_ORDER) - set(data.model_key.unique())
    if missing:
        raise ValueError(f"Missing factor loadings for: {sorted(missing)}")
    fig = plt.figure(figsize=(7.2, 7.0), facecolor="white")
    outer = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 0.10],
                              left=0.045, right=0.985, top=0.965, bottom=0.035,
                              wspace=0.20, hspace=0.25)
    for index, key in enumerate(MODEL_ORDER):
        draw_model(fig, outer[index // 4, index % 4], data.loc[data.model_key.eq(key)],
                   MODEL_LABELS[key])
    legend_ax = fig.add_subplot(outer[3, :]); legend_ax.axis("off")
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)
    legend_ax.text(0.00, 0.52, "Big Five:", fontsize=6.2, fontweight="bold", va="center")
    full = {"N": "Neuroticism", "A": "Agreeableness", "E": "Extraversion",
            "C": "Conscientiousness", "O": "Openness"}
    legend_positions = [0.08, 0.22, 0.36, 0.49, 0.66]
    for trait, x in zip(TRAIT_ORDER, legend_positions):
        legend_ax.scatter([x], [0.52], s=22, color=TRAIT_COLORS[trait], clip_on=False)
        legend_ax.text(x + 0.013, 0.52, f"{trait}  {full[trait]}", va="center", fontsize=5.2)
    cax = legend_ax.inset_axes([0.83, 0.34, 0.15, 0.30])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r"),
                      cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.35); cb.set_label("Factor loading", fontsize=5.7, labelpad=1)
    for suffix, kwargs in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
        fig.savefig(OUTPUT.with_suffix(suffix), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
