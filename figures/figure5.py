# -*- coding: utf-8 -*-
"""Compact main-text figure for participant-level slope variability."""

import os
from pathlib import Path
import argparse
import json
import hashlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / "config/paths.json").is_file())
SOURCE_SPECIFIC_RESULTS = Path(os.environ.get(
    "PERSONA_FIGURE5_INPUT", REPO / "data/reference/analysis/results/source_specific_relative_variability.csv"
))
OUTPUT_DIR = Path(os.environ.get("PERSONA_FIGURE_OUTPUT", REPO / "outputs/figures"))
OUTPUT_BASENAME = OUTPUT_DIR / "random_slope_variability_main_text"

STATE_ORDER = ["State_E", "State_A", "State_C", "State_N", "State_O"]
CONTEXT_ORDER = ["dut", "int", "adv", "mat", "pos", "neg", "dec", "soc"]
SOURCE_ORDER = [
    "human",
    "gpt4omini",
    "gpt52",
    "qwen7",
    "qwen14",
    "qwen32",
    "deepseek",
    "grok3",
    "llama",
    "phimini",
    "phismall",
    "phimedium",
]
SOURCE_FILES = {
    "human": "dat_aggregated_final.csv",
    "gpt4omini": "simulation_output_flattened-gpt4omini.csv",
    "gpt52": "simulation_output_flattened-gpt52.csv",
    "qwen7": "simulation_output_flattened-qwen2.5-7b-instruct.csv",
    "qwen14": "simulation_output_flattened-qwen2.5-14b-instruct.csv",
    "qwen32": "simulation_output_flattened-qwen2.5-32b-instruct.csv",
    "deepseek": "simulation_output_flattened-deepseek3.2.csv",
    "grok3": "simulation_output_flattened-grok3.csv",
    "llama": "simulation_output_flattened_llama3.1-8b-instruct.csv",
    "phimini": "simulation_output_flattened-phi3-mini.csv",
    "phismall": "simulation_output_flattened-phi3-small.csv",
    "phimedium": "simulation_output_flattened-phi3-medium.csv",
}
SOURCE_LABELS = {
    "human": "Human",
    "gpt4omini": "GPT-4o-mini",
    "gpt52": "GPT-5.2",
    "qwen7": "Qwen2.5-7B",
    "qwen14": "Qwen2.5-14B",
    "qwen32": "Qwen2.5-32B",
    "deepseek": "DeepSeek-3.2",
    "grok3": "Grok-3",
    "llama": "Llama-3.1-8B",
    "phimini": "Phi-3 mini",
    "phismall": "Phi-3 small",
    "phimedium": "Phi-3-medium",
}
TRAIT_LABELS = {
    "State_E": "E",
    "State_A": "A",
    "State_C": "C",
    "State_N": "N",
    "State_O": "O",
}

# Colors follow joint_distribution.py.
MODEL_COLORS = {
    "gpt4omini": "#E68613",
    "gpt52": "#D65F9E",
    "qwen7": "#168ACB",
    "qwen14": "#0868AC",
    "qwen32": "#084081",
    "phimedium": "#9A5C73",
    "deepseek": "#E64B35",
    "grok3": "#00A087",
    "llama": "#A6761D",
}
REFERENCE_COLOR = "#222222"

# Only models currently represented by files in dataset_final are eligible.
MODEL_FAMILIES = {
    "GPT": ["gpt4omini", "gpt52"],
    "Qwen2.5": ["qwen7", "qwen14", "qwen32"],
    "Phi-3": ["phimedium"],
    "DeepSeek": ["deepseek"],
    "Grok": ["grok3"],
    "Llama": ["llama"],
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.7,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def participant_slopes(df, state, context):
    """Compute one within-participant OLS slope for a matched cell."""
    values = df[["id_p", state, context]].copy()
    values[state] = pd.to_numeric(values[state], errors="coerce")
    values[context] = pd.to_numeric(values[context], errors="coerce")
    values = values.dropna()

    grouped = values.groupby("id_p", observed=True)
    values["dx"] = values[context] - grouped[context].transform("mean")
    values["dy"] = values[state] - grouped[state].transform("mean")
    values["cross"] = values["dx"] * values["dy"]
    values["sq_x"] = values["dx"] ** 2
    sums = values.groupby("id_p", observed=True)[["cross", "sq_x"]].sum()
    slopes = sums["cross"] / sums["sq_x"]
    return slopes.replace([np.inf, -np.inf], np.nan).dropna()


def compute_relative_variability():
    """Compute every source from dataset_final using one common procedure."""
    ratios = safe_read_csv(SOURCE_SPECIFIC_RESULTS)
    required_results = {
        "source", "state", "context", "relative_variability_vs_human"
    }
    missing_results = required_results - set(ratios.columns)
    if missing_results:
        raise ValueError(
            f"Source-specific results missing: {sorted(missing_results)}"
        )
    ratios = ratios.rename(
        columns={"relative_variability_vs_human": "relative_sd"}
    )
    # Frozen primary-model set and display order used throughout the paper.
    # Qwen2.5-7B is supplementary; Phi-3-medium replaces it in the main text.
    primary = [
        "deepseek", "grok3", "gpt52", "qwen32",
        "phimedium", "gpt4omini", "llama",
    ]
    ratios = ratios.loc[ratios["source"].isin(primary)].copy()
    expected = {(source, state, context) for source in primary
                for state in STATE_ORDER for context in CONTEXT_ORDER}
    observed = set(ratios[["source", "state", "context"]].itertuples(index=False, name=None))
    if observed != expected or ratios.duplicated(["source", "state", "context"]).any():
        raise ValueError("Figure 5 requires exactly 40 unique trait-context cells per primary model.")
    if not np.isfinite(ratios["relative_sd"]).all() or (ratios["relative_sd"] < 0).any():
        raise ValueError("Relative slope SDs must be finite and nonnegative.")
    return ratios


def quantile_summary(values):
    return np.quantile(values, [0.025, 0.25, 0.50, 0.75, 0.975])


def select_family_representatives(ratios):
    """Select the highest-median model from each model family.

    GPT-4o mini and GPT-5.2 are both retained so the full GPT family
    appears in the figure.
    """
    medians = ratios.groupby("source", observed=True)["relative_sd"].median()
    selected = []
    for family, candidates in MODEL_FAMILIES.items():
        available = [source for source in candidates if source in medians.index]
        if not available:
            continue
        if family == "GPT":
            for source in available:
                selected.append({
                    "family": family,
                    "source": source,
                    "median": float(medians.loc[source]),
                })
        else:
            winner = medians.loc[available].idxmax()
            selected.append({
                "family": family,
                "source": winner,
                "median": float(medians.loc[winner]),
            })

    selection = pd.DataFrame(selected)
    return selection.sort_values("median", ascending=False).reset_index(drop=True)


def fixed_jitter(n, width):
    if n <= 1:
        return np.zeros(n)
    permutation = (np.arange(n) * 17) % n
    return (permutation / (n - 1) - 0.5) * 2 * width


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(labelsize=8.2)


def draw_panel_a(ax, ratios, order):
    positions = np.arange(len(order))
    data = [
        ratios.loc[ratios["source"] == source, "relative_sd"].to_numpy()
        for source in order
    ]
    violins = ax.violinplot(
        data,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, source in zip(violins["bodies"], order):
        body.set_facecolor(MODEL_COLORS[source])
        body.set_edgecolor(MODEL_COLORS[source])
        body.set_alpha(0.28)
        body.set_linewidth(0.8)

    for x, (source, values) in enumerate(zip(order, data)):
        values = np.sort(values)
        # Exclude off-scale points before calculating jitter. Their true values
        # remain in the violin/quantiles and are shown with a labelled triangle.
        visible_values = values[values <= 1.20]
        ax.scatter(
            x + fixed_jitter(len(visible_values), 0.18),
            visible_values,
            s=7,
            color=MODEL_COLORS[source],
            alpha=0.50,
            linewidth=0,
            zorder=2,
        )
        if values.max() > 1.20:
            ax.scatter(x, 1.155, marker="^", s=24, facecolor=MODEL_COLORS[source],
                       edgecolor="black", linewidth=.75, zorder=6)
            ax.annotate(f"{values.max():.2f}", (x, 1.155), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6.2)
        q025, q25, median, q75, q975 = quantile_summary(values)
        ax.plot([x, x], [q025, q975], color="black", linewidth=0.7, zorder=4)
        ax.plot([x, x], [q25, q75], color="black", linewidth=2.6, zorder=4)
        ax.scatter(
            x,
            median,
            s=22,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

    ax.axhline(
        1,
        color=REFERENCE_COLOR,
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=0,
    )
    ax.text(
        len(order) - 0.55,
        1.018,
        "Human",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=REFERENCE_COLOR,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [SOURCE_LABELS[source] for source in order],
        rotation=42,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(0, 1.20)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Slope variability relative to Human", fontsize=8.7)
    ax.set_title("Across state–situation pairings", fontsize=9.6, pad=7)
    style_axis(ax)


def draw_panel_b(ax, ratios, order):
    trait_summary = (
        ratios.groupby(["source", "state"], observed=True)["relative_sd"]
        .median()
        .unstack()
        .reindex(index=order, columns=STATE_ORDER)
    )

    for row, source in enumerate(order):
        for column, state in enumerate(STATE_ORDER):
            value = trait_summary.loc[source, state]
            size = 30 + 180 * value
            if value < 0.35:
                size = 2 + 210 * value
            ax.scatter(
                column,
                row,
                s=size,
                color=MODEL_COLORS[source],
                alpha=0.42 + min(value, 0.8) * 0.6,
                edgecolor="white",
                linewidth=0.55,
            )

    ax.set_xlim(-0.55, len(STATE_ORDER) - 0.45)
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xticks(np.arange(len(STATE_ORDER)))
    ax.set_xticklabels([TRAIT_LABELS[state] for state in STATE_ORDER])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([SOURCE_LABELS[source] for source in order])
    ax.set_xlabel("Big Five state dimension", fontsize=8.7)
    ax.set_title("Big Five-specific variability", fontsize=9.6, pad=7)
    ax.set_xticks(np.arange(-0.5, len(STATE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="#E8E8E8", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)


def draw_panel_c(ax, ratios, order):
    context_summary = (
        ratios.groupby(["source", "context"], observed=True)["relative_sd"]
        .median()
        .unstack()
        .reindex(index=order, columns=CONTEXT_ORDER)
    )
    context_order = (
        context_summary.median(axis=0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    y_positions = np.arange(len(context_order))[::-1]
    offsets = np.linspace(-0.16, 0.16, len(order))

    for row_index, (y, context) in enumerate(zip(y_positions, context_order)):
        values = context_summary[context]
        low = float(values.min())
        high = float(values.max())
        median = float(values.median())

        if row_index % 2 == 0:
            ax.axhspan(
                y - 0.42,
                y + 0.42,
                color="#F6F6F4",
                zorder=0,
            )

        ax.plot(
            [low, high],
            [y, y],
            color="#C4C4C1",
            linewidth=4.6,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            [low, high],
            [y, y],
            color="white",
            linewidth=1.0,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )
        for offset, source in zip(offsets, order):
            value = float(values.loc[source])
            ax.scatter(
                value,
                y + offset,
                s=31,
                color=MODEL_COLORS[source],
                edgecolor="white",
                linewidth=0.75,
                zorder=3,
            )

        ax.scatter(
            median,
            y,
            marker="D",
            s=38,
            facecolor="white",
            edgecolor="#202020",
            linewidth=1.0,
            zorder=4,
        )
        ax.text(
            high + 0.016,
            y,
            f"{median:.2f}",
            ha="left",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color="#333333",
            zorder=5,
        )

    ax.set_xlim(0.0, 0.65)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(context_order, fontsize=8.2)
    ax.set_xlabel("Slope variability relative to Human", fontsize=8.3)
    ax.set_title("Situation-specific variability", fontsize=9.6, pad=7)
    ax.xaxis.grid(True, color="#E3E3E1", linewidth=0.5)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)


def plot_random_slopes_nature(dpi=600):
    all_ratios = compute_relative_variability()
    order = [
        "deepseek", "grok3", "gpt52", "qwen32",
        "phimedium", "gpt4omini", "llama",
    ]
    ratios = all_ratios[all_ratios["source"].isin(order)].copy()

    print("Primary-model source-specific REML ratios:")
    for source in order:
        median = ratios.loc[
            ratios["source"].eq(source), "relative_sd"
        ].median()
        print(f"  {SOURCE_LABELS[source]}: median={median:.3f}")

    # Reference canvas is 1110 x 406 px at 100 dpi. Use fixed panel bounds
    # rather than tight-bbox cropping, so spacing is reproducible across exports.
    fig = plt.figure(figsize=(11.10, 4.06), facecolor="white")
    def reference_axes(left, top, width, height):
        return fig.add_axes([left / 1110, (406 - top - height) / 406,
                             width / 1110, height / 406])
    ax_a = reference_axes(93, 59, 316, 266)
    ax_b = reference_axes(501, 59, 203, 266)
    ax_c = reference_axes(796, 59, 280, 266)

    draw_panel_a(ax_a, ratios, order)
    draw_panel_b(ax_b, ratios, order)
    draw_panel_c(ax_c, ratios, order)

    for ax, label in zip([ax_a, ax_b, ax_c], ["a", "b", "c"]):
        ax.text(
            -0.16,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=11.0,
            fontweight="bold",
            ha="left",
            va="top",
        )

    size_ax = fig.add_axes([491 / 1110, 11 / 406, 197 / 1110, 35 / 406])
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)
    for x, value in zip([0.14, 0.50, 0.86], [0.2, 0.5, 0.8]):
        size_ax.scatter(
            x,
            0.58,
            s=15 + 125 * value,
            color="#777777",
            alpha=0.65,
            edgecolor="white",
            linewidth=0.55,
        )
        size_ax.text(x, 0.06, f"{value:.1f}", ha="center", fontsize=6.8)
    size_ax.axis("off")


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = str(OUTPUT_BASENAME) + ".png"
    pdf_path = str(OUTPUT_BASENAME) + ".pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(str(OUTPUT_BASENAME) + ".svg", facecolor="white")
    fig_preview = str(OUTPUT_BASENAME) + "_preview.png"
    fig.savefig(fig_preview, dpi=100, facecolor="white")
    ratios.to_csv(str(OUTPUT_BASENAME) + "_source_data.csv", index=False)
    manifest = {
        "input_file": str(SOURCE_SPECIFIC_RESULTS.resolve()),
        "input_sha256": hashlib.sha256(SOURCE_SPECIFIC_RESULTS.read_bytes()).hexdigest(),
        "refitted": False,
        "models": order,
        "cells_per_model": 40,
        "panel_a": {"axis_limits": [0, 1.2], "overflow_display_y": 1.155,
                    "overflow_label": "actual maximum, not winsorized data",
                    "intervals": "2.5th-97.5th and 25th-75th percentiles across cells; not confidence intervals"},
        "bubble_legend": "Historical reference artwork uses illustrative, non-calibrated legend sizes; retained for reference matching.",
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    }
    Path(str(OUTPUT_BASENAME) + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plt.close(fig)

    print(f"PNG saved -> {png_path}")
    print(f"PDF saved -> {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE_SPECIFIC_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    SOURCE_SPECIFIC_RESULTS = args.input
    OUTPUT_DIR = args.output_dir
    OUTPUT_BASENAME = OUTPUT_DIR / "random_slope_variability_main_text"
    plot_random_slopes_nature(dpi=args.dpi)
