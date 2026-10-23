# -*- coding: utf-8 -*-
"""
8×5 filled density plot for random slopes across Human and all 11 LLMs
with HUMAN 95% interval reference background bands.

Layout
------
- 8 rows  = contexts: dut, int, adv, mat, pos, neg, dec, soc
- 5 cols  = traits:   E, A, C, N, O

This version:
1. Removes the original gray near-zero shadow band
2. Removes the big figure-level title
3. Removes the bottom "Random slope value" title
4. Adds only a HUMAN 95% interval reference background band in each panel
5. Keeps the filled density style
6. Keeps the dataset legend
7. Keeps the zero-slope dashed line

Input CSV
---------
Required columns:
    unique_id, re_slope, source, state, context

Outputs
-------
plots_random_slopes_8x5_human95_band/
    random_slopes_8x5_human95_band.png
    random_slopes_8x5_human95_band.pdf
    random_slopes_8x5_human95_band.svg
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

# 如果 csv 和脚本在同一目录，保持这个即可
# 否则改成你的实际路径，例如：
# RANDOM_SLOPES_CSV = r"./local_assets/random_slopes_8sources.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SLOPES_CSV = str(
    PROJECT_ROOT / "analysis" / "results"
    / "source_specific_blups_descriptive_only.csv"
)

OUTPUT_DIR = str(PROJECT_ROOT / "Figures_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_BASENAME = os.path.join(
    OUTPUT_DIR,
    "LMM_11"
)


# ============================================================
# 2. Orders / labels
# ============================================================

STATE_ORDER = ["State_E", "State_A", "State_C", "State_N", "State_O"]
CONTEXT_ORDER = ["dut", "int", "adv", "mat", "pos", "neg", "dec", "soc"]

SOURCE_ORDER = [
    "human",
    "deepseek",
    "grok3",
    "gpt52",
    "qwen32",
    "phimedium",
    "gpt4omini",
    "llama",
    "qwen7",
    "qwen14",
    "phimini",
    "phismall",
]

SOURCE_LABELS = {
    "human": "Human",
    "gpt4omini": "GPT-4o-mini",
    "gpt52": "GPT-5.2",
    "qwen32": "Qwen2.5-32B",
    "phimedium": "Phi-3-medium",
    "deepseek": "DeepSeek-3.2",
    "grok3": "Grok-3",
    "llama": "Llama-3.1-8B",
    "qwen7": "Qwen2.5-7B",
    "qwen14": "Qwen2.5-14B",
    "phimini": "Phi-3-mini",
    "phismall": "Phi-3-small",
}

TRAIT_SHORT = {
    "State_E": "E",
    "State_A": "A",
    "State_C": "C",
    "State_N": "N",
    "State_O": "O",
}

TRAIT_FULL = {
    "State_E": "Extraversion",
    "State_A": "Agreeableness",
    "State_C": "Conscientiousness",
    "State_N": "Neuroticism",
    "State_O": "Openness",
}

CONTEXT_DISPLAY = {
    "dut": "dut",
    "int": "int",
    "adv": "adv",
    "mat": "mat",
    "pos": "pos",
    "neg": "neg",
    "dec": "dec",
    "soc": "soc",
}


# ============================================================
# 3. Colors
# ============================================================

COLOR_MAPPING = {
    "human": "#2F6FDB",
    "gpt4omini": "#D8B48A",
    "gpt52": "#E67E22",
    "qwen7": "#C96F3D",
    "qwen32": "#D978A8",
    "phimedium": "#9A6A82",
    "deepseek": "#E74C3C",
    "grok3": "#A6504E",
    "llama": "#7F5A4E",
    "qwen7": "#C96F3D",
    "qwen14": "#6E9CC5",
    "phimini": "#D8A34B",
    "phismall": "#C57A5A",
}

HUMAN_BAND_COLOR = "#2F6FDB"


# ============================================================
# 4. Plot config
# ============================================================

X_LIM = (-0.22, 0.22)
X_TICKS = [-0.2, 0.0, 0.2]

# KDE bandwidth adjustment
KDE_BW_ADJUST = 1.0

# "panel" is recommended for readability:
# peak-height differences are preserved within each panel.
DENSITY_SCALE_MODE = "panel"

# maximum half-height of each symmetric density shape
MAX_HALF_HEIGHT = 0.38

# transparency and line width
HUMAN_FILL_ALPHA = 0.92
MODEL_FILL_ALPHA = 0.82
HUMAN_EDGE_WIDTH = 0.9
MODEL_EDGE_WIDTH = 0.7

# zero line
ZERO_LINE_COLOR = "0.45"
ZERO_LINE_STYLE = "--"
ZERO_LINE_WIDTH = 0.75

# Human 95% interval reference band
DRAW_HUMAN_95_BAND = True
HUMAN_95_ALPHA = 0.07


# ============================================================
# 5. Matplotlib style
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10.5,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# 6. Safe figure-level artist helpers
# ============================================================

def add_fig_line(fig, xdata, ydata, **kwargs):
    """
    Safely add a Line2D artist in figure coordinates.
    Avoids bbox_inches='tight' savefig errors in some Matplotlib versions.
    """
    line = Line2D(
        xdata,
        ydata,
        transform=fig.transFigure,
        **kwargs
    )
    fig.add_artist(line)
    return line


def add_fig_rect(fig, xy, width, height, **kwargs):
    """
    Safely add a Rectangle artist in figure coordinates.
    """
    rect = Rectangle(
        xy,
        width,
        height,
        transform=fig.transFigure,
        **kwargs
    )
    fig.add_artist(rect)
    return rect


# ============================================================
# 7. KDE helpers
# ============================================================

def compute_kde_density(values, x_grid, bw_adjust=1.0):
    """
    Compute KDE density from all observed values.
    Uses scipy gaussian_kde if available; otherwise falls back to
    histogram density + Gaussian smoothing.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return None

    if np.nanstd(values) == 0:
        center = float(values[0])
        bandwidth = max((x_grid.max() - x_grid.min()) / 250.0, 1e-4)
        density = np.exp(-0.5 * ((x_grid - center) / bandwidth) ** 2)
        density = density / (bandwidth * np.sqrt(2 * np.pi))
        return density

    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values)
        kde.set_bandwidth(kde.factor * bw_adjust)
        return kde(x_grid)

    except Exception:
        n_bins = 180
        bin_edges = np.linspace(x_grid.min(), x_grid.max(), n_bins + 1)
        hist, _ = np.histogram(values, bins=bin_edges, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        sigma_bins = 1.4 * bw_adjust
        radius = int(3 * sigma_bins + 0.5)
        kx = np.arange(-radius, radius + 1)
        kernel = np.exp(-(kx ** 2) / (2 * sigma_bins ** 2))
        kernel /= kernel.sum()

        smooth_hist = np.convolve(hist, kernel, mode="same")
        return np.interp(x_grid, centers, smooth_hist)


def precompute_densities(df, x_grid):
    density_table = {}
    grouped = df.groupby(["state", "context", "source"], observed=True)

    for (state, context, source), g in grouped:
        if state not in STATE_ORDER:
            continue
        if context not in CONTEXT_ORDER:
            continue
        if source not in SOURCE_ORDER:
            continue

        density = compute_kde_density(
            g["re_slope"].values,
            x_grid,
            bw_adjust=KDE_BW_ADJUST,
        )

        if density is not None and np.isfinite(density).all():
            density_table[(state, context, source)] = density

    return density_table


def get_panel_values(df):
    """
    Store raw re_slope values for each (state, context, source).
    Useful for Human 95% reference bands.
    """
    value_table = {}
    grouped = df.groupby(["state", "context", "source"], observed=True)

    for (state, context, source), g in grouped:
        if state not in STATE_ORDER:
            continue
        if context not in CONTEXT_ORDER:
            continue
        if source not in SOURCE_ORDER:
            continue

        vals = pd.to_numeric(g["re_slope"], errors="coerce").dropna().values
        if len(vals) > 0:
            value_table[(state, context, source)] = vals

    return value_table


def get_global_max_density(density_table):
    max_density = 0.0
    for density in density_table.values():
        max_density = max(max_density, float(np.nanmax(density)))
    return max_density


def get_panel_max_density(density_table, state, context):
    max_density = 0.0
    for source in SOURCE_ORDER:
        density = density_table.get((state, context, source))
        if density is not None:
            max_density = max(max_density, float(np.nanmax(density)))
    return max_density


def get_density_scale(density_table, state, context):
    if DENSITY_SCALE_MODE == "global":
        max_density = get_global_max_density(density_table)
    elif DENSITY_SCALE_MODE == "panel":
        max_density = get_panel_max_density(density_table, state, context)
    else:
        raise ValueError("DENSITY_SCALE_MODE must be 'panel' or 'global'.")

    if max_density <= 0:
        return 1.0

    return MAX_HALF_HEIGHT / max_density


def compute_human_95_interval(values):
    """
    Compute Human 95% interval from raw values.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    q025 = float(np.quantile(values, 0.025))
    q975 = float(np.quantile(values, 0.975))
    return q025, q975


# ============================================================
# 8. Main plotting function
# ============================================================

def plot_random_slopes_8x5_human95_band(random_slopes_df):
    required_cols = {"re_slope", "source", "state", "context"}
    missing_cols = required_cols - set(random_slopes_df.columns)
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {sorted(missing_cols)}")

    df = random_slopes_df.copy()
    df["re_slope"] = pd.to_numeric(df["re_slope"], errors="coerce")
    df = df.dropna(subset=["re_slope", "source", "state", "context"])

    df["source"] = pd.Categorical(df["source"], categories=SOURCE_ORDER, ordered=True)
    df["state"] = pd.Categorical(df["state"], categories=STATE_ORDER, ordered=True)
    df["context"] = pd.Categorical(df["context"], categories=CONTEXT_ORDER, ordered=True)
    df = df.dropna(subset=["source", "state", "context"])

    x_grid = np.linspace(X_LIM[0], X_LIM[1], 500)
    density_table = precompute_densities(df, x_grid)
    value_table = get_panel_values(df)

    # 8 rows × 5 columns
    n_rows = len(CONTEXT_ORDER)
    n_cols = len(STATE_ORDER)
    n_sources = len(SOURCE_ORDER)

    # Human on top
    y_positions = np.arange(n_sources)[::-1]
    y_map = {source: y for source, y in zip(SOURCE_ORDER, y_positions)}

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        # Twelve source rows need substantially more physical height than the
        # earlier eight-source version.  Preserve width and scientific content
        # while giving every label and density its own readable row.
        figsize=(15.8, 19.5),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    plt.subplots_adjust(
        left=0.205,
        right=0.845,
        top=0.925,
        bottom=0.085,
        wspace=0.085,
        hspace=0.255,
    )

    for i, context in enumerate(CONTEXT_ORDER):
        for j, state in enumerate(STATE_ORDER):
            ax = axes[i, j]

            density_scale = get_density_scale(
                density_table,
                state=state,
                context=context,
            )

            # Human 95% interval background band
            if DRAW_HUMAN_95_BAND:
                human_vals = value_table.get((state, context, "human"))
                if human_vals is not None and len(human_vals) > 0:
                    interval = compute_human_95_interval(human_vals)
                    if interval is not None:
                        q025, q975 = interval
                        ax.axvspan(
                            q025,
                            q975,
                            color=HUMAN_BAND_COLOR,
                            alpha=HUMAN_95_ALPHA,
                            linewidth=0,
                            zorder=0,
                        )

            # Zero line
            ax.axvline(
                0,
                color=ZERO_LINE_COLOR,
                linestyle=ZERO_LINE_STYLE,
                linewidth=ZERO_LINE_WIDTH,
                zorder=1,
            )

            # Light horizontal guides
            for yy in y_positions:
                ax.hlines(
                    yy,
                    X_LIM[0],
                    X_LIM[1],
                    color="0.90",
                    linewidth=0.52,
                    zorder=0,
                )

            # Draw filled symmetric densities
            for source in SOURCE_ORDER:
                density = density_table.get((state, context, source))
                if density is None:
                    continue

                y0 = y_map[source]
                color = COLOR_MAPPING[source]
                half_height = density * density_scale

                if source == "human":
                    fill_alpha = HUMAN_FILL_ALPHA
                    edge_lw = HUMAN_EDGE_WIDTH
                    z = 5
                else:
                    fill_alpha = MODEL_FILL_ALPHA
                    edge_lw = MODEL_EDGE_WIDTH
                    z = 4

                ax.fill_between(
                    x_grid,
                    y0 - half_height,
                    y0 + half_height,
                    color=color,
                    alpha=fill_alpha,
                    linewidth=0,
                    zorder=z,
                )

                ax.plot(
                    x_grid,
                    y0 + half_height,
                    color=color,
                    linewidth=edge_lw,
                    alpha=0.98,
                    zorder=z + 1,
                )
                ax.plot(
                    x_grid,
                    y0 - half_height,
                    color=color,
                    linewidth=edge_lw,
                    alpha=0.98,
                    zorder=z + 1,
                )

            # Column titles = traits
            if i == 0:
                ax.set_title(
                    f"{TRAIT_SHORT[state]}\n{TRAIT_FULL[state]}",
                    fontsize=14.2,
                    fontweight="bold",
                    pad=10,
                )

            # Dataset labels only in first column
            ax.set_yticks(y_positions)
            if j == 0:
                ax.set_yticklabels(
                    [SOURCE_LABELS[source] for source in SOURCE_ORDER],
                    fontsize=10.1,
                )
                ax.tick_params(axis="y", pad=3)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)

            ax.set_xlim(X_LIM)
            ax.set_ylim(-0.70, n_sources - 0.05)

            # x ticks only on bottom row
            if i != n_rows - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
                ax.spines["bottom"].set_visible(False)
            else:
                ax.set_xticks(X_TICKS)
                ax.set_xticklabels(["−0.2", "0", "0.2"])
                ax.tick_params(axis="x", labelsize=10.4)

            # Clean spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if j != 0:
                ax.spines["left"].set_visible(False)
            else:
                ax.spines["left"].set_linewidth(0.55)

            ax.grid(False)
            ax.set_xlabel("")
            ax.set_ylabel("")

    # Left-side row labels = contexts
    for i, context in enumerate(CONTEXT_ORDER):
        pos = axes[i, 0].get_position()
        y_mid = pos.y0 + pos.height / 2

        fig.text(
            0.085,
            y_mid,
            CONTEXT_DISPLAY[context],
            ha="center",
            va="center",
            fontsize=17.5,
            fontweight="bold",
        )

    # Dataset legend
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=12,
            markerfacecolor=COLOR_MAPPING[source],
            markeredgecolor=COLOR_MAPPING[source],
            label=SOURCE_LABELS[source],
        )
        for source in SOURCE_ORDER
    ]

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.878, 0.505),
        frameon=False,
        title="Dataset",
        title_fontsize=16.0,
        fontsize=11.3,
        handlelength=0.9,
        handletextpad=0.7,
        labelspacing=0.72,
        borderaxespad=0,
    )

    # Bottom mini-legend
    add_fig_line(
        fig,
        [0.075, 0.075],
        [0.030, 0.054],
        color=ZERO_LINE_COLOR,
        linewidth=1.1,
        linestyle=ZERO_LINE_STYLE,
    )
    fig.text(
        0.085,
        0.041,
        "= Zero slope",
        ha="left",
        va="center",
        fontsize=10.4,
        color="0.25",
    )

    add_fig_rect(
        fig,
        (0.205, 0.030),
        0.015,
        0.020,
        color=HUMAN_BAND_COLOR,
        alpha=HUMAN_95_ALPHA,
        ec=HUMAN_BAND_COLOR,
    )
    fig.text(
        0.227,
        0.041,
        "= Human 95% interval",
        ha="left",
        va="center",
        fontsize=10.4,
        color="0.25",
    )

    # Save
    png_path = OUTPUT_BASENAME + ".png"
    pdf_path = OUTPUT_BASENAME + ".pdf"
    svg_path = OUTPUT_BASENAME + ".svg"

    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    fig.savefig(png_path, dpi=250, facecolor="white", bbox_inches="tight")

    plt.close(fig)

    print(f"PNG saved -> {png_path}")
    print(f"PDF saved -> {pdf_path}")
    print(f"SVG saved -> {svg_path}")


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    if not os.path.exists(RANDOM_SLOPES_CSV):
        raise FileNotFoundError(
            f"找不到 {RANDOM_SLOPES_CSV}。请把 random_slopes_8sources.csv "
            "放到脚本同一目录，或者修改 RANDOM_SLOPES_CSV 为你的实际路径。"
        )

    random_slopes = pd.read_csv(RANDOM_SLOPES_CSV)
    # Preserve the original density-plot semantics: conditional deviations
    # from each source-specific mean slope are displayed around zero. These
    # BLUPs are descriptive only and are not used to estimate dispersion in
    # the main analysis.
    random_slopes = random_slopes.rename(
        columns={"random_slope_deviation_blup": "re_slope"}
    )
    plot_random_slopes_8x5_human95_band(random_slopes)
