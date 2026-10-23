# -*- coding: utf-8 -*-
"""
MixedLM random-slope extraction + LMM_final2-style 8x5 density plot.

This script keeps the data-building logic of LMM_5.py and the plotting
layout/style of LMM_final2.py, configured for 11 model sources plus Human.

Outputs
-------
mixed_model_results_12sources.csv
random_slopes_12sources.csv
LMM/
    LMM_12.pdf
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import chi2
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

PATH_BASE = r"./local_assets/dataset"

SOURCE_FILES = {
    "human": "dat_aggregated_final.csv",
    "gpt4omini": r"llm_output\simulation_output_flattened-gpt4omini.csv",
    "gpt52": r"llm_output\simulation_output_flattened-gpt52.csv",
    "qwen05": "simulation_output_flattened-qwen2.5-0.5b.csv",
    "qwen15": "simulation_output_flattened-qwen2.5-1.5b.csv",
    "qwen3": "simulation_output_flattened-qwen2.5-3b.csv",
    "qwen7": r"llm_output\simulation_output_flattened-qwen2.5-7b-instruct.csv",
    "qwen14": r"llm_output\simulation_output_flattened-qwen2.5-14b-instruct.csv",
    "qwen32": r"llm_output\simulation_output_flattened-qwen2.5-32b-instruct.csv",
    "grok3": r"llm_output\simulation_output_flattened-grok3.csv",
    "deepseek": r"llm_output\simulation_output_flattened-deepseek3.2.csv",
    "llama": r"llm_output\simulation_output_flattened_llama3.1-8b-instruct.csv",
    "phimini": r"llm_output\simulation_output_flattened-phi3-mini.csv",
    "phismall": r"llm_output\simulation_output_flattened-phi3-small.csv",
    "phimedium": r"llm_output\simulation_output_flattened-phi3-medium.csv",
}

PLOT_DIR = "LMM"
os.makedirs(PLOT_DIR, exist_ok=True)

MIXED_RESULTS_CSV = "mixed_model_results_12sources.csv"
RANDOM_SLOPES_CSV = "random_slopes_12sources.csv"
MODEL_DIAGNOSTICS_CSV = "mixed_model_convergence_diagnostics.csv"
LRT_RESULTS_CSV = "mixed_model_lrt_results.csv"
SAMPLE_SIZES_CSV = "mixed_model_sample_sizes.csv"
OUTPUT_BASENAME = os.path.join(PLOT_DIR, "random_slopes_12sources_human95_band")
VERSIONED_OUTPUT_BASENAME = os.path.join(PLOT_DIR, "LMM_12")


# ============================================================
# 2. Orders / labels
# ============================================================

RAW_ID_COL = "id_p"
STATE_ORDER = ["State_O", "State_C", "State_E", "State_A", "State_N"]
CONTEXT_ORDER = ["dut", "int", "adv", "mat", "pos", "neg", "dec", "soc"]

SOURCE_ORDER = [
    "human",
    "gpt4omini",
    "gpt52",
    "qwen7",
    "qwen14",
    "qwen32",
    "grok3",
    "deepseek",
    "llama",
    "phimini",
    "phismall",
    "phimedium",
]

SOURCE_LABELS = {
    "human": "Human",
    "gpt4omini": "GPT-4o-mini",
    "gpt52": "GPT-5.2",
    "qwen05": "Qwen-2.5-0.5B",
    "qwen15": "Qwen-2.5-1.5B",
    "qwen3": "Qwen-2.5-3B",
    "qwen7": "Qwen-2.5-7B",
    "qwen14": "Qwen-2.5-14B",
    "qwen32": "Qwen-2.5-32B",
    "grok3": "Grok-3",
    "deepseek": "DeepSeek-3.2",
    "llama": "Llama-3.1-8B",
    "phimini": "Phi-3-mini-4k-instruct",
    "phismall": "Phi-3-small-8k-instruct",
    "phimedium": "Phi-3-medium-4k-instruct",
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
# 3. Warm expanded colors
# ============================================================

COLOR_MAPPING = {
    "human": "#2F6FDB",
    "gpt4omini": "#D8B48A",
    "gpt52": "#E67E22",
    "qwen05": "#F6B04F",
    "qwen15": "#D9822B",
    "qwen3": "#C96A28",
    "qwen7": "#C96F3D",
    "qwen14": "#B85C38",
    "qwen32": "#D978A8",
    "grok3": "#A6504E",
    "deepseek": "#E74C3C",
    "llama": "#7F5A4E",
    "phimini": "#C9973F",
    "phismall": "#B87945",
    "phimedium": "#9A8A7A",
}

HUMAN_BAND_COLOR = COLOR_MAPPING["human"]

SOURCE_X_OFFSETS = {
    source: offset
    for source, offset in zip(
        SOURCE_ORDER,
        np.linspace(-0.065, 0.065, len(SOURCE_ORDER)),
    )
}


# ============================================================
# 4. Plot config
# ============================================================

X_LIM = (-0.22, 0.22)
X_TICKS = [-0.2, 0.0, 0.2]
KDE_BW_ADJUST = 1.0
DENSITY_SCALE_MODE = "panel"
MAX_HALF_HEIGHT = 0.32

HUMAN_FILL_ALPHA = 0.92
MODEL_FILL_ALPHA = 0.80
HUMAN_EDGE_WIDTH = 0.9
MODEL_EDGE_WIDTH = 0.62

ZERO_LINE_COLOR = "0.45"
ZERO_LINE_STYLE = "--"
ZERO_LINE_WIDTH = 0.75

DRAW_HUMAN_95_BAND = True
HUMAN_95_ALPHA = 0.07

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18.0,
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
# 5. Data helpers: LMM_5 style
# ============================================================

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def load_sources():
    frames = []
    skipped = []

    for source in SOURCE_ORDER:
        filename = SOURCE_FILES.get(source)
        if not filename:
            skipped.append((source, "reserved empty source"))
            continue

        path = os.path.join(PATH_BASE, filename)
        if not os.path.exists(path):
            skipped.append((source, f"missing file: {path}"))
            continue

        frame = safe_read_csv(path)
        frame["source"] = source
        frames.append(frame)
        print(f"Loaded {SOURCE_LABELS[source]} <- {filename}")

    if not frames:
        raise FileNotFoundError("No source files were loaded. Check PATH_BASE and SOURCE_FILES.")

    if skipped:
        print("\nSkipped sources:")
        for source, reason in skipped:
            print(f"  - {SOURCE_LABELS[source]}: {reason}")

    df = pd.concat(frames, ignore_index=True)
    df["source"] = pd.Categorical(df["source"], categories=SOURCE_ORDER, ordered=True)
    df["unique_id"] = df[RAW_ID_COL].astype(str) + "_" + df["source"].astype(str)
    return df


def prepare_for_model(df_in, x_col, y_col, group_col, min_samples=30):
    if x_col not in df_in.columns or y_col not in df_in.columns:
        return None

    df = df_in.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df["x_c"] = df[x_col] - df[x_col].mean()
    df[group_col] = df[group_col].astype("category")
    df = df.dropna(subset=["x_c", y_col, group_col, "source"])

    if len(df) < min_samples:
        return None

    return df


def fit_mixed_diagnostic(data, formula, groups_col, re_formula="~x_c"):
    if data is None or len(data) < 10:
        return None, None, []

    attempts = []
    for method in ["lbfgs", "bfgs", "powell"]:
        attempt = {
            "optimizer": method,
            "converged": False,
            "warning_count": 0,
            "warnings": "",
            "error": "",
            "llf": np.nan,
        }
        try:
            md = MixedLM.from_formula(
                formula,
                groups=groups_col,
                re_formula=re_formula,
                data=data,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = md.fit(reml=False, method=method, maxiter=1000, disp=False)

            attempt["converged"] = bool(getattr(res, "converged", False))
            attempt["warning_count"] = len(caught)
            attempt["warnings"] = " | ".join(
                f"{type(w.message).__name__}: {w.message}" for w in caught
            )
            attempt["llf"] = float(res.llf)
            attempts.append(attempt)

            if attempt["converged"]:
                return res, method, attempts
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)

    return None, None, attempts


def add_multiple_testing_columns(df, p_col, prefix):
    """Add BH-FDR and Holm adjusted p-values to one prespecified test family."""
    df = df.copy()
    df[f"{prefix}_p_fdr_bh"] = np.nan
    df[f"{prefix}_reject_fdr_bh"] = False
    df[f"{prefix}_p_holm"] = np.nan
    df[f"{prefix}_reject_holm"] = False

    valid = pd.to_numeric(df[p_col], errors="coerce").notna()
    if valid.any():
        pvals = pd.to_numeric(df.loc[valid, p_col], errors="coerce").to_numpy()
        reject_bh, p_bh, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        reject_holm, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        df.loc[valid, f"{prefix}_p_fdr_bh"] = p_bh
        df.loc[valid, f"{prefix}_reject_fdr_bh"] = reject_bh
        df.loc[valid, f"{prefix}_p_holm"] = p_holm
        df.loc[valid, f"{prefix}_reject_holm"] = reject_holm
    return df


def build_random_slopes():
    print(">>> Reading source files ...")
    df = load_sources()
    group_col = "unique_id"
    results_list = []
    diagnostics_list = []
    lrt_list = []
    sample_size_list = []
    all_random_slopes = []

    print(">>> Fitting MixedLM models ...")
    for y_col in STATE_ORDER:
        for x_col in CONTEXT_ORDER:
            combined_df = prepare_for_model(df, x_col, y_col, group_col, min_samples=200)
            if combined_df is None:
                print(f"Skip {x_col} -> {y_col}: not enough valid data")
                continue

            available_sources = [
                s for s in SOURCE_ORDER
                if len(combined_df.loc[combined_df["source"] == s]) >= 30
            ]
            combined_df = combined_df[combined_df["source"].isin(available_sources)].copy()
            combined_df["source"] = combined_df["source"].cat.remove_unused_categories()

            if len(available_sources) < 2:
                print(f"Skip {x_col} -> {y_col}: fewer than 2 available sources")
                continue

            group_sizes = combined_df.groupby(group_col, observed=True).size()
            for source, source_df in combined_df.groupby("source", observed=True):
                source_group_sizes = source_df.groupby(group_col, observed=True).size()
                sample_size_list.append({
                    "state": y_col,
                    "context": x_col,
                    "source": str(source),
                    "n_observations": int(len(source_df)),
                    "n_groups": int(source_df[group_col].nunique()),
                    "observations_per_group_min": int(source_group_sizes.min()),
                    "observations_per_group_median": float(source_group_sizes.median()),
                    "observations_per_group_max": int(source_group_sizes.max()),
                })

            full_formula = f"{y_col} ~ x_c * C(source)"
            reduced_formula = f"{y_col} ~ x_c + C(source)"

            res_comb, full_optimizer, full_attempts = fit_mixed_diagnostic(
                combined_df,
                full_formula,
                group_col,
            )
            for attempt_index, attempt in enumerate(full_attempts, start=1):
                diagnostics_list.append({
                    "state": y_col,
                    "context": x_col,
                    "model_role": "full_interaction",
                    "formula": full_formula,
                    "attempt": attempt_index,
                    "n_observations": int(len(combined_df)),
                    "n_groups": int(combined_df[group_col].nunique()),
                    "n_sources": int(combined_df["source"].nunique()),
                    **attempt,
                })

            if res_comb is None:
                print(f"Skip {x_col} -> {y_col}: model did not converge")
                continue

            res_reduced, reduced_optimizer, reduced_attempts = fit_mixed_diagnostic(
                combined_df,
                reduced_formula,
                group_col,
            )
            for attempt_index, attempt in enumerate(reduced_attempts, start=1):
                diagnostics_list.append({
                    "state": y_col,
                    "context": x_col,
                    "model_role": "reduced_no_interaction",
                    "formula": reduced_formula,
                    "attempt": attempt_index,
                    "n_observations": int(len(combined_df)),
                    "n_groups": int(combined_df[group_col].nunique()),
                    "n_sources": int(combined_df["source"].nunique()),
                    **attempt,
                })

            fixed_ci = res_comb.conf_int(alpha=0.05).copy()
            fixed_ci.columns = ["ci_95_low", "ci_95_high"]
            random_cov = np.asarray(res_comb.cov_re, dtype=float)
            random_cov_eigenvalues = np.linalg.eigvalsh(random_cov)

            for param in res_comb.fe_params.index:
                results_list.append({
                    "state": y_col,
                    "context": x_col,
                    "parameter": param,
                    "parameter_family": (
                        "source_by_situation_interaction"
                        if "x_c:C(source)" in param or "C(source)" in param and ":x_c" in param
                        else "other_fixed_effect"
                    ),
                    "coef": res_comb.fe_params[param],
                    "std_error": res_comb.bse_fe[param],
                    "ci_95_low": fixed_ci.loc[param, "ci_95_low"],
                    "ci_95_high": fixed_ci.loc[param, "ci_95_high"],
                    "p_value": res_comb.pvalues[param],
                    "optimizer": full_optimizer,
                    "converged": bool(res_comb.converged),
                    "n_observations": int(res_comb.nobs),
                    "n_groups": int(combined_df[group_col].nunique()),
                    "n_sources": int(combined_df["source"].nunique()),
                    "random_intercept_variance": float(random_cov[0, 0]),
                    "random_slope_variance": float(random_cov[1, 1]),
                    "random_intercept_slope_covariance": float(random_cov[0, 1]),
                    "random_cov_min_eigenvalue": float(random_cov_eigenvalues.min()),
                    "residual_variance": float(res_comb.scale),
                })

            if res_reduced is not None:
                lr_stat = max(0.0, 2.0 * (res_comb.llf - res_reduced.llf))
                df_diff = int(len(res_comb.fe_params) - len(res_reduced.fe_params))
                lrt_list.append({
                    "state": y_col,
                    "context": x_col,
                    "full_formula": full_formula,
                    "reduced_formula": reduced_formula,
                    "full_optimizer": full_optimizer,
                    "reduced_optimizer": reduced_optimizer,
                    "full_llf": float(res_comb.llf),
                    "reduced_llf": float(res_reduced.llf),
                    "lr_statistic": lr_stat,
                    "df": df_diff,
                    "p_value": float(chi2.sf(lr_stat, df_diff)),
                    "n_observations": int(res_comb.nobs),
                    "n_groups": int(combined_df[group_col].nunique()),
                    "n_sources": int(combined_df["source"].nunique()),
                })

            re_data = []
            for uid, series in res_comb.random_effects.items():
                re_data.append({
                    group_col: uid,
                    "re_slope": series.get("x_c", np.nan),
                })

            re_df = pd.DataFrame(re_data)
            re_df = re_df.merge(
                combined_df[[group_col, "source"]].drop_duplicates(),
                on=group_col,
                how="left",
            )
            re_df["state"] = y_col
            re_df["context"] = x_col
            all_random_slopes.append(re_df)
            print(f"Done {x_col} -> {y_col}")

    if results_list:
        fixed_results = pd.DataFrame(results_list)
        interaction_mask = (
            fixed_results["parameter_family"] == "source_by_situation_interaction"
        )
        interaction_results = add_multiple_testing_columns(
            fixed_results.loc[interaction_mask],
            "p_value",
            "interaction_family",
        )
        fixed_results = fixed_results.merge(
            interaction_results[
                [
                    "state",
                    "context",
                    "parameter",
                    "interaction_family_p_fdr_bh",
                    "interaction_family_reject_fdr_bh",
                    "interaction_family_p_holm",
                    "interaction_family_reject_holm",
                ]
            ],
            on=["state", "context", "parameter"],
            how="left",
        )
        fixed_results.to_csv(MIXED_RESULTS_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved fixed effects -> {MIXED_RESULTS_CSV}")

    if diagnostics_list:
        pd.DataFrame(diagnostics_list).to_csv(
            MODEL_DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig"
        )
        print(f"Saved convergence diagnostics -> {MODEL_DIAGNOSTICS_CSV}")

    if lrt_list:
        lrt_results = add_multiple_testing_columns(
            pd.DataFrame(lrt_list), "p_value", "lrt_family"
        )
        lrt_results.to_csv(LRT_RESULTS_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved nested-model LRTs -> {LRT_RESULTS_CSV}")

    if sample_size_list:
        pd.DataFrame(sample_size_list).to_csv(
            SAMPLE_SIZES_CSV, index=False, encoding="utf-8-sig"
        )
        print(f"Saved analysis-level sample sizes -> {SAMPLE_SIZES_CSV}")

    if not all_random_slopes:
        raise RuntimeError("No random slopes were produced.")

    random_slopes = pd.concat(all_random_slopes, ignore_index=True)
    random_slopes.to_csv(RANDOM_SLOPES_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved random slopes -> {RANDOM_SLOPES_CSV}")
    return random_slopes


# ============================================================
# 6. Plot helpers: LMM_final2 style
# ============================================================

def add_fig_line(fig, xdata, ydata, **kwargs):
    line = Line2D(xdata, ydata, transform=fig.transFigure, **kwargs)
    fig.add_artist(line)
    return line


def add_fig_rect(fig, xy, width, height, **kwargs):
    rect = Rectangle(xy, width, height, transform=fig.transFigure, **kwargs)
    fig.add_artist(rect)
    return rect


def compute_kde_density(values, x_grid, bw_adjust=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return None

    if np.nanstd(values) == 0:
        center = float(values[0])
        bandwidth = max((x_grid.max() - x_grid.min()) / 250.0, 1e-4)
        density = np.exp(-0.5 * ((x_grid - center) / bandwidth) ** 2)
        return density / (bandwidth * np.sqrt(2 * np.pi))

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
        if state not in STATE_ORDER or context not in CONTEXT_ORDER or source not in SOURCE_ORDER:
            continue

        density = compute_kde_density(g["re_slope"].values, x_grid, KDE_BW_ADJUST)
        if density is not None and np.isfinite(density).all():
            density_table[(state, context, source)] = density

    return density_table


def get_panel_values(df):
    value_table = {}
    grouped = df.groupby(["state", "context", "source"], observed=True)

    for (state, context, source), g in grouped:
        if state not in STATE_ORDER or context not in CONTEXT_ORDER or source not in SOURCE_ORDER:
            continue
        vals = pd.to_numeric(g["re_slope"], errors="coerce").dropna().values
        if len(vals) > 0:
            value_table[(state, context, source)] = vals

    return value_table


def get_panel_max_density(density_table, state, context):
    max_density = 0.0
    for source in SOURCE_ORDER:
        density = density_table.get((state, context, source))
        if density is not None:
            max_density = max(max_density, float(np.nanmax(density)))
    return max_density


def get_global_max_density(density_table):
    max_density = 0.0
    for density in density_table.values():
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
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def plot_random_slopes(random_slopes_df):
    required_cols = {"re_slope", "source", "state", "context"}
    missing_cols = required_cols - set(random_slopes_df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")

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

    n_rows = len(STATE_ORDER)
    n_cols = len(CONTEXT_ORDER)
    n_sources = len(SOURCE_ORDER)

    y_positions = np.arange(n_sources)[::-1]
    y_map = {source: y for source, y in zip(SOURCE_ORDER, y_positions)}

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(25.6, 21.2),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    plt.subplots_adjust(
        left=0.235,
        right=0.880,
        top=0.920,
        bottom=0.095,
        wspace=0.115,
        hspace=0.180,
    )

    for i, state in enumerate(STATE_ORDER):
        for j, context in enumerate(CONTEXT_ORDER):
            ax = axes[i, j]
            density_scale = get_density_scale(density_table, state, context)

            if DRAW_HUMAN_95_BAND:
                human_vals = value_table.get((state, context, "human"))
                if human_vals is not None:
                    interval = compute_human_95_interval(human_vals)
                    if interval is not None:
                        ax.axvspan(
                            interval[0],
                            interval[1],
                            color=HUMAN_BAND_COLOR,
                            alpha=HUMAN_95_ALPHA,
                            linewidth=0,
                            zorder=0,
                        )

            ax.axvline(
                0,
                color=ZERO_LINE_COLOR,
                linestyle=ZERO_LINE_STYLE,
                linewidth=ZERO_LINE_WIDTH,
                zorder=1,
            )

            for yy in y_positions:
                ax.hlines(yy, X_LIM[0], X_LIM[1], color="0.90", linewidth=0.50, zorder=0)

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

            if i == 0:
                ax.set_title(
                    CONTEXT_DISPLAY[context],
                    fontsize=18.0,
                    fontweight="bold",
                    pad=10,
                )

            ax.set_yticks(y_positions)
            if j == 0:
                ax.set_yticklabels([SOURCE_LABELS[source] for source in SOURCE_ORDER], fontsize=18.0)
                ax.tick_params(axis="y", pad=3)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)

            ax.set_xlim(X_LIM)
            ax.set_ylim(-0.70, n_sources - 0.05)

            if i != n_rows - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
                ax.spines["bottom"].set_visible(False)
            else:
                ax.set_xticks(X_TICKS)
                ax.set_xticklabels(["-0.2", "0", "0.2"])
                ax.tick_params(axis="x", labelsize=18.0)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if j != 0:
                ax.spines["left"].set_visible(False)
            else:
                ax.spines["left"].set_linewidth(0.55)

            ax.grid(False)
            ax.set_xlabel("")
            ax.set_ylabel("")

    for i, state in enumerate(STATE_ORDER):
        pos = axes[i, 0].get_position()
        y_mid = pos.y0 + pos.height / 2
        fig.text(
            0.145,
            y_mid,
            TRAIT_SHORT[state],
            ha="center",
            va="center",
            fontsize=26.0,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.30,rounding_size=0.12",
                "facecolor": "#F0F0F0",
                "edgecolor": "#D6D6D6",
                "linewidth": 0.8,
            },
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
        markersize=11,
            markerfacecolor=COLOR_MAPPING[source],
            markeredgecolor=COLOR_MAPPING[source],
            label=SOURCE_LABELS[source],
        )
        for source in SOURCE_ORDER
    ]

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.900, 0.505),
        frameon=False,
        title="Dataset",
        title_fontsize=18.0,
        fontsize=18.0,
        handlelength=0.9,
        handletextpad=0.7,
        labelspacing=0.66,
        borderaxespad=0,
    )

    add_fig_line(
        fig,
        [0.075, 0.075],
        [0.030, 0.054],
        color=ZERO_LINE_COLOR,
        linewidth=1.1,
        linestyle=ZERO_LINE_STYLE,
    )
    fig.text(0.085, 0.041, "= Zero slope", ha="left", va="center", fontsize=18.0, color="0.25")

    add_fig_rect(
        fig,
        (0.235, 0.030),
        0.015,
        0.020,
        color=HUMAN_BAND_COLOR,
        alpha=HUMAN_95_ALPHA,
        ec=HUMAN_BAND_COLOR,
    )
    fig.text(0.257, 0.041, "= Human 95% interval", ha="left", va="center", fontsize=18.0, color="0.25")

    pdf_path = VERSIONED_OUTPUT_BASENAME + ".pdf"
    fig.savefig(pdf_path, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"PDF saved -> {pdf_path}")


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":
    random_slopes = build_random_slopes()
    plot_random_slopes(random_slopes)
