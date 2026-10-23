#!/usr/bin/env python3
"""Rebuild two original-style SAE UMAP figures with held-out validation.

Figure 1: six independent draws of ten situations, with joint persona-situation
representations grouped by situation. Figure 2: baseline versus trait-amplified representations in one
shared discovery-fitted projection. UMAP is descriptive; quantitative claims
are computed in the pre-UMAP PCA space on held-out personas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from adjustText import adjust_text
    import umap
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: "
        "python -m pip install scikit-learn umap-learn matplotlib adjustText"
    ) from exc


TRAITS = ("E", "A", "C", "N", "O")
TRAIT_NAMES = {
    "E": "Extraversion",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "N": "Neuroticism",
    "O": "Openness",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = "./local_assets/llama31_opensae_outputs/sae"
    parser.add_argument("--database", default=f"{root}/persona_situation_sparse_features.sqlite")
    parser.add_argument("--personas", default="./local_assets/code/dataset/persona_profile_600_sampled.csv")
    parser.add_argument("--population-stats", default=f"{root}/feature_population_statistics.npz")
    parser.add_argument("--selected-features", default=f"{root}/selected_persona_features.csv")
    parser.add_argument("--output-dir", default=f"{root}/original_like_umaps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--figure1-personas", type=int, default=80)
    parser.add_argument("--figure1-situations", type=int, default=10)
    parser.add_argument("--figure2-personas", type=int, default=60)
    parser.add_argument("--figure2-situations", type=int, default=6)
    parser.add_argument("--modulation-scale", type=float, default=1.0)
    parser.add_argument("--skip-sensitivity", action="store_true")
    return parser.parse_args()


def stable_key(seed: int, *values: object) -> int:
    payload = "|".join([str(seed), *map(str, values)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def robust_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeError(f"Unable to decode CSV: {path}")


def id_column(table: pd.DataFrame) -> str:
    if "id_p" in table.columns:
        return "id_p"
    if "name" in table.columns:
        return "name"
    raise ValueError("Persona table requires id_p or name")


def load_joint_matrix(
    database: Path, eligible: np.ndarray, feature_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    connection = sqlite3.connect(database)
    n = int(connection.execute("SELECT COUNT(*) FROM sparse_features").fetchone()[0])
    if n != 600 * 62:
        raise RuntimeError(f"Expected 37,200 sparse records, found {n}")
    mapping = np.full(feature_size, -1, dtype=np.int32)
    mapping[eligible] = np.arange(len(eligible), dtype=np.int32)
    matrix = np.zeros((n, len(eligible)), dtype=np.float32)
    personas = np.empty(n, dtype=object)
    situations = np.empty(n, dtype=object)
    splits = np.empty(n, dtype=object)
    cursor = connection.execute(
        "SELECT id_p,id_s,split,indices,values_blob "
        "FROM sparse_features ORDER BY id_p,id_s"
    )
    for row, (id_p, id_s, split, indices_blob, values_blob) in enumerate(cursor):
        indices = np.frombuffer(indices_blob, dtype=np.int32)
        values = np.frombuffer(values_blob, dtype=np.float32)
        if len(indices) != len(values):
            raise RuntimeError(f"Corrupt sparse arrays at row {row}")
        valid_range = indices < feature_size
        columns = np.full(len(indices), -1, dtype=np.int32)
        columns[valid_range] = mapping[indices[valid_range]]
        keep = (columns >= 0) & (values > 0) & np.isfinite(values)
        np.add.at(matrix[row], columns[keep], values[keep])
        personas[row], situations[row], splits[row] = str(id_p), str(id_s), str(split)
    connection.close()
    return matrix, personas.astype(str), situations.astype(str), splits.astype(str)


def fit_discovery_projection(
    matrix: np.ndarray, discovery: np.ndarray, args: argparse.Namespace
) -> tuple[StandardScaler, PCA, object, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    discovery_scaled = scaler.fit_transform(matrix[discovery])
    components = min(args.pca_components, discovery_scaled.shape[0] - 1, discovery_scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=args.seed)
    discovery_pca = pca.fit_transform(discovery_scaled)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
        transform_seed=args.seed,
        n_jobs=1,
    )
    discovery_umap = reducer.fit_transform(discovery_pca)
    all_pca = pca.transform(scaler.transform(matrix))
    all_umap = np.empty((len(matrix), 2), dtype=np.float32)
    all_umap[discovery] = discovery_umap
    all_umap[~discovery] = reducer.transform(all_pca[~discovery])
    return scaler, pca, reducer, all_pca, all_umap


def variance_components(
    representation: np.ndarray, personas: np.ndarray, situations: np.ndarray
) -> dict[str, float]:
    grand = representation.mean(axis=0)
    total = float(np.sum((representation - grand) ** 2))

    def between(labels: np.ndarray) -> float:
        value = 0.0
        for label in np.unique(labels):
            group = representation[labels == label]
            value += len(group) * float(np.sum((group.mean(axis=0) - grand) ** 2))
        return value

    situation_ss = between(situations)
    persona_ss = between(personas)
    return {
        "total_ss": total,
        "situation_ss": situation_ss,
        "persona_ss": persona_ss,
        "situation_eta2": situation_ss / total,
        "persona_eta2": persona_ss / total,
        "situation_to_persona_ss_ratio": situation_ss / max(persona_ss, 1e-12),
    }


def neighborhood_purity(
    representation: np.ndarray,
    personas: np.ndarray,
    situations: np.ndarray,
    neighbors: int = 15,
) -> dict[str, float]:
    index = NearestNeighbors(n_neighbors=neighbors + 1).fit(representation).kneighbors(
        representation, return_distance=False
    )[:, 1:]
    return {
        "neighbors": neighbors,
        "same_situation_fraction": float(np.mean(situations[index] == situations[:, None])),
        "same_persona_fraction": float(np.mean(personas[index] == personas[:, None])),
    }


def mpl_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )


def save_bundle(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def confidence_ellipse(
    axis: plt.Axes,
    xy: np.ndarray,
    color: object,
    probability: float = 0.80,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Draw a robust ellipse and return its centre and full data-space bounds."""
    if len(xy) < 3:
        centre = np.median(xy, axis=0)
        return centre, (float(centre[0]), float(centre[0]), float(centre[1]), float(centre[1]))
    # Estimate the display ellipse from the central 90% under a coordinate-wise
    # median/MAD distance. Raw observations are never removed from the scatter.
    location = np.median(xy, axis=0)
    mad = np.median(np.abs(xy - location), axis=0)
    fallback = np.std(xy, axis=0, ddof=1)
    scale_axes = np.where(mad > 1e-8, 1.4826 * mad, np.maximum(fallback, 1e-8))
    robust_distance = np.sum(((xy - location) / scale_axes) ** 2, axis=1)
    cutoff = np.quantile(robust_distance, 0.90)
    core = xy[robust_distance <= cutoff]
    centre = np.mean(core, axis=0)
    covariance = np.cov(core, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return centre, (
            float(np.min(core[:, 0])), float(np.max(core[:, 0])),
            float(np.min(core[:, 1])), float(np.max(core[:, 1])),
        )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Chi-square quantile with 2 degrees of freedom: q = -2 log(1-p).
    scale = np.sqrt(-2.0 * np.log(1.0 - probability))
    width, height = 2.0 * scale * np.sqrt(eigenvalues)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = Ellipse(
        xy=centre,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.065,
        linewidth=1.30,
        zorder=1,
    )
    axis.add_patch(ellipse)
    theta = np.radians(angle)
    half_x = 0.5 * np.sqrt((width * np.cos(theta)) ** 2 + (height * np.sin(theta)) ** 2)
    half_y = 0.5 * np.sqrt((width * np.sin(theta)) ** 2 + (height * np.cos(theta)) ** 2)
    return centre, (
        float(centre[0] - half_x), float(centre[0] + half_x),
        float(centre[1] - half_y), float(centre[1] + half_y),
    )


def figure_one(
    coordinates: np.ndarray,
    personas: np.ndarray,
    situations: np.ndarray,
    heldout: np.ndarray,
    args: argparse.Namespace,
    output: Path,
) -> tuple[pd.DataFrame, list[list[str]]]:
    heldout_personas = sorted(np.unique(personas[heldout]), key=lambda x: stable_key(args.seed, "f1p", x))
    heldout_personas = heldout_personas[: args.figure1_personas]
    situation_order = np.asarray(sorted(np.unique(situations)), dtype=object)
    if args.figure1_situations > len(situation_order):
        raise ValueError("figure1-situations exceeds the number of available situations")
    # Six independent, reproducible draws. Situations may recur across draws,
    # which is intentional because each panel is a robustness resample.
    panels = []
    for panel_index in range(6):
        panel_rng = np.random.default_rng(stable_key(args.seed, "f1s", panel_index))
        chosen = panel_rng.choice(
            situation_order, size=args.figure1_situations, replace=False
        )
        panels.append(list(map(str, chosen)))
    plot_mask = heldout & np.isin(personas, heldout_personas)
    source_rows: list[pd.DataFrame] = []
    mpl_style()
    fig, axes = plt.subplots(
        2, 3, figsize=(183 / 25.4, 5.55), constrained_layout=True,
        sharex=True, sharey=True,
    )
    axes = axes.ravel()
    colors = [
        "#0F4D92", "#B64342", "#2F855A", "#8B5EA7", "#D97706",
        "#42949E", "#A65D57", "#667085", "#C45A87", "#7A8F27",
    ]
    displayed_situations = sorted({value for panel in panels for value in panel})
    global_xy = coordinates[plot_mask & np.isin(situations, displayed_situations)]
    display_bounds = [(
        float(global_xy[:, 0].min()), float(global_xy[:, 0].max()),
        float(global_xy[:, 1].min()), float(global_xy[:, 1].max()),
    )]
    panel_label_artists: list[list[object]] = []
    panel_label_anchors: list[np.ndarray] = []
    for panel_index, (axis, panel_situations) in enumerate(zip(axes, panels)):
        label_artists: list[object] = []
        label_anchors: list[np.ndarray] = []
        for color, situation in zip(colors, panel_situations):
            mask = plot_mask & (situations == situation)
            xy = coordinates[mask]
            centre, ellipse_bounds = confidence_ellipse(axis, xy, color, probability=0.80)
            display_bounds.append(ellipse_bounds)
            axis.scatter(
                xy[:, 0], xy[:, 1], s=7.5, color=color, alpha=0.56,
                linewidths=0, rasterized=False, zorder=2,
            )
            display_label = re.sub(r"\.(?:jpg|jpeg|png)$", "", str(situation), flags=re.I)
            label_anchors.append(np.asarray(centre, dtype=float))
            label_artists.append(
                axis.text(
                    float(centre[0]), float(centre[1]), display_label,
                    fontsize=5.0, fontweight="medium", color=color,
                    ha="left", va="bottom", zorder=6, clip_on=True,
                    bbox={
                        "boxstyle": "round,pad=0.11",
                        "fc": "white", "ec": "none", "alpha": 0.82,
                    },
                )
            )
            ids = np.flatnonzero(mask)
            source_rows.append(
                pd.DataFrame(
                    {
                        "panel": chr(97 + panel_index),
                        "id_p": personas[ids],
                        "id_s": situations[ids],
                        "display_label": display_label,
                        "color_hex": color,
                        "UMAP1": coordinates[ids, 0],
                        "UMAP2": coordinates[ids, 1],
                    }
                )
            )
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_title(
            f"Resample {panel_index + 1}", loc="left", fontsize=7.2,
            fontweight="normal", pad=3,
        )
        axis.text(-0.10, 1.04, chr(97 + panel_index), transform=axis.transAxes,
                  fontsize=9, fontweight="bold", ha="left", va="bottom")
        panel_label_artists.append(label_artists)
        panel_label_anchors.append(np.vstack(label_anchors))
    x_min = min(value[0] for value in display_bounds)
    x_max = max(value[1] for value in display_bounds)
    y_min = min(value[2] for value in display_bounds)
    y_max = max(value[3] for value in display_bounds)
    x_pad = max((x_max - x_min) * 0.045, 0.6)
    y_pad = max((y_max - y_min) * 0.045, 0.6)
    for axis in axes:
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
    # Adjust labels only after fixing the shared limits. Each panel is solved
    # independently; short leader lines identify labels that move noticeably.
    fig.canvas.draw()
    for axis, texts, anchors in zip(axes, panel_label_artists, panel_label_anchors):
        adjust_text(
            texts,
            x=anchors[:, 0], y=anchors[:, 1], ax=axis,
            expand=(1.18, 1.30),
            force_text=(0.45, 0.75),
            force_static=(0.20, 0.35),
            force_pull=(0.012, 0.020),
            force_explode=(0.35, 0.50),
            explode_radius="auto",
            max_move=(28, 34),
            ensure_inside_axes=True,
            expand_axes=False,
            prevent_crossings=True,
            min_arrow_len=5,
            only_move={
                "text": "xy", "static": "xy",
                "explode": "xy", "pull": "xy",
            },
            arrowprops={
                "arrowstyle": "-", "color": "#7A7A7A",
                "linewidth": 0.45, "alpha": 0.72,
                "shrinkA": 1.5, "shrinkB": 1.5,
            },
        )
    figure_key = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=3.5,
               markerfacecolor="#667085", markeredgewidth=0,
               label="persona–situation representation"),
        Ellipse((0, 0), 1, 0.5, facecolor="#667085", edgecolor="#667085",
                alpha=0.08, label="central 80% ellipse"),
    ]
    fig.legend(
        handles=figure_key, loc="outside upper center", ncol=3,
        fontsize=5.8, handletextpad=0.5, columnspacing=1.2,
    )
    save_bundle(fig, output / "umap_6subplots_nature_optimized")
    source = pd.concat(source_rows, ignore_index=True)
    source.to_csv(output / "umap_6subplots_nature_optimized_source_data.csv", index=False, encoding="utf-8-sig")
    return source, panels


def persona_trait_zscores(persona_table: pd.DataFrame, discovery_ids: set[str]) -> pd.DataFrame:
    result = persona_table.copy().set_index("id_p")
    for trait in TRAITS:
        column = f"{trait}_trait"
        reference = result.loc[result.index.isin(discovery_ids), column].astype(float)
        scale = float(reference.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid discovery SD for {column}")
        result[f"{trait}_z"] = ((result[column].astype(float) - reference.mean()) / scale).clip(-3, 3)
    return result


def modulation_deltas(
    persona_ids: np.ndarray,
    persona_scores: pd.DataFrame,
    selected: pd.DataFrame,
    eligible: np.ndarray,
    population_sd: np.ndarray,
    scale: float,
) -> np.ndarray:
    column_map = {int(feature): index for index, feature in enumerate(eligible)}
    unique_personas = np.unique(persona_ids)
    row_map = {persona: row for row, persona in enumerate(unique_personas)}
    deltas = np.zeros((len(unique_personas), len(eligible)), dtype=np.float32)
    for trait in TRAITS:
        rows = selected[selected["trait"] == trait]
        z = persona_scores.loc[unique_personas, f"{trait}_z"].to_numpy(dtype=np.float32)
        for _, feature in rows.iterrows():
            feature_id = int(feature["feature_id"])
            if feature_id not in column_map:
                continue
            delta = (
                z
                * float(feature["direction_sign"])
                * float(population_sd[feature_id])
                * float(scale)
            )
            deltas[:, column_map[feature_id]] += delta
    return deltas[np.asarray([row_map[value] for value in persona_ids])]


def figure_two(
    baseline_coordinates: np.ndarray,
    post_coordinates: np.ndarray,
    personas: np.ndarray,
    situations: np.ndarray,
    selected_indices: np.ndarray,
    baseline_dispersion: np.ndarray,
    post_dispersion: np.ndarray,
    args: argparse.Namespace,
    output: Path,
) -> pd.DataFrame:
    ids = selected_indices
    selected_situations = sorted(np.unique(situations[ids]), key=lambda x: stable_key(args.seed, "f2s", x))
    colors = plt.cm.Set2(np.linspace(0.05, 0.95, len(selected_situations)))
    mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 3.05), constrained_layout=True, sharex=True, sharey=True)
    all_xy = np.vstack([baseline_coordinates, post_coordinates])
    xpad = max(np.ptp(all_xy[:, 0]) * 0.05, 0.5)
    ypad = max(np.ptp(all_xy[:, 1]) * 0.05, 0.5)
    limits = (all_xy[:, 0].min() - xpad, all_xy[:, 0].max() + xpad,
              all_xy[:, 1].min() - ypad, all_xy[:, 1].max() + ypad)
    source = []
    for axis, coordinates, condition, panel in zip(
        axes, (baseline_coordinates, post_coordinates), ("Baseline", "Trait-amplified"), ("a", "b")
    ):
        for color, situation in zip(colors, selected_situations):
            local = situations[ids] == situation
            xy = coordinates[local]
            axis.scatter(xy[:, 0], xy[:, 1], s=8, color=color, alpha=0.68, linewidths=0, label=situation)
            centre = xy.mean(axis=0)
            source.append(
                pd.DataFrame(
                    {
                        "condition": condition,
                        "id_p": personas[ids][local],
                        "id_s": situations[ids][local],
                        "UMAP1": xy[:, 0],
                        "UMAP2": xy[:, 1],
                    }
                )
            )
        axis.set_xlim(limits[:2])
        axis.set_ylim(limits[2:])
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_title(condition, loc="left", fontsize=8)
        axis.text(-0.08, 1.04, panel, transform=axis.transAxes, fontsize=9,
                  fontweight="bold", ha="left", va="bottom")
    median_change = 100 * (np.median(post_dispersion / baseline_dispersion) - 1)
    axes[1].text(0.98, 0.03, f"Median within-situation dispersion\nchange: {median_change:+.1f}%",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=6.5,
                 bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#B8B8B8", "lw": 0.5})
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(6, len(labels)), fontsize=6)
    save_bundle(fig, output / "baseline_vs_trait_amplified_umap")
    table = pd.concat(source, ignore_index=True)
    table.to_csv(output / "baseline_vs_trait_amplified_umap_source_data.csv", index=False, encoding="utf-8-sig")
    return table


def sensitivity_analysis(
    pca_scores: np.ndarray,
    personas: np.ndarray,
    situations: np.ndarray,
    discovery: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    discovery_people = sorted(np.unique(personas[discovery]), key=lambda x: stable_key(seed, "sens_train", x))[:100]
    heldout_people = sorted(np.unique(personas[~discovery]), key=lambda x: stable_key(seed, "sens_test", x))[:60]
    train = discovery & np.isin(personas, discovery_people)
    test = (~discovery) & np.isin(personas, heldout_people)
    configurations = [(42, 15), (42, 30), (42, 50), (123, 30), (2026, 30)]
    rows = []
    for random_state, neighbors in configurations:
        reducer = umap.UMAP(
            n_components=2, n_neighbors=neighbors, min_dist=0.15, metric="cosine",
            random_state=random_state, transform_seed=random_state, n_jobs=1,
        )
        reducer.fit(pca_scores[train])
        embedded = reducer.transform(pca_scores[test])
        purity = neighborhood_purity(
            embedded, personas[test], situations[test], neighbors=15
        )
        rng = np.random.default_rng(stable_key(seed, "trust", random_state, neighbors))
        sample = rng.choice(len(embedded), size=min(1500, len(embedded)), replace=False)
        rows.append(
            {
                "random_state": random_state,
                "n_neighbors": neighbors,
                "min_dist": 0.15,
                "train_personas": len(discovery_people),
                "heldout_personas": len(heldout_people),
                "trustworthiness_15nn": float(
                    trustworthiness(pca_scores[test][sample], embedded[sample], n_neighbors=15, metric="cosine")
                ),
                **purity,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = arguments()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    population = np.load(args.population_stats, allow_pickle=False)
    eligible = np.flatnonzero(
        (population["activation_rate"] >= 0.01) & (population["sd_activation"] > 0)
    ).astype(np.int32)
    matrix, personas, situations, splits = load_joint_matrix(
        Path(args.database), eligible, len(population["activation_rate"])
    )
    discovery = splits == "discovery"
    if discovery.sum() != 427 * 62 or (~discovery).sum() != 173 * 62:
        raise RuntimeError("Unexpected discovery/held-out record counts")

    persona_table = robust_csv(Path(args.personas))
    persona_id = id_column(persona_table)
    persona_table[persona_id] = persona_table[persona_id].astype(str)
    persona_table = persona_table.rename(columns={persona_id: "id_p"})
    required = ["id_p", *[f"{trait}_trait" for trait in TRAITS]]
    missing = set(required) - set(persona_table.columns)
    if missing:
        raise ValueError(f"Persona table lacks columns: {sorted(missing)}")
    persona_table = persona_table[required].drop_duplicates("id_p")
    if persona_table.set_index("id_p").loc[np.unique(personas), required[1:]].isna().any().any():
        raise ValueError("Missing Big Five trait scores")

    scaler, pca, reducer, pca_scores, umap_coordinates = fit_discovery_projection(
        matrix, discovery, args
    )
    heldout = ~discovery
    rng = np.random.default_rng(args.seed)
    trust_sample = rng.choice(np.flatnonzero(heldout), size=min(2000, heldout.sum()), replace=False)
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "records": len(matrix), "personas": int(len(np.unique(personas))),
            "situations": int(len(np.unique(situations))), "eligible_features": int(len(eligible)),
            "discovery_personas": int(len(np.unique(personas[discovery]))),
            "heldout_personas": int(len(np.unique(personas[heldout]))),
        },
        "projection": {
            "fit": "StandardScaler, PCA and UMAP fitted on discovery personas only; held-out personas transformed",
            "pca_components": int(pca.n_components_),
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "umap_n_neighbors": args.n_neighbors, "umap_min_dist": args.min_dist,
            "umap_metric": "cosine", "random_state": args.seed,
            "heldout_trustworthiness_15nn": float(
                trustworthiness(pca_scores[trust_sample], umap_coordinates[trust_sample],
                                n_neighbors=15, metric="cosine")
            ),
        },
        "heldout_baseline_variance": variance_components(
            pca_scores[heldout], personas[heldout], situations[heldout]
        ),
        "heldout_baseline_neighborhood": neighborhood_purity(
            pca_scores[heldout], personas[heldout], situations[heldout], neighbors=15
        ),
    }

    _, panel_situations = figure_one(
        umap_coordinates, personas, situations, heldout, args, output
    )

    discovery_ids = set(np.unique(personas[discovery]))
    scores = persona_trait_zscores(persona_table, discovery_ids)
    selected = robust_csv(Path(args.selected_features))
    heldout_indices = np.flatnonzero(heldout)
    delta = modulation_deltas(
        personas[heldout], scores, selected, eligible,
        population["sd_activation"], args.modulation_scale,
    )
    modulated_matrix = np.maximum(matrix[heldout] + delta, 0.0)
    modulated_pca = pca.transform(scaler.transform(modulated_matrix))

    baseline_dispersion, post_dispersion = [], []
    dispersion_rows = []
    for situation in sorted(np.unique(situations)):
        local = situations[heldout] == situation
        baseline_value = float(np.mean(pdist(pca_scores[heldout][local], metric="euclidean")))
        post_value = float(np.mean(pdist(modulated_pca[local], metric="euclidean")))
        baseline_dispersion.append(baseline_value)
        post_dispersion.append(post_value)
        dispersion_rows.append(
            {"id_s": situation, "baseline_mean_pairwise_distance": baseline_value,
             "post_mean_pairwise_distance": post_value, "ratio": post_value / baseline_value}
        )
    baseline_dispersion = np.asarray(baseline_dispersion)
    post_dispersion = np.asarray(post_dispersion)
    try:
        test = wilcoxon(post_dispersion, baseline_dispersion, alternative="greater")
    except TypeError:
        test = wilcoxon(post_dispersion, baseline_dispersion)
    dispersion_table = pd.DataFrame(dispersion_rows)
    dispersion_table.to_csv(output / "within_situation_dispersion_heldout.csv", index=False, encoding="utf-8-sig")
    metrics["heldout_trait_amplification"] = {
        "definition": "Add z-scored persona trait values times discovery-feature direction and SD to top-20 features per trait",
        "scale": args.modulation_scale,
        "median_dispersion_ratio": float(np.median(post_dispersion / baseline_dispersion)),
        "mean_dispersion_ratio": float(np.mean(post_dispersion / baseline_dispersion)),
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p_value": float(test.pvalue),
        "n_situations": len(baseline_dispersion),
        "post_variance": variance_components(
            modulated_pca, personas[heldout], situations[heldout]
        ),
    }

    chosen_people = sorted(np.unique(personas[heldout]), key=lambda x: stable_key(args.seed, "f2p", x))[: args.figure2_personas]
    chosen_situations = sorted(np.unique(situations), key=lambda x: stable_key(args.seed, "f2s", x))[: args.figure2_situations]
    selected_global = np.flatnonzero(
        heldout & np.isin(personas, chosen_people) & np.isin(situations, chosen_situations)
    )
    heldout_position = {global_index: local for local, global_index in enumerate(heldout_indices)}
    selected_local = np.asarray([heldout_position[index] for index in selected_global], dtype=int)
    post_umap = reducer.transform(modulated_pca[selected_local])
    figure_two(
        umap_coordinates[selected_global], post_umap, personas, situations,
        selected_global, baseline_dispersion, post_dispersion, args, output,
    )

    if not args.skip_sensitivity:
        sensitivity = sensitivity_analysis(pca_scores, personas, situations, discovery, args.seed)
        sensitivity.to_csv(output / "umap_parameter_sensitivity.csv", index=False, encoding="utf-8-sig")
        metrics["sensitivity"] = sensitivity.to_dict(orient="records")

    metrics["figure1_situation_panels"] = panel_situations
    metrics["interpretation_limits"] = [
        "UMAP is a descriptive nonlinear projection; quantitative claims use held-out pre-UMAP PCA representations.",
        "Trait amplification is a prespecified representational perturbation, not evidence that the unmodified model naturally uses those features causally.",
        "No UMAP parameter was selected by visual appearance.",
    ]
    (output / "original_like_umap_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved two figure bundles and source data to: {output}")


if __name__ == "__main__":
    main()
