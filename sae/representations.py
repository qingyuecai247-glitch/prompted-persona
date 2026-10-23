#!/usr/bin/env python3
"""Confirmatory persona-by-situation analyses for Llama-3.1/OpenSAE.

This single file consumes the sparse-feature SQLite database produced by
``sae/pipeline.py extract`` and implements the complete analysis
chain used in the manuscript:

1. two-factor representational variance decomposition and distance contrasts;
2. held-out trait association validation and trait decoding;
3. held-out trait-by-situation interaction tests;
4. feature overlap, situation profiles and top-activating prompt examples;
5. causal-intervention summaries when ``intervention_results_long.csv`` exists.

The discovery split is used only for preprocessing, feature selection inputs
and decoder training.  Confirmatory statistics are computed on held-out
personas.  Pooled estimates are emitted separately as descriptive sensitivity
analyses.  UMAP is intentionally not used for inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

# The shared server's home partition is full.  Keep all plotting/joblib caches
# out of ~/.cache without overwriting an explicit user setting.
_SERVER_ROOT = Path("./local_assets")
_CACHE_ROOT = _SERVER_ROOT / ".cache" if _SERVER_ROOT.exists() else Path(tempfile.gettempdir()) / "sae-confirmatory-cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_CACHE_ROOT / "joblib"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - exercised on the server
    raise SystemExit(
        "Missing dependency. Install inside the opensae environment with:\n"
        "python -m pip install numpy pandas scipy scikit-learn matplotlib"
    ) from exc


# Editable text in SVG is mandatory for manuscript figures.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

TRAITS = ("E", "A", "C", "N", "O")
TRAIT_NAMES = {
    "E": "Extraversion",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "N": "Neuroticism",
    "O": "Openness",
}
TRAIT_ALIASES = {
    "E": ("E_trait", "Extraversion", "extraversion"),
    "A": ("A_trait", "Agreeableness", "agreeableness"),
    "C": ("C_trait", "Conscientiousness", "conscientiousness"),
    "N": ("N_trait", "Neuroticism", "neuroticism"),
    "O": ("O_trait", "Openness", "openness"),
}
DIAMONDS = ("dut", "int", "adv", "mat", "pos", "neg", "dec", "soc")
DIAMOND_NAMES = {
    "dut": "Duty",
    "int": "Intellect",
    "adv": "Adversity",
    "mat": "Mating",
    "pos": "pOsitivity",
    "neg": "Negativity",
    "dec": "Deception",
    "soc": "Sociality",
}
PALETTE = {
    "blue": "#0F4D92",
    "orange": "#D97706",
    "red": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "grey": "#767676",
    "light": "#D8D8D8",
    "black": "#272727",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_rng(seed: int, label: str) -> np.random.Generator:
    import hashlib

    digest = hashlib.blake2b(f"{seed}|{label}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def read_csv_robust(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"Could not decode {path}: {' | '.join(errors)}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    result = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    return tuple(np.quantile(finite, [alpha, 1 - alpha]).tolist())


def ols_hc3(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients with HC3 standard errors, without statsmodels."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, bread, x)
    adjusted = residual / np.maximum(1.0 - leverage, 1e-8)
    meat = x.T @ (x * np.square(adjusted)[:, None])
    covariance = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2 * stats.norm.sf(np.abs(z))
    return beta, se, p, covariance


def ols_two_way_cluster(
    y: np.ndarray,
    design: np.ndarray,
    cluster_a: np.ndarray,
    cluster_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """OLS with Cameron--Gelbach--Miller two-way cluster covariance."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    scores = x * residual[:, None]

    def cluster_meat(labels: np.ndarray) -> tuple[np.ndarray, int]:
        _, inverse = np.unique(labels.astype(str), return_inverse=True)
        groups = int(inverse.max()) + 1
        sums = np.zeros((groups, k), dtype=float)
        np.add.at(sums, inverse, scores)
        return sums.T @ sums, groups

    meat_a, groups_a = cluster_meat(cluster_a)
    meat_b, groups_b = cluster_meat(cluster_b)
    pair_labels = np.char.add(np.char.add(cluster_a.astype(str), "\x1f"), cluster_b.astype(str))
    meat_pair, groups_pair = cluster_meat(pair_labels)

    def correction(groups: int) -> float:
        if groups <= 1 or n <= k:
            return 1.0
        return (groups / (groups - 1.0)) * ((n - 1.0) / (n - k))

    meat = correction(groups_a) * meat_a + correction(groups_b) * meat_b - correction(groups_pair) * meat_pair
    covariance = bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2.0
    return beta, covariance


def bootstrap_metric(
    y: np.ndarray,
    prediction: np.ndarray,
    metric,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(y)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        ids = rng.integers(0, n, size=n)
        try:
            estimates[index] = metric(y[ids], prediction[ids])
        except Exception:
            estimates[index] = np.nan
    return percentile_ci(estimates)


def persona_id_column(table: pd.DataFrame) -> str:
    for candidate in ("id_p", "name", "persona_id"):
        if candidate in table.columns:
            return candidate
    raise ValueError(f"Persona table requires id_p or name; found {list(table.columns)}")


def numeric_trait(series: pd.Series, column: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().all():
        return converted.astype(float)
    mapping = {
        "very low": 1.0,
        "low": 2.0,
        "medium": 3.0,
        "moderate": 3.0,
        "high": 4.0,
        "very high": 5.0,
    }
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(mapping)
    if mapped.isna().any():
        examples = series[mapped.isna()].astype(str).unique()[:5]
        raise ValueError(f"Trait column {column!r} is not numeric; unknown values: {examples}")
    return mapped.astype(float)


def load_personas(path: Path) -> pd.DataFrame:
    table = read_csv_robust(path)
    identifier = persona_id_column(table)
    table = table.rename(columns={identifier: "id_p"}).copy()
    table["id_p"] = table["id_p"].astype(str)
    for trait in TRAITS:
        source = next((name for name in TRAIT_ALIASES[trait] if name in table.columns), None)
        if source is None:
            raise ValueError(f"No score column found for {TRAIT_NAMES[trait]}")
        table[f"{trait}_trait"] = numeric_trait(table[source], source)
    if table["id_p"].duplicated().any():
        raise ValueError("Persona identifiers must be unique")
    return table


def load_situation_features(path: Path) -> pd.DataFrame:
    table = read_csv_robust(path)
    if "id_s" not in table.columns:
        raise ValueError("Situation-feature table requires id_s")
    table["id_s"] = table["id_s"].astype(str)
    lower = {str(column).lower(): column for column in table.columns}
    renamed = {}
    for dimension in DIAMONDS:
        if dimension not in lower:
            raise ValueError(f"Missing DIAMONDS dimension {dimension!r} in {path}")
        renamed[lower[dimension]] = dimension
    table = table.rename(columns=renamed)
    table[list(DIAMONDS)] = table[list(DIAMONDS)].apply(pd.to_numeric, errors="raise")
    if table["id_s"].duplicated().any():
        raise ValueError("Situation identifiers must be unique")
    return table


@dataclass
class JointData:
    personas: np.ndarray
    situations: np.ndarray
    splits: np.ndarray
    feature_ids: np.ndarray
    raw: np.ndarray  # shape: persona x situation x feature


def database_metadata(database: Path) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sparse_features'"
        ).fetchone()
        if not exists:
            raise RuntimeError(f"{database} does not contain table sparse_features")
        rows = connection.execute(
            "SELECT id_p,id_s,split FROM sparse_features ORDER BY id_p,id_s"
        ).fetchall()
    if not rows:
        raise RuntimeError("Sparse-feature database is empty")
    return pd.DataFrame(rows, columns=["id_p", "id_s", "split"]).astype(str)


def load_joint_tensor(database: Path, feature_ids: np.ndarray) -> JointData:
    meta = database_metadata(database)
    personas = np.sort(meta["id_p"].unique())
    situations = np.sort(meta["id_s"].unique())
    expected = len(personas) * len(situations)
    if len(meta) != expected or meta.duplicated(["id_p", "id_s"]).any():
        counts = meta.groupby("id_p")["id_s"].nunique()
        raise RuntimeError(
            "Two-factor decomposition requires a complete balanced persona-by-situation "
            f"design. Found {len(meta)} rows, expected {expected}; persona counts range "
            f"from {counts.min()} to {counts.max()}."
        )
    split_counts = meta.groupby("id_p")["split"].nunique()
    if (split_counts != 1).any():
        raise RuntimeError("Split leakage: at least one persona appears in multiple splits")
    split_map = meta.groupby("id_p")["split"].first().to_dict()
    p_index = {value: index for index, value in enumerate(personas)}
    s_index = {value: index for index, value in enumerate(situations)}
    feature_ids = np.asarray(sorted(set(map(int, feature_ids))), dtype=np.int32)
    if len(feature_ids) == 0:
        raise ValueError("No feature IDs supplied")
    feature_map = {int(feature): index for index, feature in enumerate(feature_ids)}
    tensor = np.zeros((len(personas), len(situations), len(feature_ids)), dtype=np.float32)
    with sqlite3.connect(database) as connection:
        cursor = connection.execute(
            "SELECT id_p,id_s,indices,values_blob FROM sparse_features ORDER BY id_p,id_s"
        )
        for id_p, id_s, indices_blob, values_blob in cursor:
            indices = np.frombuffer(indices_blob, dtype=np.int32)
            values = np.frombuffer(values_blob, dtype=np.float32)
            if len(indices) != len(values):
                raise RuntimeError(f"Corrupt sparse row: {id_p}, {id_s}")
            row = tensor[p_index[str(id_p)], s_index[str(id_s)]]
            for feature, value in zip(indices, values):
                column = feature_map.get(int(feature))
                if column is not None and np.isfinite(value) and value > 0:
                    row[column] += float(value)
    return JointData(
        personas=personas,
        situations=situations,
        splits=np.asarray([split_map[p] for p in personas]),
        feature_ids=feature_ids,
        raw=tensor,
    )


def two_factor_decomposition(tensor: np.ndarray) -> dict[str, float]:
    if tensor.ndim != 3:
        raise ValueError("Expected persona x situation x feature tensor")
    n_personas, n_situations, _ = tensor.shape
    grand = tensor.mean(axis=(0, 1), dtype=np.float64)
    persona_mean = tensor.mean(axis=1, dtype=np.float64)
    situation_mean = tensor.mean(axis=0, dtype=np.float64)
    ss_total = float(np.square(tensor.astype(np.float64) - grand).sum())
    ss_persona = float(n_situations * np.square(persona_mean - grand).sum())
    ss_situation = float(n_personas * np.square(situation_mean - grand).sum())
    ss_residual = max(ss_total - ss_persona - ss_situation, 0.0)
    if ss_total <= 0:
        raise ValueError("Representation has zero total variance")
    return {
        "ss_total": ss_total,
        "ss_persona": ss_persona,
        "ss_situation": ss_situation,
        "ss_interaction_plus_residual": ss_residual,
        "eta2_persona": ss_persona / ss_total,
        "eta2_situation": ss_situation / ss_total,
        "eta2_interaction_plus_residual": ss_residual / ss_total,
        "situation_to_persona_ratio": ss_situation / max(ss_persona, 1e-15),
    }


def bootstrap_decomposition(
    tensor: np.ndarray,
    repetitions: int,
    seed: int,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Two-way cluster bootstrap by independently resampling personas/situations."""
    p, s, _ = tensor.shape
    rng = stable_rng(seed, "two_way_variance_bootstrap")
    rows: list[dict[str, float]] = []
    for start in range(0, repetitions, batch_size):
        current = min(batch_size, repetitions - start)
        for _ in range(current):
            p_ids = rng.integers(0, p, size=p)
            s_ids = rng.integers(0, s, size=s)
            sample = tensor[p_ids][:, s_ids]
            rows.append(two_factor_decomposition(sample))
    return pd.DataFrame(rows)


def standardize_and_project(
    data: JointData,
    pca_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    p, s, f = data.raw.shape
    flat = data.raw.reshape(p * s, f)
    discovery_rows = np.repeat(data.splits == "discovery", s)
    if discovery_rows.sum() < 20:
        raise RuntimeError("At least 20 discovery rows are required")
    scaler = StandardScaler().fit(flat[discovery_rows])
    z = scaler.transform(flat).astype(np.float32)
    components = min(pca_components, f, int(discovery_rows.sum()) - 1)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(z[discovery_rows])
    scores = pca.transform(z).astype(np.float32).reshape(p, s, components)
    return z.reshape(p, s, f), scores, scaler, pca


def distance_group_means(tensor: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    p, s, _ = tensor.shape
    within_situation = np.asarray(
        [float(pdist(tensor[:, j, :], metric=metric).mean()) for j in range(s)]
    )
    within_persona = np.asarray(
        [float(pdist(tensor[i, :, :], metric=metric).mean()) for i in range(p)]
    )
    return within_situation, within_persona


def distance_contrasts(
    tensor: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    source_rows = []
    for metric in ("euclidean", "cosine"):
        same_situation, same_persona = distance_group_means(tensor, metric)
        for index, value in enumerate(same_situation):
            source_rows.append(
                {"metric": metric, "comparison": "different_personas_same_situation", "group": index, "mean_distance": value}
            )
        for index, value in enumerate(same_persona):
            source_rows.append(
                {"metric": metric, "comparison": "same_persona_different_situations", "group": index, "mean_distance": value}
            )
        rng = stable_rng(seed, f"distance_{metric}")
        boot = np.empty((repetitions, 3), dtype=float)
        for b in range(repetitions):
            a = same_situation[rng.integers(0, len(same_situation), len(same_situation))].mean()
            c = same_persona[rng.integers(0, len(same_persona), len(same_persona))].mean()
            boot[b] = (a, c, c - a)
        a_mean, c_mean = same_situation.mean(), same_persona.mean()
        difference = c_mean - a_mean
        p_value = 2 * min(
            (1 + np.sum(boot[:, 2] <= 0)) / (repetitions + 1),
            (1 + np.sum(boot[:, 2] >= 0)) / (repetitions + 1),
        )
        a_ci = percentile_ci(boot[:, 0])
        c_ci = percentile_ci(boot[:, 1])
        d_ci = percentile_ci(boot[:, 2])
        summary_rows.append(
            {
                "metric": metric,
                "different_personas_same_situation_mean": a_mean,
                "different_personas_same_situation_ci_low": a_ci[0],
                "different_personas_same_situation_ci_high": a_ci[1],
                "same_persona_different_situations_mean": c_mean,
                "same_persona_different_situations_ci_low": c_ci[0],
                "same_persona_different_situations_ci_high": c_ci[1],
                "cross_situation_minus_within_situation": difference,
                "contrast_ci_low": d_ci[0],
                "contrast_ci_high": d_ci[1],
                "distance_ratio": c_mean / max(a_mean, 1e-15),
                "bootstrap_two_sided_p": min(float(p_value), 1.0),
                "bootstrap_repetitions": repetitions,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(source_rows)


def aggregate_personas(data: JointData) -> pd.DataFrame:
    means = data.raw.mean(axis=1)
    result = pd.DataFrame(means, columns=[f"feature_{f}" for f in data.feature_ids])
    result.insert(0, "split", data.splits)
    result.insert(0, "id_p", data.personas)
    return result


def association_validation(
    persona_features: pd.DataFrame,
    personas: pd.DataFrame,
    selected: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    merged = persona_features.merge(personas, on="id_p", validate="one_to_one")
    discovery = merged[merged["split"] == "discovery"].copy()
    heldout = merged[merged["split"] == "heldout"].copy()
    trait_cols = [f"{trait}_trait" for trait in TRAITS]
    trait_mean = discovery[trait_cols].mean()
    trait_sd = discovery[trait_cols].std(ddof=0).replace(0, np.nan)
    for frame in (discovery, heldout):
        frame.loc[:, trait_cols] = (frame[trait_cols] - trait_mean) / trait_sd
    rows = []
    for record in selected.itertuples(index=False):
        trait = str(record.trait)
        feature_id = int(record.feature_id)
        feature = f"feature_{feature_id}"
        if feature not in heldout.columns:
            raise KeyError(f"Selected feature {feature_id} was not loaded")
        ref_mean = float(discovery[feature].mean())
        ref_sd = float(discovery[feature].std(ddof=0))
        if ref_sd <= 0:
            continue
        y = ((heldout[feature] - ref_mean) / ref_sd).to_numpy(dtype=float)
        target = heldout[f"{trait}_trait"].to_numpy(dtype=float)
        if np.std(target) <= 0 or np.std(y) <= 0:
            marginal_r, marginal_p = np.nan, np.nan
        else:
            marginal_r, marginal_p = stats.pearsonr(target, y)
        design = np.column_stack([np.ones(len(heldout)), heldout[trait_cols].to_numpy(dtype=float)])
        beta, se, p_values, _ = ols_hc3(y, design)
        coefficient_index = 1 + TRAITS.index(trait)
        partial_beta = float(beta[coefficient_index])
        partial_p = float(p_values[coefficient_index])
        analytic_ci = (
            partial_beta - 1.96 * float(se[coefficient_index]),
            partial_beta + 1.96 * float(se[coefficient_index]),
        )
        rng = stable_rng(seed, f"association_{trait}_{feature_id}")
        boot_r = np.empty(repetitions, dtype=float)
        n = len(y)
        for b in range(repetitions):
            ids = rng.integers(0, n, size=n)
            if np.std(target[ids]) <= 0 or np.std(y[ids]) <= 0:
                boot_r[b] = np.nan
            else:
                boot_r[b] = np.corrcoef(target[ids], y[ids])[0, 1]
        r_ci = percentile_ci(boot_r)
        rows.append(
            {
                "trait": trait,
                "trait_name": TRAIT_NAMES[trait],
                "feature_id": feature_id,
                "discovery_rank": int(record.rank),
                "discovery_r": float(record.correlation),
                "heldout_n_personas": len(heldout),
                "heldout_marginal_r": marginal_r,
                "heldout_marginal_r_ci_low": r_ci[0],
                "heldout_marginal_r_ci_high": r_ci[1],
                "heldout_marginal_p": marginal_p,
                "heldout_partial_beta": partial_beta,
                "heldout_partial_ci_low": float(analytic_ci[0]),
                "heldout_partial_ci_high": float(analytic_ci[1]),
                "heldout_partial_p": partial_p,
                "direction_replicated": bool(np.sign(marginal_r) == np.sign(record.correlation)) if np.isfinite(marginal_r) else False,
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["heldout_marginal_q"] = bh_adjust(result["heldout_marginal_p"])
        result["heldout_partial_q"] = bh_adjust(result["heldout_partial_p"])
    return result


def trait_decoding(
    persona_features: pd.DataFrame,
    personas: pd.DataFrame,
    selected: pd.DataFrame,
    repetitions: int,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = persona_features.merge(personas, on="id_p", validate="one_to_one")
    discovery = merged[merged["split"] == "discovery"].copy()
    heldout = merged[merged["split"] == "heldout"].copy()
    rows, predictions = [], []
    alphas = np.logspace(-4, 4, 33)
    for trait in TRAITS:
        feature_ids = selected.loc[selected["trait"] == trait, "feature_id"].astype(int).tolist()
        columns = [f"feature_{feature}" for feature in feature_ids]
        scaler = StandardScaler().fit(discovery[columns])
        x_train = scaler.transform(discovery[columns])
        x_test = scaler.transform(heldout[columns])
        y_train = discovery[f"{trait}_trait"].to_numpy(dtype=float)
        y_test = heldout[f"{trait}_trait"].to_numpy(dtype=float)
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        model = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error").fit(x_train, y_train)
        predicted = model.predict(x_test)
        r2 = float(r2_score(y_test, predicted)) if np.std(y_test) > 0 else np.nan
        pearson = float(stats.pearsonr(y_test, predicted).statistic) if np.std(y_test) > 0 and np.std(predicted) > 0 else np.nan
        spearman = float(stats.spearmanr(y_test, predicted).statistic) if np.std(y_test) > 0 and np.std(predicted) > 0 else np.nan
        mae = float(mean_absolute_error(y_test, predicted))
        rng = stable_rng(seed, f"decode_{trait}")
        r2_ci = bootstrap_metric(y_test, predicted, r2_score, repetitions, rng)
        r_ci = bootstrap_metric(
            y_test,
            predicted,
            lambda a, b: stats.pearsonr(a, b).statistic if np.std(a) > 0 and np.std(b) > 0 else np.nan,
            repetitions,
            rng,
        )
        null = np.empty(permutations, dtype=float)
        for index in range(permutations):
            permuted = rng.permutation(y_test)
            null[index] = stats.pearsonr(permuted, predicted).statistic if np.std(permuted) > 0 and np.std(predicted) > 0 else 0.0
        permutation_p = float((1 + np.sum(np.abs(null) >= abs(pearson))) / (permutations + 1)) if np.isfinite(pearson) else np.nan
        rows.append(
            {
                "trait": trait,
                "trait_name": TRAIT_NAMES[trait],
                "n_discovery": len(discovery),
                "n_heldout": len(heldout),
                "n_features": len(columns),
                "ridge_alpha_selected_in_discovery": float(model.alpha_),
                "heldout_r2": r2,
                "heldout_r2_ci_low": r2_ci[0],
                "heldout_r2_ci_high": r2_ci[1],
                "heldout_pearson_r": pearson,
                "heldout_pearson_ci_low": r_ci[0],
                "heldout_pearson_ci_high": r_ci[1],
                "heldout_spearman_rho": spearman,
                "heldout_mae": mae,
                "permutation_p": permutation_p,
            }
        )
        predictions.extend(
            {
                "id_p": id_p,
                "trait": trait,
                "observed": observed,
                "predicted": estimate,
            }
            for id_p, observed, estimate in zip(heldout["id_p"], y_test, predicted)
        )
    result = pd.DataFrame(rows)
    result["permutation_q"] = bh_adjust(result["permutation_p"])
    return result, pd.DataFrame(predictions)


def group_folds(groups: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    rng = stable_rng(seed, "interaction_group_folds")
    unique = rng.permutation(unique)
    chunks = np.array_split(unique, folds)
    result = []
    for chunk in chunks:
        test = np.isin(groups, chunk)
        result.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return result


def crossvalidated_delta_r2(
    additive: np.ndarray,
    focal_interactions: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[float, float, float]:
    base_predictions = np.empty_like(y, dtype=float)
    full_predictions = np.empty_like(y, dtype=float)
    for train, test in group_folds(groups, folds, seed):
        base = Ridge(alpha=1.0).fit(additive[train], y[train])
        full = Ridge(alpha=1.0).fit(
            np.column_stack([additive[train], focal_interactions[train]]), y[train]
        )
        base_predictions[test] = base.predict(additive[test])
        full_predictions[test] = full.predict(np.column_stack([additive[test], focal_interactions[test]]))
    base_r2 = float(r2_score(y, base_predictions))
    full_r2 = float(r2_score(y, full_predictions))
    return base_r2, full_r2, full_r2 - base_r2


def conditional_interactions(
    data: JointData,
    personas: pd.DataFrame,
    situation_features: pd.DataFrame,
    selected: pd.DataFrame,
    folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    held_mask = data.splits == "heldout"
    discovery_mask = data.splits == "discovery"
    held_ids = data.personas[held_mask]
    situation_ids = data.situations
    persona_lookup = personas.set_index("id_p").loc[data.personas]
    situation_lookup = situation_features.set_index("id_s").loc[situation_ids]
    trait_cols = [f"{trait}_trait" for trait in TRAITS]
    trait_mean = persona_lookup.loc[data.personas[discovery_mask], trait_cols].mean().to_numpy()
    trait_sd = persona_lookup.loc[data.personas[discovery_mask], trait_cols].std(ddof=0).replace(0, np.nan).to_numpy()
    if not np.isfinite(trait_sd).all():
        raise ValueError("At least one Big Five trait has zero variance in discovery personas")
    held_traits = (persona_lookup.loc[held_ids, trait_cols].to_numpy(dtype=float) - trait_mean) / trait_sd
    situation_matrix = situation_lookup[list(DIAMONDS)].to_numpy(dtype=float)
    situation_matrix = StandardScaler().fit_transform(situation_matrix)
    n_p, n_s = len(held_ids), len(situation_ids)
    t_rows = np.repeat(held_traits, n_s, axis=0)
    d_rows = np.tile(situation_matrix, (n_p, 1))
    interactions = (t_rows[:, :, None] * d_rows[:, None, :]).reshape(n_p * n_s, -1)
    additive = np.column_stack([np.ones(n_p * n_s), t_rows, d_rows])
    full_design = np.column_stack([additive, interactions])
    persona_clusters = np.repeat(held_ids, n_s)
    situation_clusters = np.tile(situation_ids, n_p)
    selected_ids = sorted(set(selected["feature_id"].astype(int)))
    feature_to_col = {int(feature): i for i, feature in enumerate(data.feature_ids)}
    raw_heldout = data.raw[held_mask].reshape(n_p * n_s, -1)
    discovery_flat = data.raw[discovery_mask].reshape(-1, data.raw.shape[-1])
    coefficient_rows, joint_rows = [], []
    for feature_id in selected_ids:
        column = feature_to_col[feature_id]
        mean = float(discovery_flat[:, column].mean())
        sd = float(discovery_flat[:, column].std(ddof=0))
        if sd <= 0:
            continue
        y = (raw_heldout[:, column] - mean) / sd
        beta_all, covariance = ols_two_way_cluster(y, full_design, persona_clusters, situation_clusters)
        for trait in TRAITS:
            if not ((selected["feature_id"] == feature_id) & (selected["trait"] == trait)).any():
                continue
            t_index = TRAITS.index(trait)
            interaction_start = additive.shape[1] + t_index * len(DIAMONDS)
            ids = np.arange(interaction_start, interaction_start + len(DIAMONDS))
            beta = beta_all[ids]
            cov = covariance[np.ix_(ids, ids)]
            inverse = np.linalg.pinv(cov)
            statistic = float(beta @ inverse @ beta)
            rank = int(np.linalg.matrix_rank(cov))
            joint_p = float(stats.chi2.sf(statistic, max(rank, 1)))
            focal = interactions[:, t_index * len(DIAMONDS) : (t_index + 1) * len(DIAMONDS)]
            base_r2, full_r2, delta_r2 = crossvalidated_delta_r2(
                additive,
                focal,
                y,
                persona_clusters,
                folds,
                seed + feature_id + t_index,
            )
            joint_rows.append(
                {
                    "trait": trait,
                    "trait_name": TRAIT_NAMES[trait],
                    "feature_id": feature_id,
                    "n_personas": n_p,
                    "n_situations": n_s,
                    "wald_chi2": statistic,
                    "wald_df": rank,
                    "joint_interaction_p": joint_p,
                    "cv_additive_r2": base_r2,
                    "cv_interaction_r2": full_r2,
                    "cv_delta_r2": delta_r2,
                }
            )
            standard_errors = np.sqrt(np.maximum(np.diag(covariance)[ids], 0))
            z_values = np.divide(beta, standard_errors, out=np.full_like(beta, np.nan), where=standard_errors > 0)
            p_values = 2 * stats.norm.sf(np.abs(z_values))
            for dimension, estimate, se, p_value in zip(DIAMONDS, beta, standard_errors, p_values):
                coefficient_rows.append(
                    {
                        "trait": trait,
                        "trait_name": TRAIT_NAMES[trait],
                        "feature_id": feature_id,
                        "diamond": dimension,
                        "diamond_name": DIAMOND_NAMES[dimension],
                        "interaction_beta": float(estimate),
                        "cluster_se": float(se),
                        "ci_low": float(estimate - 1.96 * se),
                        "ci_high": float(estimate + 1.96 * se),
                        "z": float(z_values[list(DIAMONDS).index(dimension)]),
                        "p": float(p_value),
                    }
                )
    coefficients = pd.DataFrame(coefficient_rows)
    joints = pd.DataFrame(joint_rows)
    if len(coefficients):
        coefficients["q_global"] = bh_adjust(coefficients["p"])
        coefficients["q_within_trait"] = coefficients.groupby("trait")["p"].transform(bh_adjust)
    if len(joints):
        joints["joint_interaction_q"] = bh_adjust(joints["joint_interaction_p"])
    return joints, coefficients


def feature_interpretability(
    data: JointData,
    selected: pd.DataFrame,
    personas: pd.DataFrame,
    situations: pd.DataFrame,
    situation_text_path: Path | None,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    membership = selected.groupby("feature_id").agg(
        traits=("trait", lambda x: ",".join(sorted(x))),
        n_traits=("trait", "nunique"),
        signs=("direction_sign", lambda x: ",".join(map(lambda v: f"{int(v):+d}", x))),
        max_abs_discovery_r=("correlation", lambda x: float(np.max(np.abs(x)))),
    ).reset_index()
    sign_conflict = selected.groupby("feature_id")["direction_sign"].nunique()
    membership["direction_conflict"] = membership["feature_id"].map(sign_conflict).gt(1)

    overlap = pd.DataFrame(index=TRAITS, columns=TRAITS, dtype=int)
    sets = {trait: set(selected.loc[selected["trait"] == trait, "feature_id"].astype(int)) for trait in TRAITS}
    for left in TRAITS:
        for right in TRAITS:
            overlap.loc[left, right] = len(sets[left] & sets[right])
    overlap.index.name = "trait"
    overlap = overlap.reset_index()

    situation_text: dict[str, str] = {}
    if situation_text_path and situation_text_path.exists():
        value = json.loads(situation_text_path.read_text(encoding="utf-8"))
        situation_text = {
            str(key): str(item.get("description", item)) if isinstance(item, dict) else str(item)
            for key, item in value.items()
        }
    persona_meta = personas.set_index("id_p")
    feature_to_col = {int(feature): index for index, feature in enumerate(data.feature_ids)}
    held = np.flatnonzero(data.splits == "heldout")
    top_rows = []
    for feature_id in sorted(set(selected["feature_id"].astype(int))):
        column = feature_to_col[feature_id]
        values = data.raw[held, :, column]
        count = min(top_k, values.size)
        flat_ids = np.argpartition(values.ravel(), -count)[-count:]
        flat_ids = flat_ids[np.argsort(values.ravel()[flat_ids])[::-1]]
        assignments = ",".join(sorted(selected.loc[selected["feature_id"] == feature_id, "trait"].astype(str)))
        for rank, flat_id in enumerate(flat_ids, 1):
            p_local, s_local = np.unravel_index(flat_id, values.shape)
            id_p, id_s = str(data.personas[held[p_local]]), str(data.situations[s_local])
            row = {
                "feature_id": feature_id,
                "trait_assignments": assignments,
                "rank": rank,
                "activation": float(values[p_local, s_local]),
                "id_p": id_p,
                "id_s": id_s,
                "situation_text": situation_text.get(id_s, ""),
            }
            for field in ("description", "first_person"):
                if field in persona_meta.columns:
                    row[f"persona_{field}"] = str(persona_meta.loc[id_p, field])
            top_rows.append(row)

    situation_lookup = situations.set_index("id_s").loc[data.situations]
    held_mean_by_situation = data.raw[held].mean(axis=0)
    profile_rows = []
    for feature_id in sorted(set(selected["feature_id"].astype(int))):
        y = held_mean_by_situation[:, feature_to_col[feature_id]]
        for dimension in DIAMONDS:
            x = situation_lookup[dimension].to_numpy(dtype=float)
            if np.std(x) > 0 and np.std(y) > 0:
                correlation, p_value = stats.pearsonr(x, y)
            else:
                correlation, p_value = np.nan, np.nan
            profile_rows.append(
                {
                    "feature_id": feature_id,
                    "diamond": dimension,
                    "diamond_name": DIAMOND_NAMES[dimension],
                    "heldout_situation_profile_r": correlation,
                    "p": p_value,
                }
            )
    profiles = pd.DataFrame(profile_rows)
    profiles["q"] = bh_adjust(profiles["p"])
    return membership, overlap, pd.DataFrame(top_rows), profiles


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value: str,
    cluster: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    grouped = {key: group[value].to_numpy(dtype=float) for key, group in frame.groupby(cluster)}
    keys = np.asarray(list(grouped), dtype=object)
    rng = stable_rng(seed, f"cluster_{value}")
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(keys, len(keys), replace=True)
        estimates[index] = np.mean(np.concatenate([grouped[key] for key in sampled]))
    return percentile_ci(estimates)


def intervention_analysis(
    path: Path,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = read_csv_robust(path)
    if "success" in table.columns:
        table = table[table["success"] == 1].copy()
    state_columns = [f"State_{trait}" for trait in TRAITS]
    missing = set(state_columns) - set(table.columns)
    if missing:
        raise ValueError(f"Intervention table lacks state scores: {sorted(missing)}")
    keys = ["id_p", "id_s", "target_trait", "condition", "scale"]
    averaged = table.groupby(keys, as_index=False)[state_columns].mean()
    baseline = averaged[averaged["condition"] == "baseline"].groupby(
        ["id_p", "id_s"], as_index=False
    )[state_columns].mean()
    baseline = baseline.rename(columns={column: f"baseline_{column}" for column in state_columns})
    changed = averaged[averaged["condition"] != "baseline"].merge(
        baseline, on=["id_p", "id_s"], how="inner", validate="many_to_one"
    )
    changed["target_delta"] = [
        getattr(row, f"State_{row.target_trait}") - getattr(row, f"baseline_State_{row.target_trait}")
        for row in changed.itertuples(index=False)
    ]
    changed["non_target_delta"] = [
        np.mean(
            [
                getattr(row, f"State_{trait}") - getattr(row, f"baseline_State_{trait}")
                for trait in TRAITS
                if trait != row.target_trait
            ]
        )
        for row in changed.itertuples(index=False)
    ]
    changed["specificity"] = changed["target_delta"] - changed["non_target_delta"]
    rows = []
    for (trait, condition, scale), group in changed.groupby(["target_trait", "condition", "scale"]):
        for outcome in ("target_delta", "non_target_delta", "specificity"):
            ci = cluster_bootstrap_mean(group, outcome, "id_p", repetitions, seed + int(100 * float(scale)))
            values = group[outcome].to_numpy(dtype=float)
            p_value = float(stats.wilcoxon(values).pvalue) if np.any(values != 0) else 1.0
            rows.append(
                {
                    "trait": trait,
                    "condition": condition,
                    "scale": scale,
                    "outcome": outcome,
                    "n_pairs": len(group),
                    "n_personas": group["id_p"].nunique(),
                    "mean": float(values.mean()),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "wilcoxon_p": p_value,
                }
            )
    summary = pd.DataFrame(rows)
    summary["fdr_q"] = bh_adjust(summary["wilcoxon_p"])

    controls = changed[changed["condition"] == "random_control"][
        ["id_p", "id_s", "target_trait", "scale", "target_delta", "specificity"]
    ].rename(columns={"target_delta": "control_target_delta", "specificity": "control_specificity"})
    causal = changed[changed["condition"].isin(["positive", "negative"])].merge(
        controls, on=["id_p", "id_s", "target_trait", "scale"], how="inner"
    )
    contrast_rows = []
    for outcome, control, output in (
        ("target_delta", "control_target_delta", "target_vs_control"),
        ("specificity", "control_specificity", "specificity_vs_control"),
    ):
        if len(causal):
            causal[output] = causal[outcome] - causal[control]
    for (trait, condition, scale), group in causal.groupby(["target_trait", "condition", "scale"]):
        for outcome in ("target_vs_control", "specificity_vs_control"):
            values = group[outcome].to_numpy(dtype=float)
            ci = cluster_bootstrap_mean(group, outcome, "id_p", repetitions, seed)
            p_value = float(stats.wilcoxon(values).pvalue) if np.any(values != 0) else 1.0
            contrast_rows.append(
                {
                    "trait": trait,
                    "condition": condition,
                    "scale": scale,
                    "outcome": outcome,
                    "n_pairs": len(group),
                    "mean": float(values.mean()),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "wilcoxon_p": p_value,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    if len(contrasts):
        contrasts["fdr_q"] = bh_adjust(contrasts["wilcoxon_p"])
    return changed, summary, contrasts


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_core_results(
    variance: pd.DataFrame,
    distances: pd.DataFrame,
    decoding: pd.DataFrame,
    interactions: pd.DataFrame,
    output: Path,
) -> None:
    width = 183 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(width, width * 0.72), constrained_layout=True)
    ax = axes[0, 0]
    held = variance[(variance["analysis_split"] == "heldout") & (variance["space"] == "pca")]
    order = ["persona", "situation", "interaction_plus_residual"]
    labels = ["Persona", "Situation", "Interaction + residual"]
    values = [float(held[f"eta2_{key}"].iloc[0]) for key in order]
    lows = [float(held[f"eta2_{key}_ci_low"].iloc[0]) for key in order]
    highs = [float(held[f"eta2_{key}_ci_high"].iloc[0]) for key in order]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["light"]]
    ax.bar(labels, values, color=colors, edgecolor="none")
    ax.errorbar(range(3), values, yerr=[np.asarray(values) - lows, np.asarray(highs) - values], fmt="none", ecolor=PALETTE["black"], capsize=2, lw=0.8)
    ax.set_ylabel("Proportion of variance")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    x = np.arange(len(distances))
    a = distances["different_personas_same_situation_mean"].to_numpy()
    b = distances["same_persona_different_situations_mean"].to_numpy()
    ax.bar(x - 0.18, a, 0.36, color=PALETTE["teal"], label="Different personas, same situation")
    ax.bar(x + 0.18, b, 0.36, color=PALETTE["violet"], label="Same persona, different situations")
    ax.set_xticks(x, [value.capitalize() for value in distances["metric"]])
    ax.set_ylabel("Mean distance")
    ax.legend(fontsize=6)

    ax = axes[1, 0]
    y = np.arange(len(decoding))[::-1]
    estimates = decoding["heldout_pearson_r"].to_numpy()
    low = decoding["heldout_pearson_ci_low"].to_numpy()
    high = decoding["heldout_pearson_ci_high"].to_numpy()
    ax.axvline(0, color=PALETTE["grey"], lw=0.7, ls="--")
    for yy, estimate, lo, hi in zip(y, estimates, low, high):
        ax.plot([lo, hi], [yy, yy], color=PALETTE["blue"], lw=1)
        ax.plot(estimate, yy, "o", color=PALETTE["blue"], ms=3)
    ax.set_yticks(y, decoding["trait_name"])
    ax.set_xlabel("Held-out Pearson r (95% bootstrap CI)")

    ax = axes[1, 1]
    interaction_plot = interactions.groupby("trait", as_index=False).agg(
        median_delta_r2=("cv_delta_r2", "median"),
        significant=("joint_interaction_q", lambda x: int(np.sum(x <= 0.05))),
        total=("feature_id", "size"),
    )
    interaction_plot = interaction_plot.set_index("trait").reindex(TRAITS).reset_index()
    bars = ax.bar(interaction_plot["trait"], interaction_plot["median_delta_r2"], color=PALETTE["red"])
    for bar, significant, total in zip(bars, interaction_plot["significant"], interaction_plot["total"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{significant}/{total}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color=PALETTE["grey"], lw=0.7)
    ax.set_ylabel("Median held-out CV ΔR²")
    ax.set_xlabel("Trait (labels: FDR-significant/selected features)")

    for label, axis in zip("abcd", axes.ravel()):
        axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="bottom")
    save_figure(fig, output / "figures" / "core_confirmatory_results")


def plot_overlap(overlap: pd.DataFrame, output: Path) -> None:
    matrix = overlap.set_index("trait").loc[list(TRAITS), list(TRAITS)].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(89 / 25.4, 78 / 25.4), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=np.max(matrix))
    for i in range(len(TRAITS)):
        for j in range(len(TRAITS)):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(TRAITS)), TRAITS)
    ax.set_yticks(range(len(TRAITS)), TRAITS)
    ax.set_xlabel("Trait-associated top-feature set")
    ax.set_ylabel("Trait-associated top-feature set")
    fig.colorbar(image, ax=ax, label="Number of shared features", fraction=0.046, pad=0.04)
    save_figure(fig, output / "figures" / "trait_feature_overlap")


def plot_intervention(summary: pd.DataFrame, output: Path) -> None:
    target = summary[(summary["outcome"] == "target_delta") & summary["condition"].isin(["positive", "negative", "random_control"])]
    if target.empty:
        return
    fig, axes = plt.subplots(1, 5, figsize=(183 / 25.4, 47 / 25.4), sharey=True, constrained_layout=True)
    colors = {"positive": PALETTE["red"], "negative": PALETTE["blue"], "random_control": PALETTE["grey"]}
    for ax, trait in zip(axes, TRAITS):
        group = target[target["trait"] == trait]
        for condition in ("negative", "random_control", "positive"):
            line = group[group["condition"] == condition].sort_values("scale")
            if line.empty:
                continue
            ax.plot(line["scale"], line["mean"], marker="o", ms=2.5, lw=1, color=colors[condition], label=condition.replace("_", " "))
            ax.fill_between(line["scale"], line["ci_low"], line["ci_high"], color=colors[condition], alpha=0.14, linewidth=0)
        ax.axhline(0, color=PALETTE["grey"], lw=0.6)
        ax.set_title(trait)
        ax.set_xlabel("Intervention scale")
    axes[0].set_ylabel("Target-state change")
    axes[-1].legend(fontsize=5, loc="best")
    save_figure(fig, output / "figures" / "heldout_intervention_dose_response")


def latex_table(frame: pd.DataFrame, path: Path, columns: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = frame[columns].copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(lambda x: f"{x:.3g}" if np.isfinite(x) else "NA")
    def escape(value: object) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        return "".join(replacements.get(character, character) for character in str(value))

    header = " & ".join(escape(column) for column in columns) + r" \\"
    body = [" & ".join(escape(value) for value in row) + r" \\" for row in shown.itertuples(index=False, name=None)]
    text = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{" + "l" * len(columns) + "}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = "./local_assets/llama31_opensae_outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=f"{root}/sae/persona_situation_sparse_features.sqlite")
    parser.add_argument("--personas", default="./local_assets/code/dataset/persona_profile_600_sampled.csv")
    parser.add_argument("--situation-features", default="./local_assets/code/dataset/situation2.csv")
    parser.add_argument("--situation-text", default="./local_assets/code/dataset/simulate_descriptions_study1.json")
    parser.add_argument("--population-stats", default=f"{root}/sae/feature_population_statistics.npz")
    parser.add_argument("--selected-features", default=f"{root}/sae/selected_persona_features.csv")
    parser.add_argument("--intervention-results", default=f"{root}/intervention/intervention_results_long.csv")
    parser.add_argument("--output-dir", default=f"{root}/sae/confirmatory_analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--interaction-folds", type=int, default=5)
    parser.add_argument("--top-examples", type=int, default=10)
    parser.add_argument("--skip-full-space-sensitivity", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = read_csv_robust(Path(args.selected_features))
    required = {"trait", "rank", "feature_id", "correlation", "direction_sign"}
    if missing := required - set(selected.columns):
        raise ValueError(f"Selected-feature table lacks {sorted(missing)}")
    selected["trait"] = selected["trait"].astype(str)
    selected["feature_id"] = selected["feature_id"].astype(int)
    with np.load(args.population_stats, allow_pickle=False) as population:
        activation_rate = np.asarray(population["activation_rate"]).copy()
        sd_activation = np.asarray(population["sd_activation"]).copy()
    eligible = np.flatnonzero((activation_rate >= 0.01) & (sd_activation > 0))
    selected_ids = selected["feature_id"].to_numpy(dtype=int)
    feature_ids = np.unique(np.concatenate([eligible, selected_ids])).astype(np.int32)

    personas = load_personas(Path(args.personas))
    situations = load_situation_features(Path(args.situation_features))
    data = load_joint_tensor(Path(args.database), feature_ids)
    missing_personas = set(data.personas) - set(personas["id_p"])
    missing_situations = set(data.situations) - set(situations["id_s"])
    if missing_personas or missing_situations:
        raise KeyError(
            f"Missing metadata: personas={list(missing_personas)[:3]}, situations={list(missing_situations)[:3]}"
        )
    discovery_n = int(np.sum(data.splits == "discovery"))
    heldout_n = int(np.sum(data.splits == "heldout"))
    if discovery_n == 0 or heldout_n == 0:
        raise RuntimeError(f"Both discovery and heldout splits are required; found {discovery_n}/{heldout_n}")

    standardized, pca_tensor, _, pca = standardize_and_project(data, args.pca_components, args.seed)
    variance_rows = []
    masks = {"heldout": data.splits == "heldout", "pooled_descriptive": np.ones(len(data.personas), dtype=bool)}
    for split_name, mask in masks.items():
        for space, tensor in (("pca", pca_tensor[mask]),):
            point = two_factor_decomposition(tensor)
            boot = bootstrap_decomposition(tensor, args.bootstrap, args.seed)
            row = {"analysis_split": split_name, "space": space, "n_personas": int(mask.sum()), "n_situations": len(data.situations), **point}
            for key in ("eta2_persona", "eta2_situation", "eta2_interaction_plus_residual", "situation_to_persona_ratio"):
                low, high = percentile_ci(boot[key].to_numpy())
                row[f"{key}_ci_low"], row[f"{key}_ci_high"] = low, high
            variance_rows.append(row)
        if not args.skip_full_space_sensitivity:
            point = two_factor_decomposition(standardized[mask])
            variance_rows.append(
                {"analysis_split": split_name, "space": "full_standardized_point_sensitivity", "n_personas": int(mask.sum()), "n_situations": len(data.situations), **point}
            )
    variance = pd.DataFrame(variance_rows)
    variance.to_csv(output / "two_factor_variance_decomposition.csv", index=False, encoding="utf-8-sig")

    distance_summary, distance_source = distance_contrasts(pca_tensor[data.splits == "heldout"], args.bootstrap, args.seed)
    distance_summary.to_csv(output / "heldout_distance_contrasts.csv", index=False, encoding="utf-8-sig")
    distance_source.to_csv(output / "heldout_distance_source_data.csv", index=False, encoding="utf-8-sig")

    persona_features = aggregate_personas(data)
    associations = association_validation(persona_features, personas, selected, args.bootstrap, args.seed)
    associations.to_csv(output / "heldout_trait_feature_validation.csv", index=False, encoding="utf-8-sig")
    decoding, predictions = trait_decoding(persona_features, personas, selected, args.bootstrap, args.permutations, args.seed)
    decoding.to_csv(output / "heldout_trait_decoding.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "heldout_trait_decoding_predictions.csv", index=False, encoding="utf-8-sig")

    interaction_joint, interaction_coefficients = conditional_interactions(
        data, personas, situations, selected, args.interaction_folds, args.seed
    )
    interaction_joint.to_csv(output / "heldout_trait_situation_joint_tests.csv", index=False, encoding="utf-8-sig")
    interaction_coefficients.to_csv(output / "heldout_trait_situation_coefficients.csv", index=False, encoding="utf-8-sig")

    membership, overlap, top_examples, profiles = feature_interpretability(
        data,
        selected,
        personas,
        situations,
        Path(args.situation_text) if args.situation_text else None,
        args.top_examples,
    )
    membership.to_csv(output / "selected_feature_membership.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(output / "selected_feature_overlap_matrix.csv", index=False, encoding="utf-8-sig")
    top_examples.to_csv(output / "selected_feature_top_activating_pairs.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(output / "selected_feature_situation_profiles.csv", index=False, encoding="utf-8-sig")

    intervention_status: dict[str, object]
    intervention_path = Path(args.intervention_results)
    if intervention_path.exists():
        changed, intervention_summary, intervention_contrasts = intervention_analysis(
            intervention_path, args.bootstrap, args.seed
        )
        changed.to_csv(output / "intervention_paired_deltas.csv", index=False, encoding="utf-8-sig")
        intervention_summary.to_csv(output / "intervention_summary.csv", index=False, encoding="utf-8-sig")
        intervention_contrasts.to_csv(output / "intervention_random_control_contrasts.csv", index=False, encoding="utf-8-sig")
        plot_intervention(intervention_summary, output)
        intervention_status = {
            "status": "completed",
            "source": str(intervention_path),
            "n_paired_rows": len(changed),
        }
    else:
        intervention_status = {
            "status": "not_run_missing_input",
            "required_file": str(intervention_path),
            "next_command": "python sae/pipeline.py intervene && python sae/pipeline.py analyze",
        }

    plot_core_results(variance, distance_summary, decoding, interaction_joint, output)
    plot_overlap(overlap, output)
    latex_table(
        variance[variance["space"] == "pca"],
        output / "tables" / "representation-variance-table.tex",
        ["analysis_split", "n_personas", "eta2_persona", "eta2_situation", "eta2_interaction_plus_residual", "situation_to_persona_ratio"],
        "Two-factor decomposition of persona--situation SAE representations.",
        "tab:sae-variance-decomposition",
    )
    latex_table(
        distance_summary,
        output / "tables" / "representation-distance-table.tex",
        ["metric", "different_personas_same_situation_mean", "same_persona_different_situations_mean", "cross_situation_minus_within_situation", "contrast_ci_low", "contrast_ci_high", "bootstrap_two_sided_p"],
        "Held-out representational distance contrasts.",
        "tab:sae-distance-contrasts",
    )
    latex_table(
        decoding,
        output / "tables" / "heldout-trait-decoding-table.tex",
        ["trait_name", "n_features", "heldout_r2", "heldout_pearson_r", "heldout_pearson_ci_low", "heldout_pearson_ci_high", "permutation_q"],
        "Prediction of Big Five scores in held-out personas.",
        "tab:sae-heldout-decoding",
    )
    interaction_summary = interaction_joint.groupby("trait_name", as_index=False).agg(
        n_selected_features=("feature_id", "size"),
        n_joint_fdr_significant=("joint_interaction_q", lambda x: int(np.sum(x <= 0.05))),
        median_cv_delta_r2=("cv_delta_r2", "median"),
        max_cv_delta_r2=("cv_delta_r2", "max"),
    )
    interaction_summary.to_csv(output / "heldout_trait_situation_summary.csv", index=False, encoding="utf-8-sig")
    latex_table(
        interaction_summary,
        output / "tables" / "trait-situation-interaction-table.tex",
        ["trait_name", "n_selected_features", "n_joint_fdr_significant", "median_cv_delta_r2", "max_cv_delta_r2"],
        "Held-out tests of trait-by-situation integration.",
        "tab:sae-trait-situation",
    )

    summary = {
        "created_at": now(),
        "analysis_contract": {
            "trait_instrument": "60-item Big Five Inventory-2 (BFI-2; 12 items per domain)",
            "trait_score_role": "persona-level BFI-2 domain scores supplied as E_trait, A_trait, C_trait, N_trait, and O_trait",
            "discovery_role": "feature selection, preprocessing, and decoder training only",
            "heldout_role": "confirmatory associations, decoding, geometry, interactions, and interventions",
            "pooled_role": "descriptive sensitivity only",
            "split_unit": "persona; all situations for a persona remain in one split",
        },
        "sample_sizes": {
            "personas_total": len(data.personas),
            "personas_discovery": discovery_n,
            "personas_heldout": heldout_n,
            "situations": len(data.situations),
            "joint_records": int(data.raw.shape[0] * data.raw.shape[1]),
        },
        "representation": {
            "eligible_features": int(len(eligible)),
            "loaded_features": int(len(data.feature_ids)),
            "pca_components": int(pca.n_components_),
            "discovery_pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "variance_residual_label": "interaction_plus_residual because each persona-situation cell has one representation",
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "matplotlib": matplotlib.__version__,
        },
        "multiple_testing": {
            "trait_feature_validation": "Benjamini-Hochberg across preselected trait-feature assignments",
            "trait_situation_joint_tests": "Benjamini-Hochberg across preselected trait-feature assignments",
            "interaction_coefficients": "global and within-trait Benjamini-Hochberg",
        },
        "uncertainty": f"{args.bootstrap}-replicate bootstrap; two-way persona/situation bootstrap for geometry and persona bootstrap elsewhere",
        "intervention": intervention_status,
        "claim_boundaries": [
            "UMAP is descriptive and is not used for inferential statistics.",
            "Decoding shows recoverable information, not causal use.",
            "Trait-by-situation effects show conditional association, not causal control.",
            "Only matched-control interventions support causal claims about selected SAE features.",
        ],
    }
    write_json(output / "confirmatory_analysis_manifest.json", summary)
    write_json(
        output / "paper_ready_numeric_summary.json",
        {
            "heldout_variance": variance[(variance["analysis_split"] == "heldout") & (variance["space"] == "pca")].to_dict("records"),
            "heldout_distances": distance_summary.to_dict("records"),
            "heldout_decoding": decoding.to_dict("records"),
            "interaction_summary": interaction_summary.to_dict("records"),
            "intervention": intervention_status,
        },
    )
    return summary


def make_self_test(root: Path, seed: int) -> argparse.Namespace:
    rng = np.random.default_rng(seed)
    n_p, n_s, n_f = 36, 8, 24
    persona_ids = np.asarray([f"P{i:03d}" for i in range(n_p)])
    situation_ids = np.asarray([f"S{i:02d}" for i in range(n_s)])
    traits = rng.normal(size=(n_p, 5))
    diamonds = rng.normal(size=(n_s, 8))
    splits = np.asarray(["discovery"] * 25 + ["heldout"] * 11)
    tensor = np.zeros((n_p, n_s, n_f), dtype=np.float32)
    for p in range(n_p):
        for s in range(n_s):
            value = 1.5 + 0.2 * rng.normal(size=n_f)
            value += diamonds[s, 0] * np.linspace(0.2, 1.0, n_f)
            for t in range(5):
                value[t] += 0.8 * traits[p, t]
                value[t] += 0.6 * traits[p, t] * diamonds[s, t % 8]
            tensor[p, s] = np.maximum(value, 0)
    db = root / "features.sqlite"
    with sqlite3.connect(db) as connection:
        connection.executescript(
            "CREATE TABLE sparse_features (id_p TEXT,id_s TEXT,split TEXT,indices BLOB,values_blob BLOB,k INTEGER,mse REAL,fvu REAL,cosine REAL,l0 INTEGER,created_at TEXT,PRIMARY KEY(id_p,id_s));"
        )
        for p, id_p in enumerate(persona_ids):
            for s, id_s in enumerate(situation_ids):
                ids = np.arange(n_f, dtype=np.int32)
                values = tensor[p, s].astype(np.float32)
                connection.execute(
                    "INSERT INTO sparse_features VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (id_p, id_s, splits[p], ids.tobytes(), values.tobytes(), n_f, 0, 0, 1, n_f, now()),
                )
    persona_table = pd.DataFrame({"id_p": persona_ids})
    for index, trait in enumerate(TRAITS):
        persona_table[f"{trait}_trait"] = traits[:, index]
    persona_table.to_csv(root / "personas.csv", index=False)
    situations = pd.DataFrame({"id_s": situation_ids, **{d: diamonds[:, i] for i, d in enumerate(DIAMONDS)}})
    situations.to_csv(root / "situations.csv", index=False)
    (root / "situation_text.json").write_text(json.dumps({s: f"Synthetic situation {s}" for s in situation_ids}), encoding="utf-8")
    selected = pd.DataFrame(
        [
            {"trait": trait, "rank": 1, "feature_id": i, "correlation": 0.5, "direction_sign": 1, "p_value": 0.001, "q_value": 0.01}
            for i, trait in enumerate(TRAITS)
        ]
    )
    selected.to_csv(root / "selected.csv", index=False)
    np.savez_compressed(
        root / "population.npz",
        activation_rate=np.ones(n_f),
        sd_activation=np.ones(n_f),
    )
    return argparse.Namespace(
        database=str(db),
        personas=str(root / "personas.csv"),
        situation_features=str(root / "situations.csv"),
        situation_text=str(root / "situation_text.json"),
        population_stats=str(root / "population.npz"),
        selected_features=str(root / "selected.csv"),
        intervention_results=str(root / "missing_intervention.csv"),
        output_dir=str(root / "output"),
        seed=seed,
        bootstrap=20,
        permutations=20,
        pca_components=10,
        interaction_folds=3,
        top_examples=3,
        skip_full_space_sensitivity=False,
        self_test=True,
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        with tempfile.TemporaryDirectory(prefix="sae_confirmatory_") as temp:
            test_args = make_self_test(Path(temp), args.seed)
            summary = run_analysis(test_args)
            files = sorted(str(path.relative_to(temp)) for path in Path(temp).rglob("*") if path.is_file())
            print(json.dumps({"self_test": "passed", "summary": summary, "files": files}, indent=2))
        return
    summary = run_analysis(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
