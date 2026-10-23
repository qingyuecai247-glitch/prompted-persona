# -*- coding: utf-8 -*-
"""Factor diagnostics and persona-level bootstrap confidence intervals for FMI."""

from __future__ import annotations

import argparse
import inspect
import json
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
import factor_analyzer.factor_analyzer as factor_analyzer_module
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.utils.validation import check_array as sklearn_check_array


if "force_all_finite" not in inspect.signature(sklearn_check_array).parameters:
    def check_array_compat(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return sklearn_check_array(*args, **kwargs)

    factor_analyzer_module.check_array = check_array_compat


DATA_ROOT = Path(r"./local_assets/dataset")
SOURCE_FILES = {
    "human": DATA_ROOT / "dat_aggregated_final.csv",
    "gpt4omini": DATA_ROOT / "llm_output" / "simulation_output_flattened-gpt4omini.csv",
    "gpt52": DATA_ROOT / "llm_output" / "simulation_output_flattened-gpt52.csv",
    "deepseek": DATA_ROOT / "llm_output" / "simulation_output_flattened-deepseek3.2.csv",
    "grok3": DATA_ROOT / "llm_output" / "simulation_output_flattened-grok3.csv",
    "llama": DATA_ROOT / "llm_output" / "simulation_output_flattened_llama3.1-8b-instruct.csv",
    "qwen7": DATA_ROOT / "llm_output" / "simulation_output_flattened-qwen2.5-7b-instruct.csv",
    "qwen14": DATA_ROOT / "llm_output" / "simulation_output_flattened-qwen2.5-14b-instruct.csv",
    "qwen32": DATA_ROOT / "llm_output" / "simulation_output_flattened-qwen2.5-32b-instruct.csv",
    "phimini": DATA_ROOT / "llm_output" / "simulation_output_flattened-phi3-mini.csv",
    "phismall": DATA_ROOT / "llm_output" / "simulation_output_flattened-phi3-small.csv",
    "phimedium": DATA_ROOT / "llm_output" / "simulation_output_flattened-phi3-medium.csv",
}
SOURCE_LABELS = {
    "human": "Human", "gpt4omini": "GPT-4o mini", "gpt52": "GPT-5.2",
    "deepseek": "DeepSeek-3.2", "grok3": "Grok-3",
    "llama": "Llama-3.1-8B-Instruct", "qwen7": "Qwen2.5-7B-Instruct",
    "qwen14": "Qwen2.5-14B-Instruct", "qwen32": "Qwen2.5-32B-Instruct",
    "phimini": "Phi-3-mini-4k-instruct", "phismall": "Phi-3-small-8k-instruct",
    "phimedium": "Phi-3-medium-4k-instruct",
}
CHARS = ["dut", "int", "adv", "mat", "pos", "neg", "dec", "soc"]
STATES = ["E", "A", "C", "N", "O"]
THRESHOLDS = [0.4, 0.5, 0.6]
FINAL_N_FACTORS = 5
BASE_SEED = 123

warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", message="No rotation will be performed.*")


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Cannot decode {path}")


def make_person_scsc(data: pd.DataFrame) -> pd.DataFrame:
    required = ["id_p", *CHARS, *[f"State_{s}" for s in STATES]]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    work = data[required].copy()
    work["id_p"] = work["id_p"].astype(str)
    numeric = CHARS + [f"State_{s}" for s in STATES]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    products = []
    for state in STATES:
        for char in CHARS:
            name = f"Sit_{state}_{char}"
            work[name] = work[f"State_{state}"] * work[char]
            products.append(name)
    result = work.groupby("id_p", sort=False)[products].mean()
    result = result.dropna(axis=0, how="all")
    return result.apply(lambda column: column.fillna(column.mean()), axis=0)


def nearest_pd_corr(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    corr = np.asarray(corr, dtype=float)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    values, vectors = np.linalg.eigh(corr)
    values[values < eps] = eps
    repaired = vectors @ np.diag(values) @ vectors.T
    repaired = (repaired + repaired.T) / 2
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def fit_factor(corr: np.ndarray, n_factors: int = FINAL_N_FACTORS) -> np.ndarray:
    model = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="minres", is_corr_matrix=True)
    model.fit(nearest_pd_corr(corr))
    return model.loadings_


def compute_fmi(loadings: np.ndarray, item_names: list[str], threshold: float) -> float:
    modularities = []
    for factor_index in range(loadings.shape[1]):
        selected = np.flatnonzero(np.abs(loadings[:, factor_index]) > threshold)
        if selected.size:
            domains = [item_names[index].split("_")[1] for index in selected]
            modularities.append(pd.Series(domains).value_counts().iloc[0] / len(domains))
    return float(np.mean(modularities)) if modularities else np.nan


def parallel_analysis(data: pd.DataFrame, iterations: int, seed: int) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    n_rows, n_columns = data.shape
    observed = np.linalg.eigvalsh(data.corr().to_numpy())[::-1]
    random_values = np.empty((iterations, n_columns))
    for iteration in range(iterations):
        random_data = rng.normal(size=(n_rows, n_columns))
        random_values[iteration] = np.linalg.eigvalsh(np.corrcoef(random_data, rowvar=False))[::-1]
    random_95 = np.percentile(random_values, 95, axis=0)
    result = pd.DataFrame({
        "component": np.arange(1, n_columns + 1),
        "observed_eigenvalue": observed,
        "random_mean_eigenvalue": random_values.mean(axis=0),
        "random_95pct_eigenvalue": random_95,
    })
    return result, int(np.sum(observed > random_95))


def vss_analysis(corr: np.ndarray, max_factors: int = 8) -> pd.DataFrame:
    corr = nearest_pd_corr(corr)
    off_diagonal = ~np.eye(corr.shape[0], dtype=bool)
    target = corr[off_diagonal]
    total_ss = np.sum((target - target.mean()) ** 2)
    rows = []
    for n_factors in range(1, max_factors + 1):
        loadings = fit_factor(corr, n_factors)
        for complexity in (1, 2):
            simplified = np.zeros_like(loadings)
            for row in range(loadings.shape[0]):
                keep = np.argsort(np.abs(loadings[row]))[::-1][:complexity]
                simplified[row, keep] = loadings[row, keep]
            predicted = (simplified @ simplified.T)[off_diagonal]
            fit = 1 - np.sum((target - predicted) ** 2) / total_ss
            rows.append({"nfactors": n_factors, "complexity": complexity, "VSS_fit": float(fit)})
    return pd.DataFrame(rows)


def bootstrap_fmi(data: pd.DataFrame, source: str, path: Path, requested: int) -> pd.DataFrame:
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    completed = set(existing.get("replicate", pd.Series(dtype=int)).astype(int))
    pending = []
    values = data.to_numpy(dtype=float)
    names = list(data.columns)
    source_seed = zlib.crc32(source.encode("utf-8"))
    for replicate in range(1, requested + 1):
        if replicate in completed:
            continue
        rng = np.random.default_rng(BASE_SEED + source_seed + replicate * 1009)
        sample = values[rng.integers(0, len(values), size=len(values))]
        try:
            loadings = fit_factor(np.corrcoef(sample, rowvar=False))
            row = {"dataset": source, "replicate": replicate, "error": ""}
            for threshold in THRESHOLDS:
                row[f"FMI_{threshold:.1f}"] = compute_fmi(loadings, names, threshold)
        except Exception as exc:
            row = {"dataset": source, "replicate": replicate, "error": f"{type(exc).__name__}: {exc}"}
            row.update({f"FMI_{threshold:.1f}": np.nan for threshold in THRESHOLDS})
        pending.append(row)
        if len(pending) == 25 or replicate == requested:
            existing = pd.concat([existing, pd.DataFrame(pending)], ignore_index=True)
            existing = existing.drop_duplicates("replicate", keep="last").sort_values("replicate")
            existing.to_csv(path, index=False, encoding="utf-8-sig")
            pending = []
            print(f"  {source}: FMI bootstrap {replicate}/{requested}", flush=True)
    return pd.read_csv(path)


def analyze_source(source: str, args: argparse.Namespace) -> dict:
    source_dir = args.output / source
    source_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {SOURCE_LABELS[source]} ...", flush=True)
    person_data = make_person_scsc(read_csv(SOURCE_FILES[source]))
    person_data.reset_index().to_csv(source_dir / "person_level_scsc.csv", index=False, encoding="utf-8-sig")
    corr = nearest_pd_corr(person_data.corr().to_numpy())
    pd.DataFrame(corr, index=person_data.columns, columns=person_data.columns).to_csv(source_dir / "scsc_correlation.csv", encoding="utf-8-sig")

    kmo_items, kmo_overall = calculate_kmo(person_data)
    bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(person_data)
    pd.DataFrame({"item": person_data.columns, "KMO_MSAi": kmo_items}).to_csv(source_dir / "kmo_item.csv", index=False, encoding="utf-8-sig")
    seed = BASE_SEED + zlib.crc32(source.encode("utf-8"))
    parallel, parallel_n = parallel_analysis(person_data, args.parallel, seed)
    parallel.to_csv(source_dir / "parallel_analysis.csv", index=False, encoding="utf-8-sig")
    vss = vss_analysis(corr)
    vss.to_csv(source_dir / "vss.csv", index=False, encoding="utf-8-sig")

    loadings = fit_factor(corr)
    pd.DataFrame(loadings, index=person_data.columns, columns=[f"F{i}" for i in range(1, 6)]).to_csv(source_dir / "five_factor_loadings.csv", encoding="utf-8-sig")
    point_fmi = {threshold: compute_fmi(loadings, list(person_data.columns), threshold) for threshold in THRESHOLDS}
    bootstrap = bootstrap_fmi(person_data, source, source_dir / "fmi_bootstrap_replicates.csv", args.bootstrap)

    vss1 = vss.loc[vss["complexity"].eq(1)].sort_values("VSS_fit").iloc[-1]
    vss2 = vss.loc[vss["complexity"].eq(2)].sort_values("VSS_fit").iloc[-1]
    summary = {
        "dataset": source, "dataset_label": SOURCE_LABELS[source],
        "n_people_or_personas": int(len(person_data)), "n_scsc_indices": int(person_data.shape[1]),
        "KMO_overall_MSA": float(kmo_overall), "Bartlett_chi2": float(bartlett_chi2),
        "Bartlett_df": 780, "Bartlett_p": float(bartlett_p),
        "parallel_iterations": int(args.parallel), "parallel_suggested_n": int(parallel_n),
        "VSS1_suggested_n": int(vss1["nfactors"]), "VSS2_suggested_n": int(vss2["nfactors"]),
        "fmi_bootstrap_requested": int(args.bootstrap),
    }
    for threshold in THRESHOLDS:
        column = f"FMI_{threshold:.1f}"
        valid = bootstrap[column].dropna()
        summary[f"FMI_{threshold:.1f}_point"] = point_fmi[threshold]
        summary[f"FMI_{threshold:.1f}_boot_valid"] = int(len(valid))
        summary[f"FMI_{threshold:.1f}_ci_low"] = float(valid.quantile(0.025))
        summary[f"FMI_{threshold:.1f}_ci_high"] = float(valid.quantile(0.975))
    (source_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--parallel", type=int, default=500)
    parser.add_argument("--only", nargs="*", choices=list(SOURCE_FILES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = args.only or list(SOURCE_FILES)
    for source in sources:
        analyze_source(source, args)
        summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.output.glob("*/summary.json")]
        pd.DataFrame(summaries).to_csv(args.output / "factor_diagnostics_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Completed factor diagnostics for {len(sources)} datasets.")


if __name__ == "__main__":
    main()
