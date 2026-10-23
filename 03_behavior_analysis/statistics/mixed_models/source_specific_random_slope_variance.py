# -*- coding: utf-8 -*-
"""Estimate source-specific between-person random-slope variability.

This analysis deliberately fits each source separately.  Its primary outcome is
the model-estimated random-slope variance (and SD) from ``MixedLM.cov_re``.
Empirical-Bayes random effects are exported separately for descriptive plots;
their empirical SD is not used as the variance-component estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.regression.mixed_linear_model import MixedLM


STATES = ["State_O", "State_C", "State_E", "State_A", "State_N"]
CONTEXTS = ["dut", "int", "adv", "mat", "pos", "neg", "dec", "soc"]
ID_COL = "id_p"
SITUATION_COL = "id_s"

SOURCE_FILES = {
    "human": "llm_output/dat_aggregated_final.csv",
    "deepseek": "llm_output/simulation_output_flattened-deepseek3.2.csv",
    "grok3": "llm_output/simulation_output_flattened-grok3.csv",
    "gpt52": "llm_output/simulation_output_flattened-gpt52.csv",
    "qwen32": "llm_output/simulation_output_flattened-qwen2.5-32b-instruct.csv",
    "phimedium": "llm_output/simulation_output_flattened-phi3-medium.csv",
    "gpt4omini": "llm_output/simulation_output_flattened-gpt4omini.csv",
    "llama": "llm_output/simulation_output_flattened_llama3.1-8b-instruct.csv",
    "qwen7": "llm_output/simulation_output_flattened-qwen2.5-7b-instruct.csv",
    "qwen14": "llm_output/simulation_output_flattened-qwen2.5-14b-instruct.csv",
    "phimini": "llm_output/simulation_output_flattened-phi3-mini.csv",
    "phismall": "llm_output/simulation_output_flattened-phi3-small.csv",
}

SOURCE_LABELS = {
    "human": "Human",
    "deepseek": "DeepSeek-3.2",
    "grok3": "Grok-3",
    "gpt52": "GPT-5.2",
    "qwen32": "Qwen2.5-32B-Instruct",
    "phimedium": "Phi-3-medium-4k-instruct",
    "gpt4omini": "GPT-4o-mini",
    "llama": "Llama-3.1-8B-Instruct",
    "qwen7": "Qwen2.5-7B-Instruct",
    "qwen14": "Qwen2.5-14B-Instruct",
    "phimini": "Phi-3-mini-4k-instruct",
    "phismall": "Phi-3-small-8k-instruct",
}

PRIMARY_SOURCES = [
    "human", "deepseek", "grok3", "gpt52", "qwen32", "phimedium",
    "gpt4omini", "llama",
]
ALL_SOURCES = list(SOURCE_FILES)


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def prepare_cell(raw: pd.DataFrame, state: str, context: str) -> tuple[pd.DataFrame, float, int]:
    required = [ID_COL, SITUATION_COL, state, context]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    cell = raw[required].copy()
    cell[state] = pd.to_numeric(cell[state], errors="coerce")
    cell[context] = pd.to_numeric(cell[context], errors="coerce")
    cell = cell.dropna(subset=required)

    # Centre the normative situation score over the situations actually used,
    # rather than over repeated participant/persona rows.
    situation_values = cell.groupby(SITUATION_COL, observed=True)[context].median()
    context_mean = float(situation_values.mean())
    n_situations = int(situation_values.size)
    cell["x_c"] = cell[context] - context_mean
    return cell, context_mean, n_situations


def fit_cell(raw: pd.DataFrame, source: str, state: str, context: str, random_structure: str = "correlated"):
    cell, context_mean, n_situations = prepare_cell(raw, state, context)
    attempts = []
    result = None
    selected_optimizer = None

    for optimizer in ("lbfgs", "bfgs", "powell"):
        record = {"optimizer": optimizer, "converged": False, "warnings": "", "error": ""}
        try:
            model_kwargs = {
                "formula": f"{state} ~ x_c",
                "groups": ID_COL,
                "data": cell,
            }
            if random_structure == "correlated":
                model_kwargs["re_formula"] = "~x_c"
            elif random_structure == "independent":
                model_kwargs["re_formula"] = "1"
                model_kwargs["vc_formula"] = {"random_slope": "0 + x_c"}
            else:
                raise ValueError(random_structure)
            model = MixedLM.from_formula(**model_kwargs)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = model.fit(reml=True, method=optimizer, maxiter=2000, disp=False)
            record["converged"] = bool(fitted.converged)
            record["warnings"] = " | ".join(str(item.message) for item in caught)
            attempts.append(record)
            if fitted.converged:
                result = fitted
                selected_optimizer = optimizer
                break
        except Exception as exc:  # retain diagnostics and try next optimizer
            record["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(record)

    base = {
        "source": source,
        "source_label": SOURCE_LABELS[source],
        "state": state,
        "context": context,
        "n_observations": int(len(cell)),
        "n_groups": int(cell[ID_COL].nunique()),
        "n_situations": n_situations,
        "context_mean_across_situations": context_mean,
        "fit_method": "REML",
        "random_structure": random_structure,
        "optimizer": selected_optimizer or "",
        "converged": bool(result is not None and result.converged),
    }

    diagnostic_rows = []
    for number, attempt in enumerate(attempts, start=1):
        diagnostic_rows.append({**base, "attempt": number, **attempt})

    if result is None:
        estimate = {
            **base,
            "fixed_intercept": np.nan,
            "fixed_slope": np.nan,
            "random_intercept_variance": np.nan,
            "random_slope_variance": np.nan,
            "random_slope_sd": np.nan,
            "random_intercept_slope_covariance": np.nan,
            "random_intercept_slope_correlation": np.nan,
            "residual_variance": np.nan,
            "log_likelihood": np.nan,
        }
        return estimate, diagnostic_rows, []

    covariance = result.cov_re
    intercept_name = covariance.index[0]
    intercept_variance = float(covariance.loc[intercept_name, intercept_name])
    if random_structure == "correlated":
        slope_name = "x_c"
        slope_variance = float(covariance.loc[slope_name, slope_name])
        intercept_slope_covariance = float(covariance.loc[intercept_name, slope_name])
    else:
        slope_variance = float(np.asarray(result.vcomp, dtype=float)[0])
        intercept_slope_covariance = 0.0
    denominator = math.sqrt(max(intercept_variance * slope_variance, 0.0))
    correlation = intercept_slope_covariance / denominator if denominator > 0 else np.nan

    estimate = {
        **base,
        "fixed_intercept": float(result.fe_params.get("Intercept", np.nan)),
        "fixed_slope": float(result.fe_params.get("x_c", np.nan)),
        "random_intercept_variance": intercept_variance,
        "random_slope_variance": slope_variance,
        "random_slope_sd": math.sqrt(max(slope_variance, 0.0)),
        "random_intercept_slope_covariance": intercept_slope_covariance,
        "random_intercept_slope_correlation": correlation,
        "residual_variance": float(result.scale),
        "log_likelihood": float(result.llf),
        "blup_available": True,
        "blup_error": "",
    }

    blup_rows = []
    source_fixed_slope = estimate["fixed_slope"]
    try:
        conditional_effects = result.random_effects
    except (ValueError, np.linalg.LinAlgError) as exc:
        conditional_effects = {}
        estimate["blup_available"] = False
        estimate["blup_error"] = f"{type(exc).__name__}: {exc}"
    for participant, effects in conditional_effects.items():
        if random_structure == "correlated":
            slope_deviation = float(effects.get("x_c", np.nan))
        else:
            slope_candidates = [value for key, value in effects.items() if "random_slope" in str(key)]
            slope_deviation = float(slope_candidates[0]) if slope_candidates else np.nan
        blup_rows.append({
            "source": source,
            "source_label": SOURCE_LABELS[source],
            "state": state,
            "context": context,
            ID_COL: participant,
            "random_slope_deviation_blup": slope_deviation,
            "total_slope_blup": source_fixed_slope + slope_deviation,
            "descriptive_only": True,
        })

    return estimate, diagnostic_rows, blup_rows


def _write_source_checkpoint(output_dir: Path, source: str, estimates, diagnostics, blups) -> None:
    checkpoint = output_dir / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(estimates).to_csv(
        checkpoint / f"{source}_estimates.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(diagnostics).to_csv(
        checkpoint / f"{source}_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(blups).to_csv(
        checkpoint / f"{source}_blups.csv", index=False, encoding="utf-8-sig"
    )


def run_source(source: str, data_root: str, output_dir: str, random_structure: str):
    path = Path(data_root) / SOURCE_FILES[source]
    if not path.exists():
        raise FileNotFoundError(path)
    raw = read_csv(path)
    checkpoint = Path(output_dir) / "checkpoints"
    estimate_path = checkpoint / f"{source}_estimates.csv"
    diagnostic_path = checkpoint / f"{source}_diagnostics.csv"
    blup_path = checkpoint / f"{source}_blups.csv"
    estimates = pd.read_csv(estimate_path).to_dict("records") if estimate_path.exists() else []
    diagnostics = pd.read_csv(diagnostic_path).to_dict("records") if diagnostic_path.exists() else []
    if blup_path.exists() and blup_path.stat().st_size > 3:
        try:
            blups = pd.read_csv(blup_path).to_dict("records")
        except pd.errors.EmptyDataError:
            blups = []
    else:
        blups = []
    completed = {(row["state"], row["context"]) for row in estimates}
    for state in STATES:
        for context in CONTEXTS:
            if (state, context) in completed:
                continue
            estimate, diagnostic_rows, blup_rows = fit_cell(
                raw, source, state, context, random_structure=random_structure
            )
            estimates.append(estimate)
            diagnostics.extend(diagnostic_rows)
            blups.extend(blup_rows)
            _write_source_checkpoint(Path(output_dir), source, estimates, diagnostics, blups)
            print(
                f"{SOURCE_LABELS[source]}: {len(estimates)}/{len(STATES) * len(CONTEXTS)} models",
                flush=True,
            )
    return source, estimates, diagnostics, blups


def add_relative_variability(estimates: pd.DataFrame) -> pd.DataFrame:
    human = estimates.loc[
        estimates["source"].eq("human"),
        ["state", "context", "random_slope_sd"],
    ].rename(columns={"random_slope_sd": "human_random_slope_sd"})
    relative = estimates.merge(human, on=["state", "context"], how="left")
    relative["relative_variability_vs_human"] = (
        relative["random_slope_sd"] / relative["human_random_slope_sd"]
    )
    return relative


def aggregate_for_plot(relative: pd.DataFrame) -> pd.DataFrame:
    nonhuman = relative.loc[~relative["source"].eq("human")].copy()
    rows = []
    grouping_specs = [
        ("overall", [], "all"),
        ("trait", ["state"], None),
        ("context", ["context"], None),
    ]
    for source, source_df in nonhuman.groupby(["source", "source_label"], observed=True):
        source_key, source_label = source
        for summary_type, dimensions, fixed_dimension in grouping_specs:
            groups = [(fixed_dimension, source_df)] if not dimensions else source_df.groupby(dimensions[0], observed=True)
            for dimension, group in groups:
                values = pd.to_numeric(group["relative_variability_vs_human"], errors="coerce").dropna()
                rows.append({
                    "source": source_key,
                    "source_label": source_label,
                    "summary_type": summary_type,
                    "dimension": dimension,
                    "n_cells": int(values.size),
                    "median_relative_variability": float(values.median()) if len(values) else np.nan,
                    "mean_relative_variability": float(values.mean()) if len(values) else np.nan,
                    "min_relative_variability": float(values.min()) if len(values) else np.nan,
                    "max_relative_variability": float(values.max()) if len(values) else np.nan,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"./local_assets/dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("primary", "all"), default="all")
    parser.add_argument(
        "--random-structure", choices=("correlated", "independent"), default="correlated"
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sources = PRIMARY_SOURCES if args.scope == "primary" else ALL_SOURCES
    all_estimates, all_diagnostics, all_blups = [], [], []

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_source, source, str(args.data_root), str(args.output), args.random_structure
            ): source
            for source in sources
        }
        for future in as_completed(futures):
            source, estimates, diagnostics, blups = future.result()
            all_estimates.extend(estimates)
            all_diagnostics.extend(diagnostics)
            all_blups.extend(blups)
            print(f"Completed {SOURCE_LABELS[source]} ({len(estimates)} models)", flush=True)

    source_order = {source: number for number, source in enumerate(sources)}
    state_order = {state: number for number, state in enumerate(STATES)}
    context_order = {context: number for number, context in enumerate(CONTEXTS)}

    estimates = pd.DataFrame(all_estimates)
    estimates["_source_order"] = estimates["source"].map(source_order)
    estimates["_state_order"] = estimates["state"].map(state_order)
    estimates["_context_order"] = estimates["context"].map(context_order)
    estimates = estimates.sort_values(["_source_order", "_state_order", "_context_order"]).drop(
        columns=["_source_order", "_state_order", "_context_order"]
    )
    diagnostics = pd.DataFrame(all_diagnostics)
    blups = pd.DataFrame(all_blups)
    if not blups.empty:
        blup_keys = set(zip(blups["source"], blups["state"], blups["context"]))
        estimates["blup_available"] = [
            (source, state, context) in blup_keys
            for source, state, context in zip(estimates["source"], estimates["state"], estimates["context"])
        ]
    else:
        estimates["blup_available"] = False
    if "blup_error" not in estimates:
        estimates["blup_error"] = ""
    estimates["blup_error"] = estimates["blup_error"].fillna("")

    # Explicit boundary diagnostics for the 2x2 random-effects covariance.
    a = pd.to_numeric(estimates["random_intercept_variance"], errors="coerce")
    d = pd.to_numeric(estimates["random_slope_variance"], errors="coerce")
    b = pd.to_numeric(estimates["random_intercept_slope_covariance"], errors="coerce")
    discriminant = np.sqrt(np.maximum((a - d) ** 2 + 4 * b ** 2, 0.0))
    estimates["random_cov_min_eigenvalue"] = (a + d - discriminant) / 2
    estimates["random_cov_boundary_flag"] = (
        estimates["random_cov_min_eigenvalue"].lt(1e-8)
        | d.lt(1e-8)
        | pd.to_numeric(estimates["random_intercept_slope_correlation"], errors="coerce").abs().gt(0.999)
    )

    relative = add_relative_variability(estimates)
    aggregate = aggregate_for_plot(relative)

    estimates.to_csv(args.output / "source_specific_random_slope_variance.csv", index=False, encoding="utf-8-sig")
    relative.to_csv(args.output / "source_specific_relative_variability.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(args.output / "source_specific_relative_variability_plot_summary.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(args.output / "source_specific_model_diagnostics.csv", index=False, encoding="utf-8-sig")
    blups.to_csv(args.output / "source_specific_blups_descriptive_only.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "source-specific random-intercept/random-slope MixedLM",
        "formula": "state ~ x_c",
        "random_effects": (
            "correlated random intercept and x_c slope by participant/persona"
            if args.random_structure == "correlated"
            else "independent random intercept and x_c slope variance components by participant/persona"
        ),
        "random_structure": args.random_structure,
        "estimation": "REML",
        "primary_estimand": (
            "sqrt(cov_re[x_c, x_c])"
            if args.random_structure == "correlated"
            else "sqrt(vcomp[random_slope])"
        ),
        "blup_role": "descriptive only; empirical BLUP SD is not used as the variance estimate",
        "centering": "DIAMONDS scores centred over unique situations used within each source",
        "scope": args.scope,
        "sources": sources,
        "models_expected": len(sources) * len(STATES) * len(CONTEXTS),
        "models_returned": int(len(estimates)),
        "models_converged": int(estimates["converged"].sum()),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    (args.output / "source_specific_random_slope_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
