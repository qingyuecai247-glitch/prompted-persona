"""Human participant-by-situation aggregation sensitivity for SCSC reliability.

The human file contains three state records per participant-situation cell.
This script averages those records, refits the same 40 source-specific random
intercept/random slope models, and recomputes the manuscript reliability index.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import math
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"./local_assets/dat_aggregated_final.csv")
OLD_RELIABILITY = Path(r"./local_assets/reliability_all_models.csv")
OUT = ROOT / "analysis" / "results" / "human_person_situation_reliability_sensitivity"

ID, SITUATION = "id_p", "id_s"
STATES = ["State_O", "State_C", "State_E", "State_A", "State_N"]
CONTEXTS = ["dut", "int", "adv", "mat", "pos", "neg", "dec", "soc"]
TRAITS = ["O_trait", "C_trait", "E_trait", "A_trait", "N_trait"]


def aggregate_human(raw: pd.DataFrame) -> pd.DataFrame:
    aggregations = {column: "mean" for column in STATES + TRAITS}
    aggregations.update({column: "median" for column in CONTEXTS})
    return (raw.groupby([ID, SITUATION], observed=True, as_index=False)
            .agg(aggregations))


def fit_cell(payload):
    data_path, state, context = payload
    data = pd.read_csv(data_path)
    cell = data[[ID, SITUATION, state, context]].dropna().copy()
    situation_values = cell.groupby(SITUATION, observed=True)[context].median()
    context_mean = float(situation_values.mean())
    cell["x_c"] = cell[context] - context_mean
    result, optimizer, fit_warnings = None, "", ""
    for method in ("lbfgs", "bfgs", "powell"):
        try:
            model = MixedLM.from_formula(f"{state} ~ x_c", groups=ID,
                                         re_formula="~x_c", data=cell)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate = model.fit(reml=True, method=method, maxiter=2000, disp=False)
            if candidate.converged:
                result, optimizer = candidate, method
                fit_warnings = " | ".join(str(item.message) for item in caught)
                break
        except Exception as exc:
            fit_warnings = f"{type(exc).__name__}: {exc}"
    if result is None:
        return {"state": state, "context": context, "converged": False,
                "optimizer": optimizer, "warnings": fit_warnings}

    covariance = result.cov_re
    slope_variance = float(covariance.loc["x_c", "x_c"])
    residual_variance = float(result.scale)
    records_per_person = cell.groupby(ID, observed=True).size()
    mean_records = float(records_per_person.mean())
    situation_variance = float(cell["x_c"].var(ddof=1))
    denominator = slope_variance + residual_variance / ((mean_records - 1) * situation_variance)
    reliability = slope_variance / denominator if denominator > 0 else np.nan
    return {
        "state": state, "context": context, "converged": True,
        "optimizer": optimizer, "warnings": fit_warnings,
        "n_observations": int(len(cell)), "n_people": int(cell[ID].nunique()),
        "n_situations": int(situation_values.size),
        "mean_records_per_person": mean_records,
        "random_slope_variance": slope_variance,
        "random_slope_sd": math.sqrt(max(slope_variance, 0)),
        "residual_variance": residual_variance,
        "situation_variance": situation_variance,
        "reliability_aggregated_human": float(np.clip(reliability, 0, 1)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW)
    cell_counts = raw.groupby([ID, SITUATION], observed=True).size()
    aggregated = aggregate_human(raw)
    aggregated_path = OUT / "human_participant_situation_profiles.csv"
    aggregated.to_csv(aggregated_path, index=False)

    payloads = [(str(aggregated_path), state, context)
                for state in STATES for context in CONTEXTS]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fit_cell, payload): payload[1:] for payload in payloads}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(f"{row['state']} x {row['context']}: converged={row['converged']}", flush=True)
    results = pd.DataFrame(rows).sort_values(["state", "context"])

    old = pd.read_csv(OLD_RELIABILITY)
    old = (old.loc[old["Group"].str.lower().eq("human"),
                   ["trait", "situation", "reliability"]]
           .rename(columns={"trait": "state", "situation": "context",
                            "reliability": "reliability_original_human"}))
    comparison = results.merge(old, on=["state", "context"], how="left")
    comparison["delta_reliability"] = (comparison["reliability_aggregated_human"]
                                        - comparison["reliability_original_human"])
    results.to_csv(OUT / "human_aggregated_reliability_40_cells.csv", index=False)
    comparison.to_csv(OUT / "human_reliability_original_vs_aggregated.csv", index=False)

    summary = {
        "raw_rows": int(len(raw)),
        "participant_situation_profiles": int(len(aggregated)),
        "participants": int(aggregated[ID].nunique()),
        "situations": int(aggregated[SITUATION].nunique()),
        "raw_rows_per_participant_situation": {
            "min": int(cell_counts.min()), "median": float(cell_counts.median()),
            "max": int(cell_counts.max()),
        },
        "profiles_per_participant": aggregated.groupby(ID).size().describe().to_dict(),
        "models_converged": int(results["converged"].sum()),
        "models_expected": 40,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__},
    }
    (OUT / "human_aggregation_sensitivity_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
