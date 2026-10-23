import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(sys.argv[1])
diagnostics = pd.read_csv(ROOT / "mixed_model_convergence_diagnostics.csv")
fixed = pd.read_csv(ROOT / "mixed_model_results_12sources.csv")
lrt = pd.read_csv(ROOT / "mixed_model_lrt_results.csv")
sample = pd.read_csv(ROOT / "mixed_model_sample_sizes.csv")
slopes = pd.read_csv(ROOT / "random_slopes_12sources.csv")

# Repair exports produced by the initial CI-column relabelling bug without
# refitting: the saved coefficient and standard error fully determine a
# two-sided 95% Wald interval.
ci_missing = fixed[["ci_95_low", "ci_95_high"]].isna().any(axis=1)
if ci_missing.any():
    z_975 = 1.959963984540054
    fixed.loc[ci_missing, "ci_95_low"] = (
        fixed.loc[ci_missing, "coef"]
        - z_975 * fixed.loc[ci_missing, "std_error"]
    )
    fixed.loc[ci_missing, "ci_95_high"] = (
        fixed.loc[ci_missing, "coef"]
        + z_975 * fixed.loc[ci_missing, "std_error"]
    )
    fixed.to_csv(
        ROOT / "mixed_model_results_12sources.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Reconstructed {int(ci_missing.sum())} Wald confidence intervals.")

diagnostics["converged"] = diagnostics["converged"].astype(str).str.lower().eq("true")
successful = diagnostics.loc[diagnostics["converged"]].copy()
model_keys = ["state", "context", "model_role"]
all_models = diagnostics[model_keys].drop_duplicates()
successful_models = successful[model_keys].drop_duplicates()
failed_models = all_models.merge(successful_models, on=model_keys, how="left", indicator=True)
failed_models = failed_models.loc[failed_models["_merge"] == "left_only", model_keys]

unique_fit = fixed.drop_duplicates(["state", "context"]).copy()
boundary_threshold = 1e-8

interaction = fixed.loc[
    fixed["parameter_family"] == "source_by_situation_interaction"
].copy()
interaction["source"] = interaction["parameter"].str.extract(
    r"C\(source\)\[T\.([^\]]+)\]"
)[0]

source_summary = (
    interaction.groupby("source", dropna=False)
    .agg(
        tests=("p_value", "size"),
        raw_significant=("p_value", lambda x: int((x < 0.05).sum())),
        fdr_significant=(
            "interaction_family_reject_fdr_bh",
            lambda x: int(pd.Series(x).astype(str).str.lower().eq("true").sum()),
        ),
        holm_significant=(
            "interaction_family_reject_holm",
            lambda x: int(pd.Series(x).astype(str).str.lower().eq("true").sum()),
        ),
    )
    .reset_index()
)

sample_summary = (
    sample.groupby("source")
    .agg(
        model_cells=("state", "size"),
        n_observations_min=("n_observations", "min"),
        n_observations_max=("n_observations", "max"),
        n_groups_min=("n_groups", "min"),
        n_groups_max=("n_groups", "max"),
        obs_per_group_min=("observations_per_group_min", "min"),
        obs_per_group_median=("observations_per_group_median", "median"),
        obs_per_group_max=("observations_per_group_max", "max"),
    )
    .reset_index()
)

summary = {
    "completed_state_context_models": int(len(unique_fit)),
    "full_and_reduced_models_expected": int(len(all_models)),
    "full_and_reduced_models_converged": int(len(successful_models)),
    "failed_models": failed_models.to_dict(orient="records"),
    "optimizer_success_counts": {
        str(k): int(v) for k, v in successful["optimizer"].value_counts().items()
    },
    "fits_requiring_fallback": int((successful["optimizer"] != "lbfgs").sum()),
    "optimizer_attempts_total": int(len(diagnostics)),
    "attempts_with_warnings": int((diagnostics["warning_count"] > 0).sum()),
    "lrt_tests": int(len(lrt)),
    "lrt_raw_significant": int((lrt["p_value"] < 0.05).sum()),
    "lrt_fdr_significant": int(
        lrt["lrt_family_reject_fdr_bh"].astype(str).str.lower().eq("true").sum()
    ),
    "lrt_holm_significant": int(
        lrt["lrt_family_reject_holm"].astype(str).str.lower().eq("true").sum()
    ),
    "lrt_not_fdr_significant": lrt.loc[
        ~lrt["lrt_family_reject_fdr_bh"].astype(str).str.lower().eq("true"),
        ["state", "context", "lr_statistic", "df", "p_value", "lrt_family_p_fdr_bh"],
    ].to_dict(orient="records"),
    "interaction_tests": int(len(interaction)),
    "interaction_raw_significant": int((interaction["p_value"] < 0.05).sum()),
    "interaction_fdr_significant": int(
        interaction["interaction_family_reject_fdr_bh"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    ),
    "interaction_holm_significant": int(
        interaction["interaction_family_reject_holm"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    ),
    "near_singular_fits_threshold": boundary_threshold,
    "near_singular_fits": int(
        (unique_fit["random_cov_min_eigenvalue"] < boundary_threshold).sum()
    ),
    "random_slope_variance_min": float(unique_fit["random_slope_variance"].min()),
    "random_slope_variance_max": float(unique_fit["random_slope_variance"].max()),
    "random_slopes_rows": int(len(slopes)),
}

(ROOT / "mixedlm_run_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
source_summary.to_csv(ROOT / "mixedlm_interaction_summary_by_source.csv", index=False)
sample_summary.to_csv(ROOT / "mixedlm_sample_summary_by_source.csv", index=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nINTERACTION SUMMARY BY SOURCE")
print(source_summary.to_string(index=False))
print("\nSAMPLE SUMMARY BY SOURCE")
print(sample_summary.to_string(index=False))
