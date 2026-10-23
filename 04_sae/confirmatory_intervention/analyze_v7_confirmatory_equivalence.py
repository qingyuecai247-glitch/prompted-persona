#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAITS = ("E", "A", "C", "N", "O")
STATES = [f"State_{trait}" for trait in TRAITS]


def crossed_bootstrap(matrix: np.ndarray, repetitions: int, seed: int) -> np.ndarray:
    """Resample persona rows and situation columns independently."""
    rng = np.random.default_rng(seed)
    n_personas, n_situations = matrix.shape
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        rows = rng.integers(0, n_personas, n_personas)
        columns = rng.integers(0, n_situations, n_situations)
        estimates[index] = np.nanmean(matrix[np.ix_(rows, columns)])
    return estimates


def matrix_for(data: pd.DataFrame, value: str) -> np.ndarray:
    return data.pivot(index="id_p", columns="id_s", values=value).to_numpy(dtype=float)


def summarize_effect(data: pd.DataFrame, value: str, repetitions: int, seed: int,
                     sesoi: float) -> dict:
    matrix = matrix_for(data, value)
    bootstrap = crossed_bootstrap(matrix, repetitions, seed)
    estimate = float(np.nanmean(matrix))
    low, high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "estimate": estimate,
        "bootstrap_ci_95": [float(low), float(high)],
        "sesoi": float(sesoi),
        "equivalent_within_sesoi": bool(low > -sesoi and high < sesoi),
        "personas": int(matrix.shape[0]),
        "situations": int(matrix.shape[1]),
        "complete_cells": int(np.isfinite(matrix).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirmatory crossed-bootstrap equivalence analysis")
    parser.add_argument("csv", nargs="+", type=Path, help="One merged CSV or all shard CSVs")
    parser.add_argument("--sesoi", type=float, default=0.15)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.sesoi <= 0 or args.bootstrap < 1000:
        raise ValueError("Require SESOI > 0 and at least 1000 bootstrap replicates")

    frames = [pd.read_csv(path) for path in args.csv]
    table = pd.concat(frames, ignore_index=True)
    key = ["id_p", "id_s", "target_trait", "condition", "scale", "repeat_index"]
    table = table.drop_duplicates(key, keep="last")
    output = args.output_dir or args.csv[0].resolve().parent / "v7_confirmatory_analysis"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "intervention_results_long_v7_confirmatory_merged.csv",
                 index=False, encoding="utf-8-sig")

    if not np.isclose(table.loc[table["condition"] != "baseline", "scale"], 8.0).all():
        raise ValueError("Confirmatory contract requires the frozen 8-SD dose")
    expected_conditions = {
        "baseline", "positive", "negative", "positive_control", "negative_control"
    }
    if set(table["condition"]) != expected_conditions:
        raise ValueError(f"Unexpected conditions: {sorted(set(table['condition']))}")

    baseline = (
        table[table["condition"] == "baseline"]
        .drop_duplicates(["id_p", "id_s", "repeat_index"])
        [["id_p", "id_s", "repeat_index", *STATES]]
        .rename(columns={state: f"{state}_baseline" for state in STATES})
    )
    changed = table[table["condition"] != "baseline"].merge(
        baseline, on=["id_p", "id_s", "repeat_index"], how="left", validate="many_to_one"
    )
    for state in STATES:
        changed[f"delta_{state}"] = changed[state] - changed[f"{state}_baseline"]
    changed["target_delta"] = [
        row[f"delta_State_{row['target_trait']}"] for _, row in changed.iterrows()
    ]
    changed["direction"] = np.where(
        changed["condition"].str.startswith("positive"), "positive", "negative"
    )
    changed["kind"] = np.where(changed["condition"].str.contains("control"), "control", "target")
    changed["signed_delta"] = np.where(
        changed["direction"] == "positive", changed["target_delta"], -changed["target_delta"]
    )
    changed["any_state_changed"] = (
        changed[[f"delta_{state}" for state in STATES]].abs().max(axis=1) > 0
    )

    pair_index = ["id_p", "id_s", "repeat_index", "target_trait", "direction", "scale"]
    pairs = changed.pivot(index=pair_index, columns="kind", values="signed_delta").reset_index()
    if pairs[["target", "control"]].isna().any().any():
        raise ValueError("Incomplete target-control pairs")
    pairs["specific_effect"] = pairs["target"] - pairs["control"]

    # The primary estimand gives every trait and direction equal weight within a cell.
    cells = pairs.groupby(["id_p", "id_s"], as_index=False)["specific_effect"].mean()
    primary = summarize_effect(cells, "specific_effect", args.bootstrap, args.seed, args.sesoi)

    by_trait = {}
    for trait in TRAITS:
        trait_cells = (
            pairs[pairs["target_trait"] == trait]
            .groupby(["id_p", "id_s"], as_index=False)["specific_effect"].mean()
        )
        by_trait[trait] = summarize_effect(
            trait_cells, "specific_effect", args.bootstrap,
            args.seed + 100 + TRAITS.index(trait), args.sesoi,
        )

    by_direction = {}
    for direction in ("positive", "negative"):
        direction_cells = (
            pairs[pairs["direction"] == direction]
            .groupby(["id_p", "id_s"], as_index=False)["specific_effect"].mean()
        )
        by_direction[direction] = summarize_effect(
            direction_cells, "specific_effect", args.bootstrap,
            args.seed + 200 + (direction == "negative"), args.sesoi,
        )

    sensitivity_bounds = {}
    low, high = primary["bootstrap_ci_95"]
    for bound in (0.10, 0.15, 0.20):
        sensitivity_bounds[str(bound)] = bool(low > -bound and high < bound)

    norm_summary = (
        changed.groupby(["target_trait", "condition"])["residual_delta_l2"]
        .mean().reset_index()
    )
    target_norm = norm_summary[norm_summary["condition"].isin(["positive", "negative"])].copy()
    control_norm = norm_summary[
        norm_summary["condition"].isin(["positive_control", "negative_control"])
    ].copy()
    control_norm["condition"] = control_norm["condition"].str.replace("_control", "", regex=False)
    norm_pairs = target_norm.merge(
        control_norm, on=["target_trait", "condition"], suffixes=("_target", "_control")
    )
    norm_pairs["ratio"] = norm_pairs["residual_delta_l2_control"] / norm_pairs["residual_delta_l2_target"]

    n_personas = int(table["id_p"].nunique())
    n_situations = int(table["id_s"].nunique())
    expected_rows = n_personas * n_situations * (1 + 5 * 4)
    quality = {
        "rows": int(len(table)),
        "expected_rows_from_observed_design": int(expected_rows),
        "complete_row_count": bool(len(table) == expected_rows),
        "personas": n_personas,
        "situations": n_situations,
        "success_rate": float(table["success"].mean()),
        "mean_valid_fraction": float(table["valid_fraction"].mean()),
        "target_any_state_change_fraction": float(
            changed[changed["kind"] == "target"]["any_state_changed"].mean()
        ),
        "control_any_state_change_fraction": float(
            changed[changed["kind"] == "control"]["any_state_changed"].mean()
        ),
        "norm_ratio_range": [float(norm_pairs["ratio"].min()), float(norm_pairs["ratio"].max())],
    }
    quality["confirmatory_assay_sensitivity_pass"] = bool(
        quality["complete_row_count"]
        and quality["success_rate"] >= 0.95
        and quality["mean_valid_fraction"] >= 0.95
        and quality["target_any_state_change_fraction"] >= 0.15
        and quality["norm_ratio_range"][0] >= 0.90
        and quality["norm_ratio_range"][1] <= 1.10
    )

    all_trait_equivalent = all(result["equivalent_within_sesoi"] for result in by_trait.values())
    conclusion = (
        "ASSAY_SENSITIVITY_NOT_REPLICATED_DO_NOT_CLAIM_NEGATIVE_CAUSAL_EFFECT"
        if not quality["confirmatory_assay_sensitivity_pass"]
        else
        "EQUIVALENT_OVERALL_AND_ALL_TRAITS"
        if primary["equivalent_within_sesoi"] and all_trait_equivalent
        else "EQUIVALENT_OVERALL_ONLY"
        if primary["equivalent_within_sesoi"]
        else "INCONCLUSIVE_OR_NONZERO_EFFECT"
    )
    report = {
        "analysis_contract": {
            "primary_estimand": "mean signed target-minus-matched-control change, equally weighted across five traits and two directions",
            "uncertainty": "two-way crossed bootstrap over personas and situations",
            "bootstrap_repetitions": args.bootstrap,
            "pre_specified_sesoi": args.sesoi,
            "equivalence_rule": "entire 95% crossed-bootstrap CI lies within [-SESOI,+SESOI]",
        },
        "quality": quality,
        "primary": primary,
        "by_trait": by_trait,
        "by_direction": by_direction,
        "primary_equivalence_sensitivity": sensitivity_bounds,
        "conclusion": conclusion,
        "claim_boundary": "Applies only to the frozen layer-16 OpenSAE, selected features, 8-SD last-four-token intervention, and measured state outcomes.",
    }
    (output / "confirmatory_equivalence_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pairs.to_csv(output / "paired_specific_effects.csv", index=False, encoding="utf-8-sig")
    cells.to_csv(output / "primary_cell_effects.csv", index=False, encoding="utf-8-sig")
    norm_pairs.to_csv(output / "effective_norm_matching.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, indent=2))
    print(f"\nAll outputs saved to: {output}")


if __name__ == "__main__":
    main()
