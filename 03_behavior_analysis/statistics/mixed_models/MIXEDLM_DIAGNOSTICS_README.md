# MixedLM diagnostics implementation

## New outputs

The enhanced `mixedlm_diagnostics.py` produces:

- `mixed_model_convergence_diagnostics.csv`: every optimizer attempt, convergence flag, warning, exception, log likelihood, observations, groups and sources.
- `mixed_model_results_12sources.csv`: fixed effects, standard errors, 95% Wald confidence intervals, raw p-values, BH-FDR/Holm adjusted interaction p-values and variance diagnostics.
- `mixed_model_lrt_results.csv`: full-versus-reduced likelihood-ratio tests with BH-FDR and Holm correction across the 40 state-by-situation tests.
- `mixed_model_sample_sizes.csv`: observations, participant/persona groups and observations-per-group summaries for each source.
- `random_slopes_12sources.csv`: participant/persona conditional random-slope means.

## Models and inference

For every Big Five state and DIAMONDS dimension:

```text
Full:    state ~ x_c * C(source)
Reduced: state ~ x_c + C(source)
Random:  correlated intercept and x_c slope by participant/persona
Fit:     maximum likelihood, reml=False
```

The likelihood-ratio statistic is `2 * (full llf - reduced llf)`. Degrees of freedom equal the difference in fixed-effect parameter counts.

Two multiple-testing families are kept separate:

1. the 40 omnibus source-by-situation likelihood-ratio tests;
2. all individual source-by-situation interaction coefficients.

Both families receive Benjamini-Hochberg FDR and Holm corrections.

## Confidence intervals

The script reports 95% Wald confidence intervals for fixed effects. Extracted random slopes are empirical-Bayes conditional means, so fixed-effect intervals must not be presented as uncertainty intervals for individual slopes. Individual random-slope intervals require a separate participant/persona-level bootstrap.

## Sample-size limitation

The script reads flattened scene-level files. It can report rows and groups used by MixedLM but cannot reconstruct five raw calls, failures, parse errors, retries or default-filled items. Those counts must come from raw response logs.

## Source-map issues

- The available `simulation_output_flattened-phi3-mini.csv` is treated as `Phi-3-mini-4k-instruct`; it is not counted twice under a second `Phi-3 mini` label.
- `qwen05`, `qwen15` and `qwen3` appear in `SOURCE_FILES` but not in `SOURCE_ORDER`, so they are not analyzed.

## Run requirement

Run the script in the original analysis environment containing `numpy`, `pandas`, `scipy`, `statsmodels` and `matplotlib`. This Codex environment could perform syntax validation but did not contain the complete statistical runtime needed for a real fit.
