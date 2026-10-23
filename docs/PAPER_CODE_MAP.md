# Paper-to-code map

| Paper component | Code |
| --- | --- |
| Persona construction | generation/personas.py |
| Baseline questionnaire assessment | generation/traits.py |
| Situation-conditioned state generation | generation/responses.py |
| Figure2 geometry rendering | figures/figure2.py |
| Figure2 centroid inference | analysis/geometry.py |
| Factor/SCSC fitting | analysis/factor_model.R |
| Factor diagnostics and tables | analysis/factor_diagnostics.py; analysis/factor_tables.R |
| Figure3 final redraw | figures/figure3.py |
| Figure4 reliability fitting and heatmaps | analysis/reliability.R |
| Figure5 variance-component fitting | analysis/random_slopes.py |
| Figure5 final redraw | figures/figure5.py |
| Multilevel diagnostics and tables | analysis/mixed_models.py; analysis/mixed_model_summary.py; analysis/mixed_model_tables.py |
| Reliability aggregation sensitivity | analysis/reliability_sensitivity.py |
| Supplementary figures | figures/supplementary/ |
| SAE input validation, checkpoint validation, extraction, discovery, intervention generation | sae/pipeline.py |
| Figure6 representation/decoding/interaction analyses | sae/representations.py |
| SAE UMAP resamples | sae/umap.py |
| Figure6d crossed-bootstrap/equivalence | sae/interventions.py; sae/reproduce_interventions.py |
| Figure6 redraw | figures/figure6.py |

Use run.py at repository root for supported task entry points. See generation/README.md for generation configuration. Data and model checkpoints are supplied separately.
