# Paper-to-code map

Only main-text and Supplementary analyses are retained. The map uses scientific content rather than provisional figure/table numbering, which may change in Overleaf.

| Paper component | Retained implementation |
| --- | --- |
| Figure2 rendering | figures/main/joint_distribution_figure2.py |
| Figure2 centroid inference; Supplementary D.6 | statistics/geometry/cluster_figure2_centroids.py |
| Figure3 final numerical redraw | figures/main/figure3.py |
| Factor/SCSC fitting and diagnostics; Supplementary E | figures/main/plot_factor_modularity_centered_diamonds_phi_medium_ordered.R; statistics/factor/ |
| Figure4 reliability | statistics/reliability/RQ_8.R |
| Figure5 fitted slope-variance components | statistics/mixed_models/source_specific_random_slope_variance.py; figures/main/random_slope_figure5.py |
| Multilevel fixed effects, diagnostics and tables; Supplementary F | statistics/mixed_models/mixedlm_diagnostics.py, summarize_mixedlm_results.py, generate_mixedlm_appendix_tables.py |
| Supplementary conditional slope distributions | figures/supplementary/plot_source_specific_random_slopes_appendix.py (descriptive only) |
| Supplementary reliability aggregation sensitivity | statistics/reliability/human_person_situation_reliability_sensitivity.py; corresponding supplementary plot |
| Supplementary all-source factor/reliability panels | figures/supplementary/plot_supplementary_factor_all_models.py, plot_supplementary_reliability_all_models.py |
| Figure6b/c and Supplementary representation/interaction analyses | 04_sae/confirmatory_representation/sae_persona_situation_confirmatory.py |
| Supplementary UMAP resamples | 04_sae/confirmatory_representation/plot_sae_original_like_umaps.py |
| Formal Figure6d intervention generation | 04_sae/confirmatory_intervention/llama31_opensae_server_v7_confirmatory.py |
| Formal Figure6d crossed-bootstrap/equivalence | 04_sae/confirmatory_intervention/analyze_v7_confirmatory_equivalence.py |
| Figure6 final numerical redraw | figures/main/figure6_polished.py, exposed as reproduce.py figure6 |

Paths beginning statistics/ or figures/ are under03_behavior_analysis/. Supporting configuration, wrappers and tests are retained. No legacy calibration or exploratory generation directories are included. The exact final Overleaf numbering has not been re-audited from a newly exported full project.
