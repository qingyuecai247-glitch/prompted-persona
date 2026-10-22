# Prompted Persona

Code accompanying **Language model personas express traits but not human personality dynamics**. Research code snapshot prepared 2026-10-22. Repository: https://github.com/qingyuecai247-glitch/prompted-persona . This is a work-in-progress snapshot, not a final reproducibility release.

This repository contains retained generation and analysis code, portable statistical/figure entry points and a synthetic demo. Full study data, SAE outputs, model weights and old server histories are distributed separately in the reviewer archive. See [analysis status](docs/ANALYSIS_STATUS.md) before using the reported results. Figure5 scale comparability and Figure2 legacy plot annotations remain unresolved.

## Quick start: synthetic demonstration

Python3.11+ recommended. The demo and formal v7 CPU statistics require NumPy and pandas only.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sae-v7-cpu.txt
python scripts/demo_cluster.py
python -m unittest discover -s tests -p test_figure2_cluster.py -v
```

The demo writes outputs/demo_cluster/summary.json using simulated observations with no human/persona identities. It illustrates clustered mean uncertainty only; it does not reproduce a paper effect. Dependencies and output are small. Installation/runtime benchmarks have not been standardized.

## Reproduce paper statistics with the separate reviewer archive

Unzip reviewer_materials_20261022_v3.zip. Set the reference/data paths to its code/data directories:

```sh
python scripts/reproduce.py figure6-v7-statistics --reference-root /path/to/reviewer_materials_20261022_v3/code/data/reference
python scripts/reproduce.py figure2-cluster --data-root /path/to/reviewer_materials_20261022_v3/code/data/raw
```

Figure6d uses 25,200 records from three formal v7 shards, 5,000 bootstrap draws, seed20260720. Expected overall estimate0.0057777778 and 95% CI[-0.0072229167,0.0188069444]. The generic intervention_results_long.csv is an old experiment and is not selected. Figure2 defaults to 199,999 whole-person draws and seed20261021. Output directories can be changed using --output-root.

## Figures and full environments

```sh
python -m pip install -r requirements.txt
python scripts/reproduce.py figure3 --reference-root /path/to/reviewer_archive/code/data/reference
python scripts/reproduce.py figure5 --reference-root /path/to/reviewer_archive/code/data/reference
python scripts/reproduce.py figure6-polished --reference-root /path/to/reviewer_archive/code/data/reference
```

These redraw retained results rather than refitting all original models. Figure3 produces SVG; PDF export additionally uses Node and package.json dependencies. Historical `figure2-legacy` has outdated record-level Hotelling stars; current inference is `figure2-cluster`. R and GPU pipelines require additional dependencies, original checkpoints and explicit configuration; they are not covered by the CPU validation. requirements.txt/environment.yml contain compatibility ranges, not a locked, universally verified environment.

## Layout

- 01_persona/: retained persona-generation code
- 02_behavior_generation/: retained state-response generation
- 03_behavior_analysis/: geometry, factor, mixed-model, reliability and figure scripts
- 04_sae/: base pipeline and formal representation/intervention analysis
- scripts/: supported portable entry points and synthetic demo
- examples/: small synthetic test data only
- docs/: limitations, environment provenance and repository release checklist

## Data and access

Human data originate from Study1 of Kuper et al. (2024), https://doi.org/10.1037/pspp0000473 ; original materials https://osf.io/4ps29/ . Third-party redistribution terms remain to be checked. Study observations and result tables are excluded from Git. The reviewer archive access URL must be supplied by the authors; a private GitHub URL alone is not reviewer access. No public data DOI is claimed.

## License and release

License is pending author approval; see LICENSE_NOTICE.md. This repository is public; it does not contain study observations, the original named-persona cohort, server logs or model weights. Read docs/RELEASE_CHECKLIST.md before designating a publication release. Do not upload the entire workspace or model weights.
