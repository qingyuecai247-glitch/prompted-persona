# Prompted Persona

Analysis and figure-generation code accompanying **Language model personas express traits but not human personality dynamics**.

## Setup

Python3.11+ is recommended. Install the plotting/statistical dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the synthetic example and Figure6d CPU statistics, the smaller requirements-sae-v7-cpu.txt is sufficient. R analyses use the packages listed in environment.yml.

## Data

Study inputs and archived numerical tables are provided separately. Use --data-root for the raw analysis inputs and --reference-root for the reviewer archive's code/data/reference directory.

Human reference: Kuper et al.(2024), https://doi.org/10.1037/pspp0000473 ; original study materials: https://osf.io/4ps29/ . Third-party data retain their original terms. This repository contains code and synthetic examples; study observations, questionnaires, situation datasets, original named personas and model weights are not included.

## Figures

```sh
python scripts/reproduce.py figure2 --data-root /path/to/reviewer_archive/code/data/raw
python scripts/reproduce.py figure3 --reference-root /path/to/reviewer_archive/code/data/reference
python scripts/reproduce.py figure5 --reference-root /path/to/reviewer_archive/code/data/reference
python scripts/reproduce.py figure6 --reference-root /path/to/reviewer_archive/code/data/reference
```

Use --output-root to choose an output directory. Figure3 produces SVG; PDF export additionally uses Node and package.json dependencies. Figure6 redraws numerical source tables. The figure commands and statistical commands are separate entry points.

## Statistical analyses

```sh
python scripts/reproduce.py figure2-cluster --data-root /path/to/reviewer_archive/code/data/raw
python scripts/reproduce.py figure6-v7-statistics --reference-root /path/to/reviewer_archive/code/data/reference
```

Figure2 centroid inference uses person-cluster Wald statistics and 199,999 whole-person bootstrap draws (seed20261021). Figure6d uses three formal v7 intervention shards, 5,000 crossed-bootstrap draws (seed20260720). Outputs are written under outputs/statistics.

## Synthetic example

```sh
python scripts/demo_cluster.py
python -m unittest discover -s tests -p test_figure2_cluster.py -v
```

The example contains48 synthetic observations from12 simulated people and writes outputs/demo_cluster/summary.json. It is an illustration of clustered mean uncertainty.

## Organization

- 03_behavior_analysis/: main and Supplementary statistical analyses and plots.
- 04_sae/: formal representation and intervention analyses.
- scripts/: command-line wrappers, example and figure export.
- config/: portable paths and Figure2 dataset map.
- examples/: synthetic demonstration input.
- docs/PAPER_CODE_MAP.md: paper-to-code correspondence.

Retained formal GPU scripts require model/SAE checkpoints and locally configured data paths. ./local_assets denotes a user-supplied local location. Full data access details should be supplied with the submission. License selection is pending author approval; see LICENSE_NOTICE.md.
