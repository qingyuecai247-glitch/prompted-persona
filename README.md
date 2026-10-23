# Prompted Persona

Code for **Language model personas express traits but not human personality dynamics**.

## Four places to look

| Folder | Purpose | Start here |
| --- | --- | --- |
| generation/ | Create personas and generate LLM questionnaire responses | generation/README.md |
| analysis/ | Behavioural statistics: PCA geometry, factor structure, reliability and random slopes | geometry.py, factor_model.R, reliability.R, random_slopes.py |
| sae/ | SAE validation/extraction, representations and interventions | pipeline.py, representations.py, interventions.py |
| figures/ | Main figures and Supplementary plots | figure2.py through figure6.py; Figure4 is produced by analysis/reliability.R |

**Use run.py at repository root to choose a task.** You do not need to navigate nested numbered directories.

## Setup

Python3.11+ recommended:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

Optional client dependencies: requirements/generation.txt. NumPy/pandas-only statistics: requirements/sae_cpu.txt. R packages: requirements/environment.yml. Node dependencies in package.json are only needed for Figure3 PDF export.

## Generate data

Configure model endpoint/credentials and input/output locations with environment variables (see generation/README.md and .env.example), then run only the required stage:

```sh
python run.py generate-personas
python run.py assess-traits
python run.py generate-responses
```

These stages invoke external models; installing dependencies and running statistical commands does not launch generation.

## Reproduce figures

Point to the separately supplied reviewer archive:

```sh
python run.py figure2 --data-root /path/to/reviewer_archive/code/data/raw
python run.py figure3 --reference-root /path/to/reviewer_archive/code/data/reference
python run.py figure4 --data-root /path/to/reviewer_archive/code/data/raw --reference-root /path/to/reviewer_archive/code/data/reference
python run.py figure5 --reference-root /path/to/reviewer_archive/code/data/reference
python run.py figure6 --reference-root /path/to/reviewer_archive/code/data/reference
```

Use --output-root to choose where results are written. Figure3 creates SVG; scripts/export_figure3.cjs additionally exports PDF/PNG.

## Reproduce statistical results

```sh
python run.py figure2-cluster --data-root /path/to/reviewer_archive/code/data/raw
python run.py figure6-v7-statistics --reference-root /path/to/reviewer_archive/code/data/reference
```

Figure2 inference uses199,999 whole-person draws, seed20261021. Figure6d uses three formal v7 shards and5,000 crossed-bootstrap draws, seed20260720. Figure redraws and statistical computations are separate tasks. See docs/PAPER_CODE_MAP.md for detailed correspondence.

## Try without study data

```sh
python scripts/demo_cluster.py
python -m unittest discover -s tests -p test_figure2_cluster.py -v
```

The demo uses48 wholly synthetic observations from12 simulated people. Study observations, questionnaires, situation datasets, named personas and model weights are provided separately.

## Supporting folders

- data/: instructions for external raw and reference inputs.
- config/: paths and Figure2 dataset mapping.
- requirements/: analysis, generation and SAE CPU dependencies.
- scripts/: small export/demo helpers.
- tests/: offline checks.
- examples/: synthetic input.
- outputs/: generated results (ignored by Git).
- docs/: paper-to-code map and environment/configuration notes.

Human source: Kuper et al.(2024), https://doi.org/10.1037/pspp0000473 ; https://osf.io/4ps29/ . Third-party terms apply. Set service credentials and local assets yourself; no private server configuration is included. A project-wide code license remains pending author approval (LICENSE_NOTICE.md).

## BFI-2 scoring

Full60-item domain scoring is shared by the generation clients. See [scoring instructions](docs/BFI2_SCORING.md). Existing item-level CSVs can be rescored with generation/score_bfi2.py.
