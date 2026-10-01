# BRFSS 2015: cardiovascular disease classification

NumPy implementation for EPFL CS-433 Project 1. The target `_MICHD` identifies
myocardial infarction or coronary heart disease reported in the BRFSS survey.
The dataset contains 328,135 labeled respondents, 321 predictors and 8.83%
positive labels.

The selected model is **Phase 14 histogram gradient boosting**, with 295
features cleaned according to the codebook, 200 trees and maximum depth 5.
It achieved the highest F1 among the completed comparisons.

| Pipeline | Development F1 |
| --- | ---: |
| Historical ridge | 0.41951 |
| Historical logistic regression | 0.42563 |
| **Phase 14 booster** | **0.44216** |
| MLP 64 → 32 | 0.42445 |
| MLP 128 → 64 | 0.41927 |

These scores use the same 262,508 development rows, three outer folds and
thresholds selected on two inner folds. Preprocessing is fitted within each
training partition. Linear scores refer to the historical median-fill
representation.

`HAREHAB1` is included. It records rehabilitation after a heart attack;
all 620 development respondents with a recorded code are positive. Removing
this feature lowers F1 to 0.43104. Its post-event information limits the
interpretation of the result as prediction before an event.

The [final report](docs/FINAL_REPORT.md), in Italian, explains the comparisons
and model choice.

## Setup

The official environment uses **Python 3.9 and NumPy 1.23.1**. From the
repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.9 -m venv .venv` and activate with
`.venv\Scripts\Activate.ps1`.

For tests and EDA plots, install the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Training uses NumPy and the Python standard library. Matplotlib is used for
EDA and its tests.

Download the official CSV files from the
[EPFL Project 1 AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and place them under `data/raw/dataset/`, following [data/README.md](data/README.md).

## Train and create a submission

```bash
python run.py --data-dir data/raw/dataset --output submission.csv
```

The command trains on all labeled rows and writes `Id,Prediction` with labels
`-1/+1`. Parameters and the decision threshold are in
[configs/final_model.json](configs/final_model.json). The threshold,
`0.21086909970855153`, is the mean of the three Phase 14 inner thresholds.

The recorded full-data run took 18 minutes 18 seconds and produced 109,379
test predictions. Its official test score remains to be recorded.

The command stores reusable caches under `data/processed/`. When changing
feature CSV values, use a new `--cache-dir`, including when IDs and shape stay
the same. Add `--checkpoint results/final_model.pkl` to save the fitted model.

## Reproduce validation

Prepare the caches, then run the selected cross-validation:

```bash
python -m tools.prepare_data
python -m src.run_boosting_cv configs/experiments/phase14_boosting_codebook_corrections_only.json
```

The fixed split is included under `results/eda/analysis/splits/`. Each outer
fold uses the threshold selected from its inner OOF predictions. The recorded
Phase 14 CV took approximately 46.6 minutes.

To reproduce the MLP comparison:

```bash
python -m src.run_mlp_cv configs/experiments/phase32_mlp.json
```

[MLP.md](docs/MLP.md) describes its preprocessing and training.
`run_mlp.py` provides the corresponding full-data training command.

## Checks

```bash
python -m tools.verify_repository
python -m unittest discover -s tests -q
git diff --check
```

The local suite passed 164 tests with Python 3.14.3 / NumPy 2.4.2.
GitHub Actions runs the checks with Python 3.9 / NumPy 1.23.1 on pushes and pull requests.
The tests use synthetic fixtures and the included text codebook.
[FINAL_AUDIT.md](docs/FINAL_AUDIT.md) records the completed checks.

## Project files

| Location | Purpose |
| --- | --- |
| `run.py` | Training and submission for the selected model |
| `implementations.py` | Six functions required by the assignment |
| `src/`, `tests/` | Models, preprocessing, evaluation and tests |
| [configs/README.md](configs/README.md) | Model and experiment configurations |
| [docs/README.md](docs/README.md) | Report and method documentation |
| [results/README.md](results/README.md) | Analyses and experiment records |
| [tools/README.md](tools/README.md) | Data preparation and study utilities |

Git includes code, configurations, reports and the fixed development split.
Data, caches, checkpoints, row-level predictions and submissions are stored
locally.
