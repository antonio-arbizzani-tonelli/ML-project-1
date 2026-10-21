# BRFSS 2015: cardiovascular disease classification

NumPy implementation for EPFL CS-433 Project 1. The target `_MICHD` identifies
myocardial infarction or coronary heart disease reported in the BRFSS survey.
The dataset contains 328,135 labeled respondents, 321 predictors and 8.83%
positive labels.

The selected model is **`combined18_24_features256` histogram gradient boosting**,
with 297 processed features, 200 trees, maximum depth 5 and 256 candidate
features per node. It combines Phase 18 diet-column exclusions with restricted
Phase 24 imputation. Its AIcrowd submission scores **F1 0.455 and accuracy 0.883**
(team-reported feedback on 14 October 2026, displayed to three decimals).
Later growth, regularization, categorical and structural trials have lower
reported F1. The earlier combined model has the same displayed F1 and accuracy
0.882; rounding does not establish an exact F1 gain or an official tie-break.

**There is no nested development evaluation of this exact combined pipeline.**
Its full-training OOF F1, **0.44407**, calibrates its threshold on the same
labels and is not an independent test result. The nested results below describe
earlier pipelines. The selected model retains the exact historical numerical
source in `src/numpy_boosting_phase14.py`; corrected Phase 34 remains available.

| Pipeline | Development F1 |
| --- | ---: |
| Historical ridge | 0.41951 |
| Historical logistic regression | 0.42563 |
| Historical Phase 14 booster | 0.44216 |
| Corrected Phase 34 booster | 0.44080 |
| MLP 64 → 32 | 0.42445 |
| MLP 128 → 64 | 0.41927 |

These scores use the same 262,508 development rows, three outer folds and
thresholds selected on two inner folds. Preprocessing is fitted within each
training partition. Linear scores refer to the historical median-fill
representation.

`HAREHAB1` is included. It records rehabilitation after a heart attack;
all 620 development respondents with a recorded code are positive. The
historical Phase 15 ablation lowered F1 to 0.43104. Its post-event information limits the
interpretation of the result as prediction before an event.

The [two-page final report](docs/FINAL_REPORT.pdf), in English, explains the
comparisons and model choice. Its [LaTeX source](docs/FINAL_REPORT.tex) is
editable; [supporting notes](docs/FINAL_REPORT.md) retain the evidence links.

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
`0.22007772525099117`, maximizes F1 on pooled three-fold OOF predictions over
all **328,135 labeled rows**, with preprocessing fitted within each training
fold. Calibration F1 is 0.444071; it uses the threshold-selection labels and
is separate from the nested development score and AIcrowd feedback.

The selected full-data run produces **109,379 predictions: 13,895 positive
and 95,484 negative**. Its source, threshold, input and CSV fingerprints are
in the [final submission record](results/eda/final_combined18_24_submission.json).
Replay with current preprocessing and the normal model module reproduces the
chosen CSV byte for byte. Full training was not repeated for the release audit.
The historical Phase 14 submission can be reproduced with
`python run.py --config configs/final_phase14_model.json`.

CSV output uses `create_csv_submission` in [helpers.py](helpers.py), adapted
from the [official EPFL starter](https://github.com/epfml/ML_course/blob/main/projects/project1/helpers.py).
Explicit UTF-8 and LF line endings preserve the selected submission's bytes.

The command stores reusable caches under `data/processed/`. When changing
feature CSV values, use a new `--cache-dir`, including when IDs and shape stay
the same. Add `--checkpoint results/final_model.pkl` to save the fitted model.

## Reproduce validation

Prepare the caches, then reproduce the historical nested reference:

```bash
python -m tools.prepare_data
python -m src.run_boosting_cv configs/experiments/final_phase14_validation.json
```

The fixed split is included under `results/eda/analysis/splits/`. Each outer
fold uses the threshold selected from its inner OOF predictions. This suite
explicitly selects the included frozen Phase 14 implementation. For the
corrected comparison, use `configs/experiments/phase34_missing_only_split.json`.
The [utility guide](tools/README.md) explains full-train calibration and all
later combined studies. Their drivers freeze local checkpoints and source
snapshots; some follow-up studies require the preceding run's local artifacts.
The ordinary `run.py` command needs only the official CSVs and tracked files.

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

The [release audit](docs/FINAL_AUDIT.md) records current test results, checks on
a clean package without official data or saved models, and submission replay.
The local verification environment is Python 3.14.3 / NumPy 2.4.2.
GitHub Actions runs the checks with Python 3.9 / NumPy 1.23.1 on pushes and pull requests.
The tests use synthetic fixtures and the included text codebook.
[FINAL_AUDIT.md](docs/FINAL_AUDIT.md) records the completed checks.

## Project files

| Location | Purpose |
| --- | --- |
| `run.py` | Training and submission for the selected model |
| `helpers.py` | Course CSV submission helper, with explicit UTF-8 and LF output |
| `implementations.py` | Six functions required by the assignment |
| `src/`, `tests/` | Models, preprocessing, evaluation and tests |
| [configs/README.md](configs/README.md) | Model and experiment configurations |
| [docs/README.md](docs/README.md) | Report and method documentation |
| [results/README.md](results/README.md) | Analyses and experiment records |
| [tools/README.md](tools/README.md) | Data preparation and study utilities |

Git includes code, configurations, reports and the fixed development split.
Data, caches, checkpoints, row-level predictions and submissions are stored
locally.
