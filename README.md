# BRFSS 2015: cardiovascular disease classification

NumPy implementation for EPFL CS-433 Project 1. The target `_MICHD` identifies
myocardial infarction or coronary heart disease reported in the BRFSS survey.
The dataset contains 328,135 labeled respondents, 321 predictors and 8.83%
positive labels.

The selected model is **Phase 14 histogram gradient boosting**, with 295
features cleaned according to the Phase 14 codebook plan, 200 trees and
maximum depth 5. Its AIcrowd submission scores **F1 0.452 and accuracy 0.878**
(team-reported feedback, displayed to three decimals). Nested development F1
is **0.44216**.

An implementation audit recovered omitted splits separating observed values
from missing responses. Phase 34 evaluates the correction with the same
295 columns, parameters and nested folds. Historical Phase 14 reaches
0.44216 with the earlier search (commit
`aa9a23e907015e91eda75d785c0425172ac32558`); the correction changes F1 by
−0.00135 while AP rises by 0.00017. After full-train OOF calibration and
submission of both versions, the team selected Phase 14 based on AIcrowd
feedback. Its exact historical source is included as
`src/numpy_boosting_phase14.py`; the corrected Phase 34 remains available.

| Pipeline | Development F1 |
| --- | ---: |
| Historical ridge | 0.41951 |
| Historical logistic regression | 0.42563 |
| **Selected Phase 14 booster** | **0.44216** |
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
`0.2106169795070252`, maximizes F1 on pooled three-fold OOF predictions over
all **328,135 labeled rows**, with preprocessing fitted within each training
fold. Calibration F1 is 0.442770; it uses the threshold-selection labels and
is separate from the nested development score and AIcrowd feedback.

The selected full-data run produces **109,379 predictions: 14,746 positive
and 94,633 negative**. Its source, threshold, input and CSV fingerprints are
in the [final submission record](results/eda/final_phase14_submission.json).
The command above uses that Phase 14 source and calibrated threshold.

CSV output uses `create_csv_submission` in [helpers.py](helpers.py), adapted
from the [official EPFL starter](https://github.com/epfml/ML_course/blob/main/projects/project1/helpers.py).
Explicit UTF-8 and LF line endings preserve the selected submission's bytes.

The command stores reusable caches under `data/processed/`. When changing
feature CSV values, use a new `--cache-dir`, including when IDs and shape stay
the same. Add `--checkpoint results/final_model.pkl` to save the fitted model.

## Reproduce validation

Prepare the caches, then run the selected cross-validation:

```bash
python -m tools.prepare_data
python -m src.run_boosting_cv configs/experiments/final_phase14_validation.json
```

The fixed split is included under `results/eda/analysis/splits/`. Each outer
fold uses the threshold selected from its inner OOF predictions. This suite
explicitly selects the included frozen Phase 14 implementation. For the
corrected comparison, use `configs/experiments/phase34_missing_only_split.json`.
The [utility guide](tools/README.md) also explains full-train calibration of
both implementations.

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

All **206 project tests** and **26 public course tests** passed locally,
including a copy of the Git-eligible package without official data or saved
models. The course URL check awaits the final commit; its optional Black
formatting check was excluded. The local verification environment is
Python 3.14.3 / NumPy 2.4.2.
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
