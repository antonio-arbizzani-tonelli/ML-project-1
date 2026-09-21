# BRFSS cardiovascular-risk classification

NumPy-only binary classification for EPFL CS-433 Project 1. The target is
myocardial infarction or coronary heart disease (`_MICHD`) in the 2015 BRFSS
survey. The positive class represents 8.83% of the training data.

## Current result

All reported finalist metrics use the same 262,508 development rows and nested
F1-threshold selection. Preprocessing and threshold fitting are restricted to
each outer fold's training rows.

| Model | F1 | Precision | Recall | Average precision | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge regression | 0.41951 | 0.33303 | 0.56665 | 0.39028 | 0.22917 |
| Logistic regression | 0.42563 | 0.35728 | 0.52632 | 0.38775 | 0.22198 |
| NumPy histogram boosting | **0.44216** | **0.36763** | **0.55457** | **0.42861** | **0.21523** |

The tree model is the current reference. It captures nonlinear thresholds and
feature interactions that the linear baseline did not recover. Accuracy is not
used alone: predicting every row as negative already gives 91.17% accuracy and
F1 zero.

The selected model contains 200 depth-5 histogram trees, samples 128 candidate
features per node, and routes missing values explicitly. Full settings and the
experiment path are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Repository layout

```text
implementations.py        six functions required by the public grader
run.py                    full-data training and submission entry point
configs/                  final model, preprocessing, and experiment settings
src/                      NumPy models, preprocessing, evaluation, and runners
tests/                    local unit and integration tests
docs/                     data handling and experiment decisions
data/README.md            dataset placement and verified schema
results/eda/              compact analyses and figures
results/experiments/      machine-readable experiment ledger
```

Raw data, NumPy caches, model checkpoints, OOF predictions, and submissions are
excluded from Git.

## Setup

The official grading environment uses Python 3.9 and NumPy 1.23.1.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Download the official AIcrowd dataset and place its four CSV files as described
in [data/README.md](data/README.md). No external dataset or ML library is used.

## Create a submission

From the repository root:

```bash
python run.py --data-dir data/raw/dataset --output submission.csv
```

The first run parses the CSV files and creates local float32 caches under
`data/processed/`. It then:

1. applies the selected 295-feature codebook-aware representation;
2. trains the 200-tree booster on all 328,135 labeled rows;
3. applies the frozen threshold `0.2108690997`;
4. writes `Id,Prediction` with labels in `{-1, +1}`.

The default model is defined in `configs/final_model.json`. To retain the fitted
model locally, add `--checkpoint results/final_model.pkl`. Checkpoints are
pickle files and must only be loaded from trusted local runs.

The entry point was verified end to end on the complete official data: it
produced all 109,379 test predictions in 18 minutes 18 seconds on the
development machine. Runtime depends on CPU speed.

## Reproduce the selected validation result

The complete three-outer-fold plus two-inner-fold evaluation is:

```bash
python -m src.run_boosting_cv configs/experiments/phase14_boosting_codebook_corrections_only.json
```

This run produced nested F1 `0.442155`, precision `0.367632`, and recall
`0.554573`. It took 46.6 minutes on the development machine. The record is
`results/experiments/20260921T105139261213Z_phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json`.

## Tests

Run the repository tests:

```bash
python -m unittest discover -s tests -q
```

The official public tests are intentionally not copied into this repository.
From the `grading_tests` directory of the official course repository, test a
local checkout with:

```bash
pytest --github_link /absolute/path/to/this/repository . -k "not github_link_format"
```

For submission, pass an immutable GitHub commit URL:

```bash
pytest --github_link https://github.com/OWNER/REPOSITORY/tree/COMMIT_HASH .
```

The grader requires root-level `README.md`, `run.py`, and `implementations.py`,
the exact six function signatures, function docstrings, NumPy-only project
logic, scalar losses, and no unfinished markers in Python files.

## Limitations and next work

- `HAREHAB1` may encode target-conditioned survey eligibility and requires a
  controlled ablation before scientific interpretation.
- Several food and exercise variables mix daily, weekly, and monthly codes;
  their standardized representation is not yet a finalist.
- The current threshold is transferred from nested development folds. A final
  AIcrowd submission has not yet been frozen in the experiment ledger.

The next experiments and their decision criteria are listed in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#highest-priority-next-tests). Dataset
semantics and known preprocessing gaps are in [docs/DATA.md](docs/DATA.md).
