# BRFSS 2015: cardiovascular disease classification

NumPy implementation for EPFL CS-433 Project 1. The target `_MICHD` identifies
self-reported myocardial infarction or coronary heart disease.

The selected model is **combined18_24_features256**: histogram gradient boosting
with 297 processed inputs, 200 trees, depth 5 and 256 candidate features per node.
The team reported **AIcrowd F1 0.455 and accuracy 0.883** on 14 October 2026.
Its full-training OOF F1 0.444071 calibrates the threshold on the same labels;
**no nested evaluation of this exact combined pipeline has been completed**.

Read the [two-page final report](docs/FINAL_REPORT.pdf), its
[LaTeX source](docs/FINAL_REPORT.tex) and [supporting evidence](docs/FINAL_REPORT.md).
The retained [experiment catalog](docs/EXPERIMENTS.md) connects the report's
comparisons to records and reproduction commands.

## Setup

The official environment is Python 3.9 / NumPy 1.23.1. From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows use `py -3.9 -m venv .venv` and `.venv\Scripts\Activate.ps1`.
For tests and EDA plots install `requirements-dev.txt`. Training uses only NumPy
and the Python standard library; Matplotlib is used for visualization.

Download the official CSVs from the
[EPFL Project 1 challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and place them under `data/raw/dataset/`, following [data/README.md](data/README.md).

## Train the final model

```bash
python run.py --data-dir data/raw/dataset --output submission.csv
```

[configs/final_model.json](configs/final_model.json) fixes the preprocessing,
historical numerical implementation and threshold. The command trains on all
328,135 labelled rows; it requires no archived checkpoints. It produces 109,379
signed predictions in official sample order: **13,895 positive, 95,484 negative**.
The [submission record](results/eda/final_combined18_24_submission.json) retains
calibration, input/source hashes and the selected CSV SHA-256. Replay during the
release audit reproduced that CSV exactly; full training was not repeated.

Caches are created under `data/processed/`, with per-split receipts tying the
feature arrays to the CSV SHA-256, feature order and array hashes. Changed inputs,
modified arrays, incomplete caches and older caches without receipts are rejected.
Use a new `--cache-dir` to rebuild them; existing files are never deleted automatically.
Add `--checkpoint results/final_model.pkl` to save the fitted model.
CSV output uses the course helper in `helpers.py`, with explicit UTF-8 and LF.

## Report evidence and optional reproduction

The [experiment catalog](docs/EXPERIMENTS.md) and [retained results](results/README.md)
organize the existing scientific evidence. The 114 historical execution records,
the final candidate summaries and the report's numerical results are preserved.

The [optional reproduction guide](docs/REPRODUCIBILITY.md) maps each report study
to its configuration, backend and command. Its runner regenerates intermediate
artifacts from official CSVs, including the historical median-fill baselines,
two-seed ensemble, feature selection, sampling and final calibrations.
All 45 workflows passed reduced checks; a full numerical rerun of the complete
catalog was not completed. See the [audit](docs/FINAL_AUDIT.md) for actual coverage.
Full execution can take days. AIcrowd scores remain external reported feedback.

## Checks and contents

```bash
python -m tools.verify_repository
python -m unittest discover -s tests -q
git diff --check
```

The [audit](docs/FINAL_AUDIT.md) describes validation scope. GitHub Actions runs the
maintained tests and integrity checks with Python 3.9 / NumPy 1.23.1. The tests
use synthetic fixtures and the included text codebook, without official datasets.

| Location | Purpose |
| --- | --- |
| `run.py`, `helpers.py`, `implementations.py` | Final training/submission and six required methods |
| `src/`, `tests/` | Maintained models, preprocessing, evaluation, comparison runners and tests |
| [configs](configs/README.md) | Selected settings and report comparison recipes |
| [docs](docs/README.md) | Report, methods, experiment evidence and audit |
| [results](results/README.md) | Compact report evidence and historical ledger |
| [tools](tools/README.md) | Data preparation, calibration and retained study utilities |

Data, caches, models, individual predictions and submissions stay local. Earlier
screening and detailed development material remain recoverable from the
[development archive](docs/ARCHIVE.md). `HAREHAB1` is retained; its post-event
information limits interpretation as prediction before a cardiovascular event.
