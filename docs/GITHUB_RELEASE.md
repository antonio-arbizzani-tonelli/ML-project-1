# Project release and submission

This release selects **combined18_24_features256**: 297 processed features,
200 trees of depth 5, 256 candidate features per node and decision threshold
**0.22007772525099117**. Reported AIcrowd F1 is **0.455**, accuracy **0.883**.
The [two-page English report](FINAL_REPORT.pdf), its [LaTeX source](FINAL_REPORT.tex)
and [supporting notes](FINAL_REPORT.md) describe the final pipeline and its limits.

## Included material

The repository contains the six required functions, executable training and
submission scripts, NumPy model implementations, experiment configurations,
tests, the text codebook, documentation, the fixed development split and
aggregate experiment evidence. The 114 historical indexed records are preserved;
17 later calibrated candidates and their reported feedback are summarized in
[the combined-study record](../results/eda/final_combined_experiments_summary.json).

Official datasets, numeric caches, checkpoints, row-level predictions,
submission CSVs, logs, duplicate local manifests and optional local scheduling
helpers are excluded. Dated compact verification and AIcrowd feedback JSONs
are retained. Original local experiment files remain unchanged.

[The selected submission record](../results/eda/final_combined18_24_submission.json)
links parameters, calibration, input/source fingerprints and the chosen CSV
fingerprint. The frozen historical numerical implementation and hashed recipes
retain their original bytes through Git attributes. Historical Phase 14 remains
reproducible using `configs/final_phase14_model.json`.

## Reproduce from a fresh checkout

Follow the [README](../README.md) for environment setup and the official CSVs.
From the repository root:

```bash
python -m tools.prepare_data
python run.py --data-dir data/raw/dataset --output submission.csv
```

This trains on all 328,135 labelled rows. The expected signed CSV has 109,379
rows, 13,895 positive and 95,484 negative predictions, in official sample order.
Its SHA-256 is `d01c3949dea567dc9aa929ba232cbe77395d236b1c4a4492624e0bec5305e883`.
Use a fresh cache directory if feature values change.

Historical nested evaluation is reproduced with:

```bash
python -m src.run_boosting_cv configs/experiments/final_phase14_validation.json
```

The exact selected combined pipeline has **no completed nested evaluation**.
Its OOF F1 0.444071 calibrates the threshold on the same labels. The separate
`configs/experiments/final_combined_validation.json` supplies an explicit
historical implementation for a fresh nested study; its configuration is
not evidence that the study has been executed. The original full-training
calibration drivers and their artifact prerequisites are in
[tools/README.md](../tools/README.md).

## Release checks and hand-in

```bash
python -m tools.verify_repository
python -m unittest discover -s tests -q
git diff --cached --check
```

[The dated audit](FINAL_AUDIT.md) records actual checks and remaining runtime
limits. GitHub Actions is configured for Python 3.9 / NumPy 1.23.1. Check the
workflow on the published commit; local checks use Python 3.14.3 / NumPy 2.4.2.

The course requires a public repository link to the exact submitted commit:
`https://github.com/antonio-arbizzani-tonelli/ML-project-1/tree/COMMIT_HASH`.
Submit that link and the report PDF through the course submission system.
AIcrowd uploads and the course hand-in are separate from a Git push.
