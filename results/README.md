# Retained results and evidence

The current tree keeps compact evidence supporting the report. Large per-column
audits, preflights, partial rankings and duplicate release summaries are recoverable
from the [development archive](../docs/ARCHIVE.md).

| Location | Evidence |
| --- | --- |
| [Final combined submission](eda/final_combined18_24_submission.json) | Selected model, calibration, reported feedback, source/input hashes and exact CSV replay |
| [Seventeen later candidates](eda/final_combined_experiments_summary.json) | Aggregate metrics, parameters and provenance of all later reported comparisons |
| [Pipeline integrity](eda/final_pipeline_integrity.json) | Separate receipt freezing numerical sources, metadata/configuration, report and compact outcomes |
| [Dataset summary](eda/dataset_summary.json) | Portable measured counts copied from the original audit |
| [Cleaning counts](eda/phase33_preprocessing_counts.json) | Recode counts behind the report's cleaning description |
| [Historical Phase 14 submission](eda/final_phase14_submission.json) | Previous full-training reference and its calibrated threshold |
| [Main ledger](experiments/index.csv) | 108 historical runs, unchanged JSON bytes/hash rows |
| [Feature selection](experiments/phase29_feature_selection/index.csv) | Two reduced-input candidates |
| [Neural comparison](experiments/mlp/index.csv) | Two neural candidates |
| [Raw-input comparison](experiments/phase33_raw_preprocessing/index.csv) | One matched raw-input run |
| [Split correction](experiments/phase34_missing_only_split/index.csv) | One matched corrected-backend run |
| [Fixed split](eda/analysis/splits/stratified_seed_20260918.npz) |Development and initial-validation row indices |

The five indices cover 114 immutable records. Additional small comparison JSONs
and retained study summaries support the [report catalog](../docs/EXPERIMENTS.md).
Dated `aicrowd_*` folders keep aggregate verification/feedback only; prediction
CSVs and machine-specific delivery manifests remain local.

Historical JSON fields can name files outside the current checkout. Such paths
are execution provenance, not promises of fresh-checkout replay. Their original
recipes and detailed artifacts are described in the archive note. Use the
[configuration catalog](../configs/README.md) for current commands and prerequisites.

Nested development F1, same-label OOF threshold calibration and reported AIcrowd
feedback have different scopes. The exact combined submission has no completed
nested evaluation. The [delivery audit](../docs/FINAL_AUDIT.md) covers maintained
code and integrity; earlier test counts are historical evidence, not current totals.
