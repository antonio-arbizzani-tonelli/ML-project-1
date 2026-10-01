# Results and evidence

Phase 14 reached development F1 **0.442155227**. The
[final report](../docs/FINAL_REPORT.md) explains the model comparison and choice.

| Location | Contents |
| --- | --- |
| `eda/` | Dataset analyses and experiment summaries |
| [Main experiment index](experiments/index.csv) | 102 recorded runs |
| [Feature-selection index](experiments/phase29_feature_selection/index.csv) | Two reduced-feature candidates |
| [MLP index](experiments/mlp/index.csv) | Two neural candidates |
| [Full-data training](eda/final_training_summary.json) | Runtime, prediction counts and submission fingerprint |
| [Project checks](eda/final_readiness.json) | Test results, integrity checks and execution environment |
| [Fixed split](eda/analysis/splits/stratified_seed_20260918.npz) | Development and initial-validation row indices |

The three indices cover **106 records** and store a SHA-256 fingerprint for
each JSON record. [EXPERIMENTS.md](../docs/EXPERIMENTS.md) connects the studies
to their results.

## Reading the scores

The final comparisons select F1 thresholds on inner OOF predictions and
score the outer folds. In historical booster JSONs, this result is stored in
`outcome.nested_threshold_evaluation.pooled_metrics`. The index F1 column and
`outcome.metrics` can instead contain fixed-threshold or exploratory scores,
including a threshold selected on the same pooled OOF rows being scored.
The experiment summaries identify the protocol used for each result.

## Generated artifacts

Training saves checkpoints, OOF arrays and logs under `boosting_artifacts/`,
`mlp_artifacts/` or the configured output directory. These files, numeric data
caches and submission CSVs are excluded from Git.

To reproduce CV, download the official CSVs and run
`python -m tools.prepare_data` from the repository root before running the
experiment configuration. Exact replay requires the saved checkpoints and
prediction arrays listed in the study report; a new CV run generates its own
artifacts.
