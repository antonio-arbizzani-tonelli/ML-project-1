# Results and evidence

The selected Phase 14 booster scores **F1 0.452 and accuracy 0.878 on AIcrowd**,
as reported by the team. Its nested development F1 is **0.442155227**;
the corrected Phase 34 comparison reaches **0.440803240**. The
[final report](../docs/FINAL_REPORT.md) explains the model comparison and choice.

| Location | Contents |
| --- | --- |
| `eda/` | Dataset analyses and experiment summaries |
| [Main experiment index](experiments/index.csv) | 108 recorded runs |
| [Feature-selection index](experiments/phase29_feature_selection/index.csv) | Two reduced-feature candidates |
| [MLP index](experiments/mlp/index.csv) | Two neural candidates |
| [Raw-input benchmark index](experiments/phase33_raw_preprocessing/index.csv) | One controlled preprocessing comparison |
| [Split-search correction index](experiments/phase34_missing_only_split/index.csv) | One controlled implementation correction |
| [Full-data training](eda/final_training_summary.json) | Historical Phase 14 runtime, prediction counts and submission fingerprint |
| [Selected submission](eda/final_phase14_submission.json) | Full-train OOF calibration, selected source and CSV fingerprints, AIcrowd feedback |
| [Final project checks](eda/final_phase14_readiness.json) | Current clean-package tests, public course tests and integrity checks |
| [Earlier project checks](eda/final_readiness.json) | Historical audit and execution environment |
| [Fixed split](eda/analysis/splits/stratified_seed_20260918.npz) | Development and initial-validation row indices |

The five indices cover **114 records** and store a SHA-256 fingerprint for
each JSON record. [EXPERIMENTS.md](../docs/EXPERIMENTS.md) connects the studies
to their results.

## Reading the scores

The final comparisons select F1 thresholds on inner OOF predictions and
score the outer folds. In booster JSONs, this result is stored in
`outcome.nested_threshold_evaluation.pooled_metrics`. The index F1 column and
`outcome.metrics` can instead contain fixed-threshold or exploratory scores,
including a threshold selected on the same pooled OOF rows being scored.
The experiment summaries identify the protocol used for each result.

The [Phase 33 comparison](experiments/phase33_raw_preprocessing_comparison.json)
keeps the historical Phase 14 booster implementation and nested folds fixed:
all 321 raw predictors give
F1 **0.441457**, versus **0.442155** with 295 cleaned predictors. The raw
pipeline loses F1 in all three folds; the small difference has a paired
conditional interval that includes zero. This measures cleaning and exclusions
together. The [study summary](eda/phase33_raw_preprocessing_summary.md) and
raw record retain metrics and execution provenance; its recorded runtime is
the parallel resume time and excludes two earlier outer fits reused from
the sequential run.

The [Phase 34 record](experiments/phase34_missing_only_split/20261007T092637381255Z_phase34-missing-only-split-depth5-features128-200trees.json)
keeps Phase 14 cleaning, parameters and folds fixed while restoring the missing-only
split candidate. Nested F1 is **0.440803**, precision **0.362001** and recall
**0.563460**, with 206 more true positives and 907 more false positives than
historical Phase 14. The mean inner-selected threshold is
**0.20559910588816863**. This describes the development comparison; final
selection on 12 October retained Phase 14 after submitting both versions
with full-train OOF thresholds. Comparing Phase 33 raw inputs with Phase 34 cleaned inputs would
also change the implementation and cannot isolate preprocessing.
The paired conditional 95% interval for Phase 34 minus Phase 14 F1 is
[−0.003005, +0.000319], including zero. It holds the fitted models and
thresholds fixed and excludes retraining and repeated selection uncertainty.
The [comparison JSON](experiments/phase34_missing_only_split_comparison.json)
and [study summary](eda/phase34_missing_only_split_summary.md) retain fold
metrics, prediction transitions and checkpoint diagnostics.

## Generated artifacts

The [Phase 35 weight study](eda/phase35_positive_weights_summary.md) compares
positive weights 1, 1.25, 1.5 and 2 on the historical implementation. The two
new candidates achieve nested F1 0.440831 and 0.440613, below the matched
weight-1 baseline 0.442155 in every fold. Their thresholds were selected
independently in the inner folds. [Comparison JSON](experiments/phase35_positive_weights_comparison.json)
and the two new records in the main ledger preserve the evidence. This study
does not evaluate weights on the corrected Phase 34 implementation;
recorded runtimes cover only the final resumed invocation.

The [Phase 36 undersampling study](eda/phase36_undersampling_summary.md)
compares two training-only 3:1 samplers against corrected Phase 34.
Random sampling achieves nested F1 **0.439389** and hybrid hard-negative
sampling **0.393535**, versus **0.440803** with all rows. All positives and
all validation rows are preserved. Random is slightly below the control
with a conditional interval including zero; hybrid is worse in every fold.
The [comparison JSON](experiments/phase36_undersampling_comparison.json)
retains the nine fit profiles per candidate and the verified local records.
Both new candidates ran from scratch; their complete runner times are
24.9 and 28.2 minutes with concurrent execution. The archived control's
resumed timings do not provide a clean full-training speed comparison.

The [Phase 37 ratio study](eda/phase37_undersampling_ratios_summary.md) tests
random undersampling at 5:1 and 7:1 on historical Phase 14. The smaller sample
is contained in the larger, all positives remain, and validation keeps all
rows. Nested F1 is **0.438226** and **0.439820**, versus **0.442155** with
all rows; both candidates are below the historical control in every fold.
All 18 models and retained-row binners were replayed and rebuilt exactly.
[Comparison JSON](experiments/phase37_undersampling_ratios_comparison.json)
and two new records in the main ledger retain the evidence. These results
use the historical numerical implementation; the corrected Phase 36 3:1
result is a separate comparison. The selected model retains all training rows.

Training saves checkpoints, OOF arrays and logs under `boosting_artifacts/`,
`mlp_artifacts/` or the configured output directory. These files, numeric data
caches and submission CSVs are excluded from Git.

To reproduce CV, download the official CSVs and run
`python -m tools.prepare_data` from the repository root before running the
experiment configuration. Exact replay requires the saved checkpoints and
prediction arrays listed in the study report; a new CV run generates its own
artifacts.
