# Raw predictors versus the selected preprocessing

Same 262,508 development rows, three outer folds, two inner threshold folds, and 200-tree Phase 14 parameters. Raw input retains all 321 predictors and original response codes/NaNs. The selected input has 295 cleaned predictors.

| Pipeline | Nested F1 | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected preprocessing | 0.442155 | 0.367632 | 0.554573 | 0.428609 | 0.215225 |
| Raw predictors | 0.441457 | 0.366631 | 0.554659 | 0.427835 | 0.215348 |

Raw minus cleaned F1: -0.000698.

| Fold | Cleaned F1 | Raw F1 | Raw inner threshold |
| --- | ---: | ---: | ---: |
| 1 | 0.439027 | 0.438233 | 0.213335 |
| 2 | 0.444054 | 0.442862 | 0.211048 |
| 3 | 0.443512 | 0.443196 | 0.206735 |

Paired 95% interval for raw minus cleaned F1: [-0.002635, +0.001226]. It includes zero; the small observed difference does not establish a stable gain. The interval conditions on fixed fits and thresholds, excluding training and model-selection uncertainty.

Joint effect of base cleaning and exclusions; 128/321 versus 128/295 candidate fraction. Does not isolate individual preprocessing operations.

This is a development comparison after repeated model selection. Both pipelines keep HAREHAB1. The benchmark measures the combined preprocessing effect, including feature exclusions; it cannot attribute the difference to one operation.

Reproduce with `python -m tools.phase33_raw_preprocessing_benchmark run`.

Raw record: [20261007T011322024298Z_phase33-raw-321-features-depth5-features128-200trees.json](../experiments/phase33_raw_preprocessing/20261007T011322024298Z_phase33-raw-321-features-depth5-features128-200trees.json).
Comparison: [JSON](../experiments/phase33_raw_preprocessing_comparison.json).
