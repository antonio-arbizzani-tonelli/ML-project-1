# Evidence for the report

This catalog retains the comparisons used in the [two-page report](FINAL_REPORT.pdf).
[Supporting notes](FINAL_REPORT.md) explain methods and interpretation;
[configuration commands](../configs/README.md) specify the correct runner,
backend and prerequisites. Earlier exploratory screens and detailed diagnostics
are available from the [development archive](ARCHIVE.md).

For fresh report executions, use the [self-contained report runner](REPRODUCIBILITY.md)
and its [claim/configuration/target map](../configs/report_experiments.json).
It regenerates required controls, median-fill baseline representations, feature
selection and sampling fits from official data. Legacy replay utilities below
retain original provenance requirements; they are not required by this runner.

## Evaluation scopes

Historical nested comparisons use the fixed 262,508 development rows, three outer
folds and thresholds chosen on two inner folds. Preprocessing is fitted within
training partitions. The later full-training candidates instead calibrate thresholds
on three OOF folds covering all 328,135 labels. Those OOF F1 values use the labels
that selected their threshold. AIcrowd values are separate team-reported display
scores at three decimals, without independent platform retrieval.

The exact selected combined pipeline has no completed nested evaluation.
Repeated development and leaderboard choices limit independent interpretation.

## Baselines and important ablations

| Comparison | Historical nested F1 | Evidence |
| --- | ---: | --- |
| Ridge / logistic | 0.419508  / 0.425628 | [Linear/ridge records](../results/experiments/index.csv) |
| Phase 14 historical booster | 0.442155 | [Selected historical record](../results/experiments/20260921T105139261213Z_phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json) |
| Raw321 predictors | 0.441457 | [Raw/clean comparison](../results/experiments/phase33_raw_preprocessing_comparison.json) |
| Corrected observed/missing split search | 0.440803 | [Split-search comparison](../results/experiments/phase34_missing_only_split_comparison.json) |
| MLP64/32 /128/64 | 0.424451  / 0.419268 | [Neural comparison](../results/experiments/mlp/phase32_comparison.json) |
| Remove HAREHAB1 | 0.431039 | [Ablation record](../results/experiments/20260921T154905753681Z_phase15-boosting-ablate-harehab1-depth5-features128-200trees.json) |
| Diet representation / activity one-hot | 0.441862  / 0.440570 | [Diet record](../results/experiments/20260928T153145004719Z_phase18-boosting-diet-frequency-features-depth5-features128-200trees.json), [activity comparison](../results/experiments/phase22_preprocessing_comparison.json) |
| Exact bins / restricted imputation | 0.440783  / 0.441791 | [Bins](../results/experiments/phase20_binning_comparison.json), [imputation](../results/experiments/phase24_original_restricted_imputation_comparison.json) |
| Select100 /60 columns | 0.439224  / 0.437584 | [Feature selection](../results/eda/phase29_feature_selection/comparison.json) |
| Leaf L2=5 / slower400trees / depth7 | 0.441393  / 0.440586  / 0.441161 | [L2](../results/experiments/phase25_l2_comparison.json), [400trees](../results/experiments/phase27_comparison.json), [depth](../results/experiments/phase31_comparison.json) |
| Mean across two model seeds | 0.441111 | [Two-seed comparison](../results/experiments/phase26_two_seeds_comparison.json) |
| Positive weights1.25 /1.5 /2 | 0.440831  / 0.440613  / 0.43912 | [Weight comparison](../results/experiments/phase35_positive_weights_comparison.json) |
| Random undersampling5:1 /7:1 | 0.438226  / 0.439820 | [Historical ratios](../results/experiments/phase37_undersampling_ratios_comparison.json) |
| Random /hybrid hard undersampling3:1 | 0.439389  / 0.393535 | [Corrected-backend sampling](../results/experiments/phase36_undersampling_comparison.json) |

The3:1 and 5:1/7:1 sampling studies use different implementations and are separate
matched comparisons. Conditional intervals hold fitted models and thresholds fixed.
`HAREHAB1` records post-heart-attack rehabilitation; its retained information limits
interpretation as prediction before an event.

## Later full-training candidates

The [17-candidate aggregate](../results/eda/final_combined_experiments_summary.json)
retains calibration values, model parameters, output counts, CSV/OOF/model hashes,
original verification scopes and links to reported feedback. It includes individual
Phase 18/24/25, combined L2 variants, response counts, bin/feature tuning, more trees,
regularization, categorical splitting and structural alternatives.

The [selected submission record](../results/eda/final_combined18_24_submission.json)
identifies `combined18_24_features256`: reported AIcrowd F1 0.455, accuracy 0.883;
OOF calibration F1 0.444071. The earlier 128-candidate combined model has the same
displayed F1 and accuracy 0.882. Rounding does not establish an exact F1 gain or
an official tie-break. Subsequent alternatives have lower reported F1. Count
candidate AIcrowd scores are unavailable and are not inferred from calibration.
The report/supporting notes contain the full candidate table.

## Historical ledger

Five indices retain114 original JSON records with unchanged SHA-256 fingerprints.
They include exploratory runs beyond the final report. Their embedded paths,
configuration references and source hashes describe the original execution; some
refer to files now present only at the archive commit or to ignored local artifacts.
The catalog distinguishes new execution from historical replay. No record was
rewritten to make the current tree appear identical to the development archive.
