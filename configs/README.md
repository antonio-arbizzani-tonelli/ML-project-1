# Configuration and reproduction catalog

For every report comparison from a clean checkout and official CSVs, use
`python -m tools.reproduce_report run --experiment report --require-complete`.
The [report catalog](report_experiments.json) maps each claim to a command,
configuration, backend and historical target; the
[reproduction guide](../docs/REPRODUCIBILITY.md) explains validation and outputs.
This new runner regenerates controls and intermediate artifacts and explicitly
preserves historical median-fill baselines. It does not require the original
local archives described for legacy drivers in the table below.

[final_model.json](final_model.json) is the single executable definition of the
selected full-data model. Its evidence is in the
[submission record](../results/eda/final_combined18_24_submission.json).
[final_phase14_model.json](final_phase14_model.json) retains the earlier submission;
[feature_metadata.json](feature_metadata.json) supplies codebook semantics.

From the repository root, prepare official inputs with `python -m tools.prepare_data`.
For a row naming a standard runner, use `python -m RUNNER configs/experiments/FILE.json`.
For historical tree recipes, add the **listed implementation**; the unqualified
standard runner defaults to the corrected backend and is not an exact historical
reproduction. Explicit overrides record source hashes and use backend-specific
artifact subdirectories. Use a new `--output-dir` for new experiment records.

Calibration drivers use `prepare`, then `run`; details and prerequisite order are
in [tools/README.md](../tools/README.md). **Local replay** means that saved original
models, OOF arrays or numerical snapshots are required in addition to official data.
Those artifacts are not included in Git. A shared preprocessing plan is not a CLI.
**Fresh chain (explicit override)** means completing the combined/tuning sequence
and passing its new tuning manifest to the follow-up driver. Without that override,
the driver checks the pinned original manifest and requires the original archive.
The linked tools guide provides concrete `prepare`/`run` commands and output paths.
The catalog below includes all 54 retained experiment recipes and shared plans.

| Recipe | Runner/driver | Availability | Backend or prerequisites |
| --- | --- | --- | --- |
| [all_negative_baseline.json](experiments/all_negative_baseline.json) | `src.run_experiment` | Fresh data | Initial baseline; historical linear fill differs from current categorical mode fill |
| [combined18_24_bins128.json](experiments/combined18_24_bins128.json) | `tools.final_combined_tuning` | Fresh data | Full-training calibration; follow-up studies need preceding combined artifacts |
| [combined18_24_depth4.json](experiments/combined18_24_depth4.json) | `tools.final_combined_regularization` | Fresh chain (explicit override); original replay by default | Pass completed tuning manifest with `--base-manifest` |
| [combined18_24_features256.json](experiments/combined18_24_features256.json) | `tools.final_combined_tuning` | Fresh data | Full-training calibration; follow-up studies need preceding combined artifacts |
| [combined18_24_features297.json](experiments/combined18_24_features297.json) | `tools.final_combined_growth` | Fresh chain (explicit override); original replay by default | Pass completed tuning manifest with `--origin-manifest` |
| [combined18_24_l2_1.json](experiments/combined18_24_l2_1.json) | `tools.final_combined_18_24` | Fresh data | Full-training calibration; follow-up studies need preceding combined artifacts |
| [combined18_24_l2_5.json](experiments/combined18_24_l2_5.json) | `tools.final_combined_18_24` | Fresh data | Full-training calibration; follow-up studies need preceding combined artifacts |
| [combined18_24_leaf400.json](experiments/combined18_24_leaf400.json) | `tools.final_combined_regularization` | Fresh chain (explicit override); original replay by default | Pass completed tuning manifest with `--base-manifest` |
| [combined18_24_leafwise32_depth8.json](experiments/combined18_24_leafwise32_depth8.json) | `tools.final_combined_structural` | Fresh chain (explicit override); original replay by default | Pass completed tuning manifest with `--base-manifest` |
| [combined18_24_native_categories.json](experiments/combined18_24_native_categories.json) | `tools.final_combined_categorical` | Fresh chain (explicit override); original replay by default | Pass completed tuning manifest with `--base-manifest` |
| [combined18_24_preprocessing.json](experiments/combined18_24_preprocessing.json) | Used by the matching CV/calibration recipe | Shared plan | Not a standalone model comparison |
| [combined18_24_subsample08.json](experiments/combined18_24_subsample08.json) | `tools.final_combined_structural` | Fresh chain (explicit override); original replay by default | Pass completed tuning manifest with `--base-manifest` |
| [combined18_24_trees300.json](experiments/combined18_24_trees300.json) | `tools.final_combined_growth` | Fresh chain (explicit override); original replay by default | Pass completed tuning manifest with `--origin-manifest`; continues completed features256 checkpoints |
| [count_a.json](experiments/count_a.json) | `tools.final_count_candidates` | Fresh data | Raw-response counts, full-training calibration |
| [count_ab.json](experiments/count_ab.json) | `tools.final_count_candidates` | Fresh data | Raw-response counts, full-training calibration |
| [count_b.json](experiments/count_b.json) | `tools.final_count_candidates` | Fresh data | Raw-response counts, full-training calibration |
| [count_candidates_preprocessing.json](experiments/count_candidates_preprocessing.json) | Used by the matching CV/calibration recipe | Shared plan | Not a standalone model comparison |
| [final_combined_validation.json](experiments/final_combined_validation.json) | `src.run_boosting_cv` | Fresh data | Explicit phase14; no completed nested result for this exact combined pipeline |
| [final_phase14_validation.json](experiments/final_phase14_validation.json) | `src.run_boosting_cv` | Fresh data | Explicit phase14; historical nested reference |
| [logistic_baseline.json](experiments/logistic_baseline.json) | `src.run_experiment` | Fresh data | Initial baseline; historical linear fill differs from current categorical mode fill |
| [phase14_boosting_codebook_corrections_only.json](experiments/phase14_boosting_codebook_corrections_only.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase14_codebook_corrections_only.json](experiments/phase14_codebook_corrections_only.json) | Used by the matching CV/calibration recipe | Shared plan | Not a standalone model comparison |
| [phase15_ablate_harehab1.json](experiments/phase15_ablate_harehab1.json) | Used by the matching CV/calibration recipe | Shared plan | Not a standalone model comparison |
| [phase15_boosting_ablate_harehab1.json](experiments/phase15_boosting_ablate_harehab1.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase16_selected_imputation_preprocessing.json](experiments/phase16_selected_imputation_preprocessing.json) | Used by Phase 24 and combined calibration validation | Shared plan | Original restricted-imputation plan required by the retained Phase 24 recipe |
| [phase18_boosting_diet_frequency_features.json](experiments/phase18_boosting_diet_frequency_features.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase18_diet_frequency_features.json](experiments/phase18_diet_frequency_features.json) | Used by the matching CV/calibration recipe | Shared plan | Not a standalone model comparison |
| [phase20_phase14_exact_low_cardinality.json](experiments/phase20_phase14_exact_low_cardinality.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase21_phase14_clinical_exact.json](experiments/phase21_phase14_clinical_exact.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase22_activity_one_hot_preprocessing.json](experiments/phase22_activity_one_hot_preprocessing.json) | Used by the matching CV/calibration recipe | Shared plan | Not a standalone model comparison |
| [phase22_phase14_activity_one_hot.json](experiments/phase22_phase14_activity_one_hot.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase23_missing_reason_flags_preprocessing.json](experiments/phase23_missing_reason_flags_preprocessing.json) | Used by the matching CV/calibration recipe | Shared plan | Not a standalone model comparison |
| [phase23_phase14_missing_reason_flags.json](experiments/phase23_phase14_missing_reason_flags.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase24_original_restricted_imputation.json](experiments/phase24_original_restricted_imputation.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase25_phase14_l2_5.json](experiments/phase25_phase14_l2_5.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase26_phase14_seed20260921.json](experiments/phase26_phase14_seed20260921.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase27_phase14_lr0025_400trees.json](experiments/phase27_phase14_lr0025_400trees.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase29_phase14_feature_selection.json](experiments/phase29_phase14_feature_selection.json) | `tools.phase29_feature_selection` | Local replay | Requires original Phase 14 OOF/models and matching snapshots |
| [phase30_phase14_positive_weight2.json](experiments/phase30_phase14_positive_weight2.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase31_phase14_depth7.json](experiments/phase31_phase14_depth7.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase32_mlp.json](experiments/phase32_mlp.json) | `src.run_mlp_cv` | Fresh data | Two neural architectures |
| [phase33_raw_boosting_cv.json](experiments/phase33_raw_boosting_cv.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase14` |
| [phase33_raw_tree_preprocessing.json](experiments/phase33_raw_tree_preprocessing.json) | Used by the matching CV/calibration recipe | Shared plan | Not a standalone model comparison |
| [phase34_missing_only_split.json](experiments/phase34_missing_only_split.json) | `src.run_boosting_cv` | Fresh data | `--implementation phase34` |
| [phase35_phase14_positive_weight1p25.json](experiments/phase35_phase14_positive_weight1p25.json) | `tools.phase35_positive_weights` | Local replay | Requires archived Phase 30 source snapshot and Phase 14 fits |
| [phase35_phase14_positive_weight1p5.json](experiments/phase35_phase14_positive_weight1p5.json) | `tools.phase35_positive_weights` | Local replay | Requires archived Phase 30 source snapshot and Phase 14 fits |
| [phase36_undersampling_hybrid_hard.json](experiments/phase36_undersampling_hybrid_hard.json) | `tools.phase36_undersampling` | Local replay | Requires corrected Phase 34 control models/OOF |
| [phase36_undersampling_random.json](experiments/phase36_undersampling_random.json) | `tools.phase36_undersampling` | Local replay | Requires corrected Phase 34 control models/OOF |
| [phase37_phase14_random_ratio5.json](experiments/phase37_phase14_random_ratio5.json) | `tools.phase37_undersampling_ratios` | Local replay | Requires historical Phase 35 source snapshot and Phase 14 fits |
| [phase37_phase14_random_ratio7.json](experiments/phase37_phase14_random_ratio7.json) | `tools.phase37_undersampling_ratios` | Local replay | Requires historical Phase 35 source snapshot and Phase 14 fits |
| [phase3_preprocessing_cv.json](experiments/phase3_preprocessing_cv.json) | Used by the matching CV/calibration recipe | Shared plan | Not a standalone model comparison |
| [phase7_p2_corrected_interactions_nested_cv.json](experiments/phase7_p2_corrected_interactions_nested_cv.json) | `src.run_model_cv` | Fresh data | Historical linear parameters; see evidence caveat below |
| [phase7_ridge_p2_nested_cv.json](experiments/phase7_ridge_p2_nested_cv.json) | `src.run_ridge_cv` | Fresh data | Historical linear parameters; see evidence caveat below |
| [ridge_baseline.json](experiments/ridge_baseline.json) | `src.run_experiment` | Fresh data | Initial baseline; historical linear fill differs from current categorical mode fill |

## What the records mean

The [114-record historical ledger](../results/experiments/index.csv) keeps original
JSON bytes and hashes. Earlier runs can refer to recipes and source files now
available only in the [development archive](../docs/ARCHIVE.md); their embedded paths
are provenance rather than live execution instructions. Original linear results
predate categorical mode filling, so executing their old parameters with current
preprocessing is a new comparison. Exact historical replay uses the archived
checkout and original artifacts/environment.

The presence of `final_combined_validation.json` is a runnable recipe, not evidence
of completed nested evaluation. The selected pipeline's OOF F1 is calibration.
