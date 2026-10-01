# Experiments and model selection

The selected version is **Phase 14 NumPy histogram boosting**, with 295 source
features including `HAREHAB1`, nested development F1 **0.442155227** and final
threshold **0.21086909970855153**. Model search is closed for this version.
The [final report](FINAL_REPORT.md) summarizes the selected pipeline; this
page records the experiment path and links to the detailed evidence.

## Evaluation protocol

The initial stratified 80/20 split uses seed `20260918`. Baselines inspected
the 20% partition. Later comparisons use the remaining **262,508 development
rows**; that initial partition is not an untouched final test.

Final comparisons use three outer folds, seed `20260919`. In each outer
training partition, two inner folds select the F1 threshold, with seed
`20260922 + outer fold number`. Preprocessing and model fitting use only the
corresponding training rows. The refitted outer model is evaluated on the
excluded fold with the threshold selected from its inner OOF scores.

Earlier screenings optimized a threshold on the same pooled OOF labels being
scored. Those F1 values are **exploratory**, separate from the nested results.
The early threshold-transfer diagnostic is also a screen: its OOF models can
indirectly depend on labels in the fold being assessed.

F1 of the positive class is the primary metric. A classifier that always
predicts negative already obtains about 91.17% accuracy and F1 zero.
Precision, recall, average precision (AP), log loss and confusion counts
support the interpretation of F1 changes. The later operational promotion
rule requires higher pooled F1 and improvement in at least two of three
folds; it is a practical filter rather than a significance test.

Nested threshold selection isolates the threshold of each candidate, while
repeated model choices on the same development folds can still make model
selection optimistic. Paired bootstrap intervals in the detailed reports
condition on the fitted models and thresholds; they do not include retraining
or repeated candidate selection.

## Selected model and main comparisons

| Pipeline | Nested F1 | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical ridge, lambda 0.0001 | 0.419508 | 0.33303 | 0.56665 | 0.39028 | 0.22917 |
| Historical logistic, 4,000 updates | 0.425628 | 0.35728 | 0.52632 | 0.38775 | 0.22198 |
| **Phase 14 histogram boosting** | **0.442155** | **0.367632** | **0.554573** | **0.428609** | **0.215225** |
| Phase 32 MLP 64 → 32 | 0.424451 | 0.360722 | 0.515531 | 0.387782 | 0.221674 |
| Phase 32 MLP 128 → 64 | 0.419268 | 0.340005 | 0.546721 | 0.379380 | 0.222710 |

The frozen booster uses 200 trees, depth 5, learning rate 0.05, minimum leaf
size 200, 64 quantile bins, 128 candidate features per node, leaf L2=1 and
seed `20260920`. Positive class weight is 1. Missing values are routed directly
by the trees. The active Phase 14 tree matrix contains the 295 retained source
columns; auxiliary flags configured for other preprocessing paths are not
added to this tree representation.

The submission threshold is the mean of the three nested Phase 14 thresholds:
`0.203056688916058`, `0.22010463713275993`, `0.20944597307683668`. The reported
nested F1 uses each fold's own threshold; the mean is transferred to the
full-data model. Settings are in [final_model.json](../configs/final_model.json)
and [the preprocessing plan](../configs/experiments/phase14_codebook_corrections_only.json).

The linear metrics are historical results obtained with median fills for
numeric-coded categorical columns. Current linear preprocessing uses the
training-fold mode for binary, categorical and ordinal columns, and the
median for continuous/count columns. The historical results do not measure
that updated fill policy.

## Early development: baseline, linear models and booster selection

| Study | Measured result and decision | Evidence |
| --- | --- | --- |
| Initial baselines | Fixed-threshold F1: all negative 0; ridge 0.016729; logistic with 50 updates 0.164678, on the initial 20% partition | [Main ledger](../results/experiments/index.csv) |
| Phase 3 preprocessing | Eight logistic representations, 50 updates and threshold 0.5; F1 0.038285–0.157788. This short screen did not establish the best optimized representation | [Phase 3](../results/eda/phase3_preprocessing_summary.md) |
| Phase 4–5 optimization | Learning rate, L2, training duration and four age interactions. At 4,000 updates, nested F1 0.423990 without interactions and 0.425441 with them | [Phase 4–5](../results/eda/phase4_model_cv_summary.md) |
| Phase 6 semantics | Corrected BPHIGH4 nonresponses; exploratory F1 0.426321 for the reference, 0.425354 for a combined categorical/nonlinear challenger | [Phase 6](../results/eda/phase6_preprocessing_summary.md) |
| Phase 7 linear confirmation | Corrected logistic nested F1 0.425628. Ridge lambda 0.0001/0.001/0.01: 0.419508/0.417846/0.417752; calibration fitted on inner OOF | [Phase 7](../results/eda/phase7_nested_and_ridge_summary.md) |
| Phase 8–9 targeted representations | Error analysis followed by six isolated category/age variants; exploratory F1 0.425796–0.426322, no replacement of the reference | [Errors](../results/eda/phase8_error_analysis_summary.md), [Phase 9](../results/eda/phase9_targeted_preprocessing_summary.md) |
| Booster benchmarks and Phase 10 | Feasibility checks and depth 3/5, 100/200/400-tree screening | [Main ledger](../results/experiments/index.csv) |
| Phase 11 nested grid | Twelve combinations of depth 3/5, 100/200/400 trees and 64/128 candidate features. Selected depth 5, 128 candidates, 200 trees: F1 0.441741; 400 trees: 0.441132; best depth 3: 0.440987 | [Main ledger](../results/experiments/index.csv) |
| Phase 12 paired errors | Logistic shared 91.5% of booster false negatives and 82.0% of its false positives | [Paired error analysis](../results/eda/phase12_paired_boosting_logistic_summary.md) |
| Phase 13 canonical representation | Simultaneous feature removals: nested F1 0.440287. The compact variant was interrupted and has no complete result | [Phase 13](../results/eda/phase13_feature_handling_review.md) |
| Phase 14 codebook correction | Same 295 features and booster; fourteen verified nonresponse corrections and ALCDAY5=888 → 0. F1 0.442155 | [Phase 14](../results/eda/phase14_codebook_corrections_summary.md) |

Phase 14 changes nested F1 by +0.000415 versus Phase 11. Fold differences
are +0.001631, +0.001915 and −0.002097. The small, mixed gain supports keeping
the semantically correct codes without claiming a material predictive jump.

## Imputation and the alternate branch: Phase 15–17

The Phase 15 feature-recovery study fitted imputers within each training fold
without using the cardiovascular target. The direct plan evaluated 171
categorical targets: 61 beat the mode on exact accuracy and 168 on macro F1;
4/5 continuous targets beat the median. The extended plan gave 65/181,
177/181 and 4/8 respectively. These are held-out known-response measurements,
not scores on respondents whose true missing answers are unavailable.
[Feature recovery report](../results/eda/phase15_imputation_quality_summary.md).

Phase 16 downstream screening refitted imputers and disease models within
outer folds, then selected thresholds on the pooled outer OOF labels:

| Pipeline | Exploratory F1 |
| --- | ---: |
| Matched Phase 14 booster control | 0.442078 |
| Restricted imputation booster | 0.442674 |
| Expanded imputation booster | 0.441989 |
| Restricted imputation logistic | 0.426966 |
| Expanded imputation logistic | 0.425784 |

The logistic comparison also changes the historical median-fill policy to
mode fills. An earlier diagnostic modified only inference inputs of saved
models; it did not refit the complete imputation pipeline. These diagnostics
are separate from the nested confirmation of restricted imputation in Phase 24.
[Retrained screening](../results/experiments/phase16_retrained_comparison.json),
[inference diagnostic](../results/experiments/phase16_imputation_fast_oof_inference.json),
[imputation method](IMPUTATION.md).

A separate Phase 16 representation excludes HAREHAB1 and scores nested F1
0.431370. Its configuration lists five interactions, but the tree preprocessor
does not apply them: the stored result does not measure their effect. Phase 17
on that branch gives 0.430851 with learning rate 0.03, 0.432225 with 0.07,
0.432606 with minimum leaf size 150 and 0.431225 with 250. Its control is
0.431370. These branch results are not isolated changes to Phase 14;
the transfer tests on Phase 14 completed separately in Phase 19.
[Source conformity](../results/eda/phase19_phase14_conformity.md),
[Main ledger](../results/experiments/index.csv).

## Completed comparisons against Phase 14

All values in this table use nested threshold selection on the same development
folds. All candidates were rejected for the selected version.

| Phase | Change | Nested F1 | Delta Phase 14 | Improved folds | Detailed evidence |
| --- | --- | ---: | ---: | ---: | --- |
| 15 | Remove only HAREHAB1 | 0.431039 | −0.011116 | — | [Record](../results/experiments/20260921T154905753681Z_phase15-boosting-ablate-harehab1-depth5-features128-200trees.json) |
| 18 | Replace six raw food frequencies with daily derivatives | 0.441862 | −0.000293 | — | [Record](../results/experiments/20260928T153145004719Z_phase18-boosting-diet-frequency-features-depth5-features128-200trees.json) |
| 19 | Minimum leaf size 150 | 0.439991 | −0.002165 | 0/3 | [Phase 19](../results/eda/phase19_transfer_summary.md) |
| 19 | Learning rate 0.07 | 0.440700 | −0.001455 | 0/3 | [Phase 19](../results/eda/phase19_transfer_summary.md) |
| 20 | Exact bins for columns with at most 64 training states | 0.440783 | −0.001372 | 1/3 | [Phase 20](../results/eda/phase20_binning_summary.md) |
| 21 | Exact bins only for BPHIGH4 and DIABETE3 | 0.439731 | −0.002424 | 0/3 | [Phase 21–22](../results/eda/phase21_22_preprocessing_summary.md) |
| 22 | Activity one-hot with exact indicator bins | 0.440570 | −0.001585 | 0/3 | [Phase 21–22](../results/eda/phase21_22_preprocessing_summary.md) |
| 23 | Eight missing-reason flags without fill | 0.440322 | −0.001833 | 0/3 | [Phase 23](../results/eda/phase23_missing_reason_flags_summary.md) |
| 24 | Original restricted imputation with quantile flags | 0.441791 | −0.000364 | 1/3 | [Phase 24](../results/eda/phase24_original_restricted_imputation_summary.md) |
| 25 | Leaf L2=5 | 0.441393 | −0.000762 | 1/3 | [Phase 25](../results/eda/phase25_l2_summary.md) |
| 26 | Second seed 20260921 | 0.439924 | −0.002232 | 0/3 | [Phase 26](../results/eda/phase26_two_seeds_summary.md) |
| 26 | Uniform mean of seeds 20260920/20260921 | 0.441111 | −0.001044 | 0/3 | [Phase 26](../results/eda/phase26_two_seeds_summary.md) |
| 27 | Learning rate 0.025 and 400 trees | 0.440587 | −0.001568 | 0/3 | [Phase 27–28](../results/eda/phase27_28_summary.md) |
| 28 | Remove only ALCDAY5, retain DROCDY3_ | 0.440535 | −0.001620 | 1/3 | [Phase 27–28](../results/eda/phase27_28_summary.md) |
| 29 | Fold-local selection of 100 features | 0.439224 | −0.002932 | 0/3 | [Phase 29](../results/eda/phase29_feature_selection_summary.md) |
| 29 | Fold-local selection of 60 features | 0.437584 | −0.004571 | 0/3 | [Phase 29](../results/eda/phase29_feature_selection_summary.md) |
| 30 | Positive class weight 2 | 0.439118 | −0.003037 | 0/3 | [Phase 30–31](../results/eda/phase30_31_summary.md) |
| 31 | Maximum depth 7 | 0.441161 | −0.000994 | 1/3 | [Phase 30–31](../results/eda/phase30_31_summary.md) |
| 32 | MLP 64 → 32 | 0.424451 | −0.017704 | 0/3 | [Phase 32](../results/eda/phase32_mlp_summary.md) |
| 32 | MLP 128 → 64 | 0.419268 | −0.022887 | 0/3 | [Phase 32](../results/eda/phase32_mlp_summary.md) |

The activity one-hot trial expands the input to 446 columns while keeping
128 candidates per node, changing the sampling probability of the other
features. Its result is not a universal rejection of nominal representations.

Phase 23 preserves all eight binary flags without imputing. Phase 24 replicates
the original restricted pipeline, including quantile binning that collapses
two rare flags. Three outer fits were reused after exact numerical checks;
six new inner fits selected the thresholds. The earlier exploratory gain
was not confirmed. Comparing Phase 23 and 24 does not isolate imputation,
because their binning also differs.

The Phase 26 ensemble fixes weights at 0.5/0.5 and selects the threshold on
mean inner probabilities. Phase 29 selects columns separately in every outer
and inner training partition; its development lists were not applied globally
to CV folds. Phase 30 weights gradients, Hessians and initial prevalence while
leaving evaluation metrics, binning and minimum leaf row counts unweighted.

Phase 32 uses dedicated finite preprocessing: training-only median/mode,
scaling, nominal one-hot, missing/unseen flags and frequency conversion.
Epoch count is selected on a 10% holdout inside the training partition,
followed by a complete refit. Both architectures lose F1 on all three outer
folds. The shared suite took 8.86 minutes and all 18 refit checkpoints replay
saved probabilities exactly. [MLP workflow](MLP.md).

## Retained HAREHAB1 and evidence navigation

HAREHAB1 asks about rehabilitation after a heart attack. Its availability is
conditioned by the prior CVDINFR4 response, and all 620 development rows with a
recorded code are positive. It remains in the final classification pipeline;
the isolated Phase 15 ablation quantifies the loss from removing it. The
result concerns classification of reported disease and is not a measure of
prediction before the event. [Final report](FINAL_REPORT.md#harehab1).

There are **106 model records**, including repeated runs and feasibility
benchmarks, distributed across three indices:

- [Main experiment ledger](../results/experiments/index.csv): 102 records.
- [Feature-selection ledger](../results/experiments/phase29_feature_selection/index.csv): 2 records.
- [MLP ledger](../results/experiments/mlp/index.csv): 2 records.

Each record links configuration, metrics and available artifact fingerprints.
For historical booster records, the primary nested result is in
`outcome.nested_threshold_evaluation.pooled_metrics`; `outcome.metrics` can
instead contain exploratory pooled-threshold metrics. The saved phase reports
retain fold results, runtimes, verification details and conditional bootstrap
intervals. Large checkpoints and row-level OOF arrays are local artifacts
excluded from Git.
