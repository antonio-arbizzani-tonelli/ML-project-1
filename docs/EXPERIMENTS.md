# Experiments and model selection

The selected delivery model is **combined18_24_features256**: the historical
NumPy booster with 297 input features and 256 candidate features per node.
The team reported **AIcrowd F1 0.455 and accuracy 0.883**. Its full-training
three-fold OOF threshold is **0.22007772525099117** and calibration F1 is
**0.444070696**. The exact combined pipeline has no nested evaluation.

Historical Phase 14 remains the nested reference (F1 **0.442155227**, 295
source columns), while corrected Phase 34 reaches **0.440803240**. These
historical results cannot be assigned to the combined model. The
[final report](FINAL_REPORT.md) explains the selected pipeline and limitations;
the [selected submission record](../results/eda/final_combined18_24_submission.json)
and [17-candidate overview](../results/eda/final_combined_experiments_summary.json)
retain its evidence. Model choices use feedback recorded through 16 October
2026, including the leaderboard, rather than a final independent test.

## Evaluation protocol

The initial stratified 80/20 split uses seed `20260918`. Baselines inspected
the 20% partition. Later comparisons use the remaining **262,508 development
rows**; that initial partition is not an untouched final test.

Historical nested comparisons use three outer folds, seed `20260919`. In each outer
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
rule for experimental challengers requires higher pooled F1 and improvement in at least two of three
folds; it is a practical filter rather than a significance test.

Nested threshold selection isolates the threshold of each candidate, while
repeated model choices on the same development folds can still make model
selection optimistic. Paired bootstrap intervals in the detailed reports
condition on the fitted models and thresholds; they do not include retraining
or repeated candidate selection.

## Historical nested comparisons

| Pipeline | Nested F1 | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical ridge, lambda 0.0001 | 0.419508 | 0.33303 | 0.56665 | 0.39028 | 0.22917 |
| Historical logistic, 4,000 updates | 0.425628 | 0.35728 | 0.52632 | 0.38775 | 0.22198 |
| **Historical Phase 14 histogram boosting** | **0.442155** | **0.367632** | **0.554573** | **0.428609** | **0.215225** |
| Phase 33 historical booster, 321 raw predictors | 0.441457 | 0.366631 | 0.554659 | 0.427835 | 0.215348 |
| Phase 34 corrected histogram boosting | 0.440803 | 0.362001 | 0.563460 | 0.428781 | 0.215229 |
| Phase 32 MLP 64 → 32 | 0.424451 | 0.360722 | 0.515531 | 0.387782 | 0.221674 |
| Phase 32 MLP 128 → 64 | 0.419268 | 0.340005 | 0.546721 | 0.379380 | 0.222710 |

The frozen booster uses 200 trees, depth 5, learning rate 0.05, minimum leaf
size 200, 64 quantile bins, 128 candidate features per node, leaf L2=1 and
seed `20260920`. Positive class weight is 1. Missing values are routed directly
by the historical trees; Phase 34 separately adds omitted endpoint candidates.
The tree
matrix contains the 295 retained Phase 14 source columns; auxiliary flags
configured for other preprocessing paths are not
added to this tree representation.

The historical Phase 14 submission threshold **0.2106169795070252** comes from a new three-fold
OOF calibration on all training rows. Calibration F1 is 0.442770 for Phase 14
and 0.443570 for Phase 34; these scores use the labels selecting the thresholds
and are not independent evaluations. Nested development F1 uses each outer
fold's inner-selected threshold. Its preserved settings are in [final_phase14_model.json](../configs/final_phase14_model.json)
and [the preprocessing plan](../configs/experiments/phase14_codebook_corrections_only.json).
The [historical Phase 14 CV configuration](../configs/experiments/final_phase14_validation.json)
uses the included frozen historical source; the
[Phase 34 CV configuration](../configs/experiments/phase34_missing_only_split.json)
reproduces the corrected candidate. The
[historical submission record](../results/eda/final_phase14_submission.json)
records the calibrated threshold, selected CSV fingerprint and AIcrowd feedback.

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
folds. These are historical comparisons with Phase 14, before the split-search
correction; the candidates were rejected at that stage. Their recorded deltas
remain relative to the historical Phase 14 F1.

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
| 33 | All 321 raw predictors, no semantic cleaning or exclusions | 0.441457 | −0.000698 | 0/3 | [Comparison](../results/experiments/phase33_raw_preprocessing_comparison.json) |

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

## Phase 33: contribution of the base preprocessing

This controlled benchmark compares all 321 raw predictors with the 295
Phase 14 predictors. Raw inputs preserve original finite response codes and
NaNs, without recoding, exclusions, imputation, scaling or extra features.
Both pipelines retain HAREHAB1 and use the same historical booster implementation,
parameters, dataset and seeds, with three outer folds and two inner folds for
threshold selection. Both results precede the Phase 34 split-search correction.

Raw nested F1 is **0.441457**, versus **0.442155** for the cleaned Phase 14 control:
delta −0.000698, with all three folds lower. Raw inputs add two true positives
and 99 false positives; AP falls by 0.000774 and log loss rises by 0.000123.
The paired conditional 95% interval for raw minus cleaned F1 is
[−0.002635, +0.001226]. It includes zero, so the observed preprocessing benefit
is small and uncertain. The comparison combines cleaning and exclusions and
changes the candidate fraction from 128/321 to 128/295; it does not isolate
any one preprocessing operation.

The raw Phase 33 score and corrected clean Phase 34 score use different
implementations. Their difference cannot measure a preprocessing gain; a
matched raw benchmark with the corrected source has not been run.

Two complete sequential outer checkpoints were verified and reused; the seven
remaining fits ran in two single-thread workers. The unchanged standard runner
assembled OOF scores and nested metrics from caches validated against exact
matrices, labels, parameters and partition order. The recorded 35.72 minutes
cover the parallel resume and metric assembly, excluding earlier reused fits.
[Verified comparison](../results/experiments/phase33_raw_preprocessing_comparison.json),
[study summary](../results/eda/phase33_raw_preprocessing_summary.md),
[raw record](../results/experiments/phase33_raw_preprocessing/20261007T011322024298Z_phase33-raw-321-features-depth5-features128-200trees.json).

## Phase 34: corrected observed-versus-missing splits

The quantile split search omitted the terminal candidate that can put all
observed values in one child and missing values in the other. Phase 34 adds
this candidate when missing values are present, preserving the previous
candidate order and gain tie rule. The 295-column cleaning, model parameters,
rows, seeds and nested threshold protocol remain fixed. Three outer and six
inner fits use the corrected source; saved Phase 14 OOF scores and checkpoints
provide the historical control without retraining it.

Corrected nested F1 is **0.440803**, delta **−0.001352**. Fold deltas are
−0.002305, −0.002318 and +0.000221. Recall increases from 0.554573 to
0.563460, while precision falls from 0.367632 to 0.362001: the corrected
pipeline adds 206 true positives and 907 false positives. AP changes by
+0.000172 and log loss by +0.000004. This is an implementation-correction
comparison with frozen settings. The correction restores the intended candidate set. The final competition
pipeline preserves the historical source for exact reproduction; these
scores do not establish that omitting candidates is generally preferable.

The paired conditional 95% interval for corrected minus historical F1 is
[−0.003005, +0.000319], from 20,000 class-stratified paired resamples. It
includes zero and conditions on the fitted models and thresholds; retraining
and repeated model-selection uncertainty are outside its scope. Across the
three outer models, the new terminal candidate is selected 270 times.
Historical trees could already form pure observed/missing partitions when
lower thresholds were equivalent within a node. The JSON diagnostics count
actual partitions on complete training rows and also report staged loss/AP
on matched 3,000-row training and validation samples per fold.

The [Phase 34 record](../results/experiments/phase34_missing_only_split/20261007T092637381255Z_phase34-missing-only-split-depth5-features128-200trees.json)
contains nested metrics, thresholds and frozen execution provenance. Its
index F1, 0.441740, comes from selecting a threshold on pooled outer OOF labels;
the **0.440803** result above uses thresholds selected inside each outer
training partition.
[Verified comparison](../results/experiments/phase34_missing_only_split_comparison.json),
[study summary](../results/eda/phase34_missing_only_split_summary.md).

## Additional positive weights on the historical implementation (Phase 35)

The requested weights 1.25 and 1.5 were evaluated using the exact Phase 30
numerical source snapshot, with the historical Phase 14 data, cleaning, seeds
and nested threshold protocol. Each candidate completed nine fits; weights
1 and 2 reuse verified historical results. This isolates weighting from the
subsequent missing-only split correction and does not evaluate these weights
under the corrected Phase 34 implementation.

| Positive weight | Nested F1 | Precision | Recall | Favorable folds versus weight 1 |
| --- | ---: | ---: | ---: | ---: |
| 1, historical baseline | 0.442155 | 0.367632 | 0.554573 | — |
| 1.25 | 0.440831 | 0.372562 | 0.539733 | 0/3 |
| 1.5 | 0.440613 | 0.364316 | 0.557334 | 0/3 |
| 2, previously evaluated | 0.439118 | 0.361155 | 0.560009 | 0/3 |

Neither new weight meets the operational promotion rule. Their paired
conditional F1-delta intervals include zero; the observed declines do not
establish a population-level effect. Weight 1 remains preferred within this
historical comparison. These trials do not measure weights under corrected Phase 34 or under the
later combined delivery pipeline.

[Verified study and runtime scope](../results/eda/phase35_positive_weights_summary.md),
[comparison JSON](../results/experiments/phase35_positive_weights_comparison.json).
The recorded runtime is the final resumed invocation, not the original cost
of all nine fits; checkpoint reuse and a long execution pause preclude a
simple speed comparison.

## Two undersampling methods on the corrected implementation (Phase 36)

Random undersampling and hybrid hard-negative sampling were compared with
corrected Phase 34. Both keep every positive and three negatives per positive,
retaining about 35.3% of eligible training rows. Random draws negatives
uniformly; hybrid draws half uniformly and fills the remaining budget with
the highest-risk negatives from a 24-tree, depth-3 training-only pilot.
This avoids a distance metric over mixed features and missing values.

Each candidate completed nine fresh 200-tree fits with the same 295 columns,
model parameters and seeds as Phase 34. Sampling is repeated within each
outer/inner training partition. Final preprocessing and bins use retained
rows. Inner and outer validation keep all original rows; thresholds are
selected independently from inner OOF predictions. The holdout and test were
not evaluated.

| Pipeline | Nested F1 | Precision | Recall | Favorable folds versus all rows |
| --- | ---: | ---: | ---: | ---: |
| All rows, corrected Phase 34 | 0.440803 | 0.362001 | 0.563460 | — |
| Random, 3:1 | 0.439389 | 0.363860 | 0.554487 | 1/3 |
| Hybrid hard-negative, 3:1 | 0.393535 | 0.314655 | 0.525194 | 0/3 |

Neither candidate improves nested F1. Random's paired conditional delta
interval is [-0.003356, +0.000540], including zero. Hybrid's interval is
[-0.051379, -0.043165]; AP, precision and recall also decrease. Prefer
all training rows for the current model. These results concern the two fixed
3:1 procedures and one sampling seed, not every possible undersampling policy.

The complete candidate runners took 24.9 and 28.2 minutes, respectively,
with two single-thread candidates running concurrently; hybrid includes its
pilots. Both were fresh runs without checkpoint reuse. The archived control
resumed two outer models at 125 trees and its saved tree timers contain pauses,
so its recorded runtime cannot establish a clean full-training speed ratio.

Verification reproduced all 18 final models, all nine pilots and all 18
retained-row binners exactly, including inner OOF predictions. Seventeen unit
tests passed; synthetic end-to-end and interrupted-checkpoint workflows passed
for both samplers. Neither sampler was adopted by that study.
[Study summary](../results/eda/phase36_undersampling_summary.md),
[comparison JSON](../results/experiments/phase36_undersampling_comparison.json).

## Less aggressive random undersampling on historical Phase 14 (Phase 37)

Random sampling at 5:1 and 7:1 was evaluated against historical Phase 14,
whose nested F1 is 0.442155. The verified historical Phase 35 numerical
snapshot preserves the old split search. Both candidates use the same 295
columns, model parameters, seeds and nested folds as the control; all positives
remain in each eligible training partition. A shared random permutation of
negatives makes the 5:1 sample a subset of the 7:1 sample in all nine fits.
About 53.0% and 70.6% of eligible training rows remain, respectively.
Validation rows and prevalence stay unchanged; thresholds are selected from
inner OOF predictions. Each candidate completed nine fresh 200-tree fits.

| Historical pipeline | Nested F1 | Precision | Recall | Favorable folds versus all rows |
| --- | ---: | ---: | ---: | ---: |
| All rows, Phase 14 | 0.442155 | 0.367632 | 0.554573 | — |
| Random 5:1 | 0.438226 | 0.365893 | 0.546204 | 0/3 |
| Random 7:1 | 0.439820 | 0.363932 | 0.555695 | 0/3 |

Keeping more negatives reduces the observed F1 loss relative to 5:1, but neither
candidate improves the historical all-row control. Prefer all rows within this
comparison. These are two fixed ratios with one shared sampling seed; the
development folds have already been used for multiple choices. Paired
conditional intervals and complete runner timings are saved in the study.
The 3:1 experiment in Phase 36 used corrected Phase 34 and cannot supply a
matched historical 3:1 control.

Verification replayed all 18 new models, including their inner OOF scores,
reconstructed all retained-row bin edges, reproduced thresholds and metrics,
and checked all nine sample nesting relationships. Twenty selected unit tests
passed. Synthetic full-workflow and interrupted-checkpoint checks passed for
both ratios. Holdout and test were not evaluated in that study. These ratios
were not adopted; the final Phase 14 uses all training rows.
[Historical ratio study](../results/eda/phase37_undersampling_ratios_summary.md),
[comparison JSON](../results/experiments/phase37_undersampling_ratios_comparison.json).

## Full-training candidates and leaderboard feedback, 13–16 October

These later runs calibrate thresholds on three stratified OOF folds covering
all **328,135 labelled rows** (seed `20260919`), then fit on all rows for the
109,379-row test submission. Preprocessing and imputers fit only the training
partition of each OOF fit. The pooled labels both select and describe the
threshold, so the calibration scores below are not independent validation or
nested estimates. Reported AIcrowd values are user-provided display scores
rounded to three decimals, without submission URLs or independent retrieval.
Repeated leaderboard feedback contributed to selection.

| Candidate | Features | OOF calibration F1 | Reported AIcrowd F1 | Reported accuracy |
| --- | ---: | ---: | ---: | ---: |
| Phase 18, raw diet exclusions | 289 | 0.444790 | 0.453 | 0.882 |
| Phase 24, restricted imputation | 303 | 0.443520 | 0.454 | 0.880 |
| Phase 25, leaf L2=5 | 295 | 0.444429 | 0.453 | 0.883 |
| Combined 18+24, leaf L2=1, 128 candidates | 297 | 0.443882 | 0.455 | 0.882 |
| Combined 18+24, leaf L2=5 | 297 | 0.442655 | 0.453 | 0.883 |
| Combined plus functional-limitation counts (A) | 299 | 0.443450 | Unreported | Unreported |
| Combined plus other-condition counts (B) | 299 | 0.443144 | Unreported | Unreported |
| Combined plus both count groups (A+B) | 301 | 0.443352 | Unreported | Unreported |
| Combined, 128 bins | 297 | 0.443517 | 0.452 | 0.883 |
| **Selected combined, 256 candidate features** | **297** | **0.444071** | **0.455** | **0.883** |
| 256 candidates, 300 trees | 297 | 0.445145 | 0.453 | 0.883 |
| All 297 candidate features, 200 trees | 297 | 0.442930 | 0.452 | 0.885 |
| 256 candidates, minimum leaf 400 | 297 | 0.443275 | 0.453 | 0.882 |
| 256 candidates, depth 4 | 297 | 0.442140 | 0.450 | 0.884 |
| 256 candidates, native activity categories | 297 | 0.442326 | 0.452 | 0.885 |
| 256 candidates, row subsample 0.8 per tree | 297 | 0.443789 | 0.453 | 0.878 |
| 256 candidates, leafwise, 32 leaves/depth 8 | 297 | 0.444142 | 0.452 | 0.881 |

The combined representation retains the CDC's existing daily food-frequency
columns and removes six redundant raw frequency codes (`FRUITJU1`, `FRUIT1`,
`FVBEANS`, `FVGREEN`, `FVORANG`, `VEGETAB1`). It combines these exclusions
with Phase 24 restricted imputation: Naive Bayes for documented nonresponses
in `INCOME2`, `PNEUVAC3`, `EMPLOY1`, ridge for blank height/weight, and BMI
reconstruction. Eight original missing-reason indicators remain. The matrix
has **289 source columns plus eight indicators**; remaining structural blanks
retain direct tree missing routing. `HAREHAB1` remains present.

The selected model keeps 200 trees, depth 5, learning rate 0.05, minimum leaf
size 200, 64 bins, leaf L2=1, weight 1 and seed `20260920`. Relative to the
first combined model, only candidate features per node change from 128 to
256. Both display F1 0.455; the latter's accuracy is 0.883 versus 0.882.
Rounding does not establish an exact F1 improvement, and the official
leaderboard tie-break was not verified. Later recorded alternatives do not
improve its displayed F1. The 300-tree trial has the highest local calibration
F1 but lower reported AIcrowd F1, illustrating why calibration does not provide
an independent selection score.

The count trials retain individual source features and add yes/valid-response
counts from raw codes 1/2: A uses `BLIND`, `DECIDE`, `USEEQUIP`; B uses
`ADDEPEV2`, `CHCCOPD1`, `CHCKIDNY`. They do not improve the earlier combined
model's calibration F1, and no AIcrowd feedback is recorded for them.

Six fixed probability blends of Phase 18/24/25 were explored without new fits
or submissions. Their calibration F1 values (0.444215–0.444616) do not exceed
Phase 18's 0.444790. A later activity-category blend diagnostic reaches
0.444467 at 25% categorical weight, compared with 0.444071 for the reference;
its reused OOF labels do not establish independent improvement. Neither
exploratory screen changes the selected single model.

The [aggregate evidence](../results/eda/final_combined_experiments_summary.json)
records exact thresholds, parameters, counts, hashes and archived verification
scopes for all 17 candidates. Historical verification reconstructed pooled
OOF thresholds, checked official IDs/signed labels and replayed full-model
predictions with zero difference. The selected CSV contains **13,895 positive
and 95,484 negative** predictions; its SHA-256 is in the
[selected submission record](../results/eda/final_combined18_24_submission.json).
These archived replay checks are separate from the delivery checks run now.

## Retained HAREHAB1 and evidence navigation

HAREHAB1 asks about rehabilitation after a heart attack. Its availability is
conditioned by the prior CVDINFR4 response, and all 620 development rows with a
recorded code are positive. It remains in the final classification pipeline;
the isolated Phase 15 ablation quantifies the loss from removing it. The
result concerns classification of reported disease and is not a measure of
prediction before the event. [Final report](FINAL_REPORT.md#split-search-correction-and-disease-timing).

The historical nested/screening ledgers contain **114 model records**, including repeated runs and feasibility
benchmarks, distributed across five indices:

- [Main experiment ledger](../results/experiments/index.csv): 108 records.
- [Feature-selection ledger](../results/experiments/phase29_feature_selection/index.csv): 2 records.
- [MLP ledger](../results/experiments/mlp/index.csv): 2 records.
- [Raw-input benchmark ledger](../results/experiments/phase33_raw_preprocessing/index.csv): 1 record.
- [Split-search correction ledger](../results/experiments/phase34_missing_only_split/index.csv): 1 record.

The 17 later full-training candidate deliveries above are summarized separately
and are not additional rows in those historical ledgers.

Each record links configuration, metrics and available artifact fingerprints.
For historical booster records, the primary nested result is in
`outcome.nested_threshold_evaluation.pooled_metrics`; `outcome.metrics` can
instead contain exploratory pooled-threshold metrics. The saved phase reports
retain fold results, runtimes, verification details and conditional bootstrap
intervals. Large checkpoints and row-level OOF arrays are local artifacts
excluded from Git.
