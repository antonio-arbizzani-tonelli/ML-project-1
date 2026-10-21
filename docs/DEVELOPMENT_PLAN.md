# Final model and completed studies

The selected delivery model is **combined18_24_features256**, the current
project reference in the recorded feedback through 16 October 2026. The team
reported **AIcrowd F1 0.455 and accuracy 0.883**. This document records the
frozen selection; [EXPERIMENTS.md](EXPERIMENTS.md) retains the experimental
history and [the final report](FINAL_REPORT.md) explains the conclusions.

## Delivery settings

| Component | Selected setting |
| --- | --- |
| Model | NumPy histogram gradient boosting, historical Phase 14 source |
| Representation | 289 source columns plus eight missing-reason indicators; HAREHAB1 and ALCDAY5 retained |
| Preprocessing | Phase 14 codebook corrections and exclusions, six additional raw diet exclusions, restricted Phase 24 imputation |
| Trees and depth | 200 trees, maximum depth 5 |
| Learning rate and leaf size | 0.05; at least 200 training rows per leaf |
| Binning and feature candidates | 64 quantile bins; 256 candidate features per node |
| Regularization and class weight | Leaf L2=1; positive weight=1; all training rows |
| Model seed | 20260920 |
| Final threshold | 0.22007772525099117, selected on pooled three-fold full-training OOF predictions |
| OOF calibration F1 | 0.444070696; not an independent test estimate |
| Nested F1 for exact combined pipeline | Not measured |
| Reported AIcrowd | F1 0.455; accuracy 0.883; display values rounded to three decimals |
| Test output | 109,379 rows: 13,895 positive and 95,484 negative |

[final_model.json](../configs/final_model.json) and
[the combined preprocessing plan](../configs/experiments/combined18_24_preprocessing.json)
define the path used by `run.py`. The
[selected submission record](../results/eda/final_combined18_24_submission.json)
preserves the threshold, configuration and CSV fingerprints. Official data,
models and row-level predictions stay outside Git.

HAREHAB1 records rehabilitation after a heart attack. Its isolated historical
ablation reduced nested F1 from 0.442155 to 0.431039. Retaining it supports
classification of reported disease in this dataset, and limits interpretation
as prediction before an event.

## Evidence behind the choice

The historical development program contains 114 recorded runs across five
indices, including screening and feasibility checks. It evaluated linear
models, codebook cleaning, imputation, feature selection, binning, MLPs,
positive weights, random seeds and undersampling. The historical Phase 14
nested reference scores **0.442155**, compared with **0.425628** for logistic
regression. Corrected Phase 34 scores **0.440803** under matched nested folds;
its implementation correction remains documented separately. The current
competition pipeline preserves the frozen source for exact reproduction.

The later [17-candidate overview](../results/eda/final_combined_experiments_summary.json)
covers full-training calibration and reported feedback for individual Phase
18/24/25, their combination, raw-response counts, bin/feature changes, more
trees, regularization, native activity categories, row subsampling and
leafwise growth. These candidates are recorded separately from the historical
114-run ledger. Exploratory probability blends were also inspected without
new fits or submissions and were not adopted.

The selected 256-candidate model ties the earlier 128-candidate combined model
at displayed F1 0.455 and raises displayed accuracy from 0.882 to 0.883.
An exact F1 improvement and an official leaderboard tie-break were not
established. Subsequent recorded alternatives have lower displayed F1.
The 300-tree alternative raises OOF calibration F1 to 0.445145 but reports
AIcrowd F1 0.453, reinforcing the distinction between calibration and evaluation.

## Interpretation and possible future work

Thresholds are selected and described on the same full-training pooled OOF
labels. Their F1 values are calibration summaries; the exact combined model
has no nested evaluation. Development folds and public feedback were reused
for many choices, and initial baselines consulted the local 20% partition.
Neither source is an untouched final test after this search.

No further training is required to reproduce the selected delivery. A future
study could evaluate the complete combined pipeline under a newly defined
nested protocol and reconsider post-event predictors for a prospective task.
Such work would be a separate methodological comparison, with fitting and
selection confined to the appropriate training partitions.
