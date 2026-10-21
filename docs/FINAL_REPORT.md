# Classifying reported cardiovascular disease: supporting evidence

The submission report is [FINAL_REPORT.pdf](FINAL_REPORT.pdf), with editable
[LaTeX source](FINAL_REPORT.tex). These supporting notes retain the protocols,
experimental comparisons and provenance behind the two-page report.

The final project configuration selects **`combined18_24_features256`**:
NumPy histogram gradient boosting with the Phase 18 diet representation,
Phase 24 restricted imputation and 256 candidate features per node. The team
reported **AIcrowd F1 0.455 and accuracy 0.883** on 14 October 2026. Later
reported alternatives through 16 October did not exceed that displayed F1.
The exact combined pipeline has **no completed independent nested development
evaluation**. Its full-training OOF F1 **0.4440706955530217** is threshold
calibration, not an independent test result.

The earlier combined model with 128 candidates also displays F1 0.455, with
accuracy 0.882. The chosen model has greater displayed accuracy; three-decimal
rounding does not establish that its exact F1 is higher. No official
leaderboard tie-break rule is assumed. This course project selection is
separate from the earlier public portfolio snapshot.

## Data and codebook cleaning

The target `_MICHD` indicates self-reported myocardial infarction or coronary
heart disease in BRFSS 2015. The dataset contains 328,135 labelled respondents,
321 predictors and 109,379 test rows. Positives account for 8.83% of training
labels; 44.79% of predictor cells are blank. Always predicting negative gives
91.17% accuracy and F1 zero, motivating F1 as the main criterion.

The initial codebook pipeline retains 295 source columns after excluding
identifiers, full interview date, survey-design fields and redundant
measurements. Nonresponse codes are recoded to `NaN` only where the codebook
assigns that meaning. Documented zero quantities become zero, including
`ALCDAY5=888`, `PHYSHLTH=88` and `MENTHLTH=88`. On retained development columns,
cleaning recodes 1,095,421 finite nonresponse values and 899,687 zero-quantity
values. The [count audit](../results/eda/phase33_preprocessing_counts.json)
records this representation change.

The [controlled raw-input benchmark](../results/eda/phase33_raw_preprocessing_summary.md)
keeps the historical booster and nested folds fixed. All 321 raw predictors
give F1 0.441457 versus 0.442155 with 295 cleaned predictors, a difference of
0.000698. Cleaning wins in all three folds, but the paired conditional 95%
interval for cleaned minus raw F1, [-0.001226, +0.002635], includes zero. This
comparison measures cleaning and exclusions together; candidate-feature
sampling also occurs over 321 versus 295 columns. It does not isolate each
cleaning rule.

## Final representation and model

The [combined preprocessing plan](../configs/experiments/combined18_24_preprocessing.json)
composes the existing Phase 18 representation with original restricted
Phase 24 imputation:

- Remove `FRUITJU1`, `FRUIT1`, `FVBEANS`, `FVGREEN`, `FVORANG` and `VEGETAB1`.
  Their existing CDC per-day equivalents (`FTJUDA1_`, `FRUTDA1_`, `BEANDAY_`,
  `GRENDAY_`, `ORNGDAY_`, `VEGEDA1_`) remain. This leaves 289 source columns;
  the code does not create six additional diet columns.
- Impute only documented unknown/refused responses in `INCOME2`, `PNEUVAC3`
  and `EMPLOY1` using feature-only Naive Bayes. Ridge predicts missing
  `HTM4`/`WTKG3`; missing BMI is reconstructed where possible. Anchor bins,
  imputers and clipping statistics use training rows only, without disease
  labels. Structural blanks remain missing.
- Preserve eight original missing-reason indicators, giving **297 inputs**.
  The historical quantile representation is retained, including coalescing
  rare indicator states; this is not Phase 23 exact indicator binning.

The [final model configuration](../configs/final_model.json) selects:

| Parameter | Value |
| --- | ---: |
| Trees / maximum depth | 200 / 5 |
| Learning rate / minimum rows per leaf | 0.05 / 200 |
| Quantile bins / candidate features per node | 64 / 256 |
| Leaf L2 / positive class weight | 1 / 1 |
| Model seed | 20260920 |
| Final decision threshold | 0.22007772525099117 |
| Training rows / final representation columns | 328,135 / 297 |

Each Newton tree uses gradients and curvature of logistic loss, with
regularised leaf updates. Quantile bins and missing-value directions are
fitted on training data. The exact historical numerical implementation is
`src/numpy_boosting_phase14.py`, SHA-256
`6fede57915b07a6ea7675292833f8ad79269e665f09977bffd2a9562ce1aaf98`.
Its bytes remain unchanged. The corrected Phase 34 implementation is a
separate comparison.

## Evaluation protocols

The original stratified split uses seed 20260918: 262,508 development rows
and 65,627 local validation rows. Historical final comparisons use three
outer development folds (seed 20260919). Within each outer training
partition, two inner folds select the F1 threshold from pooled OOF
probabilities. The excluded outer fold is scored with that threshold.
Preprocessing, imputation, feature selection and sampling are fitted on
each fit's training partition; validation retains all rows and prevalence.
Pooled outer F1 uses the independently selected threshold for each fold.

Submission-candidate calibration is different: three stratified OOF folds
cover all 328,135 labelled rows, with fold seed 20260919. Every fold refits
its preprocessing and model. The pooled probabilities select the F1
threshold; threshold ties are broken by accuracy. The final model then
fits all labelled rows. **The labels that select the threshold also produce
the displayed calibration F1.** No test labels enter this procedure, but
this is not nested evaluation or an unbiased estimate of a selected model.

AIcrowd feedback is a third protocol. Values below are user/team-reported
display scores rounded to three decimals; submission URLs were not supplied
and the values were not fetched independently from AIcrowd. Repeated
leaderboard feedback informed selection.

## Historical nested development results

| Pipeline | F1 | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.419508 | 0.33303 | 0.56665 | 0.39028 | 0.22917 |
| Logistic regression | 0.425628 | 0.35728 | 0.52632 | 0.38775 | 0.22198 |
| Raw-input booster, 321 predictors | 0.441457 | 0.366631 | 0.554659 | 0.427835 | 0.215348 |
| Phase 14, cleaned 295 predictors | 0.442155 | 0.367632 | 0.554573 | 0.428609 | 0.215225 |
| Phase 34, corrected split search | 0.440803 | 0.362001 | 0.563460 | 0.428781 | 0.215229 |
| MLP 64 → 32 | 0.424451 | 0.360722 | 0.515531 | 0.387782 | 0.221674 |
| MLP 128 → 64 | 0.419268 | 0.340005 | 0.546721 | 0.379380 | 0.222710 |

The historical linear runs use median imputation, predating the categorical
mode-fill correction. Logistic regression starts from zero with learning
rate 0.2, 4,000 updates and no L2; ridge uses lambda 0.0001, selected from
three values, with inner-OOF logistic score calibration. The six required
functions are in `implementations.py`; least squares and ridge solve normal
equations, linear losses use half-MSE and returned losses exclude penalties.

Phase 14 raises F1 by 0.016527 over logistic regression. It recovers 12,855
of 23,180 development positives, with 22,112 false positives and 10,325
false negatives. Its outer-fold F1 values are 0.439027, 0.444054 and 0.443512.
**These counts and nested scores describe historical Phase 14, not the
selected combined pipeline.**

The following matched studies preceded the later full-training submission
search. Values use inner-fold threshold selection.

| Study | Nested F1 | Matched control |
| --- | ---: | --- |
| Daily diet representation | 0.441862 | Phase 14, 0.442155 |
| Activity one-hot | 0.44057 | Phase 14 |
| Exact discrete bins | 0.440783 | Phase 14 |
| Eight missing-reason indicators | 0.44032 | Phase 14 |
| Original restricted imputation | 0.441791 | Phase 14 |
| 100 / 60 permutation-selected inputs | 0.439224 / 0.43758 | Phase 14 |
| Learning rate 0.025, 400 trees | 0.44059 | Phase 14 |
| Maximum depth 7 | 0.44116 | Phase 14 |
| Positive weight 1.25 / 1.5 / 2 | 0.440831 / 0.440613 / 0.43912 | Phase 14 |
| Mean probabilities across two seeds | 0.441111 | Phase 14 |
| Remove HAREHAB1 | 0.431039 | Phase 14 |
| Random undersampling 5:1 / 7:1 | 0.438226 / 0.439820 | Phase 14 |
| Random / hybrid hard undersampling 3:1 | 0.439389 / 0.393535 | Corrected Phase 34, 0.440803 |

None improves its matched control. The original restricted imputation's
exploratory F1 0.442674 selects and scores the threshold on the same OOF
labels; the nested confirmation yields 0.441791. Permutation selection of
100/60 columns, weights 1.25/1.5 and two-seed averaging lose in every fold.
Random undersampling keeps all positives and the specified number of
negatives per positive; binning uses retained training rows only. Hybrid
3:1 takes half the negatives randomly and half from the highest scores of
a training-only pilot (24 trees, depth 3). Its AP falls to 0.384225 from the
corrected control's 0.428781. The 3:1 and 5:1/7:1 experiments use different
implementations and do not form one controlled ratio curve.

MLPs use ReLU hidden layers, sigmoid output, BCE and Adam (learning rate
0.001, batch 512), with L2 0.0001. Their preprocessing includes imputation,
scaling, one-hot and frequency conversion fitted on training rows. Epoch
selection uses an inner 10% split, at most 80 epochs and patience 8, followed
by a full refit. Both networks underperform the historical booster in all
three folds. Detailed evidence is linked from [EXPERIMENTS.md](EXPERIMENTS.md)
and [MLP.md](MLP.md).

## Later full-training calibration and reported AIcrowd feedback

The [17-candidate evidence summary](../results/eda/final_combined_experiments_summary.json)
connects the following results with their recorded fingerprints, verifications
and feedback records. Original machine-specific manifests remain local.
All OOF values in this section are calibration F1;
they are not comparable to the earlier nested numbers as independent test
estimates.

| Candidate | Inputs | OOF F1 | Reported AIcrowd F1 | Reported accuracy |
| --- | ---: | ---: | ---: | ---: |
| Phase 18 diet representation | 289 | 0.444790 | 0.453 | 0.882 |
| Phase 24 restricted imputation | 303 | 0.443520 | 0.454 | 0.880 |
| Phase 25 leaf L2=5 | 295 | 0.444429 | 0.453 | 0.883 |
| Combined 18+24, 128 candidates, L2=1 | 297 | 0.443882 | 0.455 | 0.882 |
| Combined 18+24, 128 candidates, L2=5 | 297 | 0.442655 | 0.453 | 0.883 |
| Add functional-limitation counts (A) | 299 | 0.443450 | Unavailable | Unavailable |
| Add other-diagnosis counts (B) | 299 | 0.443144 | Unavailable | Unavailable |
| Add both count groups (A+B) | 301 | 0.443352 | Unavailable | Unavailable |
| Combined 18+24, 128 bins | 297 | 0.443517 | 0.452 | 0.883 |
| **Combined 18+24, 256 candidates** | **297** | **0.444071** | **0.455** | **0.883** |
| Selected base with 300 trees | 297 | 0.445145 | 0.453 | 0.883 |
| Selected base with 297 candidates | 297 | 0.442930 | 0.452 | 0.885 |
| Selected base with minimum leaf size 400 | 297 | 0.443275 | 0.453 | 0.882 |
| Selected base with maximum depth 4 | 297 | 0.442140 | 0.450 | 0.884 |
| Selected base with native categorical splits | 297 | 0.442326 | 0.452 | 0.885 |
| Selected base with 80% subsampling per tree | 297 | 0.443789 | 0.453 | 0.878 |
| Selected base with leaf-wise growth, 32 leaves, depth 8 | 297 | 0.444142 | 0.452 | 0.881 |

The individual Phase 18/24/25 submissions and the combined L2 comparison
were reported on 13 October. [Feedback for the tuning pair](../results/aicrowd_combined_tuning_20261014/reported_aicrowd_scores.json)
on 14 October selects the 256-candidate reference. Growth, regularisation
and categorical results were reported on 15 October; structural results
on 16 October. The earlier Phase 14 submission had calibration F1
0.442769549, reported AIcrowd F1 0.452 and accuracy 0.878 on 12 October.

Each later change is separate. The 300-tree model extends verified copies
of the 200-tree checkpoints; 297 candidates uses every representation
column. Native category grouping applies only to `EXRACT11`/`EXRACT21`,
with training-fitted vocabularies and G/H ordering of candidate groups.
Subsampling draws a fresh stratified 80% sample without replacement for
each tree while fitting bins on all fold training rows. Leaf-wise growth
selects the available leaf with greatest gain, at most 32 leaves and depth
8. These latter numerical backends are separate from the frozen historical
booster.

Count A uses `BLIND`, `DECIDE`, `USEEQUIP`; B uses `ADDEPEV2`, `CHCCOPD1`,
`CHCKIDNY`. Each adds a positive-response and valid-response count, using
only original 1=yes/2=no codes. With no valid responses the positive count
is `NaN` and the valid count zero. The base columns are unchanged. None
exceeds its 128-candidate base in calibration, and AIcrowd scores remain
unavailable.

The selected model ties the previous combined model at the displayed F1
precision and has 0.001 greater displayed accuracy. All seven alternatives
after the selected configuration have lower reported F1. Higher local
calibration F1 does not guarantee transfer: 300 trees reaches 0.445145
locally but loses 0.002 displayed AIcrowd F1. None of these observations
establishes a general performance ceiling or an independent performance
gain for the exact combined pipeline.

## Split-search correction and disease timing

An implementation audit found that historical quantile split search could
omit the boundary separating all observed values from missing responses.
Phase 34 restores it for recursive and levelwise growth, retaining leaf
size and gain constraints. The original audit passed 49 booster tests,
including eight regression cases, and an independent enumeration of 700
split cases found no omitted candidates after the correction.

The controlled nested comparison keeps Phase 14 features, parameters and
folds fixed. It refits all nine corrected models and compares them with the
historical control. F1 is 0.440803 versus 0.442155, with two folds lower and
one slightly higher; there are 206 more true positives and 907 more false
positives. AP is 0.428781 versus 0.428609, log loss 0.215229 versus 0.215225,
and probability correlation 0.99693. The paired conditional 95% interval
for F1 difference [-0.003005, +0.000319] includes zero. The
[comparison record](../results/experiments/phase34_missing_only_split_comparison.json)
and [study summary](../results/eda/phase34_missing_only_split_summary.md)
retain details. The final combined model preserves the historical
implementation; the correction's result does not supply a nested result
for that combination.

`HAREHAB1` concerns outpatient rehabilitation following a heart attack.
All 620 recorded raw development responses belong to positive labels, including
13 unknown/refused codes; 607 are valid yes/no responses. The
[compact response counts](../results/eda/dataset_summary.json) preserve the raw-code
breakdown and original audit fingerprint. Its
[historical Phase 14 ablation](../results/experiments/20260921T154905753681Z_phase15-boosting-ablate-harehab1-depth5-features128-200trees.json)
reduces F1 from 0.442155 to 0.431039. It remains in the final model for
classification of reported disease. A prospective model would require
excluding it and checking when every other response becomes available;
these scores do not demonstrate prediction before a cardiovascular event.

## Reproduction and final submission fingerprint

From the repository root, install the documented dependencies and supply
the four official dataset CSVs. Then run:

```bash
python -m tools.prepare_data
python run.py --data-dir data/raw/dataset --output submission.csv
```

`configs/final_model.json` fixes the selected combined representation,
historical implementation, model parameters and threshold. The selected
full fit uses all 328,135 labelled rows and generates 109,379 predictions:
**13,895 positive and 95,484 negative**. The CSV has header `Id,Prediction`,
labels -1/+1 and IDs in official sample order. Its recorded SHA-256 is
`d01c3949dea567dc9aa929ba232cbe77395d236b1c4a4492624e0bec5305e883`.

The [compact evidence summary](../results/eda/final_combined_experiments_summary.json)
retains configuration, checkpoint, OOF and submission fingerprints from the
original local manifests. The
[original artifact verification](../results/aicrowd_combined_tuning_20261014/verification.json)
reconstructed OOF thresholds, confirmed unchanged preprocessing relative
to the combined base and reproduced final checkpoint probabilities with
maximum difference zero. The [current selected-pipeline check](../results/eda/final_combined18_24_submission.json)
reconstructs the normal preprocessing and exactly reproduces the selected
checkpoint probabilities and CSV. No expensive full training was repeated
for that checkpoint check. Delivery scope and current integrity checks are documented in
[FINAL_AUDIT.md](FINAL_AUDIT.md). Raw data, checkpoints, individual
probabilities and submission CSVs remain local, outside the Git deliverable.

To recalibrate the tuning candidates from official data, first prepare the
original combined base and then the tuning study:

```bash
python -m tools.final_combined_18_24 prepare
python -m tools.final_combined_18_24 run
python -m tools.final_combined_tuning prepare
python -m tools.final_combined_tuning run
```

These commands are computationally substantial and use at most two
single-thread fitting workers. The [utility guide](../tools/README.md)
documents run manifests and reuse requirements. Historical Phase 14 nested
evaluation remains reproducible with:

```bash
python -m src.run_boosting_cv configs/experiments/final_phase14_validation.json
```

This command evaluates the historical 295-input pipeline, not the selected
combined submission. Suite files for later combined candidates also retain
development settings, but a configuration's presence is not evidence that
its nested evaluation was executed.

## Limits of the evidence

Development folds were reused for many preprocessing and model decisions;
initial baselines also consulted the local 20% validation partition.
Later full-training calibration uses all labelled rows, and AIcrowd
feedback contributed to final selection. Inner threshold selection reduces
one source of optimism in historical comparisons without removing model
selection effects. Calibration F1 and the selected leaderboard result are
not untouched estimates after this search. Confidence intervals from
historical comparisons condition on fixed models, thresholds and sampled
rows; they exclude retraining and repeated-selection variability.

The report distinguishes recorded results, team-reported platform feedback
and the absence of an exact combined nested evaluation. No unavailable
AIcrowd score, final rank or independent confirmation is inferred.
