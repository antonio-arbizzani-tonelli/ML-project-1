# Dataset and preprocessing

## Task

Classify reported myocardial infarction or coronary heart disease (`_MICHD`) from the
2015 BRFSS health survey. Training labels contain 28,975 positives and 299,160
negatives, so the positive prevalence is 8.83%.

Raw data is excluded from Git. See [`data/README.md`](../data/README.md) for the
expected files and verified sizes.

## Measured properties

- 321 predictors plus `Id`; train and test columns have the same order.
- 44.79% of feature cells are blank. Much of this missingness comes from survey
  branching and should not be replaced by one global constant.
- No duplicate rows were found within train, within test, or across the two.
- Three train column pairs are exact duplicates. `CELLFON2` and `CTELNUM1`
  differ in test, so duplicate removal must use semantics as well as train
  equality.
- `HAREHAB1` is a leakage risk: every development row with a recorded response
  is positive and the historical Phase 14 booster uses it frequently. The Phase 15
  ablation is complete: nested F1 fell from 0.44216 to 0.43104 when this
  feature alone was removed. It is retained in the selected pipeline.
  Its post-event information limits interpretation as prediction before an
  event. See [the final report](FINAL_REPORT.md).

Detailed measured outputs remain under `results/eda/`; the machine-readable
feature registry is `configs/feature_metadata.json`.

## Current tree representation

The selected `combined18_24_features256` plan has **297 columns: 289 retained
source columns and eight missing-reason indicators**. It starts from Phase 14
codebook cleaning and exclusions, then combines the Phase 18 diet exclusions
with the restricted Phase 24 imputation plan. The exact settings are in
[the preprocessing configuration](../configs/experiments/combined18_24_preprocessing.json)
and [final_model.json](../configs/final_model.json).

The historical 295-column Phase 14 representation removes identifiers,
interview dates, survey-design fields and six explicitly excluded columns:
`CTELNUM1`, `CELLFON2`, `_AGEG5YR`, `WEIGHT2`, `HEIGHT3`, `HTIN4`.
The combined plan additionally removes six raw diet-frequency columns:
`FRUITJU1`, `FRUIT1`, `FVBEANS`, `FVGREEN`, `FVORANG`, `VEGETAB1`.
Their existing CDC-derived daily-frequency equivalents remain; no new diet
columns are computed by this plan.

Documented nonresponses become `NaN` per variable. Documented zero-day codes
become zero. Phase 14 cleaning corrects 14 parsing gaps and maps `ALCDAY5=888`
(no drinking) to zero. The combined plan imputes only documented nonresponses
in `INCOME2`, `PNEUVAC3`, `EMPLOY1` and blank height/weight, then reconstructs
missing BMI when valid. Each imputer fits only the current training partition.
Eight indicators preserve the corresponding original missing states; remaining
blanks keep direct tree missing routing. [Imputation details](IMPUTATION.md).
The historical booster source preserves its original split search, including
the omission of some observed-versus-missing candidates; Phase 34 separately
measures the corrected search. Quantile bins fit the training rows.

Historical representation tests provide context, not an independent evaluation
of the combined pipeline. Isolated diet exclusions scored nested F1 0.441862;
Phase 24 imputation scored 0.441791 against Phase 14's 0.442155. They were
subsequently combined and selected using full-training OOF calibration and
reported AIcrowd feedback. The combined pipeline has no nested evaluation.
`ALCDAY5` removal in Phase 28 scored 0.440535 and was not adopted.

Activity codes `EXRACT11` and `EXRACT21` remain numeric in the selected model.
One-hot activity encoding scored nested F1 0.440570 in Phase 22; the later
native-category alternative scored reported AIcrowd F1 0.452, below the
selected model's displayed 0.455. Feature-specific `555=Never` and `888=Never`
mappings still require checking against each codebook meaning. The MLP's
separate conversions and one-hot encoding are described in [MLP.md](MLP.md).
