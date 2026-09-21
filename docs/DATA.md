# Dataset and preprocessing

## Task

Predict myocardial infarction or coronary heart disease (`_MICHD`) from the
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
  is positive and the selected booster uses it frequently. Its ablation is
  required before interpreting the model scientifically.

Detailed measured outputs remain under `results/eda/`; the machine-readable
feature registry is `configs/feature_metadata.json`.

## Current tree representation

The selected Phase 14 plan retains 295 source features. It removes identifiers,
interview dates, survey-design fields, and six explicitly redundant raw
height/weight or telephone columns.

Documented nonresponses are converted to `NaN` per variable. Histogram trees
test both missing-value directions at every split, so missing values are not
median-imputed. Documented zero-day codes become zero. Phase 14 also corrects
14 codebook parsing gaps and maps `ALCDAY5=888` (no drinking) to zero.

Known representation work remains:

- daily, weekly, and monthly frequency codes should be harmonized or replaced
  by their standardized BRFSS derivatives in isolated tests;
- `555=Never` and `888=Never` require feature-specific zero mappings where the
  current metadata label is not recognized automatically;
- activity identifiers such as `EXRACT11` and `EXRACT21` are nominal codes,
  not ordered quantities.

These are hypotheses for controlled experiments, not transformations claimed
by the current result.
