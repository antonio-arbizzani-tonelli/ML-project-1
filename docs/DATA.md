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
  is positive and the selected booster uses it frequently. The Phase 15
  ablation is complete: nested F1 fell from 0.44216 to 0.43104 when this
  feature alone was removed. It is retained in the selected pipeline.
  Its post-event information limits interpretation as prediction before an
  event. See [the final report](FINAL_REPORT.md).

Detailed measured outputs remain under `results/eda/`; the machine-readable
feature registry is `configs/feature_metadata.json`.

## Current tree representation

The selected Phase 14 plan retains 295 source features. It removes identifiers,
interview dates, survey-design fields, and six explicitly excluded columns:
`CTELNUM1`, `CELLFON2`, `_AGEG5YR`, `WEIGHT2`, `HEIGHT3` and `HTIN4`.

Documented nonresponses are converted to `NaN` per variable. Histogram trees
test both missing-value directions at every split, so missing values are not
median-imputed. Documented zero-day codes become zero. Phase 14 also corrects
14 codebook parsing gaps and maps `ALCDAY5=888` (no drinking) to zero.

Representation limits and completed checks:

- food-frequency replacement was tested in Phase 18 (F1 0.441862), and isolated
  removal of `ALCDAY5` in Phase 28 (F1 0.440535); neither was adopted.
  Exercise-frequency replacement remains a deferred hypothesis;
- `555=Never` and `888=Never` require feature-specific zero mappings where the
  current metadata label is not recognized automatically;
- activity identifiers such as `EXRACT11` and `EXRACT21` are nominal codes.
  Phase 22 tested one-hot with exact binary bins (F1 0.440570); it was not adopted.

The selected tree representation retains the raw frequency/activity coding;
the MLP's conversions and one-hot encoding are documented in [MLP.md](MLP.md).
