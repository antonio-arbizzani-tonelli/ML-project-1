# Phase 15 — feature imputation validation

## Question and protocol

This phase checks whether the new feature imputers recover known answers
before they are used by the `_MICHD` model. No disease-model F1 was calculated
and no cardiovascular labels were loaded by the audit.

The imputers were fitted once in each of the three existing Phase 14 folds,
using only the other two development folds. Predictions were evaluated on
known feature values in the held-out fold. Categorical predictions are
compared with the training-fold mode; continuous predictions are compared
with the training-fold median. Metrics are aggregated across the three held
out folds; fold-level files are retained separately.

This is a proxy for imputing nonresponders: the reported recovery metrics are
scored on known responses in the held-out folds, not on the unknown/refused
rows listed below. The true response for a person who said “don't know” or
refused is unavailable, so accuracy on respondents may not transfer to them.
The validation does not establish that an imputed feature improves
cardiovascular classification.

## Results

| Imputation set | Categorical targets evaluated | Better exact accuracy than mode | Better macro F1 than mode | Better balanced accuracy in all 3 folds | Continuous targets better than median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct questionnaire responses + 5 selected continuous blanks | 171 | 61 | 168 | 158 | 4 / 5 |
| Extended set including derived and administrative values | 181 | 65 | 177 | 167 | 4 / 8 |

The categorical imputer recovers minority response classes much better than a
mode-only guess on many fields. Exact accuracy improves on fewer than half of
the targets, so the macro metrics alone do not justify imputing every field.

### Stronger direct candidates

| Feature | Unknown/refused rows to impute | Observed responses scored | Exact accuracy: imputer / mode | Balanced accuracy: imputer / mode | Macro F1: imputer / mode |
| --- | ---: | ---: | ---: | ---: | ---: |
| `INCOME2` | 45,073 | 215,488 | 0.372 / 0.320 | 0.263 / 0.125 | 0.245 / 0.061 |
| `PNEUVAC3` | 21,380 | 216,178 | 0.737 / 0.568 | 0.730 / 0.500 | 0.731 / 0.362 |
| `EMPLOY1` | 2,250 | 260,258 | 0.611 / 0.412 | 0.370 / 0.125 | 0.333 / 0.073 |
| `BPHIGH4` | 757 | 261,751 | 0.706 / 0.580 | 0.363 / 0.250 | 0.365 / 0.183 |
| `GENHLTH` | 709 | 261,797 | 0.399 / 0.331 | 0.372 / 0.200 | 0.375 / 0.099 |
| `HIVTST6` | 8,540 | 228,167 | 0.699 / 0.703 | 0.660 / 0.500 | 0.653 / 0.413 |

`INCOME2`, `PNEUVAC3`, and `EMPLOY1` improve over the mode in both exact and
class-balanced measures. `HIVTST6` is mixed: exact accuracy is slightly lower,
though it predicts the less common response class more often. These results
are consistent across folds, but remain respondent-only proxy measurements.

### Continuous values

| Feature | Missing development rows | MAE: imputer / median baseline | Assessment |
| --- | ---: | ---: | --- |
| `HTM4` | 9,096 | 0.0327 / 0.0862 | Clear improvement |
| `WTKG3` | 18,364 | 5.20 / 15.80 | Clear improvement |
| `_BMI5` | 21,655 | 1.492 / 4.637 | Clear improvement |
| `_VEGESUM` | 30,202 | 0.914 / 0.923 | Negligible improvement |
| `_FRUTSUM` | 25,821 | 0.826 / 0.801 | Worse |

In the extended run, the activity-frequency features `PAFREQ1_`, `PAFREQ2_`,
and `STRFREQ_` also lose to the median baseline. Their missingness often
reflects survey branching, so keep them missing. Do not enable the extended
variant wholesale.

## Decision before disease-model testing

The next reasonable challenger is a **restricted** imputation set: try direct
features with repeatable gains, beginning with `INCOME2`, `PNEUVAC3`, and
`EMPLOY1`, plus the continuous `HTM4`, `WTKG3`, and `_BMI5`. Keep the original
unknown/refused indicators. Leave fruit and activity-frequency measures
unimputed; handle derived variables through documented formulas where
possible. Include an indicators-only comparison to separate the value of
imputation from the value of knowing a response was withheld.

Only after this feature-level check should the restricted inputs be tested in
the disease model. That later step needs the nested `_MICHD` evaluation; this
phase intentionally stops before it.

## Artifacts

- `results/eda/phase15_imputation_quality.csv`: initial saved-validation
  feature check, retained for reference.
- `results/eda/phase15_imputation_direct_quality_fold1.csv` through
  `phase15_imputation_direct_quality_fold3.csv`: direct imputation folds.
- `results/eda/phase15_imputation_all_quality_fold1.csv` through
  `phase15_imputation_all_quality_fold3.csv`: extended folds.
- `src/audit_imputation_quality.py`: fold selection and quality metrics.
