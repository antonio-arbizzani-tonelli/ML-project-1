# Phase 13 — concrete feature-handling audit

This is a read-only audit of the saved development partition (262,508 rows). It
uses the retained P2 feature set, the current `FeaturePreprocessor` code maps,
the raw numeric cache, and the staged official BRFSS 2015 codebook. No model was
refitted and no F1 gain is claimed. The companion candidate list is
`analysis/phase13_feature_code_review_candidates.csv`.

## Verified coding gaps

The metadata import lost labels for some wrapped codebook rows. The current P2
plan therefore leaves the following documented nonresponses as ordinary numeric
values. Counts are development rows containing the indicated codes; a row can
appear under multiple features.

| Feature | Meaning | Codes to convert to NaN | Affected development rows | Codebook lines |
| --- | --- | --- | ---: | --- |
| `HIVTST6` | Ever tested for HIV | 7 unknown, 9 refused | 8,540 | 2429–2442 |
| `BLOODCHO` | Cholesterol ever checked | 7 unknown, 9 refused | 5,592 | 780–791 |
| `HPVTEST` | Ever had HPV test | 7 unknown, 9 refused | 3,894 | 4061–4073 |
| `ALCDAY5` | Drinking frequency | 777 unknown, 999 refused | 3,382 | 1628–1644 |
| `EXRACT21` | Second exercise activity | 77 unknown, 99 refused | 2,223 | 2152–2164 |
| `SMOKE100` | Ever smoked 100 cigarettes | 7 unknown, 9 refused | 1,944 | 1512–1524 |
| `FLUSHOT6` | Flu vaccination | 7 unknown, 9 refused | 1,795 | 2335–2350 |
| `CASTHDX2` | Child ever had asthma | 7 unknown, 9 refused | 1,490 | 4702–4715 |
| `ASTHMA3` | Ever told of asthma | 7 unknown, 9 refused | 807 | 889–900 |
| `EXRACT11` | Main exercise activity | 77 unknown, 99 refused | 710 | 1987–1997 |
| `HPVADVC2` | HPV vaccination | 7 unknown, 9 refused | 623 | 3906–3922 |
| `NUMHHOL2` | Multiple household phones | 7 unknown, 9 refused | 579 | 1149–1163 |
| `IMFVPLAC` | Flu vaccination location | 77 unknown | 327 | 2379–2399 |
| `SMOKDAY2` | Current smoking frequency | 7 unknown, 9 refused | 198 | 1530–1544 |

Together these 14 verified gaps affect 26,856 distinct development rows (10.23%),
containing 32,104 affected feature cells. This is a coding defect, but the
affected-row count is not an estimate of model improvement. The candidate CSV
has 54 observed feature/code pairs, including valid `IDAY=7/9`. Any other
candidate needs codebook verification before recoding.

## Different treatment for different variables

| Group | Current treatment | Concrete treatment to test |
| --- | --- | --- |
| `GENHLTH`, `INCOME2` | Valid ordered responses are retained; documented unknown/refused codes become NaN. | Keep. Their order has meaning; the current missing-code treatment is correct. |
| `PHYSHLTH`, `MENTHLTH` | `88` becomes zero days; `77`/`99` become NaN. | Keep. Zero is a real measurement here. |
| `_BMI5` | Numeric value or NaN; boosting routes NaNs at each split. | Keep as baseline. Test any BMI imputation only as an isolated challenger, not as a default. |
| `BPHIGH4`, `DIABETE3` | Their `7`/`9` codes already become NaN through existing corrections. | Keep the correction. No new recode is needed for these codes. |
| `_DRNKWEK`, `_RFDRHV5`, `_TOTINDA` | Their documented unknown sentinels (`99900` or `9`) already become NaN. | Keep; do not mistake their raw sentinel counts for a new coding gap. |
| Binary/short categorical questions above | Documented nonresponses are still ordinary codes. | Add feature-specific missing-code overrides, leaving real responses intact. Boosting can then learn a missing branch; logistic still uses its fold-fitted fill policy. |
| `HPVTEST` metadata | The registry calls this a count, although the source question is yes/no. | Correct its semantic type in an isolated metadata revision; the immediate `7`/`9` recode is needed independently. |
| `ALCDAY5` | `101`–`107` denote days per week, `201`–`230` days in 30, `888` no drinking, `777`/`999` nonresponse. All are currently treated as numeric values. | Convert `777`/`999` to NaN; convert `888` to zero; test one harmonized frequency in days per 30 days: `101`–`107` as approximately `30*(code-100)/7`, `201`–`230` as `code-200`. Keep the raw feature only in the baseline side of the comparison. |
| `EXRACT11`, `EXRACT21` | Activity codes are numeric even though their numbers identify nominal activities. | Correct `77`/`99` first. Preserve `EXRACT21=88`, which means no second activity (54,700 development rows). Test activity grouping or an explicit no-second-activity flag separately; do not one-hot every rare activity without evidence. |
| `IMONTH`, `IDAY`, `IYEAR` | Retained because metadata marks them categorical, although `IDATE` is excluded as a date. | Test excluding these three interview-time columns as one procedural-feature ablation. `IDAY=7/9` are valid days, not nonresponses; `IYEAR` is 2015 or 2016. |

The current `ALCDAY5=888` group alone has 123,530 development rows; the weekly
and monthly encodings appear in 34,388 and 91,717 rows respectively. The
boosting tree can isolate these code ranges, but their numeric distances do not
represent drinking frequency. `_DRNKWEK` is a different derived quantity
(total drinks per week), so it does not by itself replace this frequency signal.

## Proposed evaluation order

1. Freeze the current boosting configuration and baseline artifacts. Apply only
   the 14 verified nonresponse-code corrections in a new preprocessing plan.
   Screen on one saved development fold, then confirm a promising result on the
   same three outer folds with thresholds fitted wholly inside each training
   partition. Primary comparison: nested F1; also report AP, log loss, FP/FN,
   and per-fold deltas.
2. Separately test the normalized `ALCDAY5` frequency against the corrected
   codebook baseline. This isolates its effect from the missing-code corrections.
3. Separately test exclusion of `IMONTH`/`IDAY`/`IYEAR`. Treat any improvement as
   empirical, not guaranteed by their procedural meaning.
4. Only if these changes are useful, inspect nominal activity grouping and
   other candidate codebook gaps. Do not convert all 7/9/77/99 codes globally:
   `IDAY=7/9` are valid, and `EXRACT21=88` has a valid meaning.

All comparisons should use the saved development split and the same boosting
hyperparameters. The outer validation partition is not used for this audit.
If the corrected codes are later evaluated with logistic regression, compare
its current median fill with explicit missing-category indicators; converting
codes to NaN alone removes the distinction between nonresponse and a median
response in that model.

## Implementation status

The custom `ALCDAY5` conversion was removed after confirming that `DROCDY3_`
already supplies the standardized BRFSS frequency. The current two variants
are in `configs/experiments/phase13_canonical_preprocessing.json`; both exclude
`ALCDAY5` and use the selected depth-5, 128-feature, 200-tree booster.
`phase13_canonical_derived` completed nested evaluation with F1 0.44029,
versus 0.44174 for the Phase 11 reference. The `phase13_canonical_compact`
run was stopped during its first outer fit at the user's request, so it has no
complete evaluation result. The original P2 configuration remains unchanged.

A separate Phase 14 plan now applies the 14 verified nonresponse recodes and
`ALCDAY5=888 -> 0` while retaining all 295 P2 features. Its preprocessing and
boosting suite are `configs/experiments/phase14_codebook_corrections_only.json`
and `configs/experiments/phase14_boosting_codebook_corrections_only.json`.
The mappings were verified on both train and test caches; the model has not
been rerun with this plan.
