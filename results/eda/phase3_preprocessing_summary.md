# Phase 3 preprocessing screening

Run date: 2026-09-19.

## Protocol

All results use three deterministic stratified folds inside the saved 80%
development partition (262,508 rows, seed `20260919`). The outer 20%
validation partition was not read. Every run uses the same unregularized
full-batch logistic regression (50 updates, `gamma=0.1`) and the same fixed
probability threshold of 0.5. The threshold is deliberately not tuned in this
screening; threshold selection belongs to the next phase.

The command is:

```powershell
py -3 -m src.run_preprocessing_cv configs/experiments/phase3_preprocessing_cv.json
```

## Results

| Variant | Output features | F1 | Accuracy | Precision | Recall | Log loss | Average precision | Max design memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 raw median/scale baseline | 321 | 0.15779 | 0.91285 | 0.53804 | 0.09245 | 0.33198 | 0.34447 | 322 MiB |
| P1 structural exclusions | 295 | 0.15686 | 0.91281 | 0.53695 | 0.09185 | 0.33207 | 0.34358 | 296 MiB |
| P2 codebook-clean values | 296 | 0.03829 | 0.91196 | 0.54118 | 0.01984 | 0.24152 | 0.30471 | 297 MiB |
| P3 indicators for features missing at least 50% | 436 | 0.04758 | 0.91200 | 0.53724 | 0.02489 | 0.23954 | 0.31004 | 438 MiB |
| P3b indicators for features missing at least 1% | 545 | 0.04852 | 0.91200 | 0.53594 | 0.02541 | 0.23930 | 0.31088 | 547 MiB |
| P4 sparse indicators plus one-hot `_STATE`, `DIABETE3` | 491 | 0.04093 | 0.91198 | 0.54116 | 0.02127 | 0.23975 | 0.30916 | 493 MiB |
| P5 detailed missing states for six reviewed fields | 508 | 0.04093 | 0.91199 | 0.54176 | 0.02127 | 0.23975 | 0.30920 | 510 MiB |
| P6 broad indicators plus one-hot `_STATE`, `DIABETE3` | 600 | 0.04194 | 0.91193 | 0.53207 | 0.02183 | 0.23952 | 0.30997 | 602 MiB |

The numbers are out-of-fold metrics pooled across all development rows. The
full JSON records, with metrics for each fold and exact configurations, are in
`results/experiments/`.

## What was implemented

- Identifiers, interview dates, survey-design fields, raw mixed-unit weight and
  height fields, and redundant age-bin/height fields are excluded in the
  semantic configurations. `Id` was already excluded from the cached model
  inputs and remains available for submission assembly.
- `WTKG3`, `HTM4`, `_BMI5`, and `_AGE80` are retained in their existing project
  units. Project caches already contain `WTKG3` in kilograms and `_BMI5` as a
  conventional BMI value, so neither is divided by 100 again.
- Documented response codes such as `7`, `9`, `77`, `99`, and `14` become
  missing only for the columns where the codebook defines them as a non-response.
  `7` and `9` remain valid values in columns such as age groups or income when
  the codebook says they are valid.
- `PHYSHLTH=88` and `MENTHLTH=88` become zero days. `77` and `99` in those
  variables become missing.
- Missing values are filled with a statistic fitted on each training fold.
  Indicators are calculated before filling. `_AGEG5YR=14` contributes an
  auxiliary `age-not-reported` flag while `_AGEG5YR` itself is excluded because
  `_AGE80` is the preferred age measure.
- `_STATE` and `DIABETE3` were also tested as one-hot categories. The category
  list is learned from each training fold, and their missing state receives a
  separate feature.

## Interpretation and next action

P1 is effectively tied with P0 while removing 26 columns and about 26 MiB from
the retained design matrices. It is a reasonable lean reference, but it still
treats response codes as numbers and is not an acceptable final semantic
pipeline.

P2 makes probabilities markedly better under log loss, from 0.332 to 0.242,
but reduces average precision and recall at the fixed 0.5 threshold. This shows
that raw numeric response codes were carrying predictive signal, not that they
were meaningful measurements. The semantic pipeline needs a validation-only
threshold study before it can be judged by F1; its current probabilities are
more conservative.

Combined missingness indicators improve P2 slightly. Expanding them from the
50% rule to the 1% rule adds 109 columns and about 109 MiB, for an F1 gain of
only 0.00094 and an average-precision gain of 0.00085. The broad version is not
worth retaining at this stage. One-hot state/diabetes and detailed
unknown/refusal/blank indicators increase the matrix further and regress on all
screening metrics.

The next controlled work is to tune the regularized logistic optimizer and the
classification threshold within development-only cross-validation, beginning
with P1 and the compact codebook-clean representation P2. This will determine
whether P2's lower log loss converts into a better F1 trade-off without using
the outer validation partition for model selection.
