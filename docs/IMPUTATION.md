# Codebook-aware imputation experiments

Phase 15 validated feature recovery. The selected Phase 16 downstream
challengers have been retrained in an exploratory outer-fold comparison.
Their thresholds were selected on the same pooled OOF labels being scored.
Phase 24 has now completed nested threshold selection for the original
restricted booster pipeline: F1 0.441791 versus 0.442155 for Phase 14, with
one of three folds improving. The exploratory gain is not confirmed. The
Phase 14 booster and submission threshold remain the current reference.

See [the Phase 24 report](../results/eda/phase24_original_restricted_imputation_summary.md).
The expanded variant and the revised restricted variant with exact indicator
bins have not received this nested evaluation. This document covers conditional
feature imputation for the booster and the historical logistic experiments.
The MLP uses a separate median/mode preprocessing pipeline described in
[docs/MLP.md](MLP.md); its Phase 32 F1 scores, 0.424451 and 0.419268, did not
replace Phase 14.

## Scope

- `phase15_missing_reason_flags`: leave all cleaned values missing and add
  only the same original-state indicators used by the direct imputation plan.
  This isolates the effect of the indicators from the predicted values.
- `phase15_imputation_direct`: predict documented `unknown` and `refused`
  answers in retained core and optional questionnaire columns. Predict blanks
  only in `HTM4`, `WTKG3`, `_BMI5`, `_FRUTSUM`, and `_VEGESUM`.
- `phase15_imputation_all`: additionally predict documented nonresponse codes
  in retained administrative and derived columns. This is a stress test: an
  unknown derived value can mean that upstream information was unavailable.
- All other blank cells remain `NaN`, including skipped questions and activity
  measures that are undefined when activity was not reported.
- Combined codebook labels such as “Don't know/Refused/Missing” remain `NaN`:
  they do not tell us whether the question was actually asked.
- An indicator records the original unknown, refused, or selected blank state.
  Observed values never change.

The five continuous blank targets are selected explicitly. The codebook groups
some blank cells under “Not asked or Missing,” so it cannot prove the exact
reason for every blank. This experiment does not claim that all remaining
missing values are structural.

## Method and order

Every target is predicted from the **original**, codebook-cleaned anchor
features. Predicted continuous values are not fed to the nonresponse imputers,
and predicted questionnaire responses are not fed to the continuous imputers.
This removes any dependency on whether one family runs first.

Categorical, binary, ordinal, and count targets use a smoothed Naive Bayes
classifier over small anchor bins. It emits an observed valid response code.
Continuous targets use ridge regression over the same frozen anchor bins and
are clipped to the central 99.8% of observed training responses. These are
cheap, NumPy-only candidate imputers, not assumptions that all fields can be
recovered accurately.

For every outer and inner fold, anchor bins and imputers are fitted only on the
corresponding training rows. Neither `_MICHD` nor validation labels are passed
to the imputers. The downstream booster and F1 threshold follow the existing
nested evaluation protocol.

## Selected downstream variants

The restricted variant imputes documented `unknown`/`refused` codes only for
`INCOME2`, `PNEUVAC3`, and `EMPLOY1`. The expanded variant adds `ARTHDIS2`,
`BPHIGH4`, `EDUCA`, `FLUSHOT6`, `GENHLTH`, `HAVARTH3`, `HPVTEST`, `JOINPAIN`,
`LMTJOIN3`, `MARITAL`, `PDIABTST`, `QLACTLM2`, `RCSRLTN2`, `RENTHOM1`,
`SMOKE100`, and `TOLDHI2`. Each variant imputes blank `HTM4` and `WTKG3`
using the feature-only ridge imputers. The original nonresponse and blank
indicators stay in the model inputs.

Missing `_BMI5` is reconstructed as `WTKG3 / HTM4**2` after height and weight
imputation, rounded to two decimals. It is filled only when both source values
are positive and the result falls in the observed training BMI range. Original
BMI values remain untouched. This formula is documented in the feature
metadata and avoids inconsistent independent BMI predictions.

The categorical and height/weight imputers use original cleaned anchors and
are fitted separately inside each training partition. Only BMI is derived
after those predictions. Questionnaire blanks that indicate a skipped or
unasked question remain `NaN`; for a selected categorical feature, only its
documented, unambiguous unknown/refused codes are substituted.

The variants are in `configs/experiments/phase16_selected_imputation_preprocessing.json`.
Each has its own booster suite and artifact directory. The Phase 15 broad direct
and extended suites remain available as feature-audit configurations.

## Data prerequisites

Install the dependencies and obtain the four official CSV files as described
in the [README](../README.md) and [data/README.md](../data/README.md).
The experiment runners read prepared NumPy caches. From the repository root:

```powershell
python -m tools.prepare_data
```

This checks ID alignment and creates `data/processed/` without training a model.
If feature CSV values change, use a new cache directory: existing feature
caches are checked for dimensions and IDs, not every value. The development
split indices are included in the repository. Numeric predictions, caches and
checkpoints are local artifacts excluded from Git.

## First downstream test: retrain and choose the best OOF threshold

The optimistic comparison retrains each disease model and each imputer inside
the three outer training folds. It saves one probability for every development
row and chooses the F1-maximizing threshold on those same pooled OOF rows.
That threshold's F1 is useful for screening, but is optimistic. The logistic
runner also saves the original missing-state flags as model inputs. The
historical screening can be rerun from the repository root with:

```powershell
python -m src.run_phase16_logistic_cv phase16_imputation_restricted --optimistic-only
python -m src.run_phase16_logistic_cv phase16_imputation_expanded --optimistic-only
python -m src.run_boosting_cv configs/experiments/phase16_imputation_restricted_boosting_optimistic.json
python -m src.run_boosting_cv configs/experiments/phase16_imputation_expanded_boosting_optimistic.json
```

The logistic outputs are written to
`results/linear_imputation_artifacts/phase16_imputation_*_optimistic/`.
The tree outputs and fold model checkpoints are written to
`results/boosting_artifacts/phase16_imputation_*_optimistic/`.
These directories are separate from the full nested runs. The optional
`python -m src.summarize_phase16_retrained --optimistic-only` checks row,
label and fold alignment across the four variants and both controls. It also
requires the historical Phase 7 logistic and Phase 14 booster OOF artifacts,
which are excluded from Git; it cannot run from a fresh clone using only the
four commands above. It writes a combined NPZ and overwrites
`results/experiments/phase16_retrained_comparison.json` with the new summary.
The historical linear reference in that comparison used median fills for
categorical missing values; the newly trained linear models use the current
mode-fill preprocessing. Their difference from the historical reference is
therefore not attributable solely to imputation.

## Original nested commands and completed Phase 24 confirmation

Feature recovery is already summarized in
`results/eda/phase15_imputation_variable_decisions.md`. The full nested
comparison uses two inner folds to select a threshold for each outer fold:

The commands below describe the original full-retraining route. The restricted
booster confirmation is complete through Phase 24; its command reproduces a
completed experiment. The other commands expose available experimental variants.

```powershell
python -m src.run_boosting_cv configs/experiments/phase16_imputation_restricted_boosting.json
python -m src.run_boosting_cv configs/experiments/phase16_imputation_expanded_boosting.json
python -m src.run_phase16_logistic_cv phase16_imputation_restricted
python -m src.run_phase16_logistic_cv phase16_imputation_expanded
```

`python -m src.summarize_phase16_retrained` requires all four nested variant
OOFs and the same two historical control artifacts described above.

The completed exploratory retraining gives F1 0.442674 for the restricted
boosting variant and 0.441989 for the expanded variant, compared with 0.442078
for the original booster under the same optimistic threshold protocol. These
are not nested F1 estimates. The historical record is
`results/experiments/phase16_retrained_comparison.json`.

The feature recovery audit scores observed answers, so its accuracy may not
transfer to people who actually declined to answer. Selection must therefore
use downstream nested F1 together with average precision, log loss, and fold
differences. Phase 23 tested only the restricted indicators with exact binary
bins and scored 0.440322. Phase 24 retained original quantile bins to confirm
the actual earlier restricted pipeline and scored 0.441791. Both are below
Phase 14; their mutual difference does not isolate imputation because their
binning differs. The revised combination with exact bins is a separate,
unrun candidate. The Phase 15 flags-only configuration is also not automatically
equivalent to either restricted indicator treatment.

Phase 24 reused the three original outer checkpoints after exact reproduction
of bin edges, training scores and external probabilities, then trained the six
missing inner models with training-only imputers. The new inner predictions
and row boundaries are saved under
`results/boosting_artifacts/phase24_original_restricted_imputation/`.
The dedicated configuration is
`configs/experiments/phase24_original_restricted_imputation.json`.
The complete result is
`results/experiments/phase24_original_restricted_imputation_comparison.json`.
The original restricted imputation is not adopted. Phase 14 retains its
295-feature tree representation, direct missing-value routing and frozen
submission threshold `0.2108690997`. Later booster and MLP comparisons are
summarized in [docs/EXPERIMENTS.md](EXPERIMENTS.md).
