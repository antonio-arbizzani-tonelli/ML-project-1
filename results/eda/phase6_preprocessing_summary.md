# Phase 6: BPHIGH4 correction and semantic nonlinear preprocessing

Run date: 2026-09-20.

## Protocol

Both candidates use the same three deterministic stratified folds inside the
saved development partition (262,508 rows; fold seed `20260919`). The outer
20% partition was not read. Every preprocessing statistic, category list,
missing-value fill, hinge knot, interaction scale, and model fit is learned
from the corresponding training fold only. Both candidates are unregularized
full-batch NumPy logistic regression with `gamma=0.2` and 4,000 updates.

F1, accuracy, precision, and recall below use the threshold that maximizes F1
on pooled OOF probabilities. This is useful for screening but optimistic,
because the same OOF labels select and describe that threshold. AP and log loss
do not depend on the threshold. The cross-fit threshold diagnostic is reported
as a transfer check only; it is not nested CV.

## BPHIGH4 correction

The official BRFSS codebook defines `BPHIGH4=7` as “Don't know/Not Sure” and
`BPHIGH4=9` as “Refused”. The metadata parser had retained both as unlabeled
numeric values, so P2 treated them as substantive categories. Their labels and
special-value status are now corrected in `configs/feature_metadata.json`.
The corrected compact reference converts them to missing values before its
training-fold median fill. This correction applies to all future P2 runs.

## Candidates

The corrected reference keeps P2 plus the four existing age interactions.
The semantic-nonlinear challenger makes one controlled representation change:

- one-hot categories for `BPHIGH4`, `DIABETE3`, `EMPLOY1`, `GENHLTH`, and
  `INCOME2`;
- separate unknown, refused, and blank indicators for those five variables;
- two training-fold quantile hinge terms each for `_AGE80` and `_BMI5`;
- age squared, age by smoking, and age by each non-reference substantive
  category of `BPHIGH4` and `DIABETE3`.

It has 342 named feature columns excluding the intercept, compared with 300
for the corrected reference. No 8,000-update run was performed.

## Results

| Candidate | F1 | Accuracy | Precision | Recall | AP | Log loss | Pooled threshold | Matrix pair memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Corrected P2 + 4 interactions | 0.42632 | 0.86945 | 0.34831 | 0.54935 | 0.38775 | 0.22198 | 0.19491 | 301.42 MiB |
| Semantic-nonlinear challenger | 0.42535 | 0.86994 | 0.34873 | 0.54513 | 0.38727 | 0.22206 | 0.19675 | 343.48 MiB |

The challenger changes F1 by -0.00097, AP by -0.00047, and log loss by
+0.00008, while increasing accuracy by +0.00049 and memory by 42.06 MiB. Its
cross-fit-threshold F1 is 0.42501, versus 0.42596 for the corrected reference.
The added representation therefore does not justify its cost or replace the
corrected P2 interaction candidate.

The correction itself is semantically required and is retained. Its direct
comparison with the previous, uncorrected Phase 5 interaction result is only
descriptive: pooled OOF F1 rises from 0.42624 to 0.42632, AP from 0.38761 to
0.38775, and log loss improves from 0.22201 to 0.22198. The affected 7/9
responses are rare, so the small change is expected.

## Decision

Keep the corrected P2 plus four interactions as the development leader. Do
not run nested threshold fitting for the challenger, since it loses on F1, AP,
and log loss in the screening comparison. Future feature tests should isolate
one component at a time, beginning with the categorical encoding group; the
combined challenger cannot identify whether one subcomponent has a small
offsetting benefit.

## Reproduction

```powershell
py -3 -m src.run_model_cv configs/experiments/phase6_bphigh4_corrected_reference_cv.json
py -3 -m src.run_model_cv configs/experiments/phase6_semantic_nonlinear_preprocessing_cv.json
py -3 -m unittest discover -s tests -q
```
