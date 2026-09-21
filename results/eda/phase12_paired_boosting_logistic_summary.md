# Phase 12 — paired error analysis

The same development rows and outer folds are compared with each model's own nested F1 threshold. No model was retrained. Threshold shifts and group rankings below are descriptive and must not be used to choose a new threshold on these rows.

## Overall results

| Model | F1 | Precision | Recall | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boosting | 0.44174 | 0.36886 | 0.55052 | 21835 | 10419 |
| Logistic | 0.42563 | 0.35728 | 0.52632 | 21947 | 10980 |

## Paired predictions

| Actual class | Both correct | Only boosting correct | Only logistic correct | Both wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| Positive | 11316 | 1445 | 884 | 9535 |
| Negative | 213458 | 4035 | 3923 | 17912 |

## Boosting errors by distance from its fold threshold

| Absolute distance | FN, logistic correct | FN, both wrong | FP, logistic correct | FP, both wrong |
| --- | ---: | ---: | ---: | ---: |
| <0.01 | 183 | 295 | 931 | 771 |
| 0.01-<0.03 | 297 | 637 | 1231 | 1629 |
| 0.03-<0.10 | 354 | 3333 | 1530 | 5887 |
| >=0.10 | 50 | 5270 | 231 | 9625 |

A distance of at least 0.10 means far from the decision threshold, not a calibrated confidence level. Strict tails are reported separately: false negatives with probability <0.05 and false positives with probability >=0.50.

## Diagnostic threshold perturbation

| Change to each fold threshold | F1 | False positives | False negatives |
| ---: | ---: | ---: | ---: |
| -0.01 | 0.44055 | 23683 | 9941 |
| +0.00 | 0.44174 | 21835 | 10419 |
| +0.01 | 0.44049 | 20133 | 10946 |

## Group detail

The JSON and CSV contain paired false-positive and false-negative counts for each predefined feature group, plus a fold breakdown. A group can overlap other groups; its error count is not an independent contribution. Differences here are diagnostic and are subject to model-selection reuse of the development data.

## Checkpoint split usage

Split counts describe how often a feature was used, not its causal effect or an ablation result.

| Feature | Fold | Trees using it | Total splits | Root splits |
| --- | ---: | ---: | ---: | ---: |
| HAREHAB1 | 1 | 96/200 | 96 | 59 |
| HAREHAB1 | 2 | 97/200 | 97 | 61 |
| HAREHAB1 | 3 | 101/200 | 101 | 60 |

## Interpretation checks

The logistic model also misses 91.5% of boosting's false negatives and 82.0% of its false positives. At distance >=0.10 those fractions are 99.1% and 97.7% respectively.

The two +/-0.01 threshold perturbations lower F1 on these same outer predictions. This is a local diagnostic, not a newly selected threshold or an unbiased comparison of alternative threshold policies.

HAREHAB1 has a recorded response code in 620 development rows; 620 are positive. The feature asks about rehabilitation after a heart attack and is a target-conditioned leakage concern. Split counts and these associations do not quantify its causal contribution; a fit without it is required for that comparison.
