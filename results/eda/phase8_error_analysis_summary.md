# Phase 8 — nested logistic error analysis

This report uses the nested OOF predictions of the corrected P2 logistic reference. Each row is classified with the threshold selected from inner folds of its own outer fold. The 20% local validation partition is not loaded.

## Error counts

| Outcome | Rows |
| --- | ---: |
| true negative | 217381 |
| false positive | 21947 |
| false negative | 10980 |
| true positive | 12200 |

## Probability distribution by outcome

| Outcome | Rows | Mean | Median | Q25 | Q75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| true negative | 217381 | 0.04384 | 0.02292 | 0.00755 | 0.06426 |
| false positive | 21947 | 0.33919 | 0.29962 | 0.24410 | 0.39764 |
| false negative | 10980 | 0.10694 | 0.10763 | 0.06125 | 0.15300 |
| true positive | 12200 | 0.41410 | 0.37318 | 0.27824 | 0.51871 |

## Groups with the highest miss rate among actual positives

| Feature | Group | Rows | Positive rows | FN rate |
| --- | --- | ---: | ---: | ---: |
| GENHLTH | 1: Excellent | 45464 | 919 | 0.926 |
| _AGE80 | 18-44 | 70160 | 876 | 0.897 |
| GENHLTH | 2: Very good | 86578 | 3657 | 0.819 |
| EMPLOY1 | 1: Employed for wages | 107133 | 3391 | 0.800 |
| BPHIGH4 | 3: No | 151719 | 5689 | 0.721 |
| EMPLOY1 | 2: Self-employed | 21827 | 1202 | 0.691 |
| _AGE80 | 45-54 | 42835 | 2037 | 0.680 |
| INCOME2 | 8: $75,000 or more | 68975 | 3336 | 0.660 |
| PHYSHLTH | 0 days | 163506 | 9634 | 0.630 |
| _BMI5 | missing | 21655 | 1312 | 0.597 |

## Groups with the highest false-positive rate among actual negatives

| Feature | Group | Rows | Negative rows | FP rate |
| --- | --- | ---: | ---: | ---: |
| GENHLTH | 5: Poor | 13543 | 9234 | 0.555 |
| DIABETE3 | 1: Yes | 33872 | 26408 | 0.326 |
| PHYSHLTH | 30 days | 25838 | 20289 | 0.325 |
| EMPLOY1 | 8: Unable to work | 18764 | 14836 | 0.299 |
| GENHLTH | 4: Fair | 34676 | 28014 | 0.293 |
| PHYSHLTH | 14-29 days | 14362 | 11787 | 0.237 |
| INCOME2 | 2: Less than $15,000 ($10,000 to less than $15,000) | 11564 | 9672 | 0.218 |
| _AGE80 | 65+ | 90699 | 75444 | 0.216 |
| EMPLOY1 | 7: Retired | 78482 | 65803 | 0.210 |
| BPHIGH4 | 1: Yes | 105481 | 88299 | 0.208 |

## Crossed groups

The JSON result includes each requested pair and its three outer-fold breakdown. The paired analysis is used to decide whether an apparent single-feature error pattern persists after considering the second variable.

The group tables are diagnostic, not causal evidence: many features overlap and the same development data guided the model search. A promising follow-up needs to change one representation element at a time and be checked across outer folds.

The full group table with false-negative and false-positive rates is stored in `results/eda/analysis/phase8_nested_error_groups.csv`.
