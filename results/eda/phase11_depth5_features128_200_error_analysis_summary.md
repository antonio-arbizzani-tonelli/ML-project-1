# Nested out-of-fold error analysis

This report uses nested out-of-fold predictions from the configured experiment. Each row is classified with the threshold selected from inner folds of its own outer fold. The 20% local validation partition is not loaded.

## Error counts

| Outcome | Rows |
| --- | ---: |
| true negative | 217493 |
| false positive | 21835 |
| false negative | 10419 |
| true positive | 12761 |

## Probability distribution by outcome

| Outcome | Rows | Mean | Median | Q25 | Q75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| true negative | 217493 | 0.04313 | 0.01994 | 0.00678 | 0.06209 |
| false positive | 21835 | 0.33066 | 0.29954 | 0.24898 | 0.38266 |
| false negative | 10419 | 0.10976 | 0.11097 | 0.06070 | 0.15934 |
| true positive | 12761 | 0.42272 | 0.37407 | 0.28397 | 0.50328 |

## Groups with the highest miss rate among actual positives

| Feature | Group | Rows | Positive rows | FN rate |
| --- | --- | ---: | ---: | ---: |
| GENHLTH | 1: Excellent | 45464 | 919 | 0.877 |
| _AGE80 | 18-44 | 70160 | 876 | 0.850 |
| GENHLTH | 2: Very good | 86578 | 3657 | 0.795 |
| EMPLOY1 | 1: Employed for wages | 107133 | 3391 | 0.768 |
| BPHIGH4 | 3: No | 151719 | 5689 | 0.679 |
| EMPLOY1 | 2: Self-employed | 21827 | 1202 | 0.657 |
| _AGE80 | 45-54 | 42835 | 2037 | 0.633 |
| INCOME2 | 8: $75,000 or more | 68975 | 3336 | 0.614 |
| PHYSHLTH | 0 days | 163506 | 9634 | 0.609 |
| EMPLOY1 | 5: A homemaker | 16111 | 1031 | 0.590 |

## Groups with the highest false-positive rate among actual negatives

| Feature | Group | Rows | Negative rows | FP rate |
| --- | --- | ---: | ---: | ---: |
| GENHLTH | 5: Poor | 13543 | 9234 | 0.514 |
| PHYSHLTH | 30 days | 25838 | 20289 | 0.323 |
| DIABETE3 | 1: Yes | 33872 | 26408 | 0.315 |
| EMPLOY1 | 8: Unable to work | 18764 | 14836 | 0.301 |
| GENHLTH | 4: Fair | 34676 | 28014 | 0.300 |
| PHYSHLTH | 14-29 days | 14362 | 11787 | 0.240 |
| _AGE80 | 65+ | 90699 | 75444 | 0.215 |
| EMPLOY1 | 7: Retired | 78482 | 65803 | 0.208 |
| BPHIGH4 | 1: Yes | 105481 | 88299 | 0.205 |
| INCOME2 | 2: Less than $15,000 ($10,000 to less than $15,000) | 11564 | 9672 | 0.204 |

## Crossed groups

The JSON result includes each requested pair and its three outer-fold breakdown. The paired analysis is used to decide whether an apparent single-feature error pattern persists after considering the second variable.

The group tables are diagnostic, not causal evidence: many features overlap and the same development data guided the model search. A promising follow-up needs to change one representation element at a time and be checked across outer folds.

The configured CSV output contains the full group table with false-negative and false-positive rates.
