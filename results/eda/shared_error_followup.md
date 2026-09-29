# Shared false negatives: Phase 14 booster and Phase 7 logistic

The targeted feature review for older respondents reporting excellent or very
good health is in `elderly_feature_audit.md`.

This is a descriptive analysis of the 262,508 development rows. It uses the
saved **nested-threshold** decisions for `tree_original` (Phase 14 corrected
booster) and `linear_original` (Phase 7 logistic) from
`results/experiments/phase16_optimistic_all_oof_predictions.npz`. The raw
feature values come from `data/processed/x_train_float32.npy`, aligned by
`development_indices` and the `x_train.csv` header. The 20% local validation
partition is not used. No model is retrained.

The previous Phase 12 analysis used the Phase 11 booster and found 9,535
positives missed by both models. With the Phase 14 booster, that count is
9,464: Phase 14 recovers 237 of the previous shared misses and newly misses
166 positives that the Phase 11 booster had found, for a net reduction of 71.

## Paired positive outcomes

| Outcome | Rows | Share of 23,180 positives |
| --- | ---: | ---: |
| Both correct | 11,339 | 48.92% |
| Booster correct only | 1,516 | 6.54% |
| Logistic correct only | 861 | 3.71% |
| Both miss | 9,464 | 40.83% |

Taking the union of the two existing positive decisions recovers the 861
positives found only by logistic, but also adds 3,991 false positives that the
booster correctly rejected. Its descriptive F1 would be 0.43544, below the
booster's nested-threshold F1 of 0.44216.

The shared-miss rate by outer fold is 40.78%, 41.49%, and 40.21%. Of the
9,464 shared misses, 4,437 (46.88%) are at least 0.10 below **both** models'
own fold thresholds. Another diagnostic is that 1,716 (18.13%) receive
probability below 0.05 from both models. Only 1,423 (15.04%) are within 0.03
of at least one model's threshold. These sets overlap and must not be summed.
Distance from a decision threshold is not a calibrated confidence measure.

## Prespecified groups

The percentages below condition on the **actual positives** in each group.
Group rows overlap, so shared-miss counts across rows must not be summed.

| Group | Positive / all rows | Shared misses | Miss rate among positives | Share of all shared misses | Fold miss rates |
| --- | ---: | ---: | ---: | ---: | --- |
| Age 18–44 | 876 / 70,160 | 731 | 83.4% | 7.7% | 84.5%, 84.2%, 81.6% |
| Age 65+ | 15,255 / 90,699 | 5,069 | 33.2% | 53.6% | 33.3%, 33.4%, 33.0% |
| General health excellent | 919 / 45,464 | 790 | 86.0% | 8.3% | 84.6%, 83.8%, 89.5% |
| General health very good | 3,657 / 86,578 | 2,784 | 76.1% | 29.4% | 75.1%, 76.8%, 76.5% |
| General health poor | 4,309 / 13,543 | 418 | 9.7% | 4.4% | 9.1%, 10.5%, 9.5% |
| No diagnosed high blood pressure | 5,689 / 151,719 | 3,677 | 64.6% | 38.9% | 64.3%, 65.6%, 64.1% |
| Zero poor physical-health days | 9,634 / 163,506 | 5,437 | 56.4% | 57.4% | 56.1%, 56.5%, 56.7% |
| Age 18–44 and health excellent/very good | 201 / 41,679 | 193 | 96.0% | 2.0% | 93.4%, 93.9%, 100.0% |
| Age 65+ and health excellent/very good | 3,219 / 39,414 | 2,332 | 72.4% | 24.6% | 71.3%, 72.9%, 73.1% |
| No diagnosed high blood pressure and health very good | 1,244 / 56,006 | 1,115 | 89.6% | 11.8% | 90.4%, 89.0%, 89.5% |

The young, excellent/very-good-health intersection has positive prevalence
0.48%. Its booster average precision is 0.0797, compared with 0.0048 for its
prevalence; this indicates some within-group ranking signal despite the high
miss rate at the global threshold. For the 65+ excellent/very-good-health
intersection, prevalence is 8.17% and booster average precision is 0.2574.
The error pattern therefore does not establish that these features were
miscoded or that the models contain no information in these groups.

Within the 65+ excellent/very-good-health cohort, the shared misses resemble
negative respondents more closely than the positives detected by the booster
on several recorded risk factors. These are raw response-code comparisons,
restricted to a broad age band; they are not an exact matched analysis.

| Recorded characteristic | 2,332 shared-missed positives | 773 booster-detected positives | 36,195 negatives |
| --- | ---: | ---: | ---: |
| Mean age | 73.8 | 76.7 | 72.4 |
| High blood pressure: yes | 59.1% | 83.7% | 47.4% |
| Diabetes: yes | 14.3% | 36.6% | 9.3% |
| Told cholesterol high: yes | 55.7% | 83.4% | 44.0% |
| Ever smoked 100 cigarettes: yes | 46.5% | 64.6% | 43.1% |
| More than zero poor physical-health days | 21.4% | 32.2% | 19.3% |
| Zero or one `yes` among blood pressure, diabetes, cholesterol | 57.4% | 18.9% | 70.5% |
| Median booster probability | 0.100 | 0.270 | 0.052 |

The age pattern remains within this cohort: shared-miss rates among positives
are 88.0% at ages 65–69, 77.6% at 70–74, 69.0% at 75–79, and 60.1% at 80+.
Those bands contain respectively 725, 776, 667, and 1,051 positives.

## Descriptive threshold perturbations

Starting from the Phase 14 nested predictions (F1 0.442155), a fixed downward
shift in the existing fold threshold only within a prespecified group gives:

| Group and shift | F1 on the same OOF rows | Added true positives | Added false positives |
| --- | ---: | ---: | ---: |
| Age 18–44, −0.03 | 0.441847 | 16 | 97 |
| Age 18–44, −0.05 | 0.441909 | 45 | 191 |
| Health excellent/very good, −0.03 | 0.441398 | 283 | 1,099 |
| Health excellent/very good, −0.05 | 0.439728 | 498 | 2,088 |
| Age <55 and health excellent/very good, −0.03 | 0.442141 | 4 | 16 |

These fixed shifts are diagnostics on already generated predictions, not a
validated subgroup-threshold policy. Choosing a group and shift on these same
OOF labels would make its reported F1 optimistic. The observed tradeoff does
not support simply lowering thresholds in the prominent missed groups.

## Interpretation and follow-up

The models agree on many low-risk-looking positives. This may reflect the
survey's limited ability to distinguish a past infarction/coronary diagnosis
from people with similar current self-reported health. It is not evidence that
those labels are wrong. Young healthy respondents have a very high miss rate
but represent only 2% of all shared misses in the strict intersection above.
Older respondents reporting excellent or very good health account for nearly
one quarter, making that a more consequential cohort to inspect.

The coarse within-cohort comparison above explains part of the model's low
scores: many missed positives report fewer conventional risk markers. A next
diagnostic would match more narrowly by age and reported risk factors, then
look for residual differences in existing cardiovascular history, exercise,
and medication responses. Check any candidate pattern in each fold. Keep the
suspected target-conditioned `HAREHAB1` feature out of scientific
interpretation until its ablation is measured. Feature selection based on
these rows needs separate validation before a gain is claimed.
