# Feature audit for older respondents reporting good health

## Data and scope

This is a descriptive follow-up to `shared_error_followup.md`. It reads the
Phase 14 booster's nested-threshold OOF probabilities and decisions, and the
Phase 7 logistic model's nested decisions, from
`results/experiments/phase16_optimistic_all_oof_predictions.npz`. Raw survey
columns are aligned by `development_indices` to
`data/processed/x_train_float32.npy` and the `x_train.csv` header. The 20%
local validation partition is untouched. No candidate is retrained here.

The focus is respondents aged 65+ with `GENHLTH` 1 (excellent) or 2 (very
good): 39,414 rows, 3,219 positives (8.17%), and 2,332 positives missed by
both models. The booster assigns this cohort mean probability 8.13%, close to
its observed 8.17% prevalence. Its within-cohort average precision is 0.2574,
against prevalence 0.0817. Thus the high false-negative count at the global
F1 threshold does not by itself show subgroup miscalibration or a broken
health-code transformation.

## Existing evidence

Among these older, healthy-responding participants, the 2,332 shared-missed
positives report fewer conventional risk markers than the 773 positives found
by the booster:

| Response | Shared-missed positives | Booster-detected positives | Negatives |
| --- | ---: | ---: | ---: |
| High blood pressure: yes | 59.1% | 83.7% | 47.4% |
| Told cholesterol high: yes | 55.7% | 83.4% | 44.0% |
| Diabetes: yes | 14.3% | 36.6% | 9.3% |
| Ever smoked 100 cigarettes: yes | 46.5% | 64.6% | 43.1% |
| Poor physical health on more than zero days | 21.4% | 32.2% | 19.3% |

These are not exactly matched groups. Age also differs (mean 73.8, 76.7, and
72.4 respectively). The differences explain why the observed questionnaire
answers often lead to a low predicted probability; they do not imply the
positive labels are incorrect.

## Derived-feature candidates

The following component questions are **already present individually** in the
selected 295-feature tree input. A count could make their joint burden easier
to use when a depth-5 tree samples 128 candidate features at a split. A count
must be accompanied by the number of substantive answers, so unanswered
questions are not silently treated as negative.

| Candidate | Definition: count of code `1` (`Yes`) | Cohort rows with >=1 | Positives with >=1 | Shared misses with >=1 |
| --- | --- | ---: | ---: | ---: |
| Functional limitations | `BLIND`, `DECIDE`, `USEEQUIP` | 4,765 | 679 (14.2%) | 392 |
| Other diagnosed conditions | `ADDEPEV2`, `CHCCOPD1`, `CHCKIDNY` | 6,392 | 756 (11.8%) | 421 |

The two >=1 groups together contain 689 distinct shared misses, 29.5% of
those in the older healthy cohort and 7.3% of all 9,464 shared misses. This
is an upper bound on how many of the observed shared misses are even touched
by these indicators, not a projected gain.

For an exploratory residual check, partition this cohort by five-year age
band (65–69, 70–74, 75–79, 80+), `GENHLTH` (1/2), and decile of the saved
booster OOF probability. Within each stratum, compute the expected number of
positives carrying the proposed indicator from the stratum prevalence. The
number observed is larger in every outer fold:

| Indicator | Fold 1 observed / expected | Fold 2 | Fold 3 |
| --- | ---: | ---: | ---: |
| >=1 functional limitation | 229 / 185 | 228 / 207 | 222 / 194 |
| >=1 other diagnosed condition | 243 / 216 | 257 / 231 | 256 / 226 |

The same direction holds when stratifying instead by age band, `GENHLTH`,
sex, and yes/no answers to high blood pressure, high cholesterol, and diabetes.
This supports testing the two counts separately. It does **not** show that
either would improve F1 after the booster is retrained. The cohort and
features were selected while inspecting these development labels, and the
folds are not a fresh external validation set. Several components may also
reflect illness after the cardiovascular event; no causal interpretation is
claimed.

A direct shift of the existing score threshold for the functional >=1 group
does not help: a 0.03 decrease adds 59 true positives and 210 false positives,
changing F1 from 0.442155 to 0.442139 on the same OOF rows. A 0.05 shift for
functional >=2 increases descriptive F1 only to 0.442330 while affecting 23
true positives and 58 false positives. This tiny, post-hoc change is not a
validated improvement. New feature engineering would have to change ranking,
not merely change the decision threshold.

The diabetes-onset age `DIABAGE2` is another interpretable source for a
duration feature, but only 4,001 participants in this cohort report diabetes
and `_AGE80` caps age at 80. A duration derived from these values has limited
coverage and is lower priority than the two counts above.

## Removal and validity candidates

* **Do not remove age or `GENHLTH` based on this error analysis.** Older
  respondents with excellent/very good health have 8.17% positive prevalence,
  versus 32.02% among older respondents reporting fair/poor health. The
  booster's mean probability is 8.13% in the former group and 31.71% in the
  latter. Both variables contain genuine predictive information in this
  cross-sectional survey.
* **Ablate `HAREHAB1` for validity.** Its question asks about rehabilitation
  following a heart attack. All 620 development respondents with any recorded
  answer are target-positive. In the older healthy cohort, only 60 have a
  recorded answer, all positive, and only one is a shared miss. Removing it
  cannot explain or directly repair the 2,332 shared misses, but the question
  is target-conditioned and inappropriate for a prospective risk model.
* **Review `CVDASPRN` and `RDUCHART` before prospective interpretation.**
  Aspirin use/advice can follow a cardiovascular diagnosis. They are missing
  for 37,895 (96.1%) and 38,683 (98.2%) rows of the older healthy cohort,
  respectively. Their removal is not justified by an F1 result here; test
  it only as a separate temporality/validity ablation.
* **Treat procedural survey fields as an ablation hypothesis, not a deletion.**
  Phone/residence fields and interview date parts have weak or sparse residual
  associations in this cohort. They might encode survey design or geographic
  differences, but no retrained comparison shows whether excluding them helps
  or hurts generalization. The previous Phase 13 broad removal changed many
  other features at once and cannot answer this isolated question.

A scan across many single response codes also surfaced rare procedural
categories, such as `WHRTST10=3` (hospital location of last HIV test): only
367 cohort rows, with 59 positives and 43 shared misses. Selecting such codes
from hundreds tested categories would invite multiple-testing overfit and
does not provide a defensible first feature experiment.

The [CDC 2015 calculated-variable documentation](https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf)
defines `_MICHD` as ever reporting myocardial infarction **or** coronary heart
disease. The [CDC 2015 codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/CODEBOOK15_LLCP.pdf)
states that `HAREHAB1` asks about rehabilitation following a heart attack and
is skipped when the heart-attack question was not answered positively. This
supports the target-conditioned feature concern above.

## Next controlled test

Keep the Phase 14 representation fixed. Add only the functional count plus its
answered count in one challenger, and only the other-condition count plus its
answered count in a second challenger. Screen each on the saved development
folds, then run nested threshold evaluation for a promising candidate and
compare F1, average precision, log loss, false positives/negatives, and fold
directions. Separately ablate `HAREHAB1` to assess validity. The 20% local
validation partition should remain reserved for a selected, frozen pipeline.
