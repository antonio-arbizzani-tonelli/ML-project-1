# Phase 9 — targeted preprocessing tests

Date: 2026-09-20.

## Motivation and protocol

Phase 8 showed stable crossed error patterns for `BPHIGH4`, age, diabetes,
and self-rated general health. This phase tests their representations one at a
time. Every candidate uses the corrected compact P2 plan, three fixed
development folds, logistic regression with `gamma=0.2` and 4,000 updates,
and only the listed interaction change. The local 20% partition is not read.

The pooled OOF F1 selects and describes the same probabilities, so it is used
for screening only. Cross-fit F1 selects each fold's threshold on the other
two OOF folds and is the more useful screen for whether to spend a nested run.
The established reference has exploratory OOF F1 0.42632 and cross-fit F1
0.42596.

## Results

| Candidate | Output features | OOF F1 | Accuracy | Precision | Recall | AP | Log loss | Cross-fit F1 | Cross-fit accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Corrected P2 reference | 300 | 0.42632 | 0.86945 | 0.34831 | 0.54935 | 0.38775 | 0.22198 | **0.42596** | 0.87035 |
| `GENHLTH` categories | 304 | 0.42621 | 0.87345 | 0.35540 | 0.53227 | 0.38715 | 0.22212 | 0.42529 | **0.87523** |
| `BPHIGH4` categories | 302 | 0.42604 | **0.87461** | **0.35754** | 0.52701 | 0.38733 | 0.22219 | 0.42570 | 0.87409 |
| `BPHIGH4` categories plus age interactions | 305 | **0.42632** | 0.86896 | 0.34749 | **0.55142** | **0.38779** | **0.22195** | 0.42581 | 0.86899 |
| `DIABETE3` categories | 302 | 0.42581 | 0.86934 | 0.34791 | 0.54866 | 0.38766 | 0.22197 | 0.42517 | 0.86959 |
| `DIABETE3` categories plus age interactions | 305 | 0.42580 | 0.86871 | 0.34684 | 0.55129 | 0.38771 | 0.22197 | 0.42489 | 0.86960 |
| `GENHLTH` categories plus age interactions | 308 | 0.42631 | 0.87190 | 0.35259 | 0.53900 | 0.38711 | 0.22209 | 0.42452 | 0.87131 |

## Interpretation

The crossed error analysis gave a real reason to test `BPHIGH4`: the error
profile changes sharply by age and health. For example, among over-65 rows
with high blood pressure, the false-positive rate is about 31% in every fold;
among 18–44 rows with high blood pressure, false negatives dominate. These
patterns do not automatically imply that a category representation improves
overall F1.

The category-plus-age `BPHIGH4` candidate essentially ties the exploratory
reference F1 (+0.00000 after rounding), slightly improves AP and log loss, but
loses 0.00015 cross-fit F1 and 0.00136 cross-fit accuracy. Its small change is
not sufficient to justify a new nested threshold run. Plain `BPHIGH4` and
`GENHLTH` categories trade a lower F1 for higher accuracy by reducing recall.
The diabetes and general-health age interactions lose cross-fit F1 more
clearly.

## Decision

Retain the corrected numeric P2 representation and its four original
interactions as the preprocessing reference. Do not combine these category
experiments or send them to nested evaluation. The isolated tests indicate
that this family of categorical replacements is not currently the route to an
F1 near 0.45. The next material experiment should compare the existing
NumPy-only boosting implementation with the logistic reference using the same
development folds.

## Reproducibility

The six records are the `phase9-*targeted-preprocessing-gamma-0p2-4000.json`
files under `results/experiments/`. Each contains configuration, all metrics,
three fold details, OOF scores, feature names, weights, and input checksums.
