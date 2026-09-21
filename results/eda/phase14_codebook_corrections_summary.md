# Phase 14 — codebook corrections only

## Question and protocol

This experiment isolates documented response-code corrections from feature
removal. It keeps the 295 P2 source features and the selected NumPy histogram
booster: depth 5, 200 trees, 128 candidate features per node, 64 bins, and a
minimum leaf size of 200. The only representation changes are the 14 verified
feature-specific nonresponse mappings to `NaN` and `ALCDAY5=888 -> 0`.

For each of the three outer development folds, the F1 threshold is selected
from two inner out-of-fold fits using only that outer fold's training rows.
The comparison is therefore against the Phase 11 depth-5/128/200 reference on
the same 262,508 development rows and folds.

## Nested result

| Metric | Phase 11 reference | Phase 14 corrections | Difference |
| --- | ---: | ---: | ---: |
| F1 | 0.441741 | 0.442155 | +0.000415 |
| Precision | 0.368858 | 0.367632 | -0.001225 |
| Recall | 0.550518 | 0.554573 | +0.004055 |
| Accuracy | 0.877131 | 0.876434 | -0.000697 |
| Average precision | 0.428841 | 0.428609 | -0.000232 |
| Log loss | 0.215279 | 0.215225 | -0.000054 |

The new predictions have 12,855 true positives, 10,325 false negatives, and
22,112 false positives. Relative to the reference this is 94 fewer false
negatives and 277 more false positives. That trade raises F1 very slightly
because F1 emphasizes recovering positives, while it lowers accuracy and
precision slightly.

The result is directionally positive for the primary metric but too small and
mixed across other metrics to claim a meaningful predictive improvement. The
semantic correction remains appropriate independently of its small score
effect.

## Fold stability

| Outer fold | Reference F1 | Corrections F1 | Difference | Reference threshold | Corrections threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.437396 | 0.439027 | +0.001631 | 0.212139 | 0.203057 |
| 2 | 0.442139 | 0.444054 | +0.001915 | 0.218335 | 0.220105 |
| 3 | 0.445609 | 0.443512 | -0.002097 | 0.206991 | 0.209446 |

The fold directions are mixed. The nested thresholds remain in the same narrow
range (0.2031 to 0.2201), so the small F1 difference does not come from a
radically different decision rule.

## Where the input values changed

At least one corrected value occurs on 138,209 development rows. `ALCDAY5`
accounts for 126,912 changes: 123,530 `888` no-drinking values become zero and
3,382 `777/999` nonresponses become missing. The other verified nonresponse
recodes affect 11,297 rows after accounting for overlap.

On the corrected rows, descriptive F1 changes from 0.46040 to 0.46154 and log
loss improves from 0.25210 to 0.25196. On the 124,299 unchanged rows, F1 moves
from 0.40284 to 0.40178. These subgroup metrics use the fold-specific global
thresholds and are diagnostic only; they do not establish that one individual
recoding caused the overall change.

The paired probabilities remain extremely similar (correlation 0.99685; mean
absolute difference 0.00524). Of 262,508 rows, 1,464 change from an incorrect
reference prediction to a correct corrected-data prediction, while 1,647 move
in the opposite direction. The F1 gain therefore comes from the balance of
false negatives and false positives, not from a broad change in ranking.

## Recommendation

Use the Phase 14 codebook-correct representation as the semantically sound
baseline, but do not spend another broad hyperparameter grid on this tiny
difference. The next controlled representation test should exclude only
`ALCDAY5` while retaining all other Phase 14 features and `DROCDY3_`. It will
answer whether the still mixed weekly/monthly raw alcohol coding adds useful
signal after `888`, `777`, and `999` are corrected. The prior Phase 13 result
cannot answer this because it removed several other columns simultaneously.

After that isolated ablation, test nominal treatment of `EXRACT11` and
`EXRACT21` only if the alcohol ablation does not help. Their numeric activity
codes are labels rather than an ordered quantity, but they affect far fewer
rows and should not be mixed with the alcohol test. Keep the `HAREHAB1`
ablation separate as a validity check; it may reduce F1 but is needed before
interpreting the model as learning generalizable risk factors.

## Artifacts

- Experiment record: `results/experiments/20260921T105139261213Z_phase14-boosting-codebook-corrections-only-depth5-features128-200trees.json`
- Row-level probabilities and nested predictions: `results/boosting_artifacts/phase14_codebook_corrections_only/oof_predictions/phase14_boosting_codebook_corrections_only_depth5_features128_200trees_oof.npz`
- Preprocessing plan: `configs/experiments/phase14_codebook_corrections_only.json`
