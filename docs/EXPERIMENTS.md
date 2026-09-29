# Experiments and decisions

## Validation protocol

A fixed stratified 80/20 split was generated with seed `20260918`. The initial
baselines inspected the 20% partition, so later model selection is confined to
the remaining 262,508 development rows.

Final comparisons use three outer development folds. For each outer fold, two
inner fits select the F1 threshold using only that outer fold's training rows.
The frozen threshold is then evaluated once on the outer fold. Preprocessing is
refit inside every training partition.

F1 is the primary metric because an all-negative classifier already obtains
91.17% accuracy while having F1 zero. Precision, recall, average precision,
log loss, accuracy, and confusion counts are retained for every finalist.

## Model path

| Candidate | F1 | Precision | Recall | AP | Log loss | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| All negative | 0.00000 | 0.00000 | 0.00000 | — | — | Sanity reference |
| Ridge, `lambda=0.0001` | 0.41951 | 0.33303 | 0.56665 | 0.39028 | 0.22917 | Rejected |
| Logistic, 4,000 updates | 0.42563 | 0.35728 | 0.52632 | 0.38775 | 0.22198 | Linear reference |
| Histogram boosting, Phase 14 | **0.44216** | **0.36763** | **0.55457** | **0.42861** | **0.21523** | Current reference |

Ridge and logistic numbers use nested threshold selection. The boosting result
uses the same outer folds and threshold isolation. The all-negative row comes
from the initial fixed validation split and is included only as a sanity check.

Logistic regression was retained as the linear baseline after convergence,
regularization, category encoding, and selected interaction screens. Four age
interactions produced a small repeatable improvement, but the best nested F1
remained 0.42563.

The linear baseline preprocessing now fills `binary`, `categorical`, and
`ordinal` feature gaps with the training-fold mode, and continuous/count gaps
with the training-fold median. Fill statistics remain fold-local. Previously
recorded logistic and ridge metrics were produced with median fills for every
numeric-coded column; they are historical results and have not been rerun with
this correction. Recompute the linear comparisons before treating them as
results for the updated preprocessing.

The NumPy histogram booster fits quantile bins once per training partition,
stores binned features as `uint8`, routes missing values explicitly, and fits
shallow Newton trees. Depth 5 consistently improved ranking and log loss over
depth 3. The selected configuration uses 200 trees and 128 candidate features
per node. Moving to 400 trees improved AP and log loss slightly but reduced
nested F1 from 0.44174 to 0.44113 in the Phase 11 grid.

Phase 14 kept the same 295 features and model while correcting documented
response codes. Nested F1 changed from 0.44174 to 0.44216. The three fold
changes were +0.00163, +0.00191, and -0.00210, so this is a semantic cleanup
rather than evidence of a material predictive gain.

## Current configuration

```text
trees                 200
learning rate         0.05
maximum depth         5
minimum leaf rows     200
histogram bins        64
L2 regularization     1.0
candidate features    128 per node
random seed           20260920
submission threshold  0.2108690997
```

The submission threshold is the mean of the three Phase 14 thresholds selected
inside the outer training folds: 0.2030567, 0.2201046, and 0.2094460.

## Evidence files

- `results/eda/phase7_nested_and_ridge_summary.md`: nested logistic and ridge.
- `results/eda/phase12_paired_boosting_logistic_summary.md`: paired errors.
- `results/eda/phase14_codebook_corrections_summary.md`: selected tree variant.
- `results/experiments/`: complete JSON experiment ledger.

Large checkpoints and row-level OOF arrays are reproducible and excluded from
Git.

## Results integrated after Phase 14

The Phase 15–18 configurations and nested results are now present alongside
the local Phase 16 imputation screening. They use the saved development split;
the Phase 16 imputation F1 values below use a different, optimistic threshold
protocol and must not be ranked against the nested values.

| Experiment | F1 | Interpretation |
| --- | ---: | --- |
| Phase 14, with `HAREHAB1` | **0.442155** nested | Current submission reference. |
| Phase 15, ablate only `HAREHAB1` | 0.431039 nested | About 0.01112 lower; the feature's task validity still needs a decision. |
| Phase 16, targeted interactions without `HAREHAB1` | 0.431370 nested | The preprocessing also differs from Phase 15, so this does not isolate the five interactions. |
| Phase 17, learning rate 0.07 on Phase 16 | 0.432225 nested | Better than that branch's 0.431370 control. |
| Phase 17, minimum leaf rows 150 on Phase 16 | **0.432606** nested | Best F1 among the recorded Phase 17 candidates; the commit title emphasizes 0.07, but leaf size 150 scores higher. |
| Phase 18, replace six raw diet frequencies with daily derivatives | 0.441862 nested | Below the matched Phase 14 result by about 0.00029. |
| Local Phase 16, restricted imputation | 0.442674 exploratory | Threshold selected on the same pooled outer OOF labels being scored. |
| Local Phase 16, expanded imputation | 0.441989 exploratory | Same optimistic protocol; weaker than the restricted variant. |

The Phase 17 comparisons belong to the Phase 16 representation without
`HAREHAB1`. Do not transfer their gain to the Phase 14 configuration without a
new paired test. The full record list is in `results/experiments/index.csv`;
the imputation caveats and follow-up are in [IMPUTATION.md](IMPUTATION.md).

## Highest-priority next tests

1. **Resolve the role of `HAREHAB1` →** use the completed Phase 15 ablation to
   decide which feature set defines the intended task; keep results from the
   two feature sets separate.
2. **Isolate remaining representation questions →** ablate only `ALCDAY5`,
   test nominal treatment of `EXRACT11`/`EXRACT21`, and investigate exercise
   frequency units. The six diet frequencies were already tested in Phase 18.
3. **Separate missing-state indicators from imputation →** compare Phase 14,
   matching indicators only, and restricted Phase 16 imputation using nested
   threshold selection.
4. **Test promising tree settings on the chosen feature set →** Phase 17
   suggests leaf size 150 and learning rate 0.07 on its branch. Confirm them
   against the selected baseline before combining changes.
5. **Check train-to-test transfer →** after a final configuration is frozen,
   record local artifacts and the immutable Git commit before submission.

The detailed experiment order and decision criteria are in
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md).
