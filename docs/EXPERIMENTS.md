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

## Highest-priority next tests

1. **Ablate `HAREHAB1` →** quantify suspected target-conditioned leakage →
   exclude it for scientific interpretation unless its availability at test
   time and meaning are confirmed.
2. **Correct frequency families →** harmonize food and exercise unit codes in
   isolated groups → keep a group only if nested F1 improves or remains tied
   without worsening AP/log loss across folds.
3. **Tune leaf size and learning rate →** test a small grid around 200 leaves
   and 0.05 → keep settings only for a repeatable nested F1 gain; do not reopen
   the already screened tree-count grid.
4. **Inspect confident shared errors →** target feature work at cases missed by
   both boosting and logistic → retain a change only when it reduces false
   negatives without an excessive false-positive increase.
5. **Run the frozen pipeline once on AIcrowd →** check train-to-test transfer →
   use the submission only after local artifacts and the immutable Git commit
   are recorded.
