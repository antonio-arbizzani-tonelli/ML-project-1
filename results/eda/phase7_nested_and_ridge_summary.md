# Phase 7 — nested threshold control and ridge comparison

Date: 2026-09-20.

## Protocol

All results use only the saved development partition, its fixed three outer
folds, corrected `BPHIGH4=7/9` handling, compact P2 preprocessing, and the
four established age interactions.  The 20% local validation partition is not
loaded.

For every outer fold, the two inner folds fit preprocessing and the model from
scratch.  They produce inner out-of-fold scores, choose the F1-maximizing
threshold (accuracy resolves exact F1 ties), and apply that frozen threshold
once to the outer-fold scores.  This is the reported F1 and accuracy.

Ridge produces signed linear scores, not probabilities.  In each outer fold a
two-parameter logistic Platt calibrator is fitted only to that fold's inner
OOF ridge scores.  It makes log loss meaningful and supplies the scores for
inner threshold selection.  The fitted calibrator is then frozen for the
outer fold.

## Results

| Candidate | Nested F1 | Accuracy | Precision | Recall | Average precision | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic, 4,000 updates, gamma 0.2 | **0.42563** | **0.87457** | **0.35728** | 0.52632 | 0.38775 | **0.22198** |
| Ridge, lambda 0.0001 | 0.41951 | 0.86152 | 0.33303 | **0.56665** | **0.39028** | 0.22917 |
| Ridge, lambda 0.001 | 0.41785 | 0.86634 | 0.33949 | 0.54323 | 0.38351 | 0.22956 |
| Ridge, lambda 0.01 | 0.41775 | 0.86251 | 0.33363 | 0.55858 | 0.37647 | 0.22905 |

The nested logistic thresholds are 0.20880, 0.20188, and 0.20583 (mean
0.20550; standard deviation 0.00284).  Its per-fold F1 values are 0.42566,
0.42503, and 0.42620.  This is stable enough to keep the logistic candidate
as the development reference.

The best ridge setting is lambda 0.0001.  Its thresholds are tightly grouped
around 0.151 (standard deviation 0.00067), but it gives up 0.00612 F1 and
0.01304 accuracy versus logistic.  Its higher recall and average precision
show that it ranks some positives well, but its calibrated operating point
creates more false positives and its log loss is worse by 0.00720.

## Decision

Keep corrected P2 plus the four interactions and unregularized logistic
regression at gamma 0.2 / 4,000 updates as the reference.  Ridge is a useful
implemented benchmark but is not a finalist on this shared, nested protocol.
The next model improvement should test a genuinely nonlinear learner only if
the challenge library policy permits it, or perform narrow, isolated P2
representation ablations at the fixed 4,000-update budget.

## Reproducibility

- Logistic record: `20260920T113608936798Z_phase7-p2-codebook-clean-bphigh4-corrected-interactions-gamma-0p2-nested-4000.json`.
- Ridge records: `20260920T111540545185Z_phase7-p2-codebook-clean-ridge-lambda-0p0001.json`, `20260920T111612333412Z_phase7-p2-codebook-clean-ridge-lambda-0p001.json`, and `20260920T111644061550Z_phase7-p2-codebook-clean-ridge-lambda-0p01.json`.
- Each record links its compressed OOF artifact with row indices, fold mapping,
  labels, scores, probabilities, thresholds, weights, feature names, and
  SHA-256 checksum.
