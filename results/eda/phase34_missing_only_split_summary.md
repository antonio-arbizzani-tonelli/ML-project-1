> Historical study evidence. The selected combined pipeline is described in the [current report](../../docs/FINAL_REPORT.md); original replay prerequisites are in the [archive note](../../docs/ARCHIVE.md).

# Phase 34 — corrected missing-only split candidates

One implementation correction at fixed Phase 14 data, cleaning, parameters and nested threshold protocol. Nine fresh corrected fits; historical control reused without retraining. The benchmark holds model parameters and preprocessing fixed; the historical submission is preserved.

| Pipeline | Nested F1 | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical Phase 14 | 0.442155 | 0.367632 | 0.554573 | 0.428609 | 0.215225 |
| Corrected split search | 0.440803 | 0.362001 | 0.563460 | 0.428781 | 0.215229 |

Corrected minus historical F1: -0.001352. Paired conditional 95% interval: [-0.003005, +0.000319].
The interval conditions on fixed models and thresholds. Both approaches reuse development folds previously consulted for model selection and retain HAREHAB1.

| Fold | Historical F1 | Corrected F1 | Delta | Corrected threshold |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.439027 | 0.436721 | -0.002305 | 0.213344 |
| 2 | 0.444054 | 0.441735 | -0.002318 | 0.203901 |
| 3 | 0.443512 | 0.443733 | +0.000221 | 0.199552 |

Prediction transitions: recovered positives: 499, lost positives: 293, introduced false positives: 1,902, removed false positives: 995.

Selected observed-versus-missing nodes and staged log loss/AP on identical 3,000-row training/validation samples per fold are recorded in the comparison JSON. These samples diagnose fit behaviour; they do not replace the full nested evaluation.

The original commands `python -B -m tools.phase34_missing_only_benchmark prepare`
and `python -B -m tools.phase34_missing_only_benchmark run --workers 2` use a driver
now available in the [development archive](../../docs/ARCHIVE.md). For a fresh
corrected-backend comparison, follow the retained `phase34_missing_only_split.json`
recipe with `--implementation phase34` in the [current catalog](../../configs/README.md).

Candidate record: `results\experiments\phase34_missing_only_split\20261007T092637381255Z_phase34-missing-only-split-depth5-features128-200trees.json`.
Comparison: [JSON](../experiments/phase34_missing_only_split_comparison.json).

## Decision after development evaluation (7 October 2026)

At that stage, the configuration selected the corrected Phase 34 record and the mean inner threshold 0.20559910588816863. The correction restores valid split candidates. Its F1 change is small and uncertain; AP and log loss are nearly unchanged. Historical models already used 830 observed-versus-missing nodes; corrected models use 805, including 270 global endpoint nodes, many equivalent to partitions available through other thresholds.

The September full-data Phase 14 submission predates this correction. Its counts refer to that historical run.

## Final selection (12 October 2026)

Both implementations were refitted and calibrated on identical three-fold
OOF partitions of all training rows. The team retained Phase 14 after
AIcrowd feedback: F1 0.452 and accuracy 0.878. Its final threshold is
0.2106169795070252; its exact historical source is included in the repository.
The [final submission record](final_phase14_submission.json) records this
later decision. The development results above keep their original scope.
