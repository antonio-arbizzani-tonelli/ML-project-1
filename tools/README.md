# Utilities

Run from the repository root:

```bash
python -m tools.prepare_data
python -m tools.verify_repository
```

`prepare_data` checks the four official CSV files and creates caches under
`data/processed/`. Place the CSV files as described in [data/README.md](../data/README.md).
`verify_repository` checks Git contents, documentation links, configurations
and experiment hashes. It requires a Git checkout.

For model training and CV, follow the [root README](../README.md#reproduce-validation)
and the [MLP guide](../docs/MLP.md).

## Recorded studies

Study utilities require local checkpoints, OOF arrays, source snapshots or
preflight files. Consult the [experiment history](../docs/EXPERIMENTS.md)
for the corresponding study. Some commands write comparisons and summaries
to the original study directories.

Specific prerequisites:

- `verify_phase32_mlp`: the 18 saved refits referenced by the local records;
  see [MLP.md](../docs/MLP.md).
- `phase29_feature_selection`: baseline artifacts and matching source hashes;
  see the [study README](../results/eda/phase29_feature_selection/README.md).
- `phase30_31_trials`: the archived booster source at
  `.local/phase30_31_prechange/numpy_boosting_before_weighting.py`.
- `audit_phase14_conformity`: commit `4970aa6` and the source state from
  [the original check](../results/eda/phase19_phase14_conformity.md).
- `phase33_raw_preprocessing_benchmark`: Phase 14 OOF arrays and matching
  historical source, data caches, metadata and fixed split. Its recorded raw
  and cleaned results both precede the missing-only split correction.
  `preflight` checks the shared data,
  folds and model parameters; `run` trains the raw candidate and writes the
  paired comparison; `compare` rebuilds the summary from completed artifacts.
  `python -m tools.phase33_parallel_resume run --workers 2` resumes verified
  completed outer checkpoints and trains remaining fits in separate processes.
  Stop the original sequential benchmark before using the resume command.
- `phase34_missing_only_benchmark`: a matched comparison of the corrected
  missing-only split search against historical Phase 14. `prepare` freezes
  exact rows, matrices, source hashes and environment without training;
  `run --workers 1` trains nine fits and resumes validated checkpoints;
  `compare` regenerates the analysis. It requires the historical OOF arrays,
  checkpoints and before-fix source snapshot. The original controls are
  preserved. Use the standard Phase 34 CV configuration for a fresh run
  without those local controls.

Phase 34 is complete and supplies the corrected comparison: nested F1
**0.440803240**, with mean inner-selected threshold **0.20559910588816863**.
Its [record](../results/experiments/phase34_missing_only_split/20261007T092637381255Z_phase34-missing-only-split-depth5-features128-200trees.json)
preserves source and environment fingerprints for all nine fits. The historical
Phase 14 F1 is 0.442155227. A raw/clean preprocessing comparison under the
corrected implementation would require matched new fits for both inputs.
The [comparison](../results/experiments/phase34_missing_only_split_comparison.json)
and [summary](../results/eda/phase34_missing_only_split_summary.md) document
the conditional paired interval and diagnostics from completed checkpoints.

`phase35_positive_weights` evaluates weights 1.25 and 1.5 using the archived
Phase 30 numerical implementation, so the historical weight-1/weight-2
controls remain matched. `prepare` verifies the nine partitions, binning and
all historical outer predictions; `run` uses two isolated processes and
resumable 25-tree checkpoints. It requires local data and the archived source
snapshot. Both candidates completed and their checkpoints, OOF scores and
inner-selected thresholds were verified. Neither improves historical F1;
the [study](../results/eda/phase35_positive_weights_summary.md) explains the
separate corrected Phase 34 scope and the runtime after checkpoint reuse.

`phase36_undersampling` compares random and hybrid hard-negative sampling
against corrected Phase 34. Both retain all positives and three negatives per
positive. The hybrid draws half its negative budget uniformly and fills the
rest with the highest-risk remaining negatives from a 24-tree, depth-3 pilot
trained only within that training fold. Final preprocessing and bins use
retained rows; all validation rows remain available for nested threshold
selection and evaluation. `prepare` verifies the corrected control and freezes
nine eligible partitions; `run` trains both candidates in separate single-thread
processes with 25-tree checkpoints. `report --publish` copies verified records
into the main local ledger. Requires the saved Phase 34 models/OOF arrays and
matching local input caches. Use this driver to execute the sampling configs.

`phase37_undersampling_ratios` compares historical Phase 14 with random
undersampling at 5:1 and 7:1. Each eligible training partition uses a shared
random negative permutation, so the smaller sample is contained in the larger;
all positives and all validation rows are preserved. Final preprocessing and
bins learn from retained rows. The driver imports an isolated copy of the
verified Phase 35 numerical snapshot to preserve the historical split search.
`prepare` replays the historical control and freezes nine partitions; `run`
uses two single-thread workers and resumable checkpoints every 25 trees.
`compare` verifies all nine models, retained-row bin edges, inner OOF scores
and nested thresholds per candidate. `report --publish` serially copies
verified records to the main local ledger. Requires local data caches and
the archived numerical source. The selected Phase 14 retains all training rows.

## Final full-train calibration and submissions

`final_phase14_phase34` calibrates both versions on identical stratified
three-fold OOF partitions of all training rows, then fits each full-data
model and writes signed test CSVs. It uses the included byte-exact
`src/numpy_boosting_phase14.py` and the corrected `src/numpy_boosting.py`;
a fresh run needs the official data and prepared caches, without old model
artifacts:

```bash
python -m tools.prepare_data
python -m tools.final_phase14_phase34 prepare
python -m tools.final_phase14_phase34 run --workers 2
```

Run `python -m tools.final_phase14_phase34 --help` for output and resume
options. The default artifact directory is `results/boosting_artifacts/final_phase14_phase34`.
The original dated run remains an archive with its original source
fingerprints; use a new directory for a fresh run after code changes.
For the selected model with its already calibrated threshold, use the
ordinary `python run.py` command. The [submission record](../results/eda/final_phase14_submission.json)
distinguishes calibration F1 from nested development and AIcrowd scores.
