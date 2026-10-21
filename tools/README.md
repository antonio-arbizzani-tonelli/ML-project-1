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
the archived numerical source. Both historical Phase 14 and the current
combined submission retain all training rows.

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
For the historical Phase 14 model with its already calibrated threshold, use
`python run.py --config configs/final_phase14_model.json`.
The [historical submission record](../results/eda/final_phase14_submission.json)
distinguishes calibration F1 from nested development and AIcrowd scores.

## Additional submission candidates

The current default `python run.py` selects **combined18_24_features256**:
Phase 18 diet exclusions plus restricted Phase 24 imputation, 297 output
columns and 256 candidate features per node. Its threshold is
0.22007772525099117; calibration F1 0.444071 and reported AIcrowd
F1 0.455 / accuracy 0.883 describe different evaluation scopes.
The [current submission record](../results/eda/final_combined18_24_submission.json)
and [aggregate comparison](../results/eda/final_combined_experiments_summary.json)
retain the selected pipeline and all 17 later candidate outcomes.
No nested result has been computed for this exact combined configuration.
The drivers below describe the original sequence of studies; follow-up
drivers can require frozen artifacts from their preceding local runs.

`final_promising_candidates` prepares Phase 18 (daily diet frequencies),
Phase 24 (restricted imputation), and Phase 25 (leaf L2=5) using the verified
historical booster. Each candidate has three full-train OOF fits for threshold
calibration and one fit on all labeled rows. Preprocessing and imputers learn
only from the training rows of each partition.

```bash
python -m tools.final_promising_candidates prepare
python -m tools.final_promising_candidates run --workers 2
python -m tools.final_promising_candidates status
```

The run freezes its sources, inputs and configurations under
`results/boosting_artifacts/final_candidates_20261013`. Repeating `run` resumes
matching checkpoints or reuses completed fits. The three verified CSVs and
their provenance manifest are saved in `results/aicrowd_candidates_20261013`.
OOF F1 describes threshold calibration, not an independent test score.
The command does not upload to AIcrowd.

`final_combined_18_24` combines the same six Phase 18 diet exclusions with
restricted Phase 24 imputation, producing 297 features including eight missing
reason indicators. Its two configurations differ only in leaf L2 (1 or 5).
Each pipeline runs three full-training OOF fits, calibrates its own threshold,
fits all labeled rows, and delivers its CSV immediately. At most two fitting
subprocesses run concurrently.

```bash
python -B -m tools.final_combined_18_24 prepare
python -B -m tools.final_combined_18_24 run --workers 2
python -B -m tools.final_combined_18_24 status
```

Sources, configurations, inputs and folds are frozen under
`results/boosting_artifacts/combined18_24_20261013`; resumable checkpoints are
saved every 25 trees. Delivery CSVs and their manifest are saved in
`results/aicrowd_combined18_24_20261013`. OOF F1 measures threshold calibration;
official AIcrowd scores are recorded separately after submission. The original
three candidates and the selected production configuration remain available.

`final_count_candidates` adds fixed raw-response counts to the 297-feature
Phase 18+24, L2=1 base. A counts `BLIND`, `DECIDE`, `USEEQUIP`; B counts
`ADDEPEV2`, `CHCCOPD1`, `CHCKIDNY`; A+B includes both. Each group contributes
a yes count and a valid-answer count. Raw codes 1 and 2 are valid; unanswered
groups have a NaN yes count and zero answered count. Individual source columns
remain present. A/B have 299 features; A+B has 301. Historical binning and
booster parameters are preserved.

```bash
python -B -m tools.final_count_candidates prepare
python -B -m tools.final_count_candidates run --workers 2
python -B -m tools.final_count_candidates status
```

The run freezes the helper, runner, scheduler, historical numerical sources,
configurations, official inputs and common full-training OOF folds under
`results/boosting_artifacts/count_candidates_20261013`. A and B start in
parallel; A+B starts when a pipeline frees a slot. Each of the three pipelines
calibrates its own threshold on three OOF fits, trains on all labeled rows,
then immediately delivers its signed CSV to
`results/aicrowd_count_candidates_20261013`. Matching checkpoints resume every
25 trees. Calibration F1 and AIcrowd feedback are recorded separately.

`final_combined_tuning` produces two separate candidates on the unchanged
297-feature Phase 18+24, L2=1 base. `combined18_24_bins128` changes only
`n_bins` from 64 to 128, retaining 128 candidate features per node.
`combined18_24_features256` changes only `max_features` from 128 to 256,
retaining 64 bins. Both use the historical booster with 200 trees, depth 5,
learning rate 0.05, minimum leaf size 200 and model seed 20260920.

```bash
python -B -m tools.final_combined_tuning prepare
python -B -m tools.final_combined_tuning run --workers 2
python -B -m tools.final_combined_tuning status
```

The runner rejects additional configuration changes and freezes the unchanged
base suite and shared preprocessing alongside the historical numerical sources,
scheduler, runner, official inputs and the common seed-20260919 full-training
OOF folds under `results/boosting_artifacts/combined_tuning_20261014`.
Each of the two parallel pipelines runs three OOF fits, selects its own F1
threshold, fits all labeled rows and immediately delivers its signed CSV to
`results/aicrowd_combined_tuning_20261014`. Matching checkpoints resume every
25 trees; no more than two single-thread fitting subprocesses run concurrently.
OOF F1 describes threshold calibration, not an independent test score. AIcrowd
feedback is recorded separately after the user uploads the files.

`final_combined_growth` starts from the verified Phase 18+24 variant with
256 candidate features per node, 64 bins and leaf L2=1. Its separate
`combined18_24_trees300` trial extends unchanged copies of each of the four
trusted 200-tree checkpoints by 100 trees. The original fitted binner, random
generator state, training scores and tree prefix are preserved. Original
checkpoint bytes, metadata, ordered matrix signatures, sources, inputs,
environment and folds are checked before native `continue_fit` is called.
New checkpoints identify the new run and target, with explicit origin hashes;
matching partial extensions can resume without changing the original models.

The separate `combined18_24_features297` trial fits 200 trees from scratch,
changing only candidate features per node from 256 to all 297 retained
features. Both trials keep the exact preprocessing, depth 5, learning rate
0.05, minimum leaf size 200, L2=1 and model seed 20260920 of the recorded base.

```bash
python -B -m tools.final_combined_growth prepare
python -B -m tools.final_combined_growth run --workers 2
python -B -m tools.final_combined_growth status
```

The run freezes its own runner, configurations, original checkpoint copies
and origin provenance under `results/boosting_artifacts/combined_growth_20261014`.
Execution reuses the original verified numerical source package to preserve
checkpoint class identity, with additional byte-identical source snapshots.
A global pool fills at most two single-thread fitting slots across OOF and
full-data fits; fitting on all labeled rows does not depend on the threshold.
Each candidate calibrates its own threshold only after all three full-training
OOF folds are complete, then immediately delivers its CSV once its full fit is
also complete to `results/aicrowd_combined_growth_20261014`. OOF calibration
uses fold seed 20260919 and is not an independent or nested test estimate.
AIcrowd feedback is recorded separately after submission by the user.

`final_combined_regularization` fits two separate fresh trials on the verified
Phase 18+24 features256 base: `combined18_24_leaf400` changes only minimum
leaf size to 400; `combined18_24_depth4` changes only maximum depth to 4.
Both retain 200 trees, 64 bins, 256 candidate features per node, learning rate
0.05, L2=1, model seed 20260920 and the exact 297-feature preprocessing.

```bash
python -B -m tools.final_combined_regularization prepare
python -B -m tools.final_combined_regularization run --workers 2
python -B -m tools.final_combined_regularization status
```

Sources, configs, inputs, common OOF folds and the verified base reference are
frozen under `results/boosting_artifacts/combined_regularization_20261015`.
Each real partition matrix must match its recorded base signature before fit.
A global pool fills at most two single-thread fitting slots, auditing parent
and orphan-worker locks. Full-data fits may precede their last OOF fit;
thresholds are selected only after three complete OOF folds, and each CSV is
delivered only after its calibration and full fit are ready. Delivery files
are saved in `results/aicrowd_combined_regularization_20261015`. OOF F1 is a
threshold-calibration statistic, not an independent or nested test estimate.

`final_combined_categorical` fits one fresh 200-tree candidate on the same
Phase 18+24 features256 base. Only `EXRACT11` and `EXRACT21` use native
categorical split groups; the other columns retain historical numeric
binning. The recursive categorical backend is frozen separately, while
preprocessing, inputs, folds and the four other numerical source modules
must match the verified base. Categorical indices are resolved from the
297-column preprocessing output and checked against real matrix signatures.

```bash
python -B -m tools.final_combined_categorical prepare
python -B -m tools.final_combined_categorical run --workers 2
python -B -m tools.final_combined_categorical status
```

Frozen artifacts are saved under
`results/boosting_artifacts/combined_categorical_20261015`. At most two
single-thread fitting processes run globally, starting the first OOF fold
and full-data fit together. Three full-training OOF folds select the
candidate's own threshold; its signed CSV is delivered to
`results/aicrowd_combined_categorical_20261015` only after all four fits
complete. Delivery metadata distinguishes the new booster hash from the
historical baseline hash. OOF F1 describes calibration, not an independent
test score, and the command does not upload to AIcrowd.

`final_combined_structural` fits two separate fresh 200-tree trials on the
verified Phase 18+24 features256 base. `combined18_24_subsample08` changes
only per-tree training-row subsampling to 0.8, preserving class proportions.
`combined18_24_leafwise32_depth8` uses best-first leafwise growth with at most
32 leaves and a maximum depth of 8. It keeps all training rows. Both retain
64 bins, 256 candidate features per node, learning rate 0.05, minimum leaf
size 200, L2=1, model seed 20260920 and the exact 297-feature preprocessing.
The numerical binner is fitted on every training row of each partition;
subsampling affects only tree fitting and uses a separate checkpointed RNG.

```bash
python -B -m tools.final_combined_structural prepare
python -B -m tools.final_combined_structural run --workers 2
python -B -m tools.final_combined_structural status
```

The new structural booster is frozen under
`results/boosting_artifacts/combined_structural_20261015`, alongside unchanged
historical engine, scheduler, other four numerical modules, configurations,
inputs and common three-fold seed-20260919 OOF assignments. Runtime partition
matrices must match the verified base before fitting. A global pool starts
the first OOF fold of each candidate together and runs at most two isolated
single-thread fitting processes, with parent and orphan-worker lock checks.
Matching checkpoints resume every 25 trees. Each candidate chooses its own
OOF F1 threshold and delivers its signed CSV independently once all three
OOF fits and its full-data fit are complete, under
`results/aicrowd_combined_structural_20261015`. Delivery metadata distinguishes
the structural booster hash from the historical baseline. OOF F1 describes
threshold calibration, not independent or nested validation.
