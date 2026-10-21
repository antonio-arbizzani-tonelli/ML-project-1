# Retained utilities

The ordinary `python run.py` trains the selected model from official data without
old checkpoints. These utilities support data preparation, integrity checks and
comparisons reported in the project. Every recipe is listed in
[configs/README.md](../configs/README.md).

## Fresh-checkout utilities

```bash
python -m tools.prepare_data
python -m tools.verify_repository
```

The verifier checks links, Python 3.9 syntax, JSON and retained configuration
dependencies, 114 historical ledger hashes,
selected parameters, five numerical sources and the separate pipeline integrity
receipt (including preprocessing, metadata and unchanged report PDF/LaTeX).

For standard historical CV use the explicit backend documented in the catalog:

```bash
python -m src.run_boosting_cv configs/experiments/phase33_raw_boosting_cv.json --implementation phase14 --output-dir results/reproduction/raw
```

`run_boosting_cv`, the linear/ridge/MLP runners and shared modules are retained in
`src/`. `targeted_preprocessing_analysis` and `run_targeted_preprocessing_trials`
also remain because the feature-selection/sampling drivers import their shared
partition, comparison and logging helpers.

## Full-training OOF calibration

All calibration F1 values select and describe the threshold on the same labels.
These are not independent nested test estimates. Drivers do not upload to AIcrowd.
Each uses three full-training OOF folds and a full-data fit, with at most two
single-thread fitting processes and resumable checkpoints.

| Driver | Reported candidates | Prerequisite |
| --- | --- | --- |
| `final_phase14_phase34` | Historical/corrected split-search comparison | Official data and prepared caches |
| `final_promising_candidates` | Individual Phase 18 diet, Phase 24 imputation, Phase 25 leaf L2 | Official data and prepared caches |
| `final_combined_18_24` | Diet + imputation, leaf L2 1/5 | Official data and prepared caches |
| `final_count_candidates` | Functional-limitation counts, diagnosis counts, both | Official data and prepared caches |
| `final_combined_tuning` | 128 bins / 256 candidate features | Run combined base first; source/config hashes are frozen |
| `final_combined_growth` | 300 trees / all 297 candidates | Completed features256 tuning models; explicit `--origin-manifest` for a fresh chain; 300 trees continues copied checkpoints |
| `final_combined_regularization` | Leaf size400 / depth4 | Verified features256 tuning manifest and partition signatures; explicit `--base-manifest` for a fresh chain |
| `final_combined_categorical` | Native activity categories | Same verified features256 base; explicit `--base-manifest` for a fresh chain |
| `final_combined_structural` | Row subsampling / leafwise growth | Same verified features256 base; explicit `--base-manifest` for a fresh chain |

For the selected combined study, a fresh sequence is:

```bash
python -m tools.final_combined_18_24 prepare
python -m tools.final_combined_18_24 run --workers 2
python -m tools.final_combined_tuning prepare
python -m tools.final_combined_tuning run --workers 2
```

After that fresh tuning run completes, pass its manifest explicitly to each
follow-up study. The default dependency checks pin the original manifest whose
SHA256 starts `55dd6a95`; omitting the dependency override requests replay of that
original archive, even if a new manifest occupies the same path. For example:

```bash
python -m tools.final_combined_growth prepare --origin-manifest results/boosting_artifacts/combined_tuning_20261014/manifest.json --output-dir results/boosting_artifacts/reproduction_growth
python -m tools.final_combined_growth run --manifest results/boosting_artifacts/reproduction_growth/manifest.json --workers 2
python -m tools.final_combined_regularization prepare --base-manifest results/boosting_artifacts/combined_tuning_20261014/manifest.json --output-dir results/boosting_artifacts/reproduction_regularization
python -m tools.final_combined_regularization run --manifest results/boosting_artifacts/reproduction_regularization/manifest.json --workers 2
python -m tools.final_combined_categorical prepare --base-manifest results/boosting_artifacts/combined_tuning_20261014/manifest.json --output-dir results/boosting_artifacts/reproduction_categorical
python -m tools.final_combined_categorical run --manifest results/boosting_artifacts/reproduction_categorical/manifest.json --workers 2
python -m tools.final_combined_structural prepare --base-manifest results/boosting_artifacts/combined_tuning_20261014/manifest.json --output-dir results/boosting_artifacts/reproduction_structural
python -m tools.final_combined_structural run --manifest results/boosting_artifacts/reproduction_structural/manifest.json --workers 2
```

If tuning uses a custom `--output-dir`, substitute its generated `manifest.json`
in all four dependency flags. A follow-up's `--output-dir` chooses its own output;
it does not select the prerequisite manifest. Its `run --manifest` must match the
manifest produced by its `prepare` command (or use the same `--output-dir` for both).
Explicit dependency overrides still validate the base configuration, sources,
partitions and checkpoint provenance.

These commands perform substantial training. To reproduce only the selected CSV
with its fixed threshold, use `python run.py`. New runs should use new output
locations; frozen sources, recipes and old manifests must not be reformatted.

## Studies requiring the original local archive

| Driver | Material required beyond the checkout |
| --- | --- |
| `phase29_feature_selection` | Phase 14 models, OOF arrays, control manifests and matching original sources |
| `phase35_positive_weights` | Archived Phase 30 numerical source and completed Phase 14 fits |
| `phase36_undersampling` | Corrected Phase 34 control checkpoints/OOF, then training-only sampling |
| `phase37_undersampling_ratios` | Phase 35 historical source snapshot and Phase 14 controls |

These drivers retain the algorithms behind the reported studies; their control
replay is not guaranteed from a source-only checkout. Use the
[archived complete tree](../docs/ARCHIVE.md) with the original local artifacts for
byte-exact historical replay. Their small comparison records remain in Git.
Superseded audit, preflight and resume-only scripts were archived.
