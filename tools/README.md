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
