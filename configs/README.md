# Configuration guide

[final_model.json](final_model.json) defines the Phase 14 booster used by
`python run.py`: model parameters, data filenames and submission threshold.

| File | Contents |
| --- | --- |
| [final_model.json](final_model.json) | Settings for full-data training and prediction |
| [feature_metadata.json](feature_metadata.json) | Feature names, semantic types and codebook response codes |
| [Phase 14 cleaning](experiments/phase14_codebook_corrections_only.json) | The selected 295 source columns, exclusions and code corrections |
| [Phase 14 CV](experiments/phase14_boosting_codebook_corrections_only.json) | Three outer folds and two inner folds for threshold selection |
| [Phase 32 MLP](experiments/phase32_mlp.json) | Dedicated neural preprocessing and the two evaluated architectures |

The other configurations in `experiments/` describe experimental pipelines.
The measured results are explained in [EXPERIMENTS.md](../docs/EXPERIMENTS.md).

CV uses numeric caches and the included fixed split. Place the official CSVs
as described in [data/README.md](../data/README.md), then run from the repository
root:

```bash
python -m tools.prepare_data
```

Historical checkpoint replay also requires the saved model and OOF files
identified in each study report.
