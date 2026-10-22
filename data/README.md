# Dataset files

Download the official archive from the
[EPFL Project 1 AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and place the four CSV files under `data/raw/dataset/`:

| File | Rows | Contents |
| --- | ---: | --- |
| `x_train.csv` | 328,135 | `Id` and 321 predictors |
| `y_train.csv` | 328,135 | `Id` and labels `-1/+1` |
| `x_test.csv` | 109,379 | `Id` and the same 321 predictors |
| `sample_submission.csv` | 109,379 | Test IDs and output schema |

From the repository root, prepare the caches for cross-validation:

```bash
python -m tools.prepare_data
```

The utility validates schemas and ID alignment, then stores arrays under
`data/processed/`. `run.py` uses the same loaders for full training. Each feature
split has an `x_train_cache.json` or `x_test_cache.json` receipt recording the
source CSV SHA-256, feature order, dimensions and hashes of the feature/ID arrays.

Reusing a cache checks those hashes, including when IDs and shape stay the same.
Changed CSVs, modified arrays, partial caches and older caches without receipts
are rejected with instructions to choose a new `--cache-dir`; no existing data
is deleted or silently replaced. New cache directories must contain neither
feature arrays nor receipts from an earlier run. Raw data, caches and receipts
are local files ignored by Git at the default locations.
