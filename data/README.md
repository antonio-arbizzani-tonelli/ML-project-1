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
`data/processed/`. `run.py` uses the same loaders for full training.

When changing feature CSV values, use a new `--cache-dir`, including when
IDs and shape stay the same. Raw data and caches are local files ignored by Git.
