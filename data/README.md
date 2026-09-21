# Data placement

The competition dataset is not redistributed in this repository. Download the
official AIcrowd archive and place these files in `data/raw/dataset/`:

```text
x_train.csv
y_train.csv
x_test.csv
sample_submission.csv
```

`python run.py` validates the CSV schemas and writes reusable arrays to
`data/processed/`. Both directories are ignored by Git.

Verified local dataset sizes:

| File | Rows | Contents |
| --- | ---: | --- |
| `x_train.csv` | 328,135 | `Id` and 321 predictors |
| `y_train.csv` | 328,135 | labels in `{-1, +1}` |
| `x_test.csv` | 109,379 | `Id` and the same 321 predictors |
| `sample_submission.csv` | 109,379 | required test IDs and output schema |

The SHA-256 of the downloaded archive used for the recorded experiments is
`45812b9333df561c72bed861ba5ec382ddc2b68defd64ddda64118074d85a0fd`.
