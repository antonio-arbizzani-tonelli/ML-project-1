# Dataset and final representation

The 2015 BRFSS target `_MICHD` records reported myocardial infarction or coronary
heart disease. There are 328,135 labelled respondents, 321 predictors and
109,379 test respondents. Positive prevalence is 8.83%; 44.79% of training
predictor cells are blank. [Compact dataset evidence](../results/eda/dataset_summary.json)
retains the measured counts and original audit fingerprint.

Official CSVs and caches stay outside Git; see [data/README.md](../data/README.md).
The fixed seed-20260918 split contains 262,508 development and 65,627 initial
validation rows. Historical nested comparisons use development rows only;
full-training OOF calibration later uses every labelled row.

## Final input construction

The canonical [metadata](../configs/feature_metadata.json) records semantic types,
response codes and codebook sources. The
[combined plan](../configs/experiments/combined18_24_preprocessing.json) determines
exact exclusions and imputation, with code unchanged from the selected release.

- Phase 14 excludes identifiers, full interview date, survey-design fields and
  redundant measurements, leaving 295 source columns.
- Nonresponse codes become `NaN` per variable; documented zero quantities become
  zero. The [cleaning count audit](../results/eda/phase33_preprocessing_counts.json)
  retains the measured recodings.
- Six mixed-unit raw diet columns are excluded while their existing CDC per-day
  equivalents remain, leaving 289 source columns.
- Feature-only Naive Bayes imputes documented unknown/refused responses in
  `INCOME2`, `PNEUVAC3`, `EMPLOY1`. Ridge predicts blank height/weight; missing BMI
  is reconstructed when possible. Eight indicators preserve original missing
  reasons. All statistics and imputers fit training rows only.
- The resulting representation has 297 columns. Other blanks remain missing;
  each histogram split learns its missing-value direction from training rows.

`HAREHAB1` concerns rehabilitation after a heart attack. All 620 recorded raw
responses in development belong to positives, including 13 unknown/refused codes;
607 are valid yes/no responses. The [compact response counts](../results/eda/dataset_summary.json)
retain this distinction and the archived source fingerprint. Its historical ablation lowers
nested F1 from 0.442155 to 0.431039. It is retained to classify reported disease,
with no claim of prospective clinical performance. Details and evaluation limits
are in the [supporting report](FINAL_REPORT.md).

[MLP.md](MLP.md) describes the separate median/mode, scaling and one-hot neural
representation. The [archive](ARCHIVE.md) preserves detailed per-column audits
and rejected intermediate representations.

For a fresh statistical analysis with official data and the canonical metadata:

```bash
python -m src.analyze_dataset data/raw/dataset configs/feature_metadata.json results/reproduction/dataset_analysis --cache-dir data/processed
```

This regenerates detailed outputs locally; the compact published summary remains
a copy of the original measured aggregate fields.
