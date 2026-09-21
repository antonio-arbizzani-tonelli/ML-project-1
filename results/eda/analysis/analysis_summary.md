# Dataset statistical analysis

This report extends the structural audit and feature registry. All supplied predictor columns remain available for competition modelling; flags below describe data behaviour rather than automatic exclusions.

## Target and split

- Rows: 328,135.
- Positive target: 28,975 (8.8302%).
- Development rows: 262,508; validation rows: 65,627.
- Split seed: `20260918`; validation fraction: 20.0%.
- Target-aware statistics use development rows only. Validation labels were not used in feature analysis.

## Structural results

- Constant columns including missing as a state: 0.
- Exact duplicate-column annotations already in the registry: 3.
- Duplicate train feature-row groups: 0; conflicting-label groups: 0.
- Unique feature rows shared by train and test: 0.

## Largest train-test distribution differences

| Feature | Value | Type |
| --- | ---: | --- |
| `MAXVO2_` | 0.0183 | `continuous` |
| `FC60_` | 0.0175 | `continuous` |
| `_WT2RAKE` | 0.0150 | `survey_design` |
| `_LLCPWT` | 0.0141 | `survey_design` |
| `_AGE80` | 0.0122 | `continuous` |
| `_VEGESUM` | 0.0122 | `continuous` |
| `PA1MIN_` | 0.0121 | `count` |
| `_FRUTSUM` | 0.0120 | `continuous` |
| `WTKG3` | 0.0120 | `continuous` |
| `IDATE` | 0.0117 | `date` |

TVD is total variation distance. It is exact for features with at most 256 observed values and estimated from a fixed sample with train-defined quantile bins for higher-cardinality features.

## Strongest univariate target associations

| Feature | Value | Type |
| --- | ---: | --- |
| `BPMEDS` | 0.3840 | `binary` |
| `EMPLOY1` | 0.3795 | `categorical` |
| `MAXVO2_` | 0.3787 | `continuous` |
| `FC60_` | 0.3787 | `continuous` |
| `BPHIGH4` | 0.3736 | `categorical` |
| `_RFHYPE5` | 0.3730 | `binary` |
| `_AGEG5YR` | 0.3581 | `ordinal` |
| `_AGE80` | 0.3565 | `continuous` |
| `_AGE65YR` | 0.3429 | `binary` |
| `_AGE_G` | 0.3429 | `ordinal` |

Target TVD compares the feature distribution between positive and negative development respondents. A high value indicates association, not causality or independent predictive value.

## Codebook validation

- Features with at least one nonblank value outside conservatively parsed codebook intervals: 2.
- `WTKG3`: 6 rows; examples [["22.68", 6]].
- `WEIGHT2`: 1 rows; examples [["2099", 1]].
- Machine validation was withheld for 1 feature(s) because the parsed documented domain covered less than 95% of observed nonblank values.
- `MAXVO2_`: parsed-domain mismatch in 324,192 rows; this is a codebook parsing/documentation gap, not an automatic data error.

## Figures

- `figures/01_target_distribution.png`
- `figures/02_missingness_overview.png`
- `figures/03_train_test_shift.png`
- `figures/04_target_association.png`
- `figures/05_cardinality_distribution.png`
- `figures/06_missingness_heatmap.png`
- `figures/07_top_shift_distributions.png`
- `figures/08_top_target_profiles.png`

## Interpretation limits

- The target records whether MI or CHD was ever reported; it is not a future-event or recurrent-infarction label.
- Test-set comparisons use predictors only. No test labels are available or inferred.
- High-cardinality TVD and its target profile are sample estimates; exact means, standard deviations, missingness, and low-cardinality frequencies use all applicable rows.
- Codebook validation is deliberately conservative. Features without machine-readable numeric codes are marked as not checkable rather than declared valid.
