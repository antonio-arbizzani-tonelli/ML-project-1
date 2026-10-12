# Configuration guide

[final_model.json](final_model.json) defines the selected historical Phase 14 booster
used by `python run.py`: Phase 14 model parameters and preprocessing, data
filenames, and threshold **0.2106169795070252**, selected on pooled three-fold
OOF predictions over all 328,135 training rows. The included historical
source is identified by its SHA-256; the calibration and AIcrowd scores are
in [the final submission record](../results/eda/final_phase14_submission.json).

| File | Contents |
| --- | --- |
| [final_model.json](final_model.json) | Settings for full-data training and prediction |
| [Selected Phase 14 validation](experiments/final_phase14_validation.json) | Explicit historical implementation, original development folds and inner threshold selection |
| [feature_metadata.json](feature_metadata.json) | Feature names, semantic types and codebook response codes |
| [Phase 14 cleaning](experiments/phase14_codebook_corrections_only.json) | The selected 295 source columns, exclusions and code corrections |
| [Phase 14 CV](experiments/phase14_boosting_codebook_corrections_only.json) | Historical suite: three outer folds and two inner folds for threshold selection |
| [Phase 32 MLP](experiments/phase32_mlp.json) | Dedicated neural preprocessing and the two evaluated architectures |
| [Raw tree inputs](experiments/phase33_raw_tree_preprocessing.json) | All 321 predictors, preserving original response codes and NaNs |
| [Raw preprocessing ablation CV](experiments/phase33_raw_boosting_cv.json) | Historical Phase 14 parameters and folds with raw inputs |
| [Corrected missing-only split CV](experiments/phase34_missing_only_split.json) | Same 295-column representation and nested folds, with the corrected split search |
| [Historical positive weight 1.25](experiments/phase35_phase14_positive_weight1p25.json) | Phase 14 settings with positive loss weight 1.25; execute through the Phase 35 driver |
| [Historical positive weight 1.5](experiments/phase35_phase14_positive_weight1p5.json) | Phase 14 settings with positive loss weight 1.5; execute through the Phase 35 driver |
| [Random undersampling](experiments/phase36_undersampling_random.json) | Corrected Phase 34 with three randomly selected negatives per positive; execute through the Phase 36 driver |
| [Hybrid hard-negative undersampling](experiments/phase36_undersampling_hybrid_hard.json) | Same sample budget; half random negatives and half highest-risk negatives from a training-only pilot |
| [Historical random undersampling 5:1](experiments/phase37_phase14_random_ratio5.json) | Phase 14 with all positives and five negatives per positive; execute through the Phase 37 driver |
| [Historical random undersampling 7:1](experiments/phase37_phase14_random_ratio7.json) | Same historical model and shared random negative permutation; the 5:1 sample is contained in this one |

The other configurations in `experiments/` describe experimental pipelines.
The measured results are explained in [EXPERIMENTS.md](../docs/EXPERIMENTS.md).

Configurations specify settings and input representations; the executed source
also determines the result. Recorded Phase 14 and Phase 33 scores used the
implementation before the missing-only split correction. Running those settings
with the default corrected CV code performs a new experiment. Use
`final_phase14_validation.json` for the selected historical pipeline and the
Phase 34 suite for the corrected comparison. The historical preprocessing comparison keeps
both its raw and cleaned controls on the previous implementation.

CV uses numeric caches and the included fixed split. Place the official CSVs
as described in [data/README.md](../data/README.md), then run from the repository
root:

```bash
python -m tools.prepare_data
```

Historical checkpoint replay also requires the saved model and OOF files
identified in each study report.

Phase 35 uses `python -m tools.phase35_positive_weights prepare` followed by
`python -m tools.phase35_positive_weights run` to select the exact archived
Phase 30 implementation. Running these configurations directly with the
current standard runner would test the corrected implementation instead and
would not reproduce the [historical weight study](../results/eda/phase35_positive_weights_summary.md).

Phase 36 uses `python -m tools.phase36_undersampling prepare` followed by
`python -m tools.phase36_undersampling run`. The driver applies sampling
separately inside every outer and inner training fold, refits preprocessing
and histogram bins on retained rows, and preserves the complete validation
partitions. Running its configurations directly with the standard CV runner
does not implement undersampling.

Phase 37 uses `python -m tools.phase37_undersampling_ratios prepare` and
`python -m tools.phase37_undersampling_ratios run`. It executes the verified
historical numerical snapshot from Phase 35 and compares against historical
Phase 14, rather than corrected Phase 34. Sampling is fitted inside every
outer/inner training partition; all validation rows are preserved. Running
these configurations directly with the standard CV runner neither applies
the sampling nor selects the historical implementation.
