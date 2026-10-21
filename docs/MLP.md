# Neural baseline

The MLP is a separate NumPy comparison; the final submission uses the booster.
[phase32_mlp.json](../configs/experiments/phase32_mlp.json) defines both architectures.
Their matched historical nested F1 is 0.424451 for 64/32 and 0.419268 for 128/64,
versus Phase 14's 0.442155. Both are lower in every fold. The
[comparison record](../results/experiments/mlp/phase32_comparison.json) and
[study summary](../results/eda/phase32_mlp_summary.md) retain the evidence.

## Method

The neural representation shares Phase 14 cleaning and codebook overrides.
Continuous/count features use training medians; categorical, ordinal and binary
features use training modes. Numerical/ordinal channels are standardized after
imputation. Nominal variables use full one-hot, missing and unseen-category
channels. Numerical/binary inputs have missing flags. All-missing columns use
zero plus a flag. The network has no extra intercept input and no tree binning.

Nine coded frequencies are converted to per-day values: alcohol, two exercise
frequencies and six diet variables. Weekly and monthly units divide by 7 and 30.
Unit flags preserve the original coding. Food code 300 uses numeric zero plus a
distinct less-than-monthly flag. All preprocessing is fitted per training partition.
The width varies with observed categories (1,178 inputs in the first outer fit).
Channels always zero during training start with zero weights.

Both networks use ReLU hidden layers, sigmoid output, stable binary
cross-entropy, Adam (learning rate 0.001, batch 512), seed 20261001 and L2 0.0001
on weights only. Each fit selects its epoch count on a stratified inner 90/10
split (maximum 80 epochs, patience 8, minimum improvement 0.0001), then refits
preprocessing and the network on that complete training partition.

Three outer folds use the fixed development rows and seed 20260919. Two inner
folds select F1 thresholds; their seed is 20260922 plus outer fold number.
Each architecture has nine epoch selections and nine refits. Validation rows do
not fit preprocessing, select epochs or set their own decision threshold.

## Reproduce

Supply the official data and development dependencies, then run:

```bash
python -m tools.prepare_data
python -m src.run_mlp_cv configs/experiments/phase32_mlp.json
```

The runner writes checkpoints, loss histories and OOF arrays under the configured
local `results/mlp_artifacts/` directory. Fresh CV needs official data and tracked
files; exact replay of the original checkpoint bytes needs the saved local
artifacts and original numerical environment. The historical fits used Python
3.14.3 / NumPy 2.4.2 and one OpenBLAS thread; synthetic CI tests are a separate check.

`run_mlp.py --help` describes optional full-data training. It requires an explicitly
chosen threshold; no MLP submission threshold was adopted. The code is in
`src/numpy_mlp.py`, `src/mlp_preprocessing.py` and `src/run_mlp_cv.py`.
The comparison concerns complete pipelines, not the isolated effect of network
architecture, and repeated development choices limit its independence.
