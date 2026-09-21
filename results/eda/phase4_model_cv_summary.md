# Phase 4 logistic model experiments

Run date: 2026-09-19.

## Protocol and score interpretation

The experiments use the same three stratified folds (seed `20260919`) inside
the saved 80% development partition (262,508 rows). The saved outer 20%
validation partition was not used for fitting, scoring, or selection. P1 is the compact raw-code representation;
P2 is the compact codebook-clean representation. Every preprocessing statistic
and interaction scale is fitted on the corresponding training fold only.

All runs use full-batch NumPy logistic regression with `gamma=0.1` or `0.2`. L2 uses the
course implementation's objective, mean logistic loss plus
`lambda * ||weights||²`; this implementation also penalizes the intercept.
The initial weights are zero. Warm-started checkpoints within a fold are
mathematically the same sequence of updates as one continuous fit.

The reported F1 and accuracy use the threshold that maximizes F1 on pooled
development out-of-fold (OOF) predictions. Accuracy breaks an *exact* F1 tie
between thresholds. The threshold is selected and described on these same OOF
predictions, so its reported F1 is exploratory and can be optimistic. The
outer validation partition is reserved for a later evaluation after the
pipeline and threshold are frozen. It was already scored by the initial
all-negative, ridge, and logistic baselines, but not reused in Phases 3–4;
calling it completely untouched would be inaccurate. AP (average precision) and log loss do not
depend on the threshold. The final two columns show the original fixed 0.5
threshold for reference.

The public AIcrowd [submissions page](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/submissions?baselines=false)
displays F1 and accuracy, but an official accuracy tie-break rule has not been
verified. Local comparisons use F1 first and report accuracy explicitly.

## Results

| Representation | Gamma | L2 lambda | Updates | Selected threshold | F1 | Accuracy | AP | Log loss | F1 at 0.5 | Accuracy at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 0.1 | 0 | 50 | 0.3399 | 0.40019 | 0.85867 | 0.34358 | 0.33207 | 0.15686 | 0.91281 |
| P1 | 0.1 | 0 | 150 | 0.2544 | 0.40561 | 0.86996 | 0.35830 | 0.25214 | 0.11655 | 0.91360 |
| P1 | 0.1 | 0 | 300 | 0.2181 | 0.40918 | 0.86513 | 0.36528 | 0.23495 | 0.14426 | 0.91417 |
| P1 | 0.1 | 0 | 600 | 0.2010 | 0.41163 | 0.86345 | 0.36963 | 0.22898 | 0.16792 | 0.91419 |
| P1 | 0.1 | 0 | 1000 | 0.1976 | 0.41375 | 0.86450 | 0.37141 | 0.22748 | 0.17999 | 0.91423 |
| P1 | 0.1 | 0.001 | 300 | 0.2218 | 0.40922 | 0.86536 | 0.36514 | 0.23631 | 0.14116 | 0.91415 |
| P1 | 0.1 | 0.01 | 300 | 0.2420 | 0.40767 | 0.85970 | 0.36314 | 0.25131 | 0.12170 | 0.91389 |
| P2 | 0.1 | 0 | 50 | 0.1527 | 0.37395 | 0.83199 | 0.30471 | 0.24152 | 0.03829 | 0.91196 |
| P2 | 0.1 | 0 | 150 | 0.1760 | 0.39688 | 0.85062 | 0.33802 | 0.23215 | 0.10080 | 0.91280 |
| P2 | 0.1 | 0 | 300 | 0.1956 | 0.40407 | 0.86352 | 0.35214 | 0.22877 | 0.13843 | 0.91318 |
| P2 | 0.1 | 0 | 600 | 0.1822 | 0.41166 | 0.85650 | 0.36396 | 0.22632 | 0.16977 | 0.91367 |
| P2 | 0.1 | 0 | 1000 | 0.1956 | 0.41560 | 0.86569 | 0.37157 | 0.22489 | 0.18645 | 0.91380 |
| P2 | 0.1 | 0.001 | 300 | 0.1921 | 0.40393 | 0.86153 | 0.35163 | 0.22896 | 0.13281 | 0.91309 |
| P2 | 0.1 | 0.001 | 1000 | 0.1896 | 0.41460 | 0.86219 | 0.37014 | 0.22524 | 0.17572 | 0.91376 |
| P2 | 0.1 | 0.01 | 300 | 0.1795 | 0.40122 | 0.85329 | 0.34653 | 0.23108 | 0.09650 | 0.91305 |
| P2 + 4 age interactions | 0.1 | 0 | 300 | 0.1893 | 0.40508 | 0.86001 | 0.35260 | 0.22872 | 0.14063 | 0.91317 |
| P2 + 4 age interactions | 0.1 | 0 | 1000 | 0.1832 | 0.41735 | 0.85880 | 0.37241 | 0.22459 | 0.18488 | 0.91374 |
| P2 | 0.2 | 0 | 500 | 0.1956 | 0.41558 | 0.86569 | 0.37157 | 0.22489 | 0.18650 | 0.91380 |
| P2 | 0.2 | 0 | 1000 | 0.1944 | 0.42027 | 0.86685 | 0.37974 | 0.22346 | 0.20947 | 0.91437 |
| P2 | 0.2 | 0.001 | 1000 | 0.1903 | 0.41888 | 0.86414 | 0.37742 | 0.22395 | 0.19247 | 0.91424 |
| P2 + 4 age interactions | 0.2 | 0 | 1000 | 0.1940 | 0.42141 | 0.86659 | 0.38058 | 0.22313 | 0.20796 | 0.91431 |

## Interpretation

1. Fifty updates are insufficient for comparing these representations. P1
   validation log loss falls from 0.33207 at 50 updates to 0.22748 at 1000;
   P2 falls from 0.24152 to 0.22489. AP improves for both. The early P2
   ranking deficit largely closes with more optimization. At 1000 updates,
   P2 is slightly ahead of P1 on F1, accuracy, AP, and log loss at `gamma=0.1`.
   Loss still improves slightly between 600 and 1000, so strict numerical
   convergence has not been established.
2. Threshold selection is necessary for the F1 objective. At 1000 updates,
   P2 without L2 gives F1 0.18645 and accuracy 0.91380 at threshold 0.5,
   versus F1 0.41560 and accuracy 0.86569 at threshold 0.1956. The lower
   accuracy is the expected trade-off from predicting more positive cases.
3. Doubling `gamma` on P2 made 500 updates nearly equivalent to 1000 updates
   at `gamma=0.1`, with no observed instability. At 1000 updates and
   `gamma=0.2`, P2 improves to F1 0.42027, accuracy 0.86685, AP 0.37974,
   and log loss 0.22346. The larger step still improves training loss up to
   1000 updates, so this is an optimization screen rather than proof of full
   convergence.
4. L2 did not produce a meaningful improvement in this small grid. At 300
   updates, P1 with `lambda=0.001` is essentially tied with unregularized P1;
   `lambda=0.01` is worse. For P2 at 1000 updates, `lambda=0.001` has lower
   F1 (0.41460 versus 0.41560), accuracy, AP, and slightly higher log loss.
   At `gamma=0.2`, weak L2 again trails its unregularized counterpart in all
   four metrics. This does not rule out all other penalties or learning rates.
5. Four predefined P2 interactions (`_AGE80` squared and `_AGE80` multiplied
   by `BPHIGH4`, `DIABETE3`, and `SMOKE100`) raise F1 from 0.41560 to 0.41735
   and AP from 0.37157 to 0.37241 at 1000 updates, with slightly lower log
   loss at `gamma=0.1`. AP and log loss improve in all three folds, but
   accuracy at the selected threshold falls from 0.86569 to 0.85880. At
   `gamma=0.2`, the interaction model reaches F1 0.42141 and accuracy
   0.86659, versus F1 0.42027 and accuracy 0.86685 without interactions.
   The gain remains small and does not establish a strong limit of the linear
   representation.

## Threshold stability check

The selected OOF threshold above maximizes F1 on all pooled OOF predictions,
so the same labels take part in selecting and describing it. To check whether
that choice transfers across development folds, the runner now saves the OOF
row mapping and probabilities and repeats threshold selection three times. In
each repetition, it chooses the threshold on two folds and evaluates it on
the excluded fold. This diagnostic still uses only the saved development
partition; the outer 20% validation partition was not evaluated in this check.

**Review correction:** this is a threshold-transfer diagnostic, not nested
cross-validation or an independent estimate of the complete procedure. For
example, OOF scores on fold B come from a model trained on A+C. Using those
scores to choose a threshold for A indirectly involves A's training labels.
The diagnostic excludes A's labels directly from threshold optimization, but
does not remove this dependency or the effect of earlier model selection.

| Representation | Updates | Pooled OOF threshold | Pooled OOF F1 | Pooled OOF accuracy | Per-fold optimum range | Cross-fit threshold range | Cross-fit F1 | Cross-fit accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 500 | 0.1956 | 0.41558 | 0.86569 | 0.1770–0.1958 | 0.1903–0.1958 | 0.41528 | 0.86469 |
| P2 | 1000 | 0.1944 | 0.42027 | 0.86685 | 0.1909–0.2053 | 0.1913–0.1946 | 0.41990 | 0.86619 |
| P2 + 4 age interactions | 1000 | 0.1940 | 0.42141 | 0.86659 | 0.1813–0.1991 | 0.1806–0.1947 | 0.42043 | 0.86297 |

P2 at 1000 updates has a tight diagnostic threshold interval and loses only
0.00037 F1 relative to pooled OOF optimization. This is useful descriptive
evidence, but does not establish an unbiased F1 estimate or justify freezing
the model before checking convergence.

The interaction model retains only a 0.00053 cross-fit F1 advantage over P2,
while its cross-fit accuracy is 0.00322 lower and its threshold interval is
wider. These figures alone do not establish that P2 is more robust: a flat F1
curve can move the maximizing threshold substantially while barely changing
F1. The review below checks a common threshold and retains both candidates.

The reruns persist compressed OOF artifacts under
`results/oof_predictions/`. Each contains `development_indices`, `fold_ids`,
`labels`, and `probabilities`; its SHA-256 is recorded in the corresponding
experiment JSON. This lets the threshold calculation be audited without
retraining a model.

## Review and next experiment plan (2026-09-19)

This review reads the three newly saved OOF artifacts; it does not retrain
models or score the outer validation partition. Their development row order
and fold assignments match, allowing paired descriptive comparisons.

All metrics below use the pooled OOF F1-selected threshold and remain
exploratory:

| Candidate | Updates | F1 | Accuracy | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 500 | 0.41558 | 0.86569 | 0.33744 | 0.54081 | 0.37157 | 0.22489 |
| P2 | 1000 | 0.42027 | 0.86685 | 0.34139 | 0.54655 | 0.37974 | 0.22346 |
| P2 + 4 interactions | 1000 | 0.42141 | 0.86659 | 0.34148 | 0.55017 | 0.38058 | 0.22313 |

Increasing P2 from 500 to 1000 updates improves F1 by 0.00469 and AP by
0.00817. Training and validation log loss both decrease in every fold. The
current evidence therefore supports further optimization checks; it does not
establish convergence or an intrinsic performance ceiling for logistic
regression.

At 1000 updates, interactions improve AP and log loss in all three folds.
Using the common descriptive threshold 0.194 gives F1 0.42015 / accuracy
0.86661 for P2, versus F1 0.42138 / accuracy 0.86660 for interactions. The
fold F1 differences are +0.00014, +0.00197, and +0.00158. The large accuracy
gap in the transfer diagnostic is therefore not an unavoidable consequence
of adding interactions: much of it reflects the different selected thresholds.
These observations are promising but do not establish statistical significance.

For P2, thresholds 0.19, 0.194, and 0.20 give F1 0.41981, 0.42015, and
0.41962. For interactions, the corresponding values are 0.42130, 0.42138,
and 0.42061. Both have a relatively flat local F1 curve. This grid was examined
after viewing results and is descriptive, not a newly validated threshold rule.

Proceed in this order:

1. Keep P2 as the reference and the four-interaction variant as its challenger.
   On the existing development folds, use gamma 0.2, lambda 0, and warm-started
   checkpoints 1000, 2000, and 4000 for both. The saved artifacts contain
   probabilities, not optimizer weights, so the first 1000 updates must be
   rerun. Save fold weights, OOF scores, training loss, gradient norm, runtime,
   and all six reported metrics. Use validation loss and AP alongside F1;
   neither tiny F1 changes nor a low gradient norm alone defines the best fit.
2. Establish proper threshold isolation for the two resulting finalists.
   For each existing development evaluation fold, generate inner OOF scores
   using only the other two folds (two inner splits), fit preprocessing inside
   each inner training split, and choose the threshold there. Refit on all
   non-evaluation rows, then evaluate the excluded fold once. Predeclare the
   model settings before running this check. The 80/20 outer partition is not
   involved. This removes the identified threshold dependency, while earlier
   development model selection still limits claims of complete independence.
3. If optimization gains flatten, run a small, controlled representation test:
   selected categorical encoding at the established training budget, then
   piecewise-linear age/BMI terms. Earlier encoding screens used only 50
   updates and therefore do not settle their value after sufficient training.
   Compare each addition separately before combining successful changes.
4. Consider a small NumPy neural network only if the controlled representation
   tests stop improving F1/AP. It would add optimizer and regularization choices,
   so it is not the next implementation priority. Keep all model logic within
   NumPy and the standard library; visualization libraries remain figure-only.
5. Select the pipeline and threshold on development evidence, then evaluate
   that frozen choice on the existing outer partition. Do not choose between
   candidates using that score and still call it an independent final estimate.
   Document the baseline exposure of this partition explicitly.

## Phase 5 convergence and nested threshold results (2026-09-19)

Phase 5 completed items 1–2 above with two predeclared configurations. Both
use P2 preprocessing, `gamma=0.2`, `lambda=0`, the same three outer folds,
and warm-started checkpoints at 1000, 2000, and 4000 updates. The second adds
only the four age interactions. Each OOF artifact now includes the three fold
weight vectors and feature names: P2 has 297 coefficients including the
intercept, and the interaction model has 301.

| Candidate | Updates | F1 | Accuracy | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 1000 | 0.42027 | 0.86685 | 0.34139 | 0.54655 | 0.37974 | 0.22346 |
| P2 | 2000 | 0.42358 | 0.86882 | 0.34606 | 0.54586 | 0.38449 | 0.22268 |
| P2 | 4000 | 0.42549 | 0.87059 | 0.34991 | 0.54271 | 0.38699 | 0.22229 |
| P2 + 4 interactions | 1000 | 0.42141 | 0.86659 | 0.34148 | 0.55017 | 0.38058 | 0.22313 |
| P2 + 4 interactions | 2000 | 0.42463 | 0.86738 | 0.34416 | 0.55423 | 0.38528 | 0.22236 |
| P2 + 4 interactions | 4000 | 0.42624 | 0.86842 | 0.34656 | 0.55349 | 0.38761 | 0.22201 |

Both candidates improve at every checkpoint on F1, AP, and log loss. From
2000 to 4000 updates, P2 gains 0.00191 F1 and the interaction candidate gains
0.00161. The outer-fold objective gradient norms fall from approximately
0.0029 at 1000 to 0.00082 for P2 and 0.00077 for interactions at 4000. The
optimization is progressing more slowly, but the observed gains have not yet
flattened enough to claim numerical convergence.

At 4000, threshold isolation was applied only to the two fixed candidates.
For each outer evaluation fold, two inner folds were built from its training
rows. Preprocessing, interaction scaling, and model fitting were learned in
each inner training fold; the threshold maximizing inner OOF F1 was then
applied once to the held-out outer fold.

| Candidate | Pooled OOF F1 | Nested-threshold F1 | Nested-threshold accuracy | Inner threshold range |
| --- | ---: | ---: | ---: | ---: |
| P2 | 0.42549 | 0.42399 | 0.86789 | 0.18767–0.20158 |
| P2 + 4 interactions | 0.42624 | 0.42544 | 0.87216 | 0.19419–0.21264 |

The interaction candidate has the higher nested-threshold F1 by 0.00145 and
the higher nested-threshold accuracy by 0.00427. Its threshold is higher on
two folds, which reduces recall to 0.53602 and raises precision to 0.35268;
this is the expected F1 trade-off rather than evidence of an implementation
error. The result is stronger evidence for the interactions than the earlier
pooled comparison, though it is still development evidence after a sequence
of model choices and should not be reported as a final unbiased score.

The next decision is no longer whether to discard the interaction model. Keep
it as the current development leader. The review below assesses the next
experiment. No outer-partition evaluation or competition submission has been
run in Phase 5.

## Phase 5 results review and next experiment

The two 4000-update OOF artifacts have identical development row order, labels,
and fold IDs, so the comparison is paired. With thresholds chosen only inside
each outer training partition, the interaction model improves fold F1 by
0.002114, 0.000607, and 0.001360. Its pooled F1 advantage is 0.001451.
The difference is consistent across these three folds, but small and already
influenced by previous development-based choices; it is not a final
generalization estimate.

At the same descriptive threshold 0.20, P2 has F1 0.42481 and accuracy
0.87200; P2 with interactions has F1 0.42581 and accuracy 0.87196. The
ranking in F1 survives a common threshold, while the 0.00427 accuracy gap
seen under separate nested thresholds nearly disappears. Thus, much of the
accuracy difference reflects the selected thresholds, and F1 remains the
provisional primary criterion. The official accuracy tie-break rule has not
been verified.

For the next controlled experiment, continue both candidates from their saved
4000-update fold weights to 8000 updates, refitting each fold's preprocessing
on exactly its original training rows before applying the saved weights.
Compare F1 at pooled thresholds for screening, AP, log loss, gradient norms,
and a small common-threshold grid. Run the expensive inner threshold procedure
at 8000 only if the added updates give a material and consistent gain or
change the model ranking. If the curves flatten, test selected categorical
encoding at the established training budget before a larger model family.
Weak L2 previously worsened P2 at 1000 updates; without a visible widening
of the train/validation loss gap, L2 is a lower priority than convergence and
feature representation. No model or threshold is frozen by this review.

## Reproduction and next decision

Run from the project root:

```powershell
py -3 -m src.run_model_cv configs/experiments/phase4_model_cv.json
py -3 -m src.run_model_cv configs/experiments/phase4_convergence_extension_cv.json
py -3 -m src.run_model_cv configs/experiments/phase4_convergence_final_cv.json
py -3 -m src.run_model_cv configs/experiments/phase4_interactions_cv.json
py -3 -m src.run_model_cv configs/experiments/phase4_l2_extension_cv.json
py -3 -m src.run_model_cv configs/experiments/phase4_gamma_cv.json
py -3 -m src.run_model_cv configs/experiments/phase4_interactions_gamma_cv.json
py -3 -m src.run_model_cv configs/experiments/phase4_l2_gamma_cv.json
py -3 -m src.run_model_cv configs/experiments/phase5_p2_convergence_nested_cv.json
py -3 -m src.run_model_cv configs/experiments/phase5_p2_interactions_convergence_nested_cv.json
py -3 -m unittest discover -s tests -q
```

Each run produces full JSON records and CSV index rows under
`results/experiments/`, including F1, accuracy, precision, recall, confusion
counts, AP, log loss, selected threshold, fixed-threshold metrics, configuration,
input hashes, and fold outcomes. The Phase 3 summary now also prints accuracy.

The P2 plus four-interaction model at 4000 updates is the current development
leader. Its threshold still requires a predeclared final-selection rule before
any outer-partition comparison. AIcrowd has not been used in this phase.

The experiment logic imports only Python standard-library modules, NumPy, and
local project modules. `matplotlib` remains confined to EDA figures. The
50-test suite includes a project-wide import-policy check.
