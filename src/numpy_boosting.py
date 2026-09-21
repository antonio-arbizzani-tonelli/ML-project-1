"""Small, auditable histogram gradient boosting implemented with NumPy only.

The module deliberately implements a restricted first boosting baseline rather
than a general-purpose tree package.  Continuous inputs are binned from the
training fold only, missing values have their own state, and every split tests
both possible missing-value directions.  It is intended for controlled model
screening on the Project 1 data, where third-party boosting libraries are not
permitted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_MISSING_BIN = np.uint8(255)
_CHECKPOINT_FORMAT_VERSION = 1


def _as_features(features: np.ndarray, name: str) -> np.ndarray:
    """Validate a two-dimensional float32 matrix with NaNs allowed.

    The project cache and tree-preprocessing representation already use
    float32. Keeping that dtype prevents an unnecessary full float64 copy of
    a roughly 300-feature training fold before it is binned into ``uint8``.
    """

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if np.any(np.isinf(features)):
        raise ValueError(f"{name} must not contain infinite values.")
    return features


def _as_labels(labels: np.ndarray, row_count: int) -> np.ndarray:
    """Validate internal 0/1 labels aligned with a feature matrix."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size != row_count:
        raise ValueError("labels must be one-dimensional and align with features.")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must contain only 0 and 1.")
    return labels.astype(np.float64, copy=False)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Return a stable logistic sigmoid for finite real-valued scores."""

    scores = np.asarray(scores, dtype=np.float64)
    result = np.empty_like(scores)
    positive = scores >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    exp_scores = np.exp(scores[~positive])
    result[~positive] = exp_scores / (1.0 + exp_scores)
    return result


class HistogramBinner:
    """Learn train-only quantile bins and encode missing values distinctly.

    The returned matrix uses ``uint8``.  Bin 255 is reserved for missing input,
    so ``n_bins`` can be at most 255.  The learned edges must be fitted within
    each training fold before transforming validation or test data.
    """

    def __init__(self, n_bins: int = 64) -> None:
        if type(n_bins) is not int or not 2 <= n_bins <= int(_MISSING_BIN):
            raise ValueError("n_bins must be an integer between 2 and 255.")
        self.n_bins = n_bins
        self.edges_: tuple[np.ndarray, ...] | None = None
        self.bin_counts_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "HistogramBinner":
        """Fit quantile edges from finite values in each training column."""

        features = _as_features(features, "features")
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float64)[1:-1]
        edges = []
        bin_counts = np.empty(features.shape[1], dtype=np.int16)
        for column_index in range(features.shape[1]):
            values = features[:, column_index]
            known = values[~np.isnan(values)]
            if known.size == 0:
                column_edges = np.empty(0, dtype=np.float32)
            else:
                column_edges = np.asarray(
                    np.unique(np.quantile(known, quantiles)), dtype=np.float32
                )
            edges.append(column_edges)
            bin_counts[column_index] = column_edges.size + 1
        self.edges_ = tuple(edges)
        self.bin_counts_ = bin_counts
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Encode features with frozen edges, retaining NaNs as bin 255."""

        if self.edges_ is None or self.bin_counts_ is None:
            raise RuntimeError("HistogramBinner must be fitted before transform.")
        features = _as_features(features, "features")
        if features.shape[1] != len(self.edges_):
            raise ValueError(
                "features have a different column count from the fitted binner."
            )
        result = np.full(features.shape, _MISSING_BIN, dtype=np.uint8)
        for column_index, edges in enumerate(self.edges_):
            values = features[:, column_index]
            known = ~np.isnan(values)
            result[known, column_index] = np.searchsorted(
                edges, values[known], side="right"
            ).astype(np.uint8)
        return result

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit bin edges and encode the same training rows."""

        return self.fit(features).transform(features)


@dataclass(frozen=True)
class _TreeNode:
    """One immutable node in a shallow Newton tree."""

    value: float
    feature: int | None = None
    split_bin: int | None = None
    missing_go_left: bool = False
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


@dataclass
class _FittedNode:
    """Mutable training state that is frozen into :class:`_TreeNode` at the end."""

    node_id: int
    depth: int
    row_count: int
    gradient_sum: float
    hessian_sum: float
    value: float
    feature: int | None = None
    split_bin: int | None = None
    missing_go_left: bool = False
    left_id: int | None = None
    right_id: int | None = None


def _node_depth(node: _TreeNode) -> int:
    """Return the deepest split count below ``node`` for diagnostics/tests."""

    if node.feature is None:
        return 0
    return 1 + max(_node_depth(node.left), _node_depth(node.right))


class _NewtonTree:
    """Fit one histogram tree to logistic-loss gradients and curvatures."""

    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        l2_regularization: float,
        min_gain: float,
        max_features: int | float | None,
        random_generator: np.random.Generator,
        histogram_strategy: str,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.min_gain = min_gain
        self.max_features = max_features
        self.random_generator = random_generator
        self.histogram_strategy = histogram_strategy
        self.root_: _TreeNode | None = None
        self.feature_indices_: np.ndarray | None = None

    @staticmethod
    def _score(
        gradient_sum: float, hessian_sum: float, l2_regularization: float
    ) -> float:
        return gradient_sum * gradient_sum / (hessian_sum + l2_regularization)

    def _feature_indices(self, feature_count: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(feature_count, dtype=np.int64)
        if isinstance(self.max_features, float):
            count = int(np.ceil(self.max_features * feature_count))
        else:
            count = int(self.max_features)
        if count >= feature_count:
            return np.arange(feature_count, dtype=np.int64)
        return np.sort(
            self.random_generator.choice(feature_count, size=count, replace=False)
        )

    def fit(
        self,
        binned_features: np.ndarray,
        bin_counts: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
    ) -> "_NewtonTree":
        """Fit the tree to already-binned rows and aligned derivatives."""

        binned_features = np.asarray(binned_features, dtype=np.uint8)
        bin_counts = np.asarray(bin_counts, dtype=np.int16)
        gradients = np.asarray(gradients, dtype=np.float64)
        hessians = np.asarray(hessians, dtype=np.float64)
        if binned_features.ndim != 2 or binned_features.shape[0] == 0:
            raise ValueError(
                "binned_features must be a non-empty two-dimensional array."
            )
        if bin_counts.shape != (binned_features.shape[1],):
            raise ValueError("bin_counts must describe every feature column.")
        if (
            gradients.shape != (binned_features.shape[0],)
            or hessians.shape != gradients.shape
        ):
            raise ValueError("derivatives must be one-dimensional and align with rows.")
        if (
            not np.all(np.isfinite(gradients))
            or not np.all(np.isfinite(hessians))
            or np.any(hessians <= 0)
        ):
            raise ValueError("gradients must be finite and hessians strictly positive.")
        if self.histogram_strategy == "recursive":
            self.root_ = self._build(
                binned_features,
                bin_counts,
                gradients,
                hessians,
                np.arange(binned_features.shape[0], dtype=np.int64),
                depth=0,
            )
        else:
            self.feature_indices_ = self._feature_indices(binned_features.shape[1])
            self.root_ = self._fit_levelwise(
                binned_features, bin_counts, gradients, hessians
            )
        return self

    def _build(
        self,
        binned_features: np.ndarray,
        bin_counts: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        rows: np.ndarray,
        depth: int,
    ) -> _TreeNode:
        """Recursively select the strongest split with a fresh node feature sample."""

        gradient_sum = float(np.sum(gradients[rows]))
        hessian_sum = float(np.sum(hessians[rows]))
        leaf_value = -gradient_sum / (hessian_sum + self.l2_regularization)
        if depth >= self.max_depth or rows.size < 2 * self.min_samples_leaf:
            return _TreeNode(value=leaf_value)

        parent_score = self._score(gradient_sum, hessian_sum, self.l2_regularization)
        best: tuple[float, int, int, bool] | None = None
        for feature in self._feature_indices(binned_features.shape[1]):
            values = binned_features[rows, feature]
            known = values != _MISSING_BIN
            known_values = values[known]
            count = int(bin_counts[feature])
            if count < 2 and known_values.size == rows.size:
                continue
            known_gradients = gradients[rows[known]]
            known_hessians = hessians[rows[known]]
            known_counts = np.bincount(known_values, minlength=count)
            gradient_histogram = np.bincount(
                known_values, weights=known_gradients, minlength=count
            )
            hessian_histogram = np.bincount(
                known_values, weights=known_hessians, minlength=count
            )
            cumulative_counts = np.cumsum(known_counts)
            cumulative_gradients = np.cumsum(gradient_histogram)
            cumulative_hessians = np.cumsum(hessian_histogram)
            missing_count = int(rows.size - known_values.size)
            missing_gradient = gradient_sum - float(cumulative_gradients[-1])
            missing_hessian = hessian_sum - float(cumulative_hessians[-1])
            split_bins = range(count - 1) if count >= 2 else (0,)
            for split_bin in split_bins:
                known_left_count = int(cumulative_counts[split_bin])
                known_left_gradient = float(cumulative_gradients[split_bin])
                known_left_hessian = float(cumulative_hessians[split_bin])
                for missing_go_left in (False, True):
                    left_count = known_left_count + (
                        missing_count if missing_go_left else 0
                    )
                    right_count = rows.size - left_count
                    if (
                        left_count < self.min_samples_leaf
                        or right_count < self.min_samples_leaf
                    ):
                        continue
                    left_gradient = known_left_gradient + (
                        missing_gradient if missing_go_left else 0.0
                    )
                    left_hessian = known_left_hessian + (
                        missing_hessian if missing_go_left else 0.0
                    )
                    right_gradient = gradient_sum - left_gradient
                    right_hessian = hessian_sum - left_hessian
                    gain = 0.5 * (
                        self._score(left_gradient, left_hessian, self.l2_regularization)
                        + self._score(
                            right_gradient, right_hessian, self.l2_regularization
                        )
                        - parent_score
                    )
                    candidate = (gain, int(feature), split_bin, missing_go_left)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
        if best is None or best[0] <= self.min_gain:
            return _TreeNode(value=leaf_value)

        _, feature, split_bin, missing_go_left = best
        values = binned_features[rows, feature]
        go_left = values <= split_bin
        go_left[values == _MISSING_BIN] = missing_go_left
        left_rows, right_rows = rows[go_left], rows[~go_left]
        return _TreeNode(
            value=leaf_value,
            feature=feature,
            split_bin=split_bin,
            missing_go_left=missing_go_left,
            left=self._build(
                binned_features, bin_counts, gradients, hessians, left_rows, depth + 1
            ),
            right=self._build(
                binned_features, bin_counts, gradients, hessians, right_rows, depth + 1
            ),
        )

    def _fit_levelwise(
        self,
        binned_features: np.ndarray,
        bin_counts: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
    ) -> _TreeNode:
        """Build each depth from shared per-feature, per-leaf histograms.

        At one depth every row belongs to exactly one active leaf.  Encoding
        ``(leaf, bin)`` as one integer lets ``np.bincount`` produce all leaf
        histograms for one feature in one call.  This avoids repeatedly
        slicing the same rows for sibling nodes, the dominant cost in the
        initial recursive implementation.
        """

        root_gradient = float(np.sum(gradients))
        root_hessian = float(np.sum(hessians))
        root = _FittedNode(
            node_id=0,
            depth=0,
            row_count=binned_features.shape[0],
            gradient_sum=root_gradient,
            hessian_sum=root_hessian,
            value=-root_gradient / (root_hessian + self.l2_regularization),
        )
        nodes = {0: root}
        active = [root]
        assignments = np.zeros(binned_features.shape[0], dtype=np.int16)
        max_bin_count = int(np.max(bin_counts))
        next_node_id = 1

        while active:
            splittable = [
                node
                for node in active
                if node.depth < self.max_depth
                and node.row_count >= 2 * self.min_samples_leaf
            ]
            if not splittable:
                break
            node_to_code = np.full(next_node_id, -1, dtype=np.int16)
            for code, node in enumerate(splittable):
                node_to_code[node.node_id] = code
            row_codes = node_to_code[assignments]
            best: list[tuple[float, int, int, bool, int, float, float] | None] = [
                None for _ in splittable
            ]
            for feature in self.feature_indices_:
                values = binned_features[:, feature]
                known = values != _MISSING_BIN
                included = known & (row_codes >= 0)
                if not np.any(included):
                    continue
                keys = (
                    row_codes[included].astype(np.int64) * max_bin_count
                    + values[included]
                )
                histogram_size = len(splittable) * max_bin_count
                counts = np.bincount(keys, minlength=histogram_size).reshape(
                    len(splittable), max_bin_count
                )
                gradient_histograms = np.bincount(
                    keys, weights=gradients[included], minlength=histogram_size
                ).reshape(len(splittable), max_bin_count)
                hessian_histograms = np.bincount(
                    keys, weights=hessians[included], minlength=histogram_size
                ).reshape(len(splittable), max_bin_count)
                count = int(bin_counts[feature])
                for code, node in enumerate(splittable):
                    known_counts = counts[code, :count]
                    known_count = int(np.sum(known_counts))
                    if count < 2 and known_count == node.row_count:
                        continue
                    cumulative_counts = np.cumsum(known_counts)
                    cumulative_gradients = np.cumsum(gradient_histograms[code, :count])
                    cumulative_hessians = np.cumsum(hessian_histograms[code, :count])
                    missing_count = node.row_count - known_count
                    missing_gradient = node.gradient_sum - float(
                        cumulative_gradients[-1]
                    )
                    missing_hessian = node.hessian_sum - float(cumulative_hessians[-1])
                    parent_score = self._score(
                        node.gradient_sum, node.hessian_sum, self.l2_regularization
                    )
                    split_bins = range(count - 1) if count >= 2 else (0,)
                    for split_bin in split_bins:
                        known_left_count = int(cumulative_counts[split_bin])
                        known_left_gradient = float(cumulative_gradients[split_bin])
                        known_left_hessian = float(cumulative_hessians[split_bin])
                        for missing_go_left in (False, True):
                            left_count = known_left_count + (
                                missing_count if missing_go_left else 0
                            )
                            right_count = node.row_count - left_count
                            if (
                                left_count < self.min_samples_leaf
                                or right_count < self.min_samples_leaf
                            ):
                                continue
                            left_gradient = known_left_gradient + (
                                missing_gradient if missing_go_left else 0.0
                            )
                            left_hessian = known_left_hessian + (
                                missing_hessian if missing_go_left else 0.0
                            )
                            right_gradient = node.gradient_sum - left_gradient
                            right_hessian = node.hessian_sum - left_hessian
                            gain = 0.5 * (
                                self._score(
                                    left_gradient, left_hessian, self.l2_regularization
                                )
                                + self._score(
                                    right_gradient,
                                    right_hessian,
                                    self.l2_regularization,
                                )
                                - parent_score
                            )
                            candidate = (
                                gain,
                                int(feature),
                                split_bin,
                                missing_go_left,
                                left_count,
                                left_gradient,
                                left_hessian,
                            )
                            if best[code] is None or candidate[0] > best[code][0]:
                                best[code] = candidate

            next_active = []
            for node, split in zip(splittable, best):
                if split is None or split[0] <= self.min_gain:
                    continue
                (
                    _,
                    feature,
                    split_bin,
                    missing_go_left,
                    left_count,
                    left_gradient,
                    left_hessian,
                ) = split
                right_count = node.row_count - left_count
                right_gradient = node.gradient_sum - left_gradient
                right_hessian = node.hessian_sum - left_hessian
                left = _FittedNode(
                    next_node_id,
                    node.depth + 1,
                    left_count,
                    left_gradient,
                    left_hessian,
                    -left_gradient / (left_hessian + self.l2_regularization),
                )
                right = _FittedNode(
                    next_node_id + 1,
                    node.depth + 1,
                    right_count,
                    right_gradient,
                    right_hessian,
                    -right_gradient / (right_hessian + self.l2_regularization),
                )
                next_node_id += 2
                nodes[left.node_id] = left
                nodes[right.node_id] = right
                node.feature = feature
                node.split_bin = split_bin
                node.missing_go_left = missing_go_left
                node.left_id = left.node_id
                node.right_id = right.node_id
                rows = np.flatnonzero(assignments == node.node_id)
                values = binned_features[rows, feature]
                go_left = values <= split_bin
                go_left[values == _MISSING_BIN] = missing_go_left
                assignments[rows[go_left]] = left.node_id
                assignments[rows[~go_left]] = right.node_id
                next_active.extend((left, right))
            active = next_active

        def freeze(node_id: int) -> _TreeNode:
            node = nodes[node_id]
            if node.feature is None:
                return _TreeNode(value=node.value)
            return _TreeNode(
                value=node.value,
                feature=node.feature,
                split_bin=node.split_bin,
                missing_go_left=node.missing_go_left,
                left=freeze(node.left_id),
                right=freeze(node.right_id),
            )

        return freeze(0)

    def predict(self, binned_features: np.ndarray) -> np.ndarray:
        """Return the leaf value assigned by the fitted tree to every row."""

        if self.root_ is None:
            raise RuntimeError("_NewtonTree must be fitted before predict.")
        binned_features = np.asarray(binned_features, dtype=np.uint8)
        if binned_features.ndim != 2:
            raise ValueError("binned_features must be two-dimensional.")
        output = np.empty(binned_features.shape[0], dtype=np.float64)

        def assign(node: _TreeNode, rows: np.ndarray) -> None:
            if node.feature is None:
                output[rows] = node.value
                return
            values = binned_features[rows, node.feature]
            go_left = values <= node.split_bin
            go_left[values == _MISSING_BIN] = node.missing_go_left
            assign(node.left, rows[go_left])
            assign(node.right, rows[~go_left])

        assign(self.root_, np.arange(binned_features.shape[0], dtype=np.int64))
        return output


class HistogramGradientBoostingClassifier:
    """Binary logistic gradient boosting with shallow histogram trees.

    ``max_depth`` is deliberately configurable so depth 3 and 5 candidates use
    precisely the same implementation. ``max_features`` may be ``None`` for
    all features, an integer, or a fraction in ``(0, 1]`` sampled at each node
    with the default recursive strategy. The optional levelwise strategy
    samples once per tree and builds shared histograms for a performance study.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 200,
        n_bins: int = 64,
        l2_regularization: float = 1.0,
        min_gain: float = 0.0,
        max_features: int | float | None = None,
        random_seed: int = 0,
        histogram_strategy: str = "recursive",
    ) -> None:
        if type(n_estimators) is not int or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        if not np.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be finite and in (0, 1].")
        if type(max_depth) is not int or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if type(min_samples_leaf) is not int or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer.")
        if not np.isfinite(l2_regularization) or l2_regularization < 0.0:
            raise ValueError("l2_regularization must be finite and non-negative.")
        if not np.isfinite(min_gain) or min_gain < 0.0:
            raise ValueError("min_gain must be finite and non-negative.")
        if max_features is not None:
            valid_integer = type(max_features) is int and max_features >= 1
            valid_fraction = (
                isinstance(max_features, float) and 0.0 < max_features <= 1.0
            )
            if not (valid_integer or valid_fraction):
                raise ValueError(
                    "max_features must be None, a positive integer, or a fraction in (0, 1]."
                )
        if type(random_seed) is not int:
            raise ValueError("random_seed must be an integer.")
        if histogram_strategy not in {"recursive", "levelwise"}:
            raise ValueError("histogram_strategy must be 'recursive' or 'levelwise'.")
        self.n_estimators = n_estimators
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.l2_regularization = float(l2_regularization)
        self.min_gain = float(min_gain)
        self.max_features = max_features
        self.random_seed = random_seed
        self.histogram_strategy = histogram_strategy
        self.binner_: HistogramBinner | None = None
        self.trees_: list[_NewtonTree] = []
        self.tree_fit_seconds_: list[float] = []
        self.base_score_: float | None = None
        self.n_features_in_: int | None = None
        self.training_scores_: np.ndarray | None = None
        self.random_generator_state_: dict[str, Any] | None = None

    @staticmethod
    def _checkpoint_steps(
        checkpoints: Sequence[int] | None, total_trees: int
    ) -> tuple[int, ...]:
        if checkpoints is None:
            return ()
        steps = tuple(checkpoints)
        if not steps or any(type(step) is not int or step <= 0 for step in steps):
            raise ValueError("checkpoints must be non-empty positive integers.")
        if steps != tuple(sorted(set(steps))) or steps[-1] > total_trees:
            raise ValueError(
                "checkpoints must be strictly increasing fitted-tree counts."
            )
        return steps

    def _append_trees(
        self,
        binned_features: np.ndarray,
        labels: np.ndarray,
        generator: np.random.Generator,
        tree_count: int,
        checkpoints: tuple[int, ...],
        checkpoint_callback: (
            Callable[[int, "HistogramGradientBoostingClassifier"], None] | None
        ),
    ) -> None:
        if self.training_scores_ is None:
            raise RuntimeError(
                "Training scores must be initialized before fitting trees."
            )
        checkpoint_set = set(checkpoints)
        for _ in range(tree_count):
            probabilities = _sigmoid(self.training_scores_)
            gradients = probabilities - labels
            hessians = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
            started = time.perf_counter()
            tree = _NewtonTree(
                self.max_depth,
                self.min_samples_leaf,
                self.l2_regularization,
                self.min_gain,
                self.max_features,
                generator,
                self.histogram_strategy,
            ).fit(binned_features, self.binner_.bin_counts_, gradients, hessians)
            self.training_scores_ += self.learning_rate * tree.predict(binned_features)
            self.tree_fit_seconds_.append(time.perf_counter() - started)
            self.trees_.append(tree)
            self.random_generator_state_ = copy.deepcopy(generator.bit_generator.state)
            fitted_count = len(self.trees_)
            if fitted_count in checkpoint_set and checkpoint_callback is not None:
                checkpoint_callback(fitted_count, self)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        checkpoints: Sequence[int] | None = None,
        checkpoint_callback: (
            Callable[[int, "HistogramGradientBoostingClassifier"], None] | None
        ) = None,
    ) -> "HistogramGradientBoostingClassifier":
        """Fit bins and trees using only the supplied training rows."""

        features = _as_features(features, "features")
        labels = _as_labels(labels, features.shape[0])
        checkpoint_steps = self._checkpoint_steps(checkpoints, self.n_estimators)
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise ValueError("checkpoint_callback must be callable when supplied.")
        binner = HistogramBinner(self.n_bins)
        binned_features = binner.fit_transform(features)
        prevalence = float(np.clip(np.mean(labels), 1e-12, 1.0 - 1e-12))
        self.base_score_ = float(np.log(prevalence / (1.0 - prevalence)))
        self.binner_ = binner
        self.trees_ = []
        self.tree_fit_seconds_ = []
        self.n_features_in_ = features.shape[1]
        self.training_scores_ = np.full(labels.size, self.base_score_, dtype=np.float64)
        generator = np.random.default_rng(self.random_seed)
        self.random_generator_state_ = copy.deepcopy(generator.bit_generator.state)
        self._append_trees(
            binned_features,
            labels,
            generator,
            self.n_estimators,
            checkpoint_steps,
            checkpoint_callback,
        )
        return self

    def continue_fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        additional_estimators: int,
        checkpoints: Sequence[int] | None = None,
        checkpoint_callback: (
            Callable[[int, "HistogramGradientBoostingClassifier"], None] | None
        ) = None,
    ) -> "HistogramGradientBoostingClassifier":
        """Append trees from a trusted checkpoint without changing its fitted bins.

        ``features`` and ``labels`` must be the same ordered training rows used
        to create the checkpoint. The experiment runner stores their row IDs
        and input fingerprints alongside each checkpoint for that validation.
        """

        if type(additional_estimators) is not int or additional_estimators <= 0:
            raise ValueError("additional_estimators must be a positive integer.")
        if (
            self.binner_ is None
            or self.training_scores_ is None
            or self.random_generator_state_ is None
        ):
            raise RuntimeError(
                "A fitted checkpoint with training state is required to continue fitting."
            )
        features = _as_features(features, "features")
        labels = _as_labels(labels, features.shape[0])
        if (
            features.shape[1] != self.n_features_in_
            or labels.size != self.training_scores_.size
        ):
            raise ValueError(
                "Continuation features or labels do not align with the saved training state."
            )
        checkpoint_steps = self._checkpoint_steps(
            checkpoints, len(self.trees_) + additional_estimators
        )
        generator = np.random.default_rng()
        generator.bit_generator.state = copy.deepcopy(self.random_generator_state_)
        binned_features = self.binner_.transform(features)
        self.n_estimators = len(self.trees_) + additional_estimators
        self._append_trees(
            binned_features,
            labels,
            generator,
            additional_estimators,
            checkpoint_steps,
            checkpoint_callback,
        )
        return self

    def save_checkpoint(
        self, path: Path | str, metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Atomically save a trusted local checkpoint that can resume fitting."""

        if (
            self.binner_ is None
            or self.training_scores_ is None
            or self.random_generator_state_ is None
        ):
            raise RuntimeError("Only a fitted model can be checkpointed.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "model": self,
            "metadata": {} if metadata is None else dict(metadata),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)

    @classmethod
    def load_checkpoint(
        cls, path: Path | str
    ) -> tuple["HistogramGradientBoostingClassifier", dict[str, Any]]:
        """Load a checkpoint written by :meth:`save_checkpoint`.

        Checkpoints use Python pickle and must therefore come only from this
        project's trusted local artifact directory.
        """

        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if (
            not isinstance(payload, dict)
            or payload.get("format_version") != _CHECKPOINT_FORMAT_VERSION
        ):
            raise ValueError("Checkpoint has an unsupported format.")
        model = payload.get("model")
        metadata = payload.get("metadata")
        if not isinstance(model, cls) or not isinstance(metadata, dict):
            raise ValueError("Checkpoint payload is malformed.")
        if (
            model.binner_ is None
            or model.training_scores_ is None
            or model.random_generator_state_ is None
        ):
            raise ValueError("Checkpoint lacks fitted continuation state.")
        return model, metadata

    def _binned_for_prediction(self, features: np.ndarray) -> np.ndarray:
        if (
            self.binner_ is None
            or self.base_score_ is None
            or self.n_features_in_ is None
        ):
            raise RuntimeError(
                "HistogramGradientBoostingClassifier must be fitted before prediction."
            )
        features = _as_features(features, "features")
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                "features have a different column count from the fitted model."
            )
        return self.binner_.transform(features)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return additive log-odds scores for the positive class."""

        binned_features = self._binned_for_prediction(features)
        scores = np.full(binned_features.shape[0], self.base_score_, dtype=np.float64)
        for tree in self.trees_:
            scores += self.learning_rate * tree.predict(binned_features)
        return scores

    def staged_decision_function(
        self, features: np.ndarray, checkpoints: Sequence[int]
    ):
        """Yield additive scores after each requested fitted-tree checkpoint.

        A caller can evaluate 100, 200, and 400 trees after one 400-tree fit,
        rather than retraining the first 100 or 200 trees.  Checkpoints must
        be strictly increasing and cannot exceed the fitted tree count.
        """

        requested = tuple(checkpoints)
        if not requested or any(
            type(step) is not int or step <= 0 for step in requested
        ):
            raise ValueError("checkpoints must be non-empty positive integers.")
        if requested != tuple(sorted(set(requested))):
            raise ValueError("checkpoints must be strictly increasing.")
        if len(self.trees_) < requested[-1]:
            raise ValueError("A checkpoint exceeds the fitted tree count.")
        binned_features = self._binned_for_prediction(features)
        scores = np.full(binned_features.shape[0], self.base_score_, dtype=np.float64)
        checkpoint_index = 0
        for tree_number, tree in enumerate(self.trees_, start=1):
            scores += self.learning_rate * tree.predict(binned_features)
            if tree_number == requested[checkpoint_index]:
                yield tree_number, scores.copy()
                checkpoint_index += 1
                if checkpoint_index == len(requested):
                    return

    def staged_predict_proba(self, features: np.ndarray, checkpoints: Sequence[int]):
        """Yield positive-class probabilities at requested fitted-tree checkpoints."""

        for checkpoint, scores in self.staged_decision_function(features, checkpoints):
            yield checkpoint, _sigmoid(scores)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return ``(negative, positive)`` probability columns."""

        positive = _sigmoid(self.decision_function(features))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return 0/1 labels at a finite positive-class probability threshold."""

        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite.")
        return (self.predict_proba(features)[:, 1] >= threshold).astype(np.int8)
