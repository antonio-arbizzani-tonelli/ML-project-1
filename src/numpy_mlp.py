"""Mini-batch binary MLP with ReLU, stable BCE and Adam, using NumPy only."""

from __future__ import annotations

import time

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Evaluate the sigmoid without overflowing for large finite logits."""
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


class MLPClassifier:
    """Fit a dense ReLU network with one positive-class sigmoid output.

    The objective is mean BCE + ``0.5 * l2 * sum(W**2)``; biases are not
    regularized. Validation, if supplied, controls only early stopping on BCE.
    Every call to fit starts a fresh deterministic fit, including Adam state.
    """

    def __init__(self, hidden_layer_sizes=(64, 32), learning_rate=0.001,
                 l2=0.0001, batch_size=512, max_epochs=80, patience=8,
                 min_delta=0.0001, random_seed=20261001):
        if (not hidden_layer_sizes or
                any(type(width) is not int or width < 1 for width in hidden_layer_sizes)):
            raise ValueError("hidden_layer_sizes must contain positive integers.")
        for name, value in (("batch_size", batch_size), ("max_epochs", max_epochs),
                            ("patience", patience)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(l2) or l2 < 0 or not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("l2 and min_delta must be finite and nonnegative.")
        if type(random_seed) is not int or random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer.")
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = float(min_delta)
        self.random_seed = random_seed
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.history_: list[dict] = []
        self.best_epoch_ = 0
        self.epochs_trained_ = 0

    @staticmethod
    def _features(features, expected_width=None):
        """Validate a nonempty finite floating point design matrix."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or min(features.shape) == 0:
            raise ValueError("features must be a nonempty two-dimensional matrix.")
        if expected_width is not None and features.shape[1] != expected_width:
            raise ValueError("features have a different width from the fitted network.")
        if not np.all(np.isfinite(features)):
            raise ValueError("MLP input must contain only finite values.")
        return features

    @staticmethod
    def _labels(labels, rows):
        """Require internal 0/1 labels aligned with the design matrix."""
        labels = np.asarray(labels)
        if labels.shape != (rows,) or not np.all(np.isin(labels, [0, 1])):
            raise ValueError("labels must be an aligned one-dimensional 0/1 array.")
        return labels.astype(np.float32)

    def _initialize(self, width, prevalence, rng):
        """Use He hidden weights and a prevalence-based output bias."""
        sizes = (width, *self.hidden_layer_sizes, 1)
        self.weights_ = [
            (rng.standard_normal((left, right)) * np.sqrt(
                (2.0 if layer < len(sizes) - 2 else 1.0) / left
            )).astype(np.float32)
            for layer, (left, right) in enumerate(zip(sizes[:-1], sizes[1:]))
        ]
        self.biases_ = [np.zeros(right, dtype=np.float32) for right in sizes[1:]]
        prior = np.clip(prevalence, 1e-6, 1 - 1e-6)
        self.biases_[-1][0] = np.log(prior / (1 - prior))

    def _forward(self, features):
        """Return layer activations, including raw logits in the final layer."""
        activations = [features]
        for layer, (weights, bias) in enumerate(zip(self.weights_, self.biases_)):
            values = activations[-1] @ weights + bias
            if layer < len(self.weights_) - 1:
                np.maximum(values, 0, out=values)
            activations.append(values)
        return activations

    def _loss_and_gradients(self, features, labels):
        """Differentiate the stable BCE objective; also used in gradient checks."""
        activations = self._forward(features)
        logits = activations[-1][:, 0]
        loss = float(np.mean(np.logaddexp(0, logits) - labels * logits,
                             dtype=np.float64))
        loss += 0.5 * self.l2 * sum(float(np.sum(w * w, dtype=np.float64))
                                  for w in self.weights_)
        delta = ((sigmoid(logits) - labels) / labels.size)[:, None]
        weight_gradients = [None] * len(self.weights_)
        bias_gradients = [None] * len(self.weights_)
        for layer in range(len(self.weights_) - 1, -1, -1):
            weight_gradients[layer] = activations[layer].T @ delta + self.l2 * self.weights_[layer]
            bias_gradients[layer] = np.sum(delta, axis=0)
            if layer:
                delta = (delta @ self.weights_[layer].T) * (activations[layer] > 0)
        return loss, weight_gradients, bias_gradients

    def _data_loss(self, features, labels):
        """Compute unregularized BCE in bounded batches without probability clipping."""
        total = 0.0
        for start in range(0, labels.size, 4096):
            stop = min(start + 4096, labels.size)
            logits = self._forward(features[start:stop])[-1][:, 0]
            total += float(np.sum(np.logaddexp(0, logits) - labels[start:stop] * logits,
                                  dtype=np.float64))
        return total / labels.size

    def fit(self, features, labels, validation_features=None, validation_labels=None,
            progress_callback=None):
        """Train; restore the best validation weights when early stopping is enabled."""
        features = self._features(features)
        labels = self._labels(labels, features.shape[0])
        if (validation_features is None) != (validation_labels is None):
            raise ValueError("Both validation arrays must be supplied together.")
        validation = validation_features is not None
        if validation:
            validation_features = self._features(validation_features, features.shape[1])
            validation_labels = self._labels(validation_labels, validation_features.shape[0])
        rng = np.random.default_rng(self.random_seed)
        self._initialize(features.shape[1], float(np.mean(labels)), rng)
        # A reserved unseen-category flag has no training signal. Give any
        # entirely zero training channel zero initial weights so it cannot
        # introduce arbitrary random effects if it activates at inference.
        self.weights_[0][~np.any(features, axis=0), :] = 0
        parameters = self.weights_ + self.biases_
        first_moments = [np.zeros_like(p) for p in parameters]
        second_moments = [np.zeros_like(p) for p in parameters]
        self.history_ = []
        self.best_epoch_ = 0
        best_loss, stopping_reference, stale, update = np.inf, np.inf, 0, 0
        best_parameters = None
        for epoch in range(1, self.max_epochs + 1):
            started = time.perf_counter()
            order = rng.permutation(labels.size)
            for start in range(0, labels.size, self.batch_size):
                positions = order[start:start + self.batch_size]
                _, weight_gradients, bias_gradients = self._loss_and_gradients(
                    features[positions], labels[positions])
                update += 1
                rate = self.learning_rate * np.sqrt(1 - 0.999 ** update) / (1 - 0.9 ** update)
                for parameter, gradient, first, second in zip(
                        parameters, weight_gradients + bias_gradients,
                        first_moments, second_moments):
                    first *= 0.9
                    first += 0.1 * gradient
                    second *= 0.999
                    second += 0.001 * gradient * gradient
                    # This is bias-corrected Adam, including correction of epsilon.
                    parameter -= rate * first / (np.sqrt(second) + 1e-8 * np.sqrt(1 - 0.999 ** update))
            training_loss = self._data_loss(features, labels)
            validation_loss = (self._data_loss(validation_features, validation_labels)
                               if validation else None)
            if not np.isfinite(training_loss) or (validation and not np.isfinite(validation_loss)):
                raise FloatingPointError("MLP training diverged to a non-finite loss.")
            self.epochs_trained_ = epoch
            if validation:
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    self.best_epoch_ = epoch
                    best_parameters = [p.copy() for p in parameters]
                if validation_loss < stopping_reference - self.min_delta:
                    stopping_reference, stale = validation_loss, 0
                else:
                    stale += 1
            else:
                self.best_epoch_ = epoch
            entry = {"epoch": epoch, "training_bce": training_loss,
                     "validation_bce": validation_loss,
                     "seconds": time.perf_counter() - started}
            self.history_.append(entry)
            if progress_callback is not None:
                progress_callback(dict(entry))
            if validation and stale >= self.patience:
                break
        if validation:
            for parameter, best in zip(parameters, best_parameters):
                parameter[:] = best
        return self

    def predict_proba(self, features, batch_size=4096):
        """Return a vector of positive-class probabilities, not hard labels."""
        if not self.weights_:
            raise RuntimeError("fit must be called before predict_proba.")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("prediction batch_size must be a positive integer.")
        features = self._features(features, self.weights_[0].shape[0])
        probabilities = np.empty(features.shape[0], dtype=np.float64)
        for start in range(0, features.shape[0], batch_size):
            probabilities[start:start + batch_size] = sigmoid(
                self._forward(features[start:start + batch_size])[-1][:, 0])
        if not np.all(np.isfinite(probabilities)):
            raise FloatingPointError("The fitted network produced non-finite scores.")
        return probabilities
