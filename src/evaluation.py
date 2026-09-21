"""Binary-classification evaluation utilities implemented with NumPy only.

All labels in this module use the internal convention ``0`` for negative and
``1`` for positive. Convert competition labels at the pipeline boundary.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    """Four outcomes of a binary classifier, with positive class equal to 1."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass(frozen=True)
class PrecisionRecallCurve:
    """Precision and recall evaluated at every distinct score threshold.

    Arrays are ordered from the largest threshold to the smallest. Scores
    greater than or equal to a threshold are predicted positive.
    """

    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray


@dataclass(frozen=True)
class ThresholdChoice:
    """Best development threshold, breaking equal F1 values by accuracy."""

    threshold: float
    f1: float
    accuracy: float


def _as_binary_labels(labels, name):
    """Validate and return a non-empty one-dimensional 0/1 label array."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError(f"{name} must contain only binary labels 0 and 1.")
    return labels


def _as_scores(scores):
    """Validate and return a non-empty one-dimensional finite score array."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values.")
    return scores


def _validate_same_size(first, second, first_name, second_name):
    """Raise a clear error unless two validated arrays have the same length."""

    if first.size != second.size:
        raise ValueError(
            f"{first_name} and {second_name} must have the same number of values."
        )


def confusion_counts(y_true, y_pred):
    """Count TP, FP, TN, and FN from 0/1 true and predicted labels."""

    y_true = _as_binary_labels(y_true, "y_true")
    y_pred = _as_binary_labels(y_pred, "y_pred")
    _validate_same_size(y_true, y_pred, "y_true", "y_pred")
    return ConfusionCounts(
        true_positives=int(np.sum((y_true == 1) & (y_pred == 1))),
        false_positives=int(np.sum((y_true == 0) & (y_pred == 1))),
        true_negatives=int(np.sum((y_true == 0) & (y_pred == 0))),
        false_negatives=int(np.sum((y_true == 1) & (y_pred == 0))),
    )


def metrics_from_confusion(counts):
    """Derive metrics from :class:`ConfusionCounts`.

    Precision is zero without positive predictions. Recall and specificity are
    ``nan`` if the relevant true class is absent, exposing an incomplete
    evaluation set instead of reporting a misleading number.
    """

    if not isinstance(counts, ConfusionCounts):
        raise TypeError("counts must be a ConfusionCounts instance.")
    tp, fp = counts.true_positives, counts.false_positives
    tn, fn = counts.true_negatives, counts.false_negatives
    total = tp + fp + tn + fn
    if total == 0:
        raise ValueError("At least one observation is required.")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(recall) and precision + recall
        else (0.0 if np.isfinite(recall) else np.nan)
    )
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if np.isfinite(recall) and np.isfinite(specificity)
        else np.nan
    )
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def classification_metrics(y_true, y_pred):
    """Return confusion counts and their derived metrics for one prediction set."""

    counts = confusion_counts(y_true, y_pred)
    return counts, metrics_from_confusion(counts)


def labels_from_scores(scores, threshold=0.5):
    """Convert finite scores to 0/1 labels using ``score >= threshold``."""

    scores = _as_scores(scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return (scores >= threshold).astype(np.int8)


def precision_recall_curve(y_true, scores):
    """Compute precision-recall points at every distinct score threshold.

    Tied scores are always assigned the same label, so a threshold cannot
    split observations that the model scored identically.
    """

    y_true = _as_binary_labels(y_true, "y_true")
    scores = _as_scores(scores)
    _validate_same_size(y_true, scores, "y_true", "scores")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_true = scores[order], y_true[order]
    cumulative_tp = np.cumsum(sorted_true)
    cumulative_fp = np.cumsum(1 - sorted_true)
    last_at_score = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        sorted_scores.size - 1,
    ]
    thresholds = sorted_scores[last_at_score]
    true_positives = cumulative_tp[last_at_score]
    false_positives = cumulative_fp[last_at_score]
    positive_count = int(np.sum(y_true))
    precision = true_positives / (true_positives + false_positives)
    recall = (
        true_positives / positive_count
        if positive_count
        else np.full(thresholds.size, np.nan)
    )
    return PrecisionRecallCurve(thresholds, precision, recall)


def average_precision(y_true, scores):
    """Summarize a precision-recall curve as stepwise average precision.

    Returns ``nan`` when there are no positive examples, because recall is
    undefined in that case.
    """

    curve = precision_recall_curve(y_true, scores)
    if not np.all(np.isfinite(curve.recall)):
        return np.nan
    recall_before = np.r_[0.0, curve.recall[:-1]]
    return float(np.sum((curve.recall - recall_before) * curve.precision))


def best_f1_threshold(y_true, scores):
    """Choose a score threshold by F1, then accuracy, on the supplied rows.

    Evaluate every distinct observed score and the all-negative prediction.
    The caller must reserve separate data for an unbiased final evaluation.
    """

    y_true = _as_binary_labels(y_true, "y_true")
    scores = _as_scores(scores)
    _validate_same_size(y_true, scores, "y_true", "scores")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_true = y_true[order]
    last_at_score = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        sorted_scores.size - 1,
    ]
    true_positives = np.r_[0, np.cumsum(sorted_true)[last_at_score]]
    predicted_positives = np.r_[0, last_at_score + 1]
    positive_count = int(np.sum(y_true))
    false_positives = predicted_positives - true_positives
    false_negatives = positive_count - true_positives
    true_negatives = y_true.size - positive_count - false_positives
    denominators = 2 * true_positives + false_positives + false_negatives
    f1 = np.divide(
        2 * true_positives,
        denominators,
        out=np.zeros(denominators.size, dtype=float),
        where=denominators != 0,
    )
    accuracy = (true_positives + true_negatives) / y_true.size
    thresholds = np.r_[
        np.nextafter(sorted_scores[0], np.inf), sorted_scores[last_at_score]
    ]
    best_f1 = np.max(f1)
    tied = np.flatnonzero(np.isclose(f1, best_f1, rtol=0.0, atol=1e-15))
    best = int(tied[np.argmax(accuracy[tied])])
    return ThresholdChoice(
        float(thresholds[best]), float(f1[best]), float(accuracy[best])
    )


def binary_log_loss(y_true, probabilities, epsilon=1e-15):
    """Return mean binary cross-entropy from positive-class probabilities."""

    y_true = _as_binary_labels(y_true, "y_true")
    probabilities = _as_scores(probabilities)
    _validate_same_size(y_true, probabilities, "y_true", "probabilities")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be strictly between 0 and 0.5.")
    if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
        raise ValueError("probabilities must lie between 0 and 1.")
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(
            y_true * np.log(probabilities) + (1 - y_true) * np.log1p(-probabilities)
        )
    )
