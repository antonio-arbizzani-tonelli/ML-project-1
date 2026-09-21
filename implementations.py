"""NumPy implementations of the six regression methods used in Project 1."""

import numpy as np


def _compute_mse(y, tx, w):
    """Return the mean squared error divided by two."""

    residuals = y - tx @ w
    return residuals @ residuals / (2.0 * y.size)


def _compute_mse_gradient(y, tx, w):
    """Return the gradient of the mean squared error."""

    residuals = y - tx @ w
    return -(tx.T @ residuals) / y.size


def _sigmoid(scores):
    """Evaluate the sigmoid while avoiding numerical overflow."""

    probabilities = np.empty_like(scores, dtype=float)
    non_negative = scores >= 0.0

    probabilities[non_negative] = 1.0 / (1.0 + np.exp(-scores[non_negative]))

    # For negative scores, this form avoids computing exp(-score).
    exp_scores = np.exp(scores[~non_negative])
    probabilities[~non_negative] = exp_scores / (1.0 + exp_scores)
    return probabilities


def _compute_logistic_loss(y, tx, w):
    """Return the mean logistic loss."""

    scores = tx @ w
    return np.mean(np.logaddexp(0.0, scores) - y * scores)


def _compute_logistic_gradient(y, tx, w):
    """Return the gradient of the mean logistic loss."""

    probabilities = _sigmoid(tx @ w)
    return tx.T @ (probabilities - y) / y.size


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Fit linear regression with full-batch gradient descent.

    Args:
        y: One-dimensional target array with shape ``(N,)``.
        tx: Design matrix with shape ``(N, D)``.
        initial_w: Initial one-dimensional weight array with shape ``(D,)``.
        max_iters: Number of gradient updates to perform.
        gamma: Constant gradient-descent step size.

    Returns:
        A tuple ``(w, loss)`` containing the final weights and their MSE,
        where MSE is ``||y - tx @ w||^2 / (2N)``.
    """

    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    w = np.array(initial_w, dtype=float, copy=True)

    for _ in range(max_iters):
        gradient = _compute_mse_gradient(y, tx, w)
        w -= gamma * gradient

    loss = _compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Fit linear regression with stochastic gradient descent.

    One training example is sampled uniformly with replacement at each
    iteration, so the mandatory mini-batch size is exactly one.  Calling
    ``np.random.seed`` before this function makes the sampling reproducible.

    Args:
        y: One-dimensional target array with shape ``(N,)``.
        tx: Design matrix with shape ``(N, D)``.
        initial_w: Initial one-dimensional weight array with shape ``(D,)``.
        max_iters: Number of stochastic-gradient updates to perform.
        gamma: Constant stochastic-gradient step size.

    Returns:
        A tuple ``(w, loss)`` containing the final weights and the full-data
        MSE evaluated at those weights.
    """

    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    w = np.array(initial_w, dtype=float, copy=True)

    for _ in range(max_iters):
        sample_index = np.random.randint(y.size)
        sample_y = y[sample_index : sample_index + 1]
        sample_tx = tx[sample_index : sample_index + 1]
        gradient = _compute_mse_gradient(sample_y, sample_tx, w)
        w -= gamma * gradient

    loss = _compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Fit least-squares regression through the normal equations.

    Args:
        y: One-dimensional target array with shape ``(N,)``.
        tx: Full-column-rank design matrix with shape ``(N, D)``.

    Returns:
        A tuple ``(w, loss)`` containing the normal-equation solution and its
        MSE.
    """

    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    gram_matrix = tx.T @ tx
    right_hand_side = tx.T @ y
    w = np.linalg.solve(gram_matrix, right_hand_side)
    loss = _compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Fit ridge regression through its regularized normal equations.

    The solved system is ``(X.T X + 2 N lambda I) w = X.T y``. The returned
    loss is the MSE without the regularization term.

    Args:
        y: One-dimensional target array with shape ``(N,)``.
        tx: Design matrix with shape ``(N, D)``.
        lambda_: L2 regularization parameter.

    Returns:
        A tuple ``(w, loss)`` containing the ridge solution and its
        unregularized MSE.
    """

    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    feature_count = tx.shape[1]
    penalty = 2.0 * y.size * lambda_ * np.eye(feature_count)
    gram_matrix = tx.T @ tx
    right_hand_side = tx.T @ y
    w = np.linalg.solve(gram_matrix + penalty, right_hand_side)
    loss = _compute_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit binary logistic regression with full-batch gradient descent.

    Args:
        y: Binary targets encoded as 0 or 1 with shape ``(N,)``.
        tx: Design matrix with shape ``(N, D)``.
        initial_w: Initial one-dimensional weight array with shape ``(D,)``.
        max_iters: Number of gradient updates to perform.
        gamma: Constant gradient-descent step size.

    Returns:
        A tuple ``(w, loss)`` containing the final weights and their mean
        binary cross-entropy.
    """

    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    w = np.array(initial_w, dtype=float, copy=True)

    for _ in range(max_iters):
        gradient = _compute_logistic_gradient(y, tx, w)
        w -= gamma * gradient

    loss = _compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit L2-regularized logistic regression with gradient descent.

    The optimized objective adds ``lambda_ * ||w||^2`` to mean logistic loss,
    so its gradient adds ``2 * lambda_ * w``. The returned loss does not
    include the regularization term.

    Args:
        y: Binary targets encoded as 0 or 1 with shape ``(N,)``.
        tx: Design matrix with shape ``(N, D)``.
        lambda_: L2 regularization parameter.
        initial_w: Initial one-dimensional weight array with shape ``(D,)``.
        max_iters: Number of gradient updates to perform.
        gamma: Constant gradient-descent step size.

    Returns:
        A tuple ``(w, loss)`` containing the final weights and their
        unregularized mean binary cross-entropy.
    """

    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    w = np.array(initial_w, dtype=float, copy=True)

    for _ in range(max_iters):
        data_gradient = _compute_logistic_gradient(y, tx, w)
        regularization_gradient = 2.0 * lambda_ * w
        w -= gamma * (data_gradient + regularization_gradient)

    loss = _compute_logistic_loss(y, tx, w)
    return w, loss
