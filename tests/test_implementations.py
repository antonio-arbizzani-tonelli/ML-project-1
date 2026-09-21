"""Independent contract and numerical tests for the required Project 1 methods."""

import ast
import inspect
from pathlib import Path
import unittest

import numpy as np

import implementations

Y = np.array([0.1, 0.3, 0.5])
TX = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
INITIAL_W = np.array([0.5, 1.0])
MAX_ITERS = 2
GAMMA = 0.1


class TestRequiredImplementations(unittest.TestCase):
    """Check public-reference values and additional numerical properties."""

    def test_required_signatures_and_docstrings(self):
        """Protect the exact function contracts expected by the grader."""

        expected_parameters = {
            "mean_squared_error_gd": (
                "y",
                "tx",
                "initial_w",
                "max_iters",
                "gamma",
            ),
            "mean_squared_error_sgd": (
                "y",
                "tx",
                "initial_w",
                "max_iters",
                "gamma",
            ),
            "least_squares": ("y", "tx"),
            "ridge_regression": ("y", "tx", "lambda_"),
            "logistic_regression": (
                "y",
                "tx",
                "initial_w",
                "max_iters",
                "gamma",
            ),
            "reg_logistic_regression": (
                "y",
                "tx",
                "lambda_",
                "initial_w",
                "max_iters",
                "gamma",
            ),
        }

        for function_name, parameter_names in expected_parameters.items():
            with self.subTest(function=function_name):
                function = getattr(implementations, function_name)
                actual_names = tuple(inspect.signature(function).parameters)
                self.assertEqual(actual_names, parameter_names)
                self.assertTrue(inspect.getdoc(function))

    def test_implementation_imports_follow_library_policy(self):
        """Allow only standard-library support and NumPy in the submission file."""

        implementation_path = Path(implementations.__file__)
        syntax_tree = ast.parse(implementation_path.read_text(encoding="utf-8"))
        imported_roots = set()

        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", maxsplit=1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", maxsplit=1)[0])

        self.assertEqual(imported_roots, {"numpy"})

    def test_mean_squared_error_gd_matches_public_reference(self):
        """Check the exact convention and post-update loss used by grading."""

        w, loss = implementations.mean_squared_error_gd(
            Y, TX, INITIAL_W, MAX_ITERS, GAMMA
        )

        np.testing.assert_allclose(w, [-0.050586, 0.203718], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(loss, 0.051534, rtol=1e-4, atol=1e-8)

    def test_mean_squared_error_gd_matches_zero_step_reference(self):
        """Return the supplied weights and their loss when no update is asked."""

        expected_w = np.array([0.413044, 0.875757])
        w, loss = implementations.mean_squared_error_gd(Y, TX, expected_w, 0, GAMMA)

        np.testing.assert_allclose(w, expected_w, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(loss, 2.959836, rtol=1e-4, atol=1e-8)
        self.assertEqual(w.shape, expected_w.shape)
        self.assertEqual(loss.ndim, 0)

    def test_mean_squared_error_sgd_matches_single_sample_reference(self):
        """A one-row dataset removes randomness from the SGD update."""

        w, loss = implementations.mean_squared_error_sgd(
            Y[:1], TX[:1], INITIAL_W, MAX_ITERS, GAMMA
        )

        np.testing.assert_allclose(w, [0.063058, 0.392080], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(loss, 0.844595, rtol=1e-4, atol=1e-8)

    def test_sgd_is_reproducible_when_numpy_seed_is_fixed(self):
        """Document the random-number contract inherited from Lab 2."""

        np.random.seed(7)
        first_w, first_loss = implementations.mean_squared_error_sgd(
            Y, TX, INITIAL_W, 10, 0.01
        )
        np.random.seed(7)
        second_w, second_loss = implementations.mean_squared_error_sgd(
            Y, TX, INITIAL_W, 10, 0.01
        )

        np.testing.assert_array_equal(first_w, second_w)
        np.testing.assert_equal(first_loss, second_loss)

    def test_least_squares_matches_reference_and_normal_equations(self):
        """Check reference values and the first-order optimality condition."""

        w, loss = implementations.least_squares(Y, TX)

        np.testing.assert_allclose(w, [0.218786, -0.053837], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(loss, 0.026942, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(TX.T @ (TX @ w - Y), 0.0, atol=1e-12)

    def test_ridge_regression_matches_public_references(self):
        """Check the zero-penalty and regularized normal equations."""

        cases = [
            (0.0, [0.218786, -0.053837], 0.026942),
            (1.0, [0.054303, 0.042713], 0.031750),
        ]
        for lambda_, expected_w, expected_loss in cases:
            with self.subTest(lambda_=lambda_):
                w, loss = implementations.ridge_regression(Y, TX, lambda_)
                np.testing.assert_allclose(w, expected_w, rtol=1e-4, atol=1e-8)
                np.testing.assert_allclose(loss, expected_loss, rtol=1e-4, atol=1e-8)

    def test_logistic_regression_matches_public_reference(self):
        """Check the mean-logistic-loss normalization used by grading."""

        binary_y = (Y > 0.2).astype(float)
        w, loss = implementations.logistic_regression(
            binary_y, TX, INITIAL_W, MAX_ITERS, GAMMA
        )

        np.testing.assert_allclose(w, [0.378561, 0.801131], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(loss, 1.348358, rtol=1e-4, atol=1e-8)

    def test_logistic_regression_matches_zero_step_reference(self):
        """Check the official no-update binary cross-entropy value."""

        binary_y = (Y > 0.2).astype(float)
        expected_w = np.array([0.463156, 0.939874])
        w, loss = implementations.logistic_regression(
            binary_y, TX, expected_w, 0, GAMMA
        )

        np.testing.assert_allclose(w, expected_w, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(loss, 1.533694, rtol=1e-4, atol=1e-8)
        self.assertEqual(w.shape, expected_w.shape)
        self.assertEqual(loss.ndim, 0)

    def test_regularized_logistic_regression_matches_public_reference(self):
        """Check the L2-gradient factor and unregularized returned loss."""

        binary_y = (Y > 0.2).astype(float)
        w, loss = implementations.reg_logistic_regression(
            binary_y, TX, 1.0, INITIAL_W, MAX_ITERS, GAMMA
        )

        np.testing.assert_allclose(w, [0.216062, 0.467747], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(loss, 0.972165, rtol=1e-4, atol=1e-8)

    def test_regularized_logistic_matches_zero_step_reference(self):
        """Confirm that the returned loss excludes the L2 penalty."""

        binary_y = (Y > 0.2).astype(float)
        expected_w = np.array([0.409111, 0.843996])
        w, loss = implementations.reg_logistic_regression(
            binary_y, TX, 1.0, expected_w, 0, GAMMA
        )

        np.testing.assert_allclose(w, expected_w, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(loss, 1.407327, rtol=1e-4, atol=1e-8)
        self.assertEqual(w.shape, expected_w.shape)
        self.assertEqual(loss.ndim, 0)

    def test_zero_iterations_preserves_weights_and_returns_current_loss(self):
        """All iterative methods define the zero-update case consistently."""

        binary_y = (Y > 0.2).astype(float)
        cases = [
            (
                implementations.mean_squared_error_gd,
                (Y, TX, INITIAL_W, 0, GAMMA),
            ),
            (
                implementations.mean_squared_error_sgd,
                (Y, TX, INITIAL_W, 0, GAMMA),
            ),
            (
                implementations.logistic_regression,
                (binary_y, TX, INITIAL_W, 0, GAMMA),
            ),
            (
                implementations.reg_logistic_regression,
                (binary_y, TX, 1.0, INITIAL_W, 0, GAMMA),
            ),
        ]

        for method, args in cases:
            with self.subTest(method=method.__name__):
                original_w = INITIAL_W.copy()
                w, loss = method(*args)
                np.testing.assert_array_equal(w, original_w)
                np.testing.assert_array_equal(INITIAL_W, original_w)
                self.assertIsInstance(w, np.ndarray)
                self.assertEqual(w.ndim, 1)
                self.assertEqual(np.ndim(loss), 0)

    def test_iterative_methods_do_not_modify_initial_weights(self):
        """Protect callers from aliasing side effects during updates."""

        initial_w = INITIAL_W.copy()
        implementations.mean_squared_error_gd(Y, TX, initial_w, 2, GAMMA)
        np.testing.assert_array_equal(initial_w, INITIAL_W)

    def test_mse_gradient_matches_central_finite_difference(self):
        """Numerically verify every component of the analytic MSE gradient."""

        w = np.array([0.2, -0.4])
        epsilon = 1e-6
        numerical = np.empty_like(w)
        for index in range(w.size):
            offset = np.zeros_like(w)
            offset[index] = epsilon
            upper = implementations._compute_mse(Y, TX, w + offset)
            lower = implementations._compute_mse(Y, TX, w - offset)
            numerical[index] = (upper - lower) / (2.0 * epsilon)

        analytic = implementations._compute_mse_gradient(Y, TX, w)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-9)

    def test_logistic_gradient_matches_central_finite_difference(self):
        """Numerically verify every component of the logistic gradient."""

        binary_y = (Y > 0.2).astype(float)
        w = np.array([0.2, -0.4])
        epsilon = 1e-6
        numerical = np.empty_like(w)
        for index in range(w.size):
            offset = np.zeros_like(w)
            offset[index] = epsilon
            upper = implementations._compute_logistic_loss(binary_y, TX, w + offset)
            lower = implementations._compute_logistic_loss(binary_y, TX, w - offset)
            numerical[index] = (upper - lower) / (2.0 * epsilon)

        analytic = implementations._compute_logistic_gradient(binary_y, TX, w)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-9)

    def test_logistic_loss_is_finite_for_extreme_scores(self):
        """Avoid overflow where direct log-of-sigmoid formulas fail."""

        extreme_tx = np.array([[1.0], [-1.0]])
        binary_y = np.array([1.0, 0.0])
        _, loss = implementations.logistic_regression(
            binary_y, extreme_tx, np.array([1000.0]), 0, 0.1
        )

        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
