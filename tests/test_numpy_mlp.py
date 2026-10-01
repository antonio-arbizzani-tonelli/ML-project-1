"""Numerical differentiation, nonlinear learning and stopping checks for the MLP."""

import unittest

import numpy as np

from src.numpy_mlp import MLPClassifier, sigmoid


class TestNumpyMLP(unittest.TestCase):
    def test_every_weight_and_bias_gradient_matches_finite_differences(self):
        model = MLPClassifier(hidden_layer_sizes=(3, 2), l2=0.07)
        model._initialize(2, 0.4, np.random.default_rng(5))
        model.weights_ = [w.astype(np.float64) for w in model.weights_]
        model.biases_ = [b.astype(np.float64) for b in model.biases_]
        for b in model.biases_[:-1]:
            b[:] = 2
        x = np.array([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2]], dtype=np.float64)
        y = np.array([0., 1., 1.])
        _, dw, db = model._loss_and_gradients(x, y)
        for parameter, gradient in zip(model.weights_ + model.biases_, dw + db):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + 1e-5
                positive = model._loss_and_gradients(x, y)[0]
                parameter[index] = original - 1e-5
                negative = model._loss_and_gradients(x, y)[0]
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (positive - negative) / 2e-5, places=7)

    def test_first_adam_step_matches_bias_corrected_formula(self):
        x = np.array([[0.2, 0.4], [0.7, 0.1], [0.5, 0.8]], dtype=np.float32)
        y = np.array([0, 1, 1], dtype=np.float32)
        model = MLPClassifier(hidden_layer_sizes=(3,), max_epochs=1, batch_size=10, random_seed=7)
        reference = MLPClassifier(hidden_layer_sizes=(3,), max_epochs=1, batch_size=10, random_seed=7)
        rng = np.random.default_rng(7)
        reference._initialize(2, float(np.mean(y)), rng)
        order = rng.permutation(y.size)
        _, dw, db = reference._loss_and_gradients(x[order], y[order])
        expected = [p - model.learning_rate * g / (np.sqrt(g * g) + 1e-8)
                    for p, g in zip(reference.weights_ + reference.biases_, dw + db)]
        model.fit(x, y)
        for actual, correct in zip(model.weights_ + model.biases_, expected):
            np.testing.assert_allclose(actual, correct, rtol=1e-6, atol=1e-7)

    def test_learns_xor_which_a_linear_classifier_cannot_separate(self):
        x = np.tile(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32), (16, 1))
        y = np.tile([0, 1, 1, 0], 16)
        fitted = MLPClassifier(hidden_layer_sizes=(8, 4), learning_rate=0.02,
                              max_epochs=150, batch_size=64, l2=0, random_seed=10).fit(x, y)
        np.testing.assert_array_equal(fitted.predict_proba(x) >= 0.5, y)
        self.assertLess(fitted.history_[-1]["training_bce"], 0.05)

    def test_stopping_restores_lowest_validation_bce_weights(self):
        x = np.tile(np.array([[-1], [1]], dtype=np.float32), (10, 1))
        y = np.tile([0, 1], 10)
        fitted = MLPClassifier(hidden_layer_sizes=(4,), learning_rate=0.03,
                              max_epochs=40, patience=2, min_delta=0, l2=0, random_seed=3).fit(x, y, x, 1 - y)
        self.assertLess(fitted.epochs_trained_, 40)
        best = min(fitted.history_, key=lambda entry: entry["validation_bce"])
        self.assertEqual(fitted.best_epoch_, best["epoch"])
        self.assertAlmostEqual(fitted._data_loss(x, (1 - y).astype(np.float32)), best["validation_bce"])

    def test_seed_and_repeated_fit_reset_adam_and_weights(self):
        x = np.arange(40, dtype=np.float32).reshape(20, 2) / 20
        y = np.tile([0, 1], 10)
        model = MLPClassifier(hidden_layer_sizes=(4, 2), max_epochs=3, batch_size=7)
        first = model.fit(x, y).predict_proba(x)
        second = model.fit(x, y).predict_proba(x)
        np.testing.assert_array_equal(first, second)

    def test_zero_training_channel_has_no_random_effect_when_first_seen(self):
        x = np.array([[0, -1], [0, 1], [0, -2], [0, 2]], dtype=np.float32)
        fitted = MLPClassifier(hidden_layer_sizes=(4,), max_epochs=3).fit(x, [0, 1, 0, 1])
        changed = x.copy()
        changed[:, 0] = 1
        np.testing.assert_array_equal(fitted.predict_proba(x), fitted.predict_proba(changed))

    def test_stable_sigmoid_and_bce_at_extreme_logits(self):
        np.testing.assert_allclose(sigmoid(np.array([-1000., 0., 1000.])), [0, 0.5, 1])
        model = MLPClassifier(hidden_layer_sizes=(1,), l2=0)
        model.weights_ = [np.ones((1, 1)), np.array([[1000.]])]
        model.biases_ = [np.zeros(1), np.zeros(1)]
        loss, dw, db = model._loss_and_gradients(np.array([[0.], [1.]]), np.array([0., 0.]))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.all(np.isfinite(g)) for g in dw + db))

    def test_rejects_bad_input_settings_and_unfitted_predictions(self):
        for parameters in ({"hidden_layer_sizes": ()}, {"batch_size": 0}, {"l2": -1}, {"learning_rate": float("nan")}):
            with self.assertRaises(ValueError):
                MLPClassifier(**parameters)
        with self.assertRaises(RuntimeError):
            MLPClassifier().predict_proba([[1]])
        for x, y in (([[np.nan]], [0]), ([[1]], [-1]), ([[1], [2]], [0])):
            with self.assertRaises(ValueError):
                MLPClassifier().fit(x, y)


if __name__ == "__main__":
    unittest.main()
