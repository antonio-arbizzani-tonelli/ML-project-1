"""Verify that progress reporting does not change the fitted experiment."""

import contextlib
import io
import unittest

import numpy as np

from src.run_boosting_cv import _fit_checkpoint_probabilities
from src.run_logged_boosting_cv import _logged_fit


class TestLoggedBoostingCV(unittest.TestCase):
    def test_progress_checkpoints_preserve_predictions_and_callback_steps(self):
        rng = np.random.default_rng(34)
        x = rng.normal(size=(120, 4)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 0).astype(np.int8)
        parameters = {
            "max_depth": 2, "min_samples_leaf": 5, "n_bins": 8,
            "max_features": 2, "random_seed": 91,
        }
        expected, _ = _fit_checkpoint_probabilities(x[:80], y[:80], x[80:],
                                                   parameters, [30])
        profiles = []
        calls = []
        logged = _logged_fit(_fit_checkpoint_probabilities, profiles, 3)
        with contextlib.redirect_stdout(io.StringIO()):
            actual, _ = logged(x[:80], y[:80], x[80:], parameters, [30],
                               lambda step, model: calls.append(step))
        self.assertEqual(list(actual), [30])
        self.assertEqual(calls, [30])
        np.testing.assert_array_equal(actual[30], expected[30])
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["stage"], "outer")


if __name__ == "__main__":
    unittest.main()
