"""Regression tests for finite-but-large Kalman inputs.

All inputs are valid finite floats. The mathematical update must stay finite
when the intermediate sums or differences would overflow binary64.
"""
import math
import unittest

from filters import KalmanFilter


class TestKalmanNumerics(unittest.TestCase):
    def test_equal_large_covariances_keep_half_gain(self):
        kf = KalmanFilter(
            process_noise=0.0,
            measurement_noise=1e308,
            estimated_error=1e308,
        )
        result = kf.update(10.0)
        self.assertTrue(math.isfinite(result))
        self.assertAlmostEqual(result, 5.0, places=12)
        self.assertAlmostEqual(kf.p / 1e308, 0.5, places=12)

    def test_opposite_large_measurements_do_not_overflow_difference(self):
        kf = KalmanFilter(
            process_noise=0.0,
            measurement_noise=1.0,
            estimated_error=1.0,
            initial_value=1e308,
        )
        result = kf.update(-1e308)
        self.assertTrue(math.isfinite(result))
        self.assertAlmostEqual(result, 0.0)

    def test_prediction_overflow_fails_without_corrupting_state(self):
        kf = KalmanFilter(
            process_noise=1e308,
            measurement_noise=1.0,
            estimated_error=1e308,
            initial_value=7.0,
        )
        with self.assertRaisesRegex(ValueError, "covariance overflowed"):
            kf.update(10.0)
        self.assertEqual(kf.x, 7.0)
        self.assertEqual(kf.p, 1e308)

    def test_zero_prior_covariance_preserves_estimate(self):
        kf = KalmanFilter(
            process_noise=0.0,
            measurement_noise=1.0,
            estimated_error=0.0,
            initial_value=3.0,
        )
        self.assertEqual(kf.update(10.0), 3.0)
        self.assertEqual(kf.p, 0.0)


if __name__ == "__main__":
    unittest.main()
