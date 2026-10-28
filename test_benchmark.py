import unittest

import numpy as np

from benchmark import DT, compute_metrics, compute_multiseed_metrics
from filters import AttitudeKalmanFilter


class TestBenchmark(unittest.TestCase):
    def test_default_metrics_are_deterministic(self):
        self.assertEqual(compute_metrics(), compute_metrics())

    def test_multiseed_metrics_are_deterministic_for_at_least_20_seeds(self):
        first = compute_multiseed_metrics()
        self.assertEqual(first, compute_multiseed_metrics())
        self.assertTrue(all(row["n_seeds"] >= 20 for row in first))

    def test_fused_filters_beat_raw_accelerometer_and_gyro_integration(self):
        metrics = {row["estimator"]: row for row in compute_metrics()}
        raw_rmse = metrics["Raw accelerometer"]["rmse_deg"]
        gyro_rmse = metrics["Gyro integration"]["rmse_deg"]

        self.assertLess(metrics["1-D Kalman (accel only)"]["rmse_deg"], raw_rmse)
        self.assertLess(metrics["Complementary"]["rmse_deg"], raw_rmse)
        self.assertLess(metrics["Attitude Kalman"]["rmse_deg"], raw_rmse)
        self.assertLess(metrics["Attitude Kalman"]["rmse_deg"], gyro_rmse)

    def test_attitude_kalman_recovers_constant_gyro_bias(self):
        rng = np.random.default_rng(1234)
        true_bias = 0.65
        attitude = AttitudeKalmanFilter(
            q_angle=0.000144,
            q_bias=0.005,
            r_measure=0.2**2,
            initial_angle=5.0,
            innovation_gate_sigma=4.0,
        )
        estimates = []
        for _ in range(10_000):
            estimates.append(
                attitude.update(
                    5.0 + float(rng.normal(0.0, 0.2)),
                    true_bias + float(rng.normal(0.0, 0.02)),
                    DT,
                )
            )

        self.assertAlmostEqual(attitude.bias, true_bias, delta=0.03)
        self.assertAlmostEqual(estimates[-1], 5.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
