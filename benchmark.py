"""Reproducible synthetic benchmark for the scalar IMU angle filters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from filters import (
    AttitudeKalmanFilter,
    ComplementaryFilter,
    KalmanFilter,
    LowPassFilter,
)

DEFAULT_SEED = 20261028
SAMPLE_RATE_HZ = 100.0
DURATION_SECONDS = 20.0
DT = 1.0 / SAMPLE_RATE_HZ
METRIC_FIELDS = ("estimator", "rmse_deg", "max_abs_error_deg", "lag_seconds")
MULTISEED_FIELDS = ("estimator", "n_seeds", "rmse_mean_deg", "rmse_std_deg")
MULTISEED_SEEDS = tuple(range(DEFAULT_SEED, DEFAULT_SEED + 20))

# Fixed from a coarse grid scored on three disjoint tuning seeds, never the
# evaluation seed(s). q_angle starts from the discrete gyro-noise variance
# (0.12 deg/s)^2 * 0.01 s = 0.000144 deg²/s; q_bias was selected on the grid.
ATTITUDE_KALMAN_PARAMETERS = {
    "q_angle": 0.000144,
    "q_bias": 0.005,
    "r_measure": 2.2**2,
    "innovation_gate_sigma": 4.0,
}


def _make_scenario(seed: int) -> tuple[np.ndarray, ...]:
    """Return time, true angle, accel angle, and gyro rate in degrees/seconds."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, DURATION_SECONDS, DT)

    # Smooth, known tilt trajectory with two frequencies and an analytic rate.
    f1_hz, f2_hz = 0.12, 0.035
    angle = 20.0 * np.sin(2.0 * np.pi * f1_hz * time)
    angle += 6.0 * np.sin(2.0 * np.pi * f2_hz * time)
    true_rate = 20.0 * 2.0 * np.pi * f1_hz * np.cos(2.0 * np.pi * f1_hz * time)
    true_rate += 6.0 * 2.0 * np.pi * f2_hz * np.cos(2.0 * np.pi * f2_hz * time)

    accel_angle = angle + rng.normal(0.0, 2.2, size=time.size)
    vibration_a = (time >= 4.0) & (time < 4.4)
    vibration_b = (time >= 13.0) & (time < 13.25)
    accel_angle[vibration_a] += 7.0 * np.sin(2.0 * np.pi * 18.0 * (time[vibration_a] - 4.0))
    accel_angle[vibration_b] += 5.0 * np.sin(2.0 * np.pi * 24.0 * (time[vibration_b] - 13.0))
    accel_angle[np.argmin(np.abs(time - 10.0))] += 25.0
    accel_angle[np.argmin(np.abs(time - 10.0)) + 1] -= 18.0

    gyro_bias_and_drift = 0.35 + 0.025 * time
    gyro_rate = true_rate + gyro_bias_and_drift + rng.normal(0.0, 0.12, size=time.size)
    return time, angle, accel_angle, gyro_rate


def _estimate_lag(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Find the correlation lag in seconds; positive means estimate lags truth."""
    max_shift = min(int(round(0.5 / DT)), reference.size - 2)
    best_shift = 0
    best_score = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        if shift > 0:
            expected = reference[:-shift]
            observed = estimate[shift:]
        elif shift < 0:
            expected = reference[-shift:]
            observed = estimate[:shift]
        else:
            expected = reference
            observed = estimate

        expected_centered = expected - np.mean(expected)
        observed_centered = observed - np.mean(observed)
        denominator = np.linalg.norm(expected_centered) * np.linalg.norm(observed_centered)
        score = float(np.dot(expected_centered, observed_centered) / denominator)
        if score > best_score or (score == best_score and abs(shift) < abs(best_shift)):
            best_score = score
            best_shift = shift
    return best_shift * DT


def _run(
    seed: int,
    calculate_lag: bool = True,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    time, truth, accel_angle, gyro_rate = _make_scenario(seed)
    estimates: dict[str, np.ndarray] = {"Raw accelerometer": accel_angle.copy()}

    gyro_integrated = np.empty_like(truth)
    gyro_integrated[0] = truth[0]
    gyro_integrated[1:] = truth[0] + np.cumsum(
        0.5 * (gyro_rate[:-1] + gyro_rate[1:]) * DT
    )
    estimates["Gyro integration"] = gyro_integrated

    low_pass = LowPassFilter(alpha=0.18, initial_value=float(truth[0]))
    estimates["Low-pass"] = np.array([low_pass.update(float(value)) for value in accel_angle])

    complementary = ComplementaryFilter(alpha=0.98, initial_value=float(truth[0]))
    estimates["Complementary"] = np.array(
        [
            complementary.update(float(accel), float(rate), DT)
            for accel, rate in zip(accel_angle, gyro_rate)
        ]
    )

    # These benchmark-specific values match the synthetic sensor noise scale.
    kalman = KalmanFilter(
        process_noise=0.15,
        measurement_noise=2.2**2,
        estimated_error=2.2**2,
        initial_value=float(truth[0]),
    )
    estimates["1-D Kalman (accel only)"] = np.array(
        [kalman.update(float(value)) for value in accel_angle]
    )

    attitude_kalman = AttitudeKalmanFilter(
        **ATTITUDE_KALMAN_PARAMETERS,
        initial_angle=float(truth[0]),
    )
    estimates["Attitude Kalman"] = np.array(
        [
            attitude_kalman.update(float(accel), float(rate), DT)
            for accel, rate in zip(accel_angle, gyro_rate)
        ]
    )

    metrics: list[dict[str, Any]] = []
    for name, estimate in estimates.items():
        errors = estimate - truth
        metrics.append(
            {
                "estimator": name,
                "rmse_deg": float(np.sqrt(np.mean(np.square(errors)))),
                "max_abs_error_deg": float(np.max(np.abs(errors))),
                "lag_seconds": _estimate_lag(truth, estimate) if calculate_lag else 0.0,
            }
        )
    return metrics, time, truth, estimates


def compute_metrics(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Compute deterministic benchmark metrics without writing files."""
    metrics, _, _, _ = _run(seed)
    return metrics


def compute_multiseed_metrics(
    seeds: tuple[int, ...] = MULTISEED_SEEDS,
) -> list[dict[str, Any]]:
    """Return population mean and standard deviation of RMSE across seeds."""
    if not seeds:
        raise ValueError("seeds must contain at least one evaluation seed.")
    rmse_by_estimator: dict[str, list[float]] = {}
    for seed in seeds:
        seed_metrics, _, _, _ = _run(seed, calculate_lag=False)
        for row in seed_metrics:
            rmse_by_estimator.setdefault(row["estimator"], []).append(row["rmse_deg"])
    return [
        {
            "estimator": name,
            "n_seeds": len(seeds),
            "rmse_mean_deg": float(np.mean(values)),
            "rmse_std_deg": float(np.std(values, ddof=0)),
        }
        for name, values in rmse_by_estimator.items()
    ]


def write_results(output_dir: str | Path = "results", seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Write single-seed/multi-seed metrics and the single-seed figure."""
    metrics, time, truth, estimates = _run(seed)
    multiseed_metrics = compute_multiseed_metrics()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    "estimator": row["estimator"],
                    "rmse_deg": f"{row['rmse_deg']:.6f}",
                    "max_abs_error_deg": f"{row['max_abs_error_deg']:.6f}",
                    "lag_seconds": f"{row['lag_seconds']:.6f}",
                }
            )

    with (destination / "metrics_multiseed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MULTISEED_FIELDS)
        writer.writeheader()
        for row in multiseed_metrics:
            writer.writerow(
                {
                    "estimator": row["estimator"],
                    "n_seeds": row["n_seeds"],
                    "rmse_mean_deg": f"{row['rmse_mean_deg']:.6f}",
                    "rmse_std_deg": f"{row['rmse_std_deg']:.6f}",
                }
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (trajectory_axis, error_axis, impulse_axis) = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [2, 1, 1]},
        layout="constrained",
    )
    # Only the two full-run panels share the time axis; the impulse panel is a zoom.
    error_axis.sharex(trajectory_axis)
    trajectory_axis.plot(time, truth, color="black", linewidth=2.2, label="Ground truth")
    for name, estimate in estimates.items():
        trajectory_axis.plot(time, estimate, linewidth=1.0, alpha=0.82, label=name)
        error_axis.plot(time, np.abs(estimate - truth), linewidth=0.9, label=name)
        impulse_axis.plot(
            time,
            np.abs(estimate - truth),
            linewidth=1.0,
            alpha=0.9,
            label=name,
        )
    trajectory_axis.axvspan(4.0, 4.4, color="gray", alpha=0.12, label="Vibration")
    trajectory_axis.axvline(10.0, color="gray", linestyle=":", linewidth=1.0, label="Impulse")
    trajectory_axis.set_ylabel("Tilt angle (degrees)")
    trajectory_axis.set_title(f"Synthetic IMU tilt estimation (seed {seed})")
    trajectory_axis.grid(True, alpha=0.25)
    trajectory_axis.legend(ncol=3, fontsize=8, loc="upper right")
    error_axis.set_xlabel("Time (seconds)")
    error_axis.set_ylabel("Absolute error (degrees)")
    error_axis.grid(True, alpha=0.25)
    error_axis.legend(ncol=3, fontsize=8, loc="upper right")
    impulse_axis.set_xlim(9.95, 10.10)
    impulse_axis.set_xlabel("Time (seconds)")
    impulse_axis.set_ylabel("Absolute error (degrees)")
    impulse_axis.set_title("Impulse detail at t = 10 s")
    impulse_axis.grid(True, alpha=0.25)
    impulse_axis.legend(ncol=3, fontsize=8, loc="upper right")
    figure.savefig(destination / "filter_comparison.png", dpi=170)
    plt.close(figure)
    return metrics


if __name__ == "__main__":
    for result in write_results():
        print(
            f"{result['estimator']:<20} "
            f"RMSE={result['rmse_deg']:.4f} deg  "
            f"max={result['max_abs_error_deg']:.4f} deg  "
            f"lag={result['lag_seconds']:.3f} s"
        )
