# mecha-imu-filter

Sensor filtering library (Kalman, Complementary, Lowpass) for IMU data.

## Overview

`mecha-imu-filter` is a lightweight Python library providing common filter implementations for Inertial Measurement Unit (IMU) data processing. It operates exclusively using the Python standard library, avoiding heavy external dependencies like NumPy for simplicity and portability.

## Installation

Clone the repository and run the scripts directly, or install locally if packaged. No external dependencies are required.

## Configuration Setup

The library uses a `config.py` module to handle settings, pulling from a local `config.json` file if it exists, or environment variables. It has sensible defaults.

Here is an example `config.json`:
```json
{
  "filters": {
    "kalman": {
      "process_noise": 1e-5,
      "measurement_noise": 1e-2,
      "estimated_error": 1.0
    },
    "complementary": {
      "alpha": 0.98
    },
    "lowpass": {
      "alpha": 0.5
    }
  },
  "resilience": {
    "max_retries": 3,
    "retry_delay": 0.1,
    "fallback_value": 0.0
  }
}
```

You can also override configuration values at runtime using environment variables. The pattern is `MECHA_<MODULE>_<PROPERTY>`:

```bash
# Overrides max_retries for Resilience
export MECHA_RESILIENCE_MAX_RETRIES=5

# Overrides process noise for the Kalman filter
export MECHA_KALMAN_PROCESS_NOISE=0.005
```

### Logging Configuration

The library uses the standard Python `logging` module. You can configure logging directly in your application or use the provided `setup_logging` helper function in `main.py`.

```python
from main import setup_logging
import logging

# Setup standard INFO level logging
setup_logging(level=logging.INFO)

# Or for more verbose output:
setup_logging(level=logging.DEBUG)
```

## CLI Instructions

You can run the main entry point to verify the setup:

```bash
python main.py
```

Expected output:
```
Hello from mecha-imu-filter!
```

To run the test suite and check coverage (requires `coverage` package):

```bash
coverage run -m unittest discover && coverage report -m
```

## API Reference

This library provides three standard filters inside `filters.py`:

### KalmanFilter

A 1D Kalman filter to smooth noisy signals, estimating the state of a dynamic system from a series of incomplete and noisy measurements.

**Constructor:**
`KalmanFilter(process_noise=None, measurement_noise=None, estimated_error=None, initial_value=0.0)`
If standard parameters are not given, the values fall back to the configurations handled in `config.py`.
- `process_noise` (float): The process noise variance (q). Must be >= 0.
- `measurement_noise` (float): The measurement noise variance (r). Must be > 0.
- `estimated_error` (float): The initial estimate error variance (p). Must be >= 0.
- `initial_value` (float): The initial state estimate (x).

**Methods:**
- `update(val)` / `filter(val)`: Updates the filter state with a new measurement `val`. Returns the updated state estimate.

**Usage:**
```python
from filters import KalmanFilter

# Initialize filter
kf = KalmanFilter(process_noise=1e-5, measurement_noise=1e-2, estimated_error=1.0, initial_value=0.0)

# Update filter with a new measurement
filtered_val = kf.update(10.5) 
# or kf.filter(10.5)
```

### ComplementaryFilter

A sensor fusion filter typically used to combine accelerometer and gyroscope data. It combines a high-pass filter (for gyroscope data) and a low-pass filter (for accelerometer data) to estimate a more stable angle.

**Constructor:**
`ComplementaryFilter(alpha=None, initial_value=0.0)`
If standard parameters are not given, the values fall back to the configurations handled in `config.py`.
- `alpha` (float): The filter coefficient weighting the gyroscope integration. Must be between 0.0 and 1.0 inclusive.
- `initial_value` (float): The initial angle estimate.

**Methods:**
- `update(accel_angle, gyro_rate, dt)` / `filter(accel_angle, gyro_rate, dt)`: Updates the filter state.
    - `accel_angle` (float): Angle measured by the accelerometer.
    - `gyro_rate` (float): Rate of change of angle measured by the gyroscope.
    - `dt` (float): Time delta since the last update.
    Returns the updated angle estimate.

**Usage:**
```python
from filters import ComplementaryFilter

# Initialize filter
cf = ComplementaryFilter(alpha=0.98, initial_value=0.0)

# Update filter with accelerometer angle and gyroscope rate
accel_angle = 10.0
gyro_rate = 5.0
dt = 0.01

filtered_angle = cf.update(accel_angle, gyro_rate, dt)
# or cf.filter(accel_angle, gyro_rate, dt)
```

### LowPassFilter

A simple exponential moving average filter that smooths data by applying a weighting factor to recent measurements versus previous estimates.

**Constructor:**
`LowPassFilter(alpha=None, initial_value=0.0)`
If standard parameters are not given, the values fall back to the configurations handled in `config.py`.
- `alpha` (float): The smoothing factor (weight) given to the new measurement. Must be between 0.0 and 1.0 inclusive.
- `initial_value` (float): The initial value estimate.

**Methods:**
- `update(val)` / `filter(val)`: Updates the low-pass filter with a new measurement `val`. Returns the updated filtered value.

**Usage:**
```python
from filters import LowPassFilter

# Initialize filter
lpf = LowPassFilter(alpha=0.1, initial_value=0.0)

# Update filter with a new measurement
filtered_val = lpf.update(10.5)
# or lpf.filter(10.5)
```
