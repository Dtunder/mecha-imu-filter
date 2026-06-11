# mecha-imu-filter

Sensor filtering library (Kalman, Complementary, Lowpass) for IMU data.

## Classes Available

This library provides three standard filters inside `filters.py`:

- **KalmanFilter**: A 1D Kalman filter to smooth noisy signals.
- **ComplementaryFilter**: A sensor fusion filter typically used to combine accelerometer and gyroscope data.
- **LowPassFilter**: A simple exponential moving average filter.

## Usage Example

### KalmanFilter

```python
from filters import KalmanFilter

# Initialize filter
kf = KalmanFilter(process_noise=1e-5, measurement_noise=1e-2, estimated_error=1.0, initial_value=0.0)

# Update filter with a new measurement
filtered_val = kf.update(10.5) 
# or kf.filter(10.5)
```

### ComplementaryFilter

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

```python
from filters import LowPassFilter

# Initialize filter
lpf = LowPassFilter(alpha=0.1, initial_value=0.0)

# Update filter with a new measurement
filtered_val = lpf.update(10.5)
# or lpf.filter(10.5)
```
