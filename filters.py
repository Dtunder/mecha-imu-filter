import math

def _validate_number(val, name="Value"):
    """
    Validates that a value is a finite number (int or float).
    """
    if not isinstance(val, (int, float)):
        raise TypeError(f"{name} must be a number (int or float).")
    if not math.isfinite(val):
        raise ValueError(f"{name} cannot be NaN or infinity.")
    return float(val)


class KalmanFilter:
    """
    A simple 1D Kalman Filter.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2, estimated_error=1.0, initial_value=0.0):
        self.q = _validate_number(process_noise, "process_noise")
        self.r = _validate_number(measurement_noise, "measurement_noise")
        self.p = _validate_number(estimated_error, "estimated_error")
        self.x = _validate_number(initial_value, "initial_value")

        if self.q < 0:
            raise ValueError("process_noise must be non-negative.")
        if self.r <= 0:
            raise ValueError("measurement_noise must be strictly positive.")
        if self.p < 0:
            raise ValueError("estimated_error must be non-negative.")

    def update(self, val):
        """
        Update the Kalman filter with a new measurement.
        """
        val = _validate_number(val, "measurement value")

        # Prediction step
        self.p += self.q

        # Update step
        try:
            k = self.p / (self.p + self.r)
        except ZeroDivisionError:
            k = 0.0

        self.x += k * (val - self.x)
        self.p *= (1.0 - k)

        return self.x

    def filter(self, val):
        return self.update(val)


class ComplementaryFilter:
    """
    A Complementary Filter, typically used for sensor fusion (e.g., combining accelerometer and gyroscope).
    """
    def __init__(self, alpha=0.98, initial_value=0.0):
        self.alpha = _validate_number(alpha, "alpha")
        self.angle = _validate_number(initial_value, "initial_value")

        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0 inclusive.")
        
        self._inv_alpha = 1.0 - self.alpha

        self._inv_alpha = 1.0 - self.alpha

    def update(self, accel_angle, gyro_rate, dt):
        """
        Update the complementary filter.
        accel_angle: angle measured by accelerometer (or absolute measurement)
        gyro_rate: rate of change of angle measured by gyroscope (or relative measurement rate)
        dt: time delta
        """
        accel_angle = _validate_number(accel_angle, "accel_angle")
        gyro_rate = _validate_number(gyro_rate, "gyro_rate")
        dt = _validate_number(dt, "dt")

        if dt < 0:
            raise ValueError("dt must be non-negative.")

        self.angle = self.alpha * (self.angle + gyro_rate * dt) + self._inv_alpha * accel_angle
        return self.angle

    def filter(self, accel_angle, gyro_rate, dt):
        return self.update(accel_angle, gyro_rate, dt)


class LowPassFilter:
    """
    A simple exponential moving average Low-Pass Filter.
    """
    def __init__(self, alpha=0.5, initial_value=0.0):
        self.alpha = _validate_number(alpha, "alpha")
        self.value = _validate_number(initial_value, "initial_value")

        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0 inclusive.")
        
        self._inv_alpha = 1.0 - self.alpha

        self._inv_alpha = 1.0 - self.alpha

    def update(self, val):
        """
        Update the low pass filter with a new measurement.
        """
        val = _validate_number(val, "measurement value")

        self.value = self.alpha * val + self._inv_alpha * self.value
        return self.value

    def filter(self, val):
        return self.update(val)
