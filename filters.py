class KalmanFilter:
    """
    A simple 1D Kalman Filter.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2, estimated_error=1.0, initial_value=0.0):
        self.q = process_noise
        self.r = measurement_noise
        self.p = estimated_error
        self.x = initial_value

    def update(self, val):
        """
        Update the Kalman filter with a new measurement.
        """
        # Prediction step
        self.p = self.p + self.q

        # Update step
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (val - self.x)
        self.p = (1 - k) * self.p

        return self.x

    def filter(self, val):
        return self.update(val)


class ComplementaryFilter:
    """
    A Complementary Filter, typically used for sensor fusion (e.g., combining accelerometer and gyroscope).
    """
    def __init__(self, alpha=0.98, initial_value=0.0):
        self.alpha = alpha
        self.angle = initial_value

    def update(self, accel_angle, gyro_rate, dt):
        """
        Update the complementary filter.
        accel_angle: angle measured by accelerometer (or absolute measurement)
        gyro_rate: rate of change of angle measured by gyroscope (or relative measurement rate)
        dt: time delta
        """
        self.angle = self.alpha * (self.angle + gyro_rate * dt) + (1.0 - self.alpha) * accel_angle
        return self.angle

    def filter(self, accel_angle, gyro_rate, dt):
        return self.update(accel_angle, gyro_rate, dt)


class LowPassFilter:
    """
    A simple exponential moving average Low-Pass Filter.
    """
    def __init__(self, alpha=0.5, initial_value=0.0):
        self.alpha = alpha
        self.value = initial_value

    def update(self, val):
        """
        Update the low pass filter with a new measurement.
        """
        self.value = self.alpha * val + (1.0 - self.alpha) * self.value
        return self.value

    def filter(self, val):
        return self.update(val)
