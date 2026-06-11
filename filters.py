import logging
import math

logger = logging.getLogger(__name__)


def _validate_number(val: int | float, name: str = "Value") -> float:
    """
    Validates that a value is a finite number (int or float).

    Args:
        val (int or float): The value to validate.
        name (str, optional): The name of the variable being validated, used in error messages. Defaults to "Value".

    Returns:
        float: The validated number cast to a float.

    Raises:
        TypeError: If the value is not an int or float.
        ValueError: If the value is NaN or infinity.
    """
    if not isinstance(val, (int, float)):
        logger.error("Validation failed: %s is not a number. Value: %s", name, val)
        raise TypeError(f"{name} must be a number (int or float).")
    if not math.isfinite(val):
        logger.error("Validation failed: %s is NaN or infinity. Value: %s", name, val)
        raise ValueError(f"{name} cannot be NaN or infinity.")
    return float(val)


class KalmanFilter:
    """
    A simple 1D Kalman Filter.

    This filter estimates the state of a dynamic system from a series of incomplete and noisy measurements.

    Attributes:
        q (float): The process noise covariance.
        r (float): The measurement noise covariance.
        p (float): The estimation error covariance.
        x (float): The estimated state value.
    """

    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-2,
        estimated_error: float = 1.0,
        initial_value: float = 0.0,
    ) -> None:
        """
        Initializes the Kalman filter with process and measurement noise.

        Args:
            process_noise (float, optional): The process noise variance (q). Must be >= 0. Defaults to 1e-5.
            measurement_noise (float, optional): The measurement noise variance (r). Must be > 0. Defaults to 1e-2.
            estimated_error (float, optional): The initial estimate error variance (p). Must be >= 0. Defaults to 1.0.
            initial_value (float, optional): The initial state estimate (x). Defaults to 0.0.

        Raises:
            ValueError: If process_noise < 0, measurement_noise <= 0, or estimated_error < 0.
        """
        self.q = _validate_number(process_noise, "process_noise")
        self.r = _validate_number(measurement_noise, "measurement_noise")
        self.p = _validate_number(estimated_error, "estimated_error")
        self.x = _validate_number(initial_value, "initial_value")

        if self.q < 0:
            logger.error(
                "Initialization failed: process_noise=%s must be non-negative.", self.q
            )
            raise ValueError("process_noise must be non-negative.")
        if self.r <= 0:
            logger.error(
                "Initialization failed: measurement_noise=%s must be strictly positive.",
                self.r,
            )
            raise ValueError("measurement_noise must be strictly positive.")
        if self.p < 0:
            logger.error(
                "Initialization failed: estimated_error=%s must be non-negative.",
                self.p,
            )
            raise ValueError("estimated_error must be non-negative.")

        logger.info(
            "KalmanFilter initialized with q=%s, r=%s, p=%s, x=%s",
            self.q,
            self.r,
            self.p,
            self.x,
        )

    def update(self, val: float) -> float:
        """
        Updates the Kalman filter state with a new measurement.

        Args:
            val (float): The new measurement value.

        Returns:
            float: The updated state estimate.
        """
        val = _validate_number(val, "measurement value")

        # Prediction step
        self.p += self.q

        # Update step
        try:
            k = self.p / (self.p + self.r)
        except ZeroDivisionError:
            logger.warning(
                "ZeroDivisionError in Kalman filter update, p=%s, r=%s. Setting kalman gain to 0.",
                self.p,
                self.r,
            )
            k = 0.0

        self.x += k * (val - self.x)
        self.p *= 1.0 - k

        logger.debug(
            "KalmanFilter updated: new measurement=%s, gain=%s, x=%s, p=%s",
            val,
            k,
            self.x,
            self.p,
        )

        return self.x

    def filter(self, val: float) -> float:
        """
        Alias for update(). Filters a new measurement value.

        Args:
            val (float): The new measurement value.

        Returns:
            float: The updated state estimate.
        """
        return self.update(val)


class ComplementaryFilter:
    """
    A Complementary Filter, typically used for sensor fusion (e.g., combining accelerometer and gyroscope).

    This filter combines a high-pass filter (for gyroscope data) and a low-pass filter (for accelerometer data)
    to estimate a more stable angle.

    Attributes:
        alpha (float): The filter coefficient weighting the gyroscope integration.
        angle (float): The estimated angle.
    """

    def __init__(self, alpha: float = 0.98, initial_value: float = 0.0) -> None:
        """
        Initializes the Complementary filter.

        Args:
            alpha (float, optional): The filter coefficient (between 0.0 and 1.0 inclusive). Defaults to 0.98.
            initial_value (float, optional): The initial angle estimate. Defaults to 0.0.

        Raises:
            ValueError: If alpha is not between 0.0 and 1.0 inclusive.
        """
        self.alpha = _validate_number(alpha, "alpha")
        self.angle = _validate_number(initial_value, "initial_value")

        if not (0.0 <= self.alpha <= 1.0):
            logger.error(
                "Initialization failed: alpha=%s must be between 0.0 and 1.0 inclusive.",
                self.alpha,
            )
            raise ValueError("alpha must be between 0.0 and 1.0 inclusive.")

        self._inv_alpha = 1.0 - self.alpha
        logger.info(
            "ComplementaryFilter initialized with alpha=%s, angle=%s",
            self.alpha,
            self.angle,
        )

    def update(self, accel_angle: float, gyro_rate: float, dt: float) -> float:
        """
        Updates the complementary filter state.

        Args:
            accel_angle (float): Angle measured by the accelerometer (absolute measurement).
            gyro_rate (float): Rate of change of angle measured by the gyroscope (relative measurement rate).
            dt (float): Time delta since the last update.

        Returns:
            float: The updated angle estimate.

        Raises:
            ValueError: If dt is negative.
        """
        accel_angle = _validate_number(accel_angle, "accel_angle")
        gyro_rate = _validate_number(gyro_rate, "gyro_rate")
        dt = _validate_number(dt, "dt")

        if dt < 0:
            logger.error("Update failed: dt=%s must be non-negative.", dt)
            raise ValueError("dt must be non-negative.")

        self.angle = (
            self.alpha * (self.angle + gyro_rate * dt) + self._inv_alpha * accel_angle
        )
        logger.debug(
            "ComplementaryFilter updated: accel_angle=%s, gyro_rate=%s, dt=%s, new angle=%s",
            accel_angle,
            gyro_rate,
            dt,
            self.angle,
        )
        return self.angle

    def filter(self, accel_angle: float, gyro_rate: float, dt: float) -> float:
        """
        Alias for update(). Filters the sensor data.

        Args:
            accel_angle (float): Angle measured by the accelerometer.
            gyro_rate (float): Rate of change of angle measured by the gyroscope.
            dt (float): Time delta since the last update.

        Returns:
            float: The updated angle estimate.
        """
        return self.update(accel_angle, gyro_rate, dt)


class LowPassFilter:
    """
    A simple exponential moving average Low-Pass Filter.

    This filter smooths data by applying a weighting factor to recent measurements versus previous estimates.

    Attributes:
        alpha (float): The smoothing factor (weight) given to the new measurement.
        value (float): The current filtered value.
    """

    def __init__(self, alpha: float = 0.5, initial_value: float = 0.0) -> None:
        """
        Initializes the Low-Pass filter.

        Args:
            alpha (float, optional): The smoothing factor (between 0.0 and 1.0 inclusive). Defaults to 0.5.
            initial_value (float, optional): The initial value estimate. Defaults to 0.0.

        Raises:
            ValueError: If alpha is not between 0.0 and 1.0 inclusive.
        """
        self.alpha = _validate_number(alpha, "alpha")
        self.value = _validate_number(initial_value, "initial_value")

        if not (0.0 <= self.alpha <= 1.0):
            logger.error(
                "Initialization failed: alpha=%s must be between 0.0 and 1.0 inclusive.",
                self.alpha,
            )
            raise ValueError("alpha must be between 0.0 and 1.0 inclusive.")

        self._inv_alpha = 1.0 - self.alpha
        logger.info(
            "LowPassFilter initialized with alpha=%s, value=%s", self.alpha, self.value
        )

    def update(self, val: float) -> float:
        """
        Updates the low-pass filter with a new measurement.

        Args:
            val (float): The new measurement value.

        Returns:
            float: The updated filtered value.
        """
        val = _validate_number(val, "measurement value")

        self.value = self.alpha * val + self._inv_alpha * self.value
        logger.debug(
            "LowPassFilter updated: new measurement=%s, new value=%s", val, self.value
        )
        return self.value

    def filter(self, val: float) -> float:
        """
        Alias for update(). Filters a new measurement value.

        Args:
            val (float): The new measurement value.

        Returns:
            float: The updated filtered value.
        """
        return self.update(val)
