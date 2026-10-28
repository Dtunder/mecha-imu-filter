import logging
import math

from config import settings

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
        process_noise: float | None = None,
        measurement_noise: float | None = None,
        estimated_error: float | None = None,
        initial_value: float = 0.0,
    ) -> None:
        if process_noise is None:
            process_noise = settings["filters"]["kalman"]["process_noise"]
        if measurement_noise is None:
            measurement_noise = settings["filters"]["kalman"]["measurement_noise"]
        if estimated_error is None:
            estimated_error = settings["filters"]["kalman"]["estimated_error"]
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
        predicted_p = self.p + self.q
        if not math.isfinite(predicted_p):
            raise ValueError("Kalman prediction covariance overflowed; state unchanged.")
        self.p = predicted_p

        # Update step
        try:
            # Compute the gain without overflowing p + r for large finite covariances.
            if self.p >= self.r:
                k = 1.0 / (1.0 + self.r / self.p)
            else:
                ratio = self.p / self.r
                k = ratio / (1.0 + ratio)
        except ZeroDivisionError:
            logger.warning(
                "ZeroDivisionError in Kalman filter update, p=%s, r=%s. Setting kalman gain to 0.",
                self.p,
                self.r,
            )
            k = 0.0

        # A weighted average avoids overflow in (val - x) when both are finite.
        self.x = (1.0 - k) * self.x + k * val
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


class AttitudeKalmanFilter:
    """Two-state angle and gyro-bias Kalman filter for a single tilt axis.

    Angles are in degrees, gyro rates in degrees/second, and ``dt`` in seconds.
    ``q_angle`` and ``q_bias`` are continuous-time process-noise intensities in
    degree²/second and (degree/second)²/second, respectively. ``r_measure`` is
    the accelerometer-angle measurement variance in degree².

    Set ``innovation_gate_sigma`` to a positive value to reject accelerometer
    updates whose innovation is farther than that many predicted standard
    deviations from the estimate. The default ``None`` disables gating.
    """

    def __init__(
        self,
        q_angle: float = 0.00144,
        q_bias: float = 0.0001,
        r_measure: float = 4.84,
        initial_angle: float = 0.0,
        initial_bias: float = 0.0,
        initial_angle_variance: float = 1.0,
        initial_bias_variance: float = 1.0,
        innovation_gate_sigma: float | None = None,
    ) -> None:
        self.q_angle = _validate_number(q_angle, "q_angle")
        self.q_bias = _validate_number(q_bias, "q_bias")
        self.r_measure = _validate_number(r_measure, "r_measure")
        self.angle = _validate_number(initial_angle, "initial_angle")
        self.bias = _validate_number(initial_bias, "initial_bias")
        self._p00 = _validate_number(initial_angle_variance, "initial_angle_variance")
        self._p11 = _validate_number(initial_bias_variance, "initial_bias_variance")
        self._p01 = 0.0
        self.innovation_gate_sigma = (
            None
            if innovation_gate_sigma is None
            else _validate_number(innovation_gate_sigma, "innovation_gate_sigma")
        )
        if self.q_angle < 0.0:
            raise ValueError("q_angle must be non-negative.")
        if self.q_bias < 0.0:
            raise ValueError("q_bias must be non-negative.")
        if self.r_measure <= 0.0:
            raise ValueError("r_measure must be strictly positive.")
        if self._p00 < 0.0 or self._p11 < 0.0:
            raise ValueError("initial covariance variances must be non-negative.")
        if self.innovation_gate_sigma is not None and self.innovation_gate_sigma <= 0.0:
            raise ValueError("innovation_gate_sigma must be strictly positive or None.")
        self.last_measurement_accepted = True

    def update(self, accel_angle: float, gyro_rate: float, dt: float) -> float:
        """Predict from the gyro, then correct with an accelerometer angle."""
        accel_angle = _validate_number(accel_angle, "accel_angle")
        gyro_rate = _validate_number(gyro_rate, "gyro_rate")
        dt = _validate_number(dt, "dt")
        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        # State prediction: theta' = theta + (gyro_rate - bias) * dt.
        angle_prior = self.angle + (gyro_rate - self.bias) * dt
        p00_prior = (
            self._p00
            - dt * (self._p01 + self._p01)
            + dt * dt * self._p11
            + self.q_angle * dt
        )
        p01_prior = self._p01 - dt * self._p11
        p11_prior = self._p11 + self.q_bias * dt
        innovation = accel_angle - angle_prior
        innovation_variance = p00_prior + self.r_measure
        predicted_values = (
            angle_prior,
            p00_prior,
            p01_prior,
            p11_prior,
            innovation,
            innovation_variance,
        )
        if not all(math.isfinite(value) for value in predicted_values) or innovation_variance <= 0:
            raise ValueError("Attitude Kalman prediction became non-finite; state unchanged.")

        rejected = False
        if self.innovation_gate_sigma is not None:
            normalized_innovation = abs(innovation) / math.sqrt(innovation_variance)
            rejected = normalized_innovation > self.innovation_gate_sigma

        if rejected:
            self.angle = angle_prior
            self._p00 = p00_prior
            self._p01 = p01_prior
            self._p11 = p11_prior
            self.last_measurement_accepted = False
            return self.angle

        k0 = p00_prior / innovation_variance
        k1 = p01_prior / innovation_variance
        angle_posterior = angle_prior + k0 * innovation
        bias_posterior = self.bias + k1 * innovation

        # Joseph-form covariance correction keeps the covariance symmetric and
        # non-negative despite finite-precision roundoff.
        one_minus_k0 = 1.0 - k0
        p00_posterior = (
            one_minus_k0 * one_minus_k0 * p00_prior
            + k0 * k0 * self.r_measure
        )
        p01_posterior = (
            one_minus_k0 * (p01_prior - k1 * p00_prior)
            + k0 * k1 * self.r_measure
        )
        p11_posterior = (
            p11_prior
            - 2.0 * k1 * p01_prior
            + k1 * k1 * p00_prior
            + k1 * k1 * self.r_measure
        )
        posterior_values = (
            angle_posterior,
            bias_posterior,
            p00_posterior,
            p01_posterior,
            p11_posterior,
        )
        if not all(math.isfinite(value) for value in posterior_values):
            raise ValueError("Attitude Kalman update became non-finite; state unchanged.")

        self.angle = angle_posterior
        self.bias = bias_posterior
        self._p00 = max(0.0, p00_posterior)
        self._p01 = p01_posterior
        self._p11 = max(0.0, p11_posterior)
        self.last_measurement_accepted = True
        return self.angle

    def filter(self, accel_angle: float, gyro_rate: float, dt: float) -> float:
        """Alias for :meth:`update`."""
        return self.update(accel_angle, gyro_rate, dt)


class ComplementaryFilter:
    """
    A Complementary Filter, typically used for sensor fusion (e.g., combining accelerometer and gyroscope).

    This filter combines a high-pass filter (for gyroscope data) and a low-pass filter (for accelerometer data)
    to estimate a more stable angle.

    Attributes:
        alpha (float): The filter coefficient weighting the gyroscope integration.
        angle (float): The estimated angle.
    """

    def __init__(self, alpha: float | None = None, initial_value: float = 0.0) -> None:
        if alpha is None:
            alpha = settings["filters"]["complementary"]["alpha"]
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

    def __init__(self, alpha: float | None = None, initial_value: float = 0.0) -> None:
        if alpha is None:
            alpha = settings["filters"]["lowpass"]["alpha"]
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
