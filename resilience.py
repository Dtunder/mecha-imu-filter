import logging
import time
from typing import Any, Callable

from config import settings

logger = logging.getLogger(__name__)


class SensorError(Exception):
    """Base class for sensor-related exceptions."""

    pass


class SensorConnectionError(SensorError):
    """Raised when the sensor cannot be connected to."""

    pass


class SensorTimeoutError(SensorError):
    """Raised when a sensor read operation times out."""

    pass


class BadConfigurationError(SensorError):
    """Raised when the resilience wrapper is misconfigured."""

    pass


from config import settings


class ResilientSensorWrapper:
    """
    A wrapper for sensor reading functions to provide resilience through retries and fallbacks.
    """

    def __init__(
        self,
        read_func: Callable[[], float],
        max_retries: int | None = None,
        retry_delay: float | None = None,
        fallback_value: float | None = None,
    ) -> None:
        if max_retries is None:
            max_retries = settings["resilience"]["max_retries"]
        if retry_delay is None:
            retry_delay = settings["resilience"]["retry_delay"]
        if fallback_value is None:
            fallback_value = settings["resilience"]["fallback_value"]
        """
        Initializes the ResilientSensorWrapper.

        Args:
            read_func (Callable[[], float]): The function to call to read the sensor.
            max_retries (int): Maximum number of retries before falling back. Must be >= 0.
            retry_delay (float): Delay in seconds between retries. Must be >= 0.
            fallback_value (float): Value to return if all retries fail.
        """
        if not isinstance(max_retries, int) or max_retries < 0:
            logger.error(
                "Initialization failed: max_retries=%s must be a non-negative integer.",
                max_retries,
            )
            raise BadConfigurationError("max_retries must be a non-negative integer.")
        if not isinstance(retry_delay, (int, float)) or retry_delay < 0:
            logger.error(
                "Initialization failed: retry_delay=%s must be a non-negative number.",
                retry_delay,
            )
            raise BadConfigurationError("retry_delay must be a non-negative number.")
        if not isinstance(fallback_value, (int, float)):
            logger.error(
                "Initialization failed: fallback_value=%s must be a number.",
                fallback_value,
            )
            raise BadConfigurationError("fallback_value must be a number.")

        self.read_func = read_func
        self.max_retries = max_retries
        self.retry_delay = float(retry_delay)
        self.fallback_value = float(fallback_value)

    def read(self) -> float:
        """
        Attempts to read from the sensor, retrying on connection or timeout errors.

        Returns:
            float: The sensor value, or the fallback value if all retries fail.
        """
        attempts = 0
        while attempts <= self.max_retries:
            try:
                value = self.read_func()
                if attempts > 0:
                    logger.info("Sensor read succeeded after %d retries.", attempts)
                return float(value)
            except (SensorConnectionError, SensorTimeoutError) as e:
                attempts += 1
                if attempts <= self.max_retries:
                    logger.warning(
                        "Sensor read failed: %s. Retrying in %s seconds (%d/%d)...",
                        e,
                        self.retry_delay,
                        attempts,
                        self.max_retries,
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        "Sensor read failed after %d max retries: %s. Using fallback value: %s",
                        self.max_retries,
                        e,
                        self.fallback_value,
                    )
            except Exception as e:
                logger.error(
                    "Unexpected error during sensor read: %s. Using fallback value: %s",
                    e,
                    self.fallback_value,
                )
                break

        return self.fallback_value
