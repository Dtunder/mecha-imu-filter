import logging
import unittest
from unittest.mock import MagicMock, patch

from resilience import (BadConfigurationError, ResilientSensorWrapper,
                        SensorConnectionError, SensorTimeoutError)


class TestResilientSensorWrapper(unittest.TestCase):
    def test_successful_read(self) -> None:
        mock_read = MagicMock(return_value=42.0)
        wrapper = ResilientSensorWrapper(mock_read)
        result = wrapper.read()
        self.assertEqual(result, 42.0)
        mock_read.assert_called_once()

    @patch("time.sleep")
    def test_retry_success(self, mock_sleep: MagicMock) -> None:
        mock_read = MagicMock(side_effect=[SensorTimeoutError("Timeout"), 42.0])
        wrapper = ResilientSensorWrapper(mock_read, max_retries=3, retry_delay=0.1)

        with self.assertLogs("resilience", level="WARNING") as cm:
            result = wrapper.read()
            self.assertEqual(result, 42.0)
            self.assertEqual(mock_read.call_count, 2)
            mock_sleep.assert_called_once_with(0.1)
            self.assertTrue(
                any("Sensor read failed: Timeout" in log for log in cm.output)
            )

    @patch("time.sleep")
    def test_fallback_after_max_retries(self, mock_sleep: MagicMock) -> None:
        mock_read = MagicMock(side_effect=SensorConnectionError("Connection Failed"))
        wrapper = ResilientSensorWrapper(
            mock_read, max_retries=2, retry_delay=0.1, fallback_value=-1.0
        )

        with self.assertLogs("resilience", level="ERROR") as cm:
            result = wrapper.read()
            self.assertEqual(result, -1.0)
            self.assertEqual(mock_read.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)
            self.assertTrue(
                any(
                    "Sensor read failed after 2 max retries" in log for log in cm.output
                )
            )

    def test_fallback_on_unexpected_error(self) -> None:
        mock_read = MagicMock(side_effect=ValueError("Unexpected Error"))
        wrapper = ResilientSensorWrapper(mock_read, fallback_value=99.9)

        with self.assertLogs("resilience", level="ERROR") as cm:
            result = wrapper.read()
            self.assertEqual(result, 99.9)
            mock_read.assert_called_once()
            self.assertTrue(
                any("Unexpected error during sensor read" in log for log in cm.output)
            )

    def test_bad_configuration(self) -> None:
        mock_read = MagicMock()

        with self.assertLogs("resilience", level="ERROR"):
            with self.assertRaises(BadConfigurationError):
                ResilientSensorWrapper(mock_read, max_retries=-1)

            with self.assertRaises(BadConfigurationError):
                ResilientSensorWrapper(mock_read, retry_delay=-0.5)

            with self.assertRaises(BadConfigurationError):
                ResilientSensorWrapper(mock_read, fallback_value="not a number")  # type: ignore


if __name__ == "__main__":
    unittest.main()
