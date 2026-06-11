import json
import os
import tempfile
import unittest
from unittest.mock import patch

from config import DEFAULT_CONFIG, load_config


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        config = load_config("nonexistent_file.json")
        self.assertEqual(config, DEFAULT_CONFIG)

    def test_load_config_from_file(self):
        test_config = {
            "filters": {"kalman": {"process_noise": 0.5}},
            "resilience": {"max_retries": 5},
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            self.assertEqual(config["filters"]["kalman"]["process_noise"], 0.5)
            self.assertEqual(config["resilience"]["max_retries"], 5)
            # Default should remain
            self.assertEqual(config["filters"]["complementary"]["alpha"], 0.98)
        finally:
            os.remove(temp_path)

    @patch.dict(
        os.environ,
        {"MECHA_KALMAN_PROCESS_NOISE": "0.1", "MECHA_RESILIENCE_MAX_RETRIES": "10"},
    )
    def test_load_config_from_env(self):
        config = load_config("nonexistent_file.json")
        self.assertEqual(config["filters"]["kalman"]["process_noise"], 0.1)
        self.assertEqual(config["resilience"]["max_retries"], 10)
        # Default should remain
        self.assertEqual(config["filters"]["complementary"]["alpha"], 0.98)


if __name__ == "__main__":
    unittest.main()
