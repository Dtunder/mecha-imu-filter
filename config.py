import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "filters": {
        "kalman": {
            "process_noise": 1e-5,
            "measurement_noise": 1e-2,
            "estimated_error": 1.0,
        },
        "complementary": {"alpha": 0.98},
        "lowpass": {"alpha": 0.5},
    },
    "resilience": {"max_retries": 3, "retry_delay": 0.1, "fallback_value": 0.0},
}


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Loads configuration from a JSON file, environment variables, or falls back to defaults.
    """
    # Create a deep copy of DEFAULT_CONFIG
    config: Dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))

    # Try to load from file
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
            # Simple deep update for nested dicts (2 levels deep max in our case)
            if "filters" in file_config:
                for filter_type, params in file_config["filters"].items():
                    if filter_type in config["filters"]:
                        config["filters"][filter_type].update(params)
            if "resilience" in file_config:
                config["resilience"].update(file_config["resilience"])
            logger.info("Loaded configuration from %s", config_path)
        except Exception as e:
            logger.error(
                "Failed to load config from %s: %s. Using defaults.", config_path, e
            )

    # Try to override with environment variables
    # Filters
    for filter_type, params in config["filters"].items():
        for key in params.keys():
            env_key = f"MECHA_{filter_type.upper()}_{key.upper()}"
            if env_key in os.environ:
                try:
                    config["filters"][filter_type][key] = type(params[key])(
                        os.environ[env_key]
                    )
                    logger.info("Overridden %s from env var %s", key, env_key)
                except ValueError:
                    logger.error(
                        "Invalid value for env var %s. Using default/file value.",
                        env_key,
                    )

    # Resilience
    for key in config["resilience"].keys():
        env_key = f"MECHA_RESILIENCE_{key.upper()}"
        if env_key in os.environ:
            try:
                config["resilience"][key] = type(config["resilience"][key])(
                    os.environ[env_key]
                )
                logger.info("Overridden %s from env var %s", key, env_key)
            except ValueError:
                logger.error(
                    "Invalid value for env var %s. Using default/file value.", env_key
                )

    return config


# Load globally on module import
settings = load_config()
