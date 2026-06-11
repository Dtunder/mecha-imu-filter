import logging


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures structured logging for the application.

    Sets up basic logging configuration with the specified logging level and
    a standard format including timestamp, logger name, log level, and message.

    Args:
        level (int, optional): The logging level to set (e.g., logging.INFO, logging.DEBUG).
                               Defaults to logging.INFO.
    """
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured.")


def main() -> None:
    """
    Main entry point for the mecha-imu-filter application.

    Initializes logging and prints a welcome message.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting mecha-imu-filter...")
    print("Hello from mecha-imu-filter!")


if __name__ == "__main__":
    main()
