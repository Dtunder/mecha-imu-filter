import logging

def setup_logging(level=logging.INFO):
    """
    Configures structured logging for the application.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured.")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting mecha-imu-filter...")
    print("Hello from mecha-imu-filter!")

if __name__ == "__main__":
    main()
