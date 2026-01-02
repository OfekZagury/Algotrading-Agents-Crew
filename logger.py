import logging
import os


def setup_logger(name="AlgoTrader"):
    """
    Sets up a logger that writes to both a file (trading.log) and the console.
    """
    # Create logs directory if not exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if function called twice
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    file_handler = logging.FileHandler('logs/trading.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger