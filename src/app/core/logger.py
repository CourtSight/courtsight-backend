import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logging():
    logger = logging.getLogger("app")
    logger.setLevel(LOGGING_LEVEL)

    # Prevent adding multiple handlers in reload scenarios
    if not logger.handlers:
        file_handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=10485760, backupCount=5)
        file_handler.setLevel(LOGGING_LEVEL)
        file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(LOGGING_LEVEL)
        stream_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
        logger.addHandler(stream_handler)

    return logger

logger = setup_logging()

def get_logger(name: str = None):
    """Get a logger with the given name, defaulting to 'app'."""
    return logging.getLogger(name or "app")
