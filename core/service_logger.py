from pathlib import Path
import sys
import logging


BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(exist_ok=True)


def get_service_logger(service_name: str) -> logging.Logger:
    """
    Create a reusable logger for Young60 services.

    Each service gets:
    1. Terminal logs
    2. Separate log file inside logs/

    Example:
        logger = get_service_logger("medicine_service")
        logger.info("Medicine service started")
    """

    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_file = LOG_DIR / f"{service_name}.log"

    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - [{service_name.upper()}] %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger