import logging
import os

logging.basicConfig(
    format="%(asctime)s [%(name)s] [%(levelname)s]: %(message)s",
    datefmt="%b %d %Y %H:%M:%S",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
