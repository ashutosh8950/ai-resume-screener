"""Consistent application logging."""

import logging

from resume_screener.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"resume_screener.{name}")
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
        logger.propagate = False
    return logger