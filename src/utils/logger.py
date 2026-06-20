"""
src/utils/logger.py
===================
One reusable function, get_logger(), so every part of the project logs the same
way: nice messages on screen AND saved to a rotating file in logs/.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting ingestion...")
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str = "ifrs9", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to console + logs/ifrs9.log."""
    logger = logging.getLogger(name)

    # If this logger was already set up, don't add duplicate handlers.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    # 1) Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 2) Rotating file handler (max 5 MB per file, keep 3 backups)
    log_file = settings.paths.logs / "ifrs9.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
