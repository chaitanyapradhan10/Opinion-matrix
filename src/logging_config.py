"""
logging_config.py
------------------
Centralized logging setup for the Opinion Metrix AI pipeline.

Usage:
    from logging_config import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"

_CONFIGURED = False


def _configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger once: console + rotating file output."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with console + file handlers attached."""
    _configure_root_logger(level)
    return logging.getLogger(name)
