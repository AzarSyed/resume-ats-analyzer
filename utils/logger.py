"""Shared logger for the Resume ATS Analyzer.

Writes to both the console and ``data/analyzer.log``. Use
``get_logger(__name__)`` from any module.
"""

from __future__ import annotations

import logging
from logging import Logger

from utils.settings import LOG_FILE

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured: dict[str, Logger] = {}


def get_logger(name: str = "ats") -> Logger:
    if name in _configured:
        return _configured[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _configured[name] = logger
    return logger
