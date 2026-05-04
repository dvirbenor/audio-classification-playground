"""Logging helpers for acoustic-event inference."""
from __future__ import annotations

import logging
import sys


LOGGER_NAME = "audio_classification_playground.acoustic_events.inference"


def get_logger() -> logging.Logger:
    """Return the inference logger, configured to write INFO logs to stdout."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def configure_stdout_logging(*, verbose: bool = False) -> logging.Logger:
    """Configure CLI logging. Verbose currently keeps INFO output readable."""
    logger = get_logger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger
