from __future__ import annotations

import logging

LOGGER_NAME = 'TRLAU'
logger = logging.getLogger(LOGGER_NAME)


def configure_logging(debug_enabled: bool = False) -> logging.Logger:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[TRLAU] %(levelname)s: %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug_enabled else logging.CRITICAL + 1)
    logger.propagate = False
    return logger
