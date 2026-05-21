"""
Single place to configure logging for the whole project. Callers (API,
scripts, runner) should import `get_logger` rather than calling
`logging.basicConfig` themselves.
"""
import logging
import os

_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_once()
    return logging.getLogger(name)
