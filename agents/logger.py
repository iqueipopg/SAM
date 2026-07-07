# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Structured logging for SAM pipeline.

Usage:
    from agents.logger import setup_logger, get_logger

    # Setup once (e.g., in app.py or at pipeline start):
    setup_logger("SAM", log_dir="outputs/run_20260309_120000")

    # Get logger in any module:
    logger = get_logger("Agent1Protocol")
    logger.info("Model loaded")
    logger.warning("Fallback triggered")
    logger.error("Parse failed: %s", err)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Module-level registry so we only configure once
_configured = False
_log_dir = None
_log_file = None


def setup_logger(name="SAM", log_dir="outputs", level="INFO"):
    """Create the root SAM logger with console + rotating file handlers.

    Args:
        name: Root logger name.
        log_dir: Directory for the log file (pipeline.log).
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        logging.Logger instance.
    """
    global _configured, _log_dir, _log_file

    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if _configured and _log_dir == log_dir:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (rotating, 5 MB max, keep 3 backups)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "pipeline.log")
    fh = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _configured = True
    _log_dir = log_dir
    _log_file = log_path

    logger.info("Logger initialized -> %s", log_path)
    return logger


def get_logger(name):
    """Get a child logger under the SAM root.

    If setup_logger() has not been called, returns a basic logger
    that writes to stderr (so nothing crashes).

    Args:
        name: Logger name (typically self.__class__.__name__).

    Returns:
        logging.Logger instance.
    """
    root = logging.getLogger("SAM")
    # If root has no handlers yet, add a minimal stderr handler
    if not root.handlers:
        root.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(ch)
    return root.getChild(name)


def get_log_file_path():
    """Return the path to the current log file, or None if not configured."""
    return _log_file
