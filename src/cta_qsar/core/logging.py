"""Structured logging for CTA-QSAR."""

from __future__ import annotations

import logging
import sys
from typing import Any

_LOGGERS: dict[str, logging.Logger] = {}
_AGENT_LOGGER_NAME = "cta_qsar.agent"


def get_logger(name: str = "cta_qsar") -> logging.Logger:
    """Return a module logger, cached per name."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    _LOGGERS[name] = logger
    return logger


def configure_logging(level: int = logging.INFO, json_lines: bool = False) -> None:
    """Configure root CTA-QSAR logging once (idempotent)."""
    root = logging.getLogger("cta_qsar")
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    if json_lines:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


def agent_log(step: str, message: str, **context: Any) -> None:
    """Log an agent step with structured context."""
    ctx = " ".join(f"{k}={v}" for k, v in context.items())
    get_logger(_AGENT_LOGGER_NAME).info("[%s] %s %s", step, message, ctx)


class _JsonFormatter(logging.Formatter):
    """Minimal JSON-lines formatter."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)