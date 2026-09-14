"""Package-scoped logging, inspired by dexe_agent/common/log/logger.py.

Configuration is explicit and never replaces root handlers. Default output is
stderr; CLI stdout remains available for machine-readable workflow results.
Handlers supplied by callers are borrowed, never closed by this module.
"""

import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime, timezone

PACKAGE_NAME = "embodiedforge"
DEFAULT_FORMAT = (
    "%(asctime)s|%(levelname)s|%(name)s|"
    "%(filename)s-%(funcName)s-%(lineno)04d]: %(message)s"
)
_owned_handlers: set[logging.Handler] = set()
_package_logger = logging.getLogger(PACKAGE_NAME)
if not _package_logger.handlers:
    _package_logger.addHandler(logging.NullHandler())


class JsonFormatter(logging.Formatter):
    """One JSON object per record, preserving Unicode and exception tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "source": {
                "file": record.filename,
                "function": record.funcName,
                "line": record.lineno,
            },
            "process": record.process,
            "thread": record.threadName,
            "context": getattr(record, "context", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _construct_default_formatter(with_json_format: bool = False) -> logging.Formatter:
    return JsonFormatter() if with_json_format else logging.Formatter(DEFAULT_FORMAT)


def get_logger(name: str = PACKAGE_NAME, **context: object) -> logging.LoggerAdapter:
    """Get a child logger with fixed context such as task, backend or run_id.

    Pass __name__ within this package, or a short name for external adapters.
    Context is a shallow copy; keep values small and do not log sensor arrays.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Logger name must be a nonempty string")
    if name != PACKAGE_NAME and not name.startswith(PACKAGE_NAME + "."):
        name = f"{PACKAGE_NAME}.{name}"
    return logging.LoggerAdapter(logging.getLogger(name), {"context": dict(context)})


def setup_logging(
    level: int | str = logging.WARNING,
    handlers: Sequence[logging.Handler] | None = None,
    with_json_format: bool = False,
) -> None:
    """Replace this package's handlers at application startup, not during logging.

    None creates an owned stderr handler; [] explicitly disables package output.
    Provided handlers retain their levels and existing formatters. Configuration
    does not mutate the caller's sequence, close borrowed handlers, or touch root.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    if type(level) is not int or level < 0:
        raise ValueError(
            "Log level must be a nonnegative integer or standard level name"
        )
    selected = list(handlers) if handlers is not None else [logging.StreamHandler()]
    if any(not isinstance(handler, logging.Handler) for handler in selected):
        raise TypeError("handlers must contain logging.Handler instances")
    formatter = _construct_default_formatter(with_json_format)
    for handler in selected:
        if handler.formatter is None:
            handler.setFormatter(formatter)
    for handler in list(_package_logger.handlers):
        _package_logger.removeHandler(handler)
    for handler in list(_owned_handlers):
        if handler not in selected:
            handler.close()
            _owned_handlers.remove(handler)
    if handlers is None:
        _owned_handlers.update(selected)
    for handler in selected or [logging.NullHandler()]:
        _package_logger.addHandler(handler)
    _package_logger.setLevel(level)
    _package_logger.propagate = False


def add_logger_handler(
    handler: logging.Handler, with_json_format: bool = False
) -> None:
    """Attach a borrowed handler once, retaining its existing level/formatter."""
    if not isinstance(handler, logging.Handler):
        raise TypeError("handler must be a logging.Handler")
    if handler.formatter is None:
        handler.setFormatter(_construct_default_formatter(with_json_format))
    _package_logger.addHandler(handler)


def auto_configure_debug_logging() -> bool:
    """Opt in via EMBODIEDFORGE_DEBUG / EMBODIEDFORGE_JSON_FORMAT_LOG.

    Called explicitly by CLI entry points; importing the library does not inspect
    environment flags or configure stream/file output. Returns whether configured.
    """
    truthy = {"1", "true", "yes", "on"}
    if os.getenv("EMBODIEDFORGE_DEBUG", "").lower() not in truthy:
        return False
    setup_logging(
        logging.DEBUG,
        with_json_format=os.getenv("EMBODIEDFORGE_JSON_FORMAT_LOG", "").lower()
        in truthy,
    )
    return True
