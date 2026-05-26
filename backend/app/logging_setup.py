"""Centralized logging configuration.

Call `configure_logging(level)` once at app startup (done in `app.main`).
Everywhere else, use `get_logger(__name__)` to get a module-scoped logger.
Logger names track the module path so `logging.getLogger("app.primitives")`
can be set to DEBUG to trace primitive activity without flooding from elsewhere.

Per CLAUDE.md: INFO for major operations (LLM calls, primitive invocations,
sandbox executions), DEBUG for everything else, WARNING for recoverable
anomalies, ERROR for failures the caller should know about.
"""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger. Idempotent.

    Adds a single StreamHandler to stderr with a fixed format. Calling again
    replaces the existing handler rather than stacking, so this is safe to
    invoke from create_app() on every app reload during development.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name. Callers should pass `__name__`."""
    return logging.getLogger(name)
