"""Logging set-up that cannot emit a raw contact detail.

`PiiMaskingFilter` rewrites ``record.msg``, ``record.args`` and
``record.exc_text`` before any formatter runs. Doing it in a filter rather than
at each call site is the point: a new ``logger.info`` cannot leak because there
is nothing to remember.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Any

from .safety.pii import mask_text

__all__ = [
    "HANDLER_NAME",
    "LOG_FORMAT",
    "PiiMaskingFilter",
    "RedactingAdapter",
    "log_safe",
    "setup_logging",
]

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

#: How a later `setup_logging` recognises the handler it installed earlier, so a
#: second call replaces it instead of stacking a duplicate that double-logs
#: everything. A plain handler count would be wrong: pytest and uvicorn add
#: handlers of their own to the same logger.
HANDLER_NAME = "meher_agent.pii_stream"


class PiiMaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _mask_message(record.msg)
        record.args = _mask_args(record.args)
        record.exc_text = _mask_traceback(record)
        return True


class RedactingAdapter(logging.LoggerAdapter):
    """For loggers that cannot install the filter, e.g. a third-party library."""

    def process(
        self, msg: Any, kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:  # pragma: no cover - trivial delegation
        extra = kwargs.get("extra")
        if isinstance(extra, dict):
            kwargs["extra"] = {
                key: mask_text(value) if isinstance(value, str) else value
                for key, value in extra.items()
            }
        return _mask_message(msg), kwargs


def _mask_message(msg: Any) -> Any:
    if isinstance(msg, str):
        return mask_text(msg)
    if msg is None:
        return None
    return mask_text(str(msg))


def _mask_arg(value: Any) -> Any:
    # Anything that is not a string keeps its type: `logger.info("%d", 5)` has to
    # survive the round trip through the formatter.
    return mask_text(value) if isinstance(value, str) else value


def _mask_args(args: Any) -> Any:
    if args is None:
        return None
    if isinstance(args, dict):
        return {key: _mask_arg(value) for key, value in args.items()}
    if isinstance(args, tuple):
        return tuple(_mask_arg(value) for value in args)
    return _mask_arg(args)


def _mask_traceback(record: logging.LogRecord) -> str | None:
    # `Formatter.format` renders `exc_info` only while `exc_text` is still empty,
    # so filling it in here is the one chance to mask the exception message and
    # the offending source line it quotes.
    if not record.exc_info or record.exc_text:
        return record.exc_text
    exc_type, exc_value, exc_tb = record.exc_info
    try:
        rendered = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    except Exception:  # pragma: no cover - a broken traceback must not break logging
        return record.exc_text
    return mask_text(rendered)


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).strip().upper(), logging.INFO)


def setup_logging(level: str) -> None:
    """Install one masked stream handler on the root logger.

    Idempotent: a second call replaces the handler the first one installed rather
    than adding a second, which would print every record twice.
    """
    resolved = _resolve_level(level)
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == HANDLER_NAME:
            root.removeHandler(existing)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.set_name(HANDLER_NAME)
    handler.setLevel(resolved)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(PiiMaskingFilter())

    root.addHandler(handler)
    # The filter on the root logger catches records from a logger whose own
    # handlers were installed before this one; the handler copy is the backstop
    # for records that reach the handler without passing through the root logger.
    root.addFilter(PiiMaskingFilter())
    root.setLevel(resolved)


def log_safe(logger: logging.Logger, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
    """`logger.log` with the message and its arguments masked before emission."""
    logger.log(level, mask_text(str(msg)), *_mask_args(args), **kwargs)
