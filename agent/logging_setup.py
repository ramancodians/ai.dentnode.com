"""JSON structured logging for Cloud Run.

Cloud Run captures stdout and Cloud Logging automatically parses JSON records
that match the LogEntry format. Calling setup_logging() once at startup
wires every logger.info/warning/error call to emit a single JSON object per
line so Cloud Logging indexes severity, message, and all extra fields.
"""

import json
import logging
import sys


class _JsonFormatter(logging.Formatter):
    # Fields that are internal to LogRecord and shouldn't be forwarded as
    # arbitrary extra keys — they're either redundant or already captured.
    _SKIP = frozenset({
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs", "msg",
        "name", "pathname", "process", "processName", "relativeCreated",
        "stack_info", "taskName", "thread", "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        out: dict = {
            "severity": record.levelname,
            "logger": record.name,
            "message": record.message,
        }
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key not in self._SKIP:
                out[key] = val
        return json.dumps(out, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit JSON to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    # Silence noisy third-party loggers that add no value in production.
    for name in ("httpx", "httpcore", "google.adk", "litellm", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)
