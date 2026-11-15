"""Structured JSON logging that never leaks secrets."""
from __future__ import annotations

import json
import logging
import re
import sys
import time

_SECRET_KEYS = re.compile(r"(pass(word)?|token|secret|api[_-]?key|authorization|credential)", re.I)
_SECRET_VALUE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,})")


def redact(value):
    """Recursively redact secret-looking keys and values."""
    if isinstance(value, dict):
        return {k: ("***REDACTED***" if _SECRET_KEYS.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("***REDACTED***", value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": _SECRET_VALUE.sub("***REDACTED***", record.getMessage()),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(redact(extra))
        return json.dumps(payload, default=str)


def get_logger(name: str = "mlops-agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = True
    return logger


def log_tool_call(logger: logging.Logger, *, tool: str, status: str, duration_ms: float, request_id: str, args=None):
    logger.info(
        "tool_call",
        extra={
            "fields": {
                "event": "tool_call",
                "tool": tool,
                "status": status,
                "duration_ms": round(duration_ms, 2),
                "request_id": request_id,
                "args": redact(args or {}),
            }
        },
    )
