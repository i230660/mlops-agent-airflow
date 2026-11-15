"""Tool registry: validates input, enforces timeouts/permissions, logs every call, and
always returns a ToolResult (tools never raise into the orchestrator)."""
from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

from .logging_utils import get_logger, log_tool_call
from .schemas import ToolResult

logger = get_logger()


class ToolUnavailable(Exception):
    """Raised by a tool when its backing service cannot be reached."""


class ToolDenied(Exception):
    """Raised when a request violates policy (e.g. namespace or DAG not approved)."""


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    func: Callable[..., Any]
    module: str
    mutating: bool = False  # requires explicit user confirmation
    timeout_s: Optional[float] = None

    def json_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description + (" [STATE-CHANGING: requires user confirmation]" if self.mutating else ""),
            "input_schema": self.input_model.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self, default_timeout_s: float = 20.0):
        self._tools: dict[str, Tool] = {}
        self.default_timeout_s = default_timeout_s
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def by_module(self, module: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.module == module]

    def schemas(self) -> list[dict]:
        return [t.json_schema() for t in self._tools.values()]

    def call(self, name: str, args: dict, *, request_id: str, confirmed: bool = False) -> ToolResult:
        start = time.perf_counter()

        def done(status: str, data=None, error: Optional[str] = None) -> ToolResult:
            ms = (time.perf_counter() - start) * 1000
            log_tool_call(logger, tool=name, status=status, duration_ms=ms, request_id=request_id, args=args)
            return ToolResult(tool=name, status=status, data=data, error=error, duration_ms=ms, request_id=request_id)

        tool = self._tools.get(name)
        if tool is None:
            return done("invalid_request", error=f"unknown tool '{name}'")

        try:
            parsed = tool.input_model(**(args or {}))
        except ValidationError as exc:
            return done("invalid_request", error=f"invalid arguments: {exc.errors(include_url=False)}")

        if tool.mutating and not confirmed:
            return done(
                "confirmation_required",
                error="This operation changes state. Ask the user for explicit confirmation and retry with confirm_actions=true.",
            )

        timeout = tool.timeout_s or self.default_timeout_s
        future = self._pool.submit(tool.func, parsed)
        try:
            data = future.result(timeout=timeout)
            return done("ok", data=data)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return done("timeout", error=f"tool exceeded {timeout}s timeout")
        except ToolDenied as exc:
            return done("denied", error=str(exc))
        except ToolUnavailable as exc:
            return done("unavailable", error=str(exc))
        except (ConnectionError, OSError) as exc:  # network-level failures of any backing service
            return done("unavailable", error=f"service unreachable: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            return done("error", error=f"{type(exc).__name__}: {exc}")
