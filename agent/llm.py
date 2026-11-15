"""Gemini function-calling loop (google-genai SDK, free-tier API key).

Tools are executed through the registry; the model only ever sees ToolResult JSON.
We run the loop manually (automatic function calling disabled) so that every call goes
through validation, permissions, timeouts and logging.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from .config import Settings

_KEEP = {"type", "description", "properties", "required", "items", "enum", "minimum", "maximum", "minLength", "maxLength", "pattern"}


def to_gemini_schema(schema: dict, defs: dict | None = None) -> dict:
    """Convert a pydantic JSON schema into the OpenAPI subset Gemini accepts
    (no $defs/$ref, no title/default, Optional[...] anyOf collapsed)."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        return to_gemini_schema(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:  # Optional[X] -> X
        options = [o for o in schema["anyOf"] if o.get("type") != "null"]
        merged = {k: v for k, v in schema.items() if k != "anyOf"}
        merged.update(options[0] if options else {"type": "string"})
        return to_gemini_schema(merged, defs)
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k not in _KEEP:
            continue
        if k == "properties":
            out[k] = {name: to_gemini_schema(s, defs) for name, s in v.items()}
        elif k == "items":
            out[k] = to_gemini_schema(v, defs)
        else:
            out[k] = v
    return out


def function_declarations(tools: list[dict]) -> list[dict]:
    decls = []
    for t in tools:
        params = to_gemini_schema(t["input_schema"])
        decl = {"name": t["name"], "description": t["description"]}
        if params.get("properties"):  # Gemini rejects empty-object parameters
            decl["parameters"] = params
        decls.append(decl)
    return decls


class GeminiLLM:
    def __init__(self, settings: Settings, client=None):
        self.s = settings
        if client is None:
            from google import genai

            client = genai.Client(api_key=settings.gemini_api_key)
        self.client = client

    def _generate(self, contents, config):
        """Call Gemini, retrying briefly on transient errors (rate limit / overload)."""
        delay = 2.0
        for attempt in range(4):
            try:
                return self.client.models.generate_content(model=self.s.llm_model, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001
                if getattr(exc, "code", None) in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

    def run(self, system: str, message: str, tools: list[dict], call_tool: Callable) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=function_declarations(tools))],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.1,
        )
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=message)])]
        for _ in range(self.s.max_agent_steps):
            resp = self._generate(contents, config)
            calls = resp.function_calls or []
            if not calls:
                return resp.text or ""
            contents.append(resp.candidates[0].content)
            parts = []
            for fc in calls:
                res = call_tool(fc.name, dict(fc.args or {}))
                payload = json.loads(json.dumps(res.model_dump(), default=str))
                parts.append(types.Part.from_function_response(name=fc.name, response={"result": payload}))
            contents.append(types.Content(role="user", parts=parts))
        return "Stopped: maximum number of tool steps reached before a final answer."
