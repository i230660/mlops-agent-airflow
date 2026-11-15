"""Agent service (FastAPI) and CLI entrypoint.

Run API:  uvicorn agent.main:app --port 8001
Run CLI:  python -m agent.main "Which dataset version was used for run <id>?"
"""
from __future__ import annotations

import sys
from typing import Optional

from fastapi import FastAPI

from .config import Settings, get_settings
from .orchestrator import Orchestrator
from .registry import ToolRegistry
from .schemas import ChatRequest, ChatResponse
from .tools.airflow_tools import build_airflow_tools
from .tools.dvc_tools import build_dvc_tools
from .tools.kubernetes_tools import build_kubernetes_tools
from .tools.mlflow_tools import build_mlflow_tools


def build_registry(settings: Settings, *, mlflow_client_factory=None, airflow_client=None, k8s_backend_factory=None) -> ToolRegistry:
    if mlflow_client_factory is None:
        def mlflow_client_factory():  # noqa: E306
            from mlflow.tracking import MlflowClient

            return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    registry = ToolRegistry(default_timeout_s=settings.tool_timeout_s)
    build_dvc_tools(settings, registry, mlflow_client_factory)
    build_mlflow_tools(settings, registry, mlflow_client_factory)
    build_airflow_tools(settings, registry, airflow_client)
    if k8s_backend_factory is not None:
        build_kubernetes_tools(settings, registry, k8s_backend_factory)
    else:
        build_kubernetes_tools(settings, registry)
    return registry


def build_orchestrator(settings: Optional[Settings] = None, **kw) -> Orchestrator:
    settings = settings or get_settings()
    llm = None
    if settings.gemini_api_key:
        from .llm import GeminiLLM

        llm = GeminiLLM(settings)
    return Orchestrator(settings, build_registry(settings, **kw), llm)


app = FastAPI(title="MLOps Agent", version="1.0.0")
_orch: Optional[Orchestrator] = None


def _get() -> Orchestrator:
    global _orch
    if _orch is None:
        _orch = build_orchestrator()
    return _orch


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools")
def tools():
    return {"tools": [{"name": t["name"], "description": t["description"]} for t in _get().registry.schemas()]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return _get().handle(req.message, confirm_actions=req.confirm_actions)


def cli() -> None:
    msg = " ".join(sys.argv[1:]) or input("You: ")
    resp = build_orchestrator().handle(msg)
    print(resp.answer)


if __name__ == "__main__":
    cli()
