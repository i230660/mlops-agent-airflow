"""Airflow 3 REST API tools (JWT-authenticated). Only approved DAGs may be triggered."""
from __future__ import annotations

import time
from typing import Optional

import requests
from pydantic import BaseModel, Field

from ..config import Settings
from ..registry import Tool, ToolDenied, ToolRegistry, ToolUnavailable

API = "/api/v2"


class _Empty(BaseModel):
    pass


class DagInput(BaseModel):
    dag_id: str = Field(default="training_pipeline", pattern=r"^[A-Za-z0-9_.\-]{1,100}$")


class RunsInput(BaseModel):
    dag_id: str = Field(default="training_pipeline", pattern=r"^[A-Za-z0-9_.\-]{1,100}$")
    limit: int = Field(default=5, ge=1, le=25)


class RunIdInput(BaseModel):
    dag_id: str = Field(default="training_pipeline", pattern=r"^[A-Za-z0-9_.\-]{1,100}$")
    dag_run_id: Optional[str] = Field(default=None, description="Defaults to the latest run")


class TaskLogInput(BaseModel):
    dag_id: str = Field(default="training_pipeline", pattern=r"^[A-Za-z0-9_.\-]{1,100}$")
    dag_run_id: str
    task_id: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,100}$")
    try_number: int = Field(default=1, ge=1, le=20)


class AirflowClient:
    """Thin authenticated client. Token is fetched from /auth/token (Airflow 3) and cached."""

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None):
        self.s = settings
        self.http = session or requests.Session()
        self._token: Optional[str] = None
        self._token_ts = 0.0

    def _auth(self) -> dict:
        if self._token and time.time() - self._token_ts < 600:
            return {"Authorization": f"Bearer {self._token}"}
        if not (self.s.airflow_username and self.s.airflow_password):
            raise ToolUnavailable("AIRFLOW_USERNAME/AIRFLOW_PASSWORD not configured")
        try:
            r = self.http.post(
                f"{self.s.airflow_base_url}/auth/token",
                json={"username": self.s.airflow_username, "password": self.s.airflow_password},
                timeout=10,
            )
        except requests.RequestException as exc:
            raise ToolUnavailable(f"Airflow unreachable: {exc.__class__.__name__}") from exc
        if r.status_code >= 400:
            raise ToolUnavailable(f"Airflow authentication failed (HTTP {r.status_code})")
        self._token = r.json().get("access_token")
        self._token_ts = time.time()
        return {"Authorization": f"Bearer {self._token}"}

    def request(self, method: str, path: str, **kw):
        try:
            r = self.http.request(method, f"{self.s.airflow_base_url}{API}{path}", headers=self._auth(), timeout=10, **kw)
        except requests.RequestException as exc:
            raise ToolUnavailable(f"Airflow unreachable: {exc.__class__.__name__}") from exc
        if r.status_code == 404:
            raise ValueError(f"not found: {path}")
        if r.status_code >= 400:
            raise RuntimeError(f"Airflow API error HTTP {r.status_code}: {r.text[:300]}")
        return r.json()


def build_airflow_tools(settings: Settings, registry: ToolRegistry, client: Optional[AirflowClient] = None) -> None:
    af = client or AirflowClient(settings)

    def _latest_run_id(dag_id: str) -> str:
        data = af.request("GET", f"/dags/{dag_id}/dagRuns", params={"order_by": "-logical_date", "limit": 1})
        runs = data.get("dag_runs", [])
        if not runs:
            raise ValueError(f"no runs found for DAG {dag_id}")
        return runs[0]["dag_run_id"]

    def list_dags(_: _Empty):
        data = af.request("GET", "/dags", params={"limit": 100})
        return [
            {"dag_id": d["dag_id"], "paused": d.get("is_paused"), "approved_for_trigger": d["dag_id"] in settings.airflow_allowed_dags}
            for d in data.get("dags", [])
        ]

    def dag_runs(inp: RunsInput):
        data = af.request("GET", f"/dags/{inp.dag_id}/dagRuns", params={"order_by": "-logical_date", "limit": inp.limit})
        return [
            {
                "dag_run_id": r["dag_run_id"],
                "state": r.get("state"),
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
                "run_type": r.get("run_type"),
            }
            for r in data.get("dag_runs", [])
        ]

    def trigger(inp: DagInput):
        if inp.dag_id not in settings.airflow_allowed_dags:
            raise ToolDenied(f"DAG '{inp.dag_id}' is not approved for triggering; allowed: {list(settings.airflow_allowed_dags)}")
        data = af.request("POST", f"/dags/{inp.dag_id}/dagRuns", json={"logical_date": None, "conf": {"triggered_by": "mlops-agent"}})
        return {"dag_id": inp.dag_id, "dag_run_id": data.get("dag_run_id"), "state": data.get("state")}

    def task_instances(inp: RunIdInput):
        run_id = inp.dag_run_id or _latest_run_id(inp.dag_id)
        data = af.request("GET", f"/dags/{inp.dag_id}/dagRuns/{run_id}/taskInstances")
        return {
            "dag_run_id": run_id,
            "tasks": [
                {
                    "task_id": t["task_id"],
                    "state": t.get("state"),
                    "try_number": t.get("try_number"),
                    "duration": t.get("duration"),
                    "start_date": t.get("start_date"),
                    "end_date": t.get("end_date"),
                }
                for t in data.get("task_instances", [])
            ],
        }

    def failed_tasks(inp: RunIdInput):
        info = task_instances(inp)
        failed = [t for t in info["tasks"] if t["state"] in ("failed", "upstream_failed")]
        return {"dag_run_id": info["dag_run_id"], "failed_tasks": failed, "all_ok": not failed}

    def task_logs(inp: TaskLogInput):
        data = af.request(
            "GET", f"/dags/{inp.dag_id}/dagRuns/{inp.dag_run_id}/taskInstances/{inp.task_id}/logs/{inp.try_number}"
        )
        content = data.get("content", data)
        text = content if isinstance(content, str) else str(content)
        return {"task_id": inp.task_id, "try_number": inp.try_number, "log_tail": text[-3000:]}

    reg = registry.register
    reg(Tool("airflow_list_dags", "List Airflow DAGs and whether each is approved for triggering.", _Empty, list_dags, "airflow"))
    reg(Tool("airflow_dag_runs", "Get recent DAG run states.", RunsInput, dag_runs, "airflow"))
    reg(Tool("airflow_trigger_dag", "Trigger an approved training DAG run.", DagInput, trigger, "airflow", mutating=True))
    reg(Tool("airflow_task_instances", "Get per-task states for a DAG run (default latest).", RunIdInput, task_instances, "airflow"))
    reg(Tool("airflow_failed_tasks", "Identify failed tasks in a DAG run (default latest).", RunIdInput, failed_tasks, "airflow"))
    reg(Tool("airflow_task_logs", "Retrieve the tail of a task's log for failure diagnosis.", TaskLogInput, task_logs, "airflow"))
