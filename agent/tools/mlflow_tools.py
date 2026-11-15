"""MLflow experiment-intelligence tools using MlflowClient (never scraping the UI)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..config import Settings
from ..registry import Tool, ToolRegistry, ToolUnavailable


class _Empty(BaseModel):
    pass


class ListRunsInput(BaseModel):
    experiment_name: Optional[str] = Field(default=None, description="Experiment name; default = all experiments")
    limit: int = Field(default=5, ge=1, le=50)


class RunInput(BaseModel):
    run_id: str = Field(min_length=1)


class BestRunInput(BaseModel):
    metric: str = Field(default="f1", description="Metric to maximise, e.g. f1")
    experiment_name: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100, description="How many recent runs to consider")
    minimize: bool = False


class CompareInput(BaseModel):
    experiment_names: list[str] = Field(min_length=1, max_length=10)
    metric: str = "f1"


class RegistryInput(BaseModel):
    model_name: Optional[str] = None


def _run_summary(run) -> dict:
    info, data = run.info, run.data
    return {
        "run_id": info.run_id,
        "experiment_id": info.experiment_id,
        "status": info.status,
        "start_time": info.start_time,
        "metrics": dict(data.metrics),
        "params": dict(data.params),
        "tags": {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")},
    }


def build_mlflow_tools(settings: Settings, registry: ToolRegistry, client_factory) -> None:
    def client():
        try:
            return client_factory()
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"MLflow unavailable: {exc}") from exc

    def _exp_ids(c, name: Optional[str]) -> list[str]:
        if name:
            exp = c.get_experiment_by_name(name)
            if exp is None:
                raise ValueError(f"experiment '{name}' not found")
            return [exp.experiment_id]
        return [e.experiment_id for e in c.search_experiments()]

    def list_experiments(_: _Empty):
        c = client()
        return [{"experiment_id": e.experiment_id, "name": e.name} for e in c.search_experiments()]

    def list_runs(inp: ListRunsInput):
        c = client()
        runs = c.search_runs(_exp_ids(c, inp.experiment_name), order_by=["attributes.start_time DESC"], max_results=inp.limit)
        return [_run_summary(r) for r in runs]

    def get_run(inp: RunInput):
        c = client()
        run = c.get_run(inp.run_id)
        out = _run_summary(run)
        out["artifacts"] = [a.path for a in c.list_artifacts(inp.run_id)]
        return out

    def best_run(inp: BestRunInput):
        c = client()
        runs = c.search_runs(_exp_ids(c, inp.experiment_name), order_by=["attributes.start_time DESC"], max_results=inp.limit)
        scored = [r for r in runs if inp.metric in r.data.metrics and r.info.status == "FINISHED"]
        if not scored:
            return {"best": None, "considered": len(runs), "note": f"no finished runs with metric '{inp.metric}'"}
        def key(r):
            return r.data.metrics[inp.metric]

        best = min(scored, key=key) if inp.minimize else max(scored, key=key)
        summary = _run_summary(best)
        caveats = []
        if not summary["tags"].get(settings.dataset_tag_key) and not summary["tags"].get(settings.dataset_hash_tag_key):
            caveats.append("no DVC provenance recorded; dataset unknown")
        gate = summary["tags"].get("quality_gate")
        if gate != "passed":
            caveats.append(f"quality_gate tag is '{gate}', not 'passed'")
        return {
            "best": summary,
            "metric": inp.metric,
            "considered": len(scored),
            "ranking": sorted(
                ({"run_id": r.info.run_id, inp.metric: r.data.metrics[inp.metric]} for r in scored),
                key=lambda x: x[inp.metric],
                reverse=not inp.minimize,
            ),
            "production_ready": False if caveats else None,
            "caveats": caveats,
            "note": "Highest metric is not automatically production-ready; check quality gate, provenance and deployment.",
        }

    def compare_experiments(inp: CompareInput):
        c = client()
        rows = []
        for name in inp.experiment_names:
            ids = _exp_ids(c, name)
            runs = c.search_runs(ids, max_results=200)
            vals = [r.data.metrics[inp.metric] for r in runs if inp.metric in r.data.metrics]
            rows.append(
                {
                    "experiment": name,
                    "runs": len(runs),
                    f"best_{inp.metric}": max(vals) if vals else None,
                    f"mean_{inp.metric}": (sum(vals) / len(vals)) if vals else None,
                }
            )
        return {"metric": inp.metric, "comparison": rows}

    def model_registry(inp: RegistryInput):
        c = client()
        try:
            models = c.search_registered_models()
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"registry unavailable: {exc}") from exc
        out = []
        for m in models:
            if inp.model_name and m.name != inp.model_name:
                continue
            versions = c.search_model_versions(f"name='{m.name}'")
            out.append(
                {
                    "name": m.name,
                    "versions": [
                        {"version": v.version, "run_id": v.run_id, "stage": getattr(v, "current_stage", None), "status": v.status}
                        for v in versions
                    ],
                }
            )
        return {"registered_models": out}

    reg = registry.register
    reg(Tool("mlflow_list_experiments", "List MLflow experiments.", _Empty, list_experiments, "mlflow"))
    reg(Tool("mlflow_list_runs", "List recent training runs with metrics and params.", ListRunsInput, list_runs, "mlflow"))
    reg(Tool("mlflow_get_run", "Get metrics, params, tags and artifacts of one run.", RunInput, get_run, "mlflow"))
    reg(Tool("mlflow_best_run", "Find the best recent run by a metric, with readiness caveats.", BestRunInput, best_run, "mlflow"))
    reg(Tool("mlflow_compare_experiments", "Compare experiments on a metric.", CompareInput, compare_experiments, "mlflow"))
    reg(Tool("mlflow_model_registry", "Get registered models and versions.", RegistryInput, model_registry, "mlflow"))
