"""Shared fakes: MLflow client, Airflow HTTP session, Kubernetes backend. No real services are contacted."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from agent.config import Settings
from agent.main import build_orchestrator
from agent.tools.airflow_tools import AirflowClient


def make_run(run_id, f1, gate="passed", dataset="v2.1", status="FINISHED", start=1):
    tags = {"quality_gate": gate, "mlflow.user": "x"}
    if dataset:
        tags.update({"dvc.dataset_version": dataset, "dvc.git_commit": "a4c92e1", "dvc.data_md5": "abc123"})
    return NS(
        info=NS(run_id=run_id, experiment_id="1", status=status, start_time=start),
        data=NS(metrics={"f1": f1, "accuracy": f1 - 0.01}, params={"n_estimators": "100"}, tags=tags),
    )


class FakeMlflow:
    def __init__(self, runs=None, fail=False):
        self.runs = runs if runs is not None else [
            make_run("run_a", 0.91, start=3),
            make_run("run_b", 0.88, start=2),
            make_run("run_c", 0.95, gate="failed", start=1),
        ]
        self.fail = fail

    def _chk(self):
        if self.fail:
            raise ConnectionError("mlflow down")

    def search_experiments(self):
        self._chk()
        return [NS(experiment_id="1", name="mlops-agent-training")]

    def get_experiment_by_name(self, name):
        self._chk()
        return NS(experiment_id="1", name=name) if name == "mlops-agent-training" else None

    def search_runs(self, ids, order_by=None, max_results=100):
        self._chk()
        return self.runs[:max_results]

    def get_run(self, run_id):
        self._chk()
        for r in self.runs:
            if r.info.run_id == run_id:
                return r
        raise KeyError(run_id)

    def list_artifacts(self, run_id):
        return [NS(path="model.joblib")]

    def search_registered_models(self):
        return []

    def search_model_versions(self, q):
        return []


class FakeResp:
    def __init__(self, code=200, body=None):
        self.status_code, self._b, self.text = code, body or {}, str(body)

    def json(self):
        return self._b


class FakeSession:
    """Routes Airflow API calls to canned responses. `down=True` simulates an unreachable server."""

    def __init__(self, down=False, failed=False):
        self.down, self.failed, self.calls = down, failed, []

    def post(self, url, **kw):
        self.calls.append(("POST", url))
        if self.down:
            import requests

            raise requests.ConnectionError("refused")
        return FakeResp(200, {"access_token": "tok"})

    def request(self, method, url, **kw):
        self.calls.append((method, url))
        if self.down:
            import requests

            raise requests.ConnectionError("refused")
        if url.endswith("/dags") and method == "GET":
            return FakeResp(200, {"dags": [{"dag_id": "training_pipeline", "is_paused": False}, {"dag_id": "other", "is_paused": False}]})
        if url.endswith("/dagRuns") and method == "GET":
            state = "failed" if self.failed else "success"
            return FakeResp(200, {"dag_runs": [{"dag_run_id": "run1", "state": state}]})
        if url.endswith("/dagRuns") and method == "POST":
            return FakeResp(200, {"dag_run_id": "manual__1", "state": "queued"})
        if url.endswith("/taskInstances"):
            st = "failed" if self.failed else "success"
            return FakeResp(200, {"task_instances": [{"task_id": "validate_dataset", "state": "success"}, {"task_id": "quality_gate", "state": st}]})
        return FakeResp(404, {})


def pod(name, ready=True, restarts=0, waiting=None, phase="Running"):
    state = NS(waiting=NS(reason=waiting) if waiting else None)
    cs = NS(ready=ready, restart_count=restarts, state=state, last_state=NS(terminated=None))
    return NS(
        metadata=NS(name=name, creation_timestamp=None),
        status=NS(phase=phase, container_statuses=[cs]),
        spec=NS(containers=[NS(image="mlops-inference:1.0.0")]),
    )


class FakeK8s:
    def __init__(self, pods=None, endpoints=2):
        self._pods = pods if pods is not None else [pod("ml-inference-1"), pod("ml-inference-2")]
        self.patched = []
        ep = NS(subsets=[NS(addresses=[object()] * endpoints)] if endpoints else [])
        dep = NS(
            metadata=NS(name="ml-inference", labels={"app": "ml-inference"}),
            spec=NS(replicas=2, template=NS(metadata=NS(annotations={"model-version": "v1", "dataset-version": "v2.1", "mlflow-run-id": "run_a"}), spec=NS(containers=[NS(image="mlops-inference:1.0.0")]))),
            status=NS(ready_replicas=2, available_replicas=2),
        )
        outer = self
        self.core = NS(
            list_namespaced_pod=lambda ns, label_selector=None: NS(items=outer._pods),
            read_namespaced_pod_log=lambda name, ns, tail_lines=50: "ERROR model file not found at /app/models/model.joblib\nreadiness failing",
            list_namespaced_service=lambda ns: NS(items=[NS(metadata=NS(name="ml-inference"), spec=NS(type="ClusterIP", cluster_ip="10.0.0.1", ports=[NS(port=80, target_port=8000)]))]),
            read_namespaced_endpoints=lambda n, ns: ep,
            list_namespaced_event=lambda ns: NS(items=[NS(type="Warning", involved_object=NS(name="ml-inference-1"), reason="Unhealthy", message="Readiness probe failed: 503", count=7)]),
        )
        self.apps = NS(
            list_namespaced_deployment=lambda ns, label_selector=None: NS(items=[dep]),
            patch_namespaced_deployment=lambda name, ns, body: outer.patched.append((name, ns)),
            patch_namespaced_deployment_scale=lambda name, ns, body: outer.patched.append((name, ns, body)),
        )


@pytest.fixture
def settings():
    return Settings(airflow_username="u", airflow_password="p")


@pytest.fixture
def make_agent(settings):
    def _make(mlflow=None, session=None, k8s=None):
        mlflow = mlflow or FakeMlflow()
        session = session or FakeSession()
        k8s = k8s or FakeK8s()
        return build_orchestrator(
            settings,
            mlflow_client_factory=lambda: mlflow,
            airflow_client=AirflowClient(settings, session),
            k8s_backend_factory=lambda: k8s,
        )

    return _make
