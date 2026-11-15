"""Agent, tool, safety and logging tests. All external systems are mocked."""
from __future__ import annotations

import json
import logging
import time

import pytest
from pydantic import BaseModel

from agent.logging_utils import redact
from agent.orchestrator import RulePlanner
from agent.registry import Tool, ToolDenied, ToolRegistry
from agent.tools.dvc_tools import run_allowed

from .conftest import FakeK8s, FakeMlflow, FakeSession, pod

FULL = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."


# 1. tool selection + execution across all four systems --------------------------------
def test_multi_tool_scenario_uses_all_four_systems(make_agent):
    resp = make_agent().handle(FULL)
    used = set(resp.tools_called)
    assert "mlflow_best_run" in used
    assert "dvc_dataset_for_run" in used
    assert "k8s_app_version" in used
    assert "airflow_dag_runs" in used
    assert any("dataset version v2.1" in f for f in resp.known_facts)
    assert any("git a4c92e1" in f for f in resp.known_facts)


# 2. best run by metric is not blindly declared production-ready -----------------------
def test_best_run_ignores_failed_gate_candidate_note_and_flags_caveats(make_agent):
    agent = make_agent()
    res = agent.registry.call("mlflow_best_run", {"metric": "f1"}, request_id="t")
    assert res.ok
    # highest f1 (0.95) belongs to a run whose quality gate failed -> must be flagged
    assert res.data["best"]["run_id"] == "run_c"
    assert any("quality_gate" in c for c in res.data["caveats"])
    assert res.data["production_ready"] is False


# 3. retraining must never be auto-started from an analysis question --------------------
def test_retraining_is_not_triggered_automatically(make_agent):
    session = FakeSession()
    resp = make_agent(session=session).handle(FULL)
    assert "airflow_trigger_dag" not in resp.tools_called
    assert not any(m == "POST" and u.endswith("/dagRuns") for m, u in session.calls)
    assert any("Retraining is justified" in a for a in resp.recommended_actions)  # run_c failed gate = evidence


def test_no_retrain_recommendation_without_evidence(make_agent):
    healthy = FakeMlflow(runs=[__import__("tests.conftest", fromlist=["make_run"]).make_run("ok1", 0.93)])
    resp = make_agent(mlflow=healthy).handle(FULL)
    assert any("Retraining is NOT indicated" in a for a in resp.recommended_actions)


# 4. unavailable tool => no invented results -------------------------------------------
def test_agent_does_not_invent_results_when_mlflow_down(make_agent):
    resp = make_agent(mlflow=FakeMlflow(fail=True)).handle("What is the best model by f1 in mlflow?")
    assert resp.evidence[0].status == "unavailable"
    assert not any("run_" in f for f in resp.known_facts)
    assert any("mlflow_best_run failed" in u for u in resp.uncertainties)
    assert "run_a" not in resp.answer and "0.91" not in resp.answer


def test_airflow_down_reported_as_unknown(make_agent):
    resp = make_agent(session=FakeSession(down=True)).handle("Show the airflow dag status")
    assert resp.evidence[0].status == "unavailable"
    assert any("unknown" in u for u in resp.uncertainties)
    assert resp.known_facts == []


# 5. unauthorized Kubernetes modifications rejected -------------------------------------
def test_k8s_mutation_requires_confirmation(make_agent):
    k8s = FakeK8s()
    agent = make_agent(k8s=k8s)
    res = agent.registry.call("k8s_restart_deployment", {"deployment": "ml-inference"}, request_id="t", confirmed=False)
    assert res.status == "confirmation_required"
    assert k8s.patched == []


def test_k8s_mutation_rejected_outside_approved_namespace(make_agent):
    k8s = FakeK8s()
    agent = make_agent(k8s=k8s)
    res = agent.registry.call("k8s_restart_deployment", {"deployment": "ml-inference", "namespace": "kube-system"}, request_id="t", confirmed=True)
    assert res.status == "denied"
    assert k8s.patched == []


def test_k8s_confirmed_restart_in_approved_namespace(make_agent):
    k8s = FakeK8s()
    res = make_agent(k8s=k8s).registry.call("k8s_restart_deployment", {"deployment": "ml-inference"}, request_id="t", confirmed=True)
    assert res.ok and k8s.patched == [("ml-inference", "mlops-demo")]


def test_k8s_read_in_unapproved_namespace_denied(make_agent):
    res = make_agent().registry.call("k8s_list_pods", {"namespace": "default"}, request_id="t")
    assert res.status == "denied"


# 6. Airflow trigger safety -------------------------------------------------------------
def test_airflow_trigger_needs_confirmation_and_approved_dag(make_agent):
    reg = make_agent().registry
    assert reg.call("airflow_trigger_dag", {}, request_id="t").status == "confirmation_required"
    assert reg.call("airflow_trigger_dag", {"dag_id": "other"}, request_id="t", confirmed=True).status == "denied"
    ok = reg.call("airflow_trigger_dag", {"dag_id": "training_pipeline"}, request_id="t", confirmed=True)
    assert ok.ok and ok.data["dag_run_id"] == "manual__1"


# 7. invalid requests + timeouts -------------------------------------------------------
def test_invalid_arguments_rejected(make_agent):
    reg = make_agent().registry
    assert reg.call("mlflow_list_runs", {"limit": 9999}, request_id="t").status == "invalid_request"
    assert reg.call("does_not_exist", {}, request_id="t").status == "invalid_request"
    assert reg.call("k8s_pod_logs", {"pod_name": "Bad Name; rm -rf /"}, request_id="t").status == "invalid_request"


def test_tool_timeout_is_reported():
    class In(BaseModel):
        pass

    reg = ToolRegistry(default_timeout_s=0.1)
    reg.register(Tool("slow", "slow", In, lambda _: time.sleep(1), "test"))
    res = reg.call("slow", {}, request_id="t")
    assert res.status == "timeout"


def test_unexpected_exception_becomes_error_result():
    class In(BaseModel):
        pass

    def boom(_):
        raise RuntimeError("kaboom")

    reg = ToolRegistry()
    reg.register(Tool("boom", "b", In, boom, "test"))
    res = reg.call("boom", {}, request_id="t")
    assert res.status == "error" and "kaboom" in res.error


# 8. DVC command allowlist --------------------------------------------------------------
def test_dvc_allowlist_blocks_dangerous_commands(tmp_path):
    for cmd in (["dvc", "remove", "x"], ["git", "push"], ["rm", "-rf", "/"], ["dvc", "gc", "-f"]):
        with pytest.raises(ToolDenied):
            run_allowed(cmd, str(tmp_path))


def test_dvc_unsafe_revision_rejected(make_agent):
    res = make_agent().registry.call("dvc_current_version", {"rev": "HEAD; rm -rf /"}, request_id="t")
    assert res.status == "denied"


def test_dvc_provenance_missing_is_reported_not_guessed(make_agent):
    from .conftest import make_run

    agent = make_agent(mlflow=FakeMlflow(runs=[make_run("nodvc", 0.9, dataset=None)]))
    res = agent.registry.call("dvc_dataset_for_run", {"run_id": "nodvc"}, request_id="t")
    assert res.ok and res.data["provenance_recorded"] is False and res.data["dataset_version"] is None


def test_dvc_list_datasets_reads_dvc_files(tmp_path):
    from agent.config import Settings
    from agent.main import build_registry

    (tmp_path / "data.csv.dvc").write_text("outs:\n- md5: deadbeef\n  size: 10\n  path: data.csv\n")
    reg = build_registry(Settings(repo_path=str(tmp_path)), mlflow_client_factory=lambda: FakeMlflow(), k8s_backend_factory=lambda: FakeK8s())
    res = reg.call("dvc_list_datasets", {}, request_id="t")
    assert res.ok and res.data["tracked_files"][0]["md5"] == "deadbeef"


# 9. Kubernetes diagnosis from evidence ------------------------------------------------
def test_deployment_diagnosis_uses_pods_events_logs_services(make_agent):
    k8s = FakeK8s(pods=[pod("ml-inference-1", ready=False, waiting="CrashLoopBackOff", restarts=5), pod("ml-inference-2", ready=False, restarts=4)], endpoints=0)
    resp = make_agent(k8s=k8s).handle("Why is my prediction API unavailable?")
    joined = " ".join(resp.known_facts)
    assert "CrashLoopBackOff" in joined
    assert "Readiness probe failed" in joined
    assert "model file not found" in joined  # from real (mock) pod logs
    assert any("no ready endpoints" in a for a in resp.recommended_actions)


# 10. Airflow failure diagnosis (injected failure) -----------------------------------
def test_airflow_failed_task_identified(make_agent):
    resp = make_agent(session=FakeSession(failed=True)).handle("Which airflow dag tasks failed?")
    assert any("quality_gate(failed)" in f for f in resp.known_facts)


# 11. structured logging: names, status, duration, request id, no secrets -------------
def test_logs_are_structured_and_secret_free(make_agent, caplog):
    with caplog.at_level(logging.INFO, logger="mlops-agent"):
        make_agent().registry.call("mlflow_list_experiments", {}, request_id="req-123")
    rec = [r for r in caplog.records if getattr(r, "fields", {}).get("event") == "tool_call"][0]
    f = rec.fields
    assert f["tool"] == "mlflow_list_experiments" and f["status"] == "ok" and f["request_id"] == "req-123" and f["duration_ms"] >= 0


def test_redaction():
    out = redact({"password": "hunter2", "nested": {"api_key": "x"}, "msg": "Bearer abcdefghijkl", "ok": 1})
    dumped = json.dumps(out)
    assert "hunter2" not in dumped and "abcdefghijkl" not in dumped and out["ok"] == 1


def test_airflow_password_never_logged(make_agent, caplog):
    with caplog.at_level(logging.INFO, logger="mlops-agent"):
        make_agent().registry.call("airflow_list_dags", {}, request_id="r")
    blob = " ".join(str(getattr(r, "fields", "")) + r.getMessage() for r in caplog.records)
    assert caplog.records and "tok" not in blob and "Bearer" not in blob


# 12. planner + API --------------------------------------------------------------------
def test_planner_maps_intents():
    from agent.config import Settings

    p = RulePlanner()
    assert p.plan("list airflow dags", Settings())[0][0] == "airflow_dag_runs"
    assert p.plan("trigger the training pipeline", Settings())[0][0] == "airflow_trigger_dag"
    assert p.plan("hello there", Settings()) == []


def test_unmatched_request_yields_no_facts(make_agent):
    resp = make_agent().handle("tell me a joke")
    assert resp.tools_called == [] and resp.known_facts == []
    assert any("No tool matched" in u for u in resp.uncertainties)


def test_chat_api_endpoint(make_agent, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.main as m

    monkeypatch.setattr(m, "_orch", make_agent())
    c = TestClient(m.app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/chat", json={"message": "list airflow dags status"})
    assert r.status_code == 200 and r.json()["tools_called"] == ["airflow_dag_runs"]
    assert c.post("/chat", json={"message": ""}).status_code == 422


def test_each_module_has_at_least_three_tools(make_agent):
    reg = make_agent().registry
    for module in ("dvc", "mlflow", "airflow", "kubernetes"):
        assert len(reg.by_module(module)) >= 3, module


# 13. Gemini loop (fake client, no network) ---------------------------------------------
def test_gemini_schema_conversion_strips_unsupported_keys(make_agent):
    from agent.llm import function_declarations

    decls = {d["name"]: d for d in function_declarations(make_agent().registry.schemas())}
    assert "parameters" not in decls["mlflow_list_experiments"]  # empty-object params omitted
    props = decls["mlflow_list_runs"]["parameters"]["properties"]
    assert props["limit"]["maximum"] == 50 and "default" not in props["limit"] and "title" not in props["limit"]
    assert props["experiment_name"]["type"] == "string"  # Optional[str] collapsed


def test_gemini_loop_executes_tools_through_registry(make_agent):
    from types import SimpleNamespace as NS

    from agent.llm import GeminiLLM

    class FakeModels:
        def __init__(self):
            self.n = 0

        def generate_content(self, model, contents, config):
            self.n += 1
            if self.n == 1:
                fc = NS(name="mlflow_best_run", args={"metric": "f1"})
                return NS(function_calls=[fc], candidates=[NS(content="assistant-turn")], text=None)
            return NS(function_calls=None, candidates=[], text="final answer from tool evidence")

    agent = make_agent()
    llm = GeminiLLM(agent.settings, client=NS(models=FakeModels()))
    agent.llm = llm
    resp = agent.handle("which is the best model?")
    assert resp.tools_called == ["mlflow_best_run"]
    assert resp.answer == "final answer from tool evidence"
    assert resp.evidence[0].status == "ok"
