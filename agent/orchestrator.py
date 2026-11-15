
from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from .config import Settings
from .logging_utils import get_logger
from .registry import ToolRegistry
from .schemas import ChatResponse, Evidence, ToolResult

logger = get_logger()

SYSTEM_PROMPT = """You are an MLOps assistant with tools for DVC, MLflow, Airflow and Kubernetes.
Rules:
1. Every operational claim (metrics, versions, pod status, DAG state) must come from a tool result in this conversation. Never answer from memory.
2. If a tool returns an error, timeout or unavailable status, say exactly that and what is therefore unknown. Do not guess.
3. State-changing tools (triggering DAGs, restarting/scaling deployments) require explicit user confirmation. Never trigger retraining from an analysis question.
4. A high metric alone does not make a model production-ready. Consider quality gate, provenance and deployment health.
5. Recommend retraining only with evidence: a failed quality gate, an explicit freshness-policy breach, or supplied degradation results.
6. Structure the final answer as: Known facts / Uncertainties / Recommended actions."""


class RulePlanner:
    """Deterministic planner: maps intent keywords to tool calls (sequence-aware)."""

    def plan(self, message: str, settings: Settings) -> list[tuple[str, dict]]:
        m = message.lower()
        calls: list[tuple[str, dict]] = []
        full_scenario = ("best" in m and ("dataset" in m or "version" in m) and ("deploy" in m or "kubernetes" in m or "retrain" in m))
        if full_scenario or "retrain" in m or "retraining" in m:
            return [
                ("mlflow_best_run", {"metric": "f1", "limit": 10}),
                ("DVC_FROM_BEST", {}),
                ("k8s_app_version", {}),
                ("k8s_unhealthy", {}),
                ("airflow_dag_runs", {"dag_id": "training_pipeline", "limit": 5}),
                ("airflow_failed_tasks", {"dag_id": "training_pipeline"}),
            ]
        run_id = re.search(r"\brun[\s_-]*(?:id)?[:\s]*([0-9a-f]{8,32})\b", m)
        if any(w in m for w in ("dataset", "dvc", "provenance", "data version", "reproduce")):
            if run_id:
                calls.append(("dvc_dataset_for_run", {"run_id": run_id.group(1)}))
            elif "reproduce" in m:
                calls.append(("dvc_explain_reproduce", {"rev": "HEAD"}))
            elif "changed" in m or "change" in m:
                calls.append(("dvc_detect_changes", {}))
            else:
                calls += [("dvc_list_datasets", {}), ("dvc_current_version", {"rev": "HEAD"})]
        if any(w in m for w in ("mlflow", "experiment", "metric", "f1", "best model", "compare", "runs")):
            calls.append(("mlflow_best_run", {"metric": "f1"}) if ("best" in m or "recommend" in m) else ("mlflow_list_runs", {"limit": 5}))
        if any(w in m for w in ("airflow", "dag", "pipeline", "workflow")):
            if any(w in m for w in ("trigger", "run the", "start", "launch")):
                calls.append(("airflow_trigger_dag", {"dag_id": "training_pipeline"}))
            elif "fail" in m:
                calls.append(("airflow_failed_tasks", {"dag_id": "training_pipeline"}))
            else:
                calls.append(("airflow_dag_runs", {"dag_id": "training_pipeline", "limit": 5}))
        if any(w in m for w in ("kubernetes", "k8s", "pod", "deployment", "unavailable", "api down", "prediction api", "service")):
            if "restart" in m:
                calls.append(("k8s_restart_deployment", {"deployment": "ml-inference"}))
            elif any(w in m for w in ("why", "unavailable", "unhealthy", "down", "crash")):
                calls += [("k8s_unhealthy", {}), ("k8s_list_pods", {}), ("k8s_list_services", {})]
                calls.append(("K8S_LOGS_FROM_PODS", {}))
            else:
                calls += [("k8s_list_deployments", {}), ("k8s_list_pods", {})]
                if any(w in m for w in ("version", "model", "dataset", "image")):
                    calls.append(("k8s_app_version", {}))
        return calls


class Orchestrator:
    def __init__(self, settings: Settings, registry: ToolRegistry, llm: Optional[Any] = None):
        self.settings = settings
        self.registry = registry
        self.llm = llm  # object with .run(system, message, tools, call_tool) -> str ; optional

    # ------------------------------------------------------------------ public
    def handle(self, message: str, confirm_actions: bool = False, request_id: Optional[str] = None) -> ChatResponse:
        rid = request_id or uuid.uuid4().hex[:12]
        logger.info("request_received", extra={"fields": {"event": "request", "request_id": rid, "chars": len(message)}})
        results: list[ToolResult] = []

        def call_tool(name: str, args: dict) -> ToolResult:
            res = self.registry.call(name, args, request_id=rid, confirmed=confirm_actions)
            results.append(res)
            return res

        llm_text: Optional[str] = None
        llm_note: Optional[str] = None
        if self.llm is None:
            llm_note = "No GEMINI_API_KEY loaded in the agent process; used the keyword rule planner instead of the LLM."
        else:
            try:
                llm_text = self.llm.run(SYSTEM_PROMPT, message, self.registry.schemas(), call_tool)
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None)
                logger.warning("llm_failed", extra={"fields": {"request_id": rid, "error": type(exc).__name__, "code": code}})
                llm_note = f"The LLM call failed ({type(exc).__name__}, code={code}); fell back to the keyword rule planner."
                results.clear()
        if llm_text is None:
            self._run_rule_plan(message, call_tool)

        return self._compose(rid, message, results, llm_text, llm_note)

    # ----------------------------------------------------------------- planning
    def _run_rule_plan(self, message: str, call_tool) -> None:
        plan = RulePlanner().plan(message, self.settings)
        best: Optional[ToolResult] = None
        pods: Optional[ToolResult] = None
        for name, args in plan:
            if name == "DVC_FROM_BEST":
                run_id = _best_run_id(best)
                if run_id:
                    call_tool("dvc_dataset_for_run", {"run_id": run_id})
                continue
            if name == "K8S_LOGS_FROM_PODS":
                for p in _bad_pods(pods)[:2]:
                    call_tool("k8s_pod_logs", {"pod_name": p, "tail_lines": 40})
                continue
            res = call_tool(name, args)
            if name == "mlflow_best_run":
                best = res
            if name == "k8s_list_pods":
                pods = res
            if name == "k8s_unhealthy" and res.ok:
                pods = ToolResult(tool="k8s_list_pods", status="ok", data=res.data.get("unhealthy_pods", []))

    # ----------------------------------------------------------------- answering
    def _compose(self, rid: str, message: str, results: list[ToolResult], llm_text: Optional[str], llm_note: Optional[str] = None) -> ChatResponse:
        facts, unknown, actions = analyse(results, self.settings)
        if llm_note:
            unknown.insert(0, llm_note)
        if not results:
            unknown.append("No tool matched this request, so no operational facts were collected.")
        answer = llm_text or render_answer(facts, unknown, actions)
        return ChatResponse(
            request_id=rid,
            answer=answer,
            planner="gemini" if llm_text else "rules",
            tools_called=[r.tool for r in results],
            evidence=[Evidence(tool=r.tool, status=r.status, summary=r.data if r.ok else r.error) for r in results],
            known_facts=facts,
            uncertainties=unknown,
            recommended_actions=actions,
        )


# ---------------------------------------------------------------------- helpers
def _best_run_id(res: Optional[ToolResult]) -> Optional[str]:
    if res and res.ok and isinstance(res.data, dict) and res.data.get("best"):
        return res.data["best"]["run_id"]
    return None


def _bad_pods(res: Optional[ToolResult]) -> list[str]:
    if not (res and res.ok and isinstance(res.data, list)):
        return []
    return [p["name"] for p in res.data if (not p.get("ready")) or p.get("restarts", 0) > 0 or p.get("waiting_reason")]


def analyse(results: list[ToolResult], settings: Settings) -> tuple[list[str], list[str], list[str]]:
    """Correlate evidence into (known facts, uncertainties, recommended actions)."""
    facts: list[str] = []
    unknown: list[str] = []
    actions: list[str] = []
    by_tool: dict[str, list[ToolResult]] = {}
    for r in results:
        by_tool.setdefault(r.tool, []).append(r)

    for r in results:
        if r.status == "confirmation_required":
            actions.append(f"'{r.tool}' changes state and was NOT executed. Confirm explicitly to proceed.")
        elif r.status == "denied":
            unknown.append(f"{r.tool} was denied by policy: {r.error}")
        elif not r.ok:
            unknown.append(f"{r.tool} failed ({r.status}): {r.error}. Its results are unknown.")

    quality_failed = False
    best_data = _first_ok(by_tool, "mlflow_best_run")
    if best_data is not None:
        best = best_data.get("best")
        if best:
            m = best_data["metric"]
            facts.append(f"Best recent run by {m}: {best['run_id']} ({m}={best['metrics'].get(m)}), considered {best_data['considered']} finished runs.")
            gate = best["tags"].get("quality_gate")
            if gate == "failed":
                quality_failed = True
            for c in best_data.get("caveats", []):
                unknown.append(f"Production readiness of {best['run_id']}: {c}.")
            if best["metrics"].get(m, 0) < settings.min_quality_f1:
                quality_failed = True
                facts.append(f"{m}={best['metrics'].get(m)} is below the policy minimum {settings.min_quality_f1}.")
        else:
            unknown.append(best_data.get("note", "No best run could be determined."))

    ds = _first_ok(by_tool, "dvc_dataset_for_run")
    if ds is not None:
        if ds["provenance_recorded"]:
            facts.append(f"Run {ds['run_id']} used dataset version {ds['dataset_version']} (git {ds['git_commit']}, md5 {ds['data_md5']}).")
        else:
            unknown.append(ds["note"])

    for tool in ("dvc_list_datasets", "dvc_current_version", "dvc_detect_changes", "dvc_explain_reproduce"):
        d = _first_ok(by_tool, tool)
        if d is not None:
            facts.append(f"{tool}: {_short(d)}")

    deps = _first_ok(by_tool, "k8s_app_version")
    if deps:
        for d in deps:
            facts.append(f"Deployed {d['deployment']}: image {d['image']}, model-version={d['model_version_annotation']}, dataset-version={d['dataset_version_annotation']}, mlflow-run={d['mlflow_run_annotation']}.")
        best_id = _best_run_id(by_tool.get("mlflow_best_run", [None])[0]) if "mlflow_best_run" in by_tool else None
        if best_id and deps and deps[0].get("mlflow_run_annotation") not in (None, best_id):
            unknown.append(f"Deployed model run ({deps[0]['mlflow_run_annotation']}) differs from best recent run ({best_id}).")
            actions.append("Consider promoting the best run via your normal release process (no automatic action taken).")
    unh = _first_ok(by_tool, "k8s_unhealthy")
    if unh is not None:
        if unh["healthy"]:
            facts.append("All application pods are Running, Ready, with no restarts.")
        else:
            for p in unh["unhealthy_pods"]:
                facts.append(f"Unhealthy pod {p['name']}: phase={p['phase']}, ready={p['ready']}, restarts={p['restarts']}, waiting={p['waiting_reason']}, last_terminated={p['last_terminated_reason']}.")
            for e in unh["warning_events"][:5]:
                facts.append(f"Warning event on {e['object']}: {e['reason']} - {e['message']}")
            actions.append("Investigate the pod events/logs above; a restart is available but needs your explicit confirmation.")
    for r in by_tool.get("k8s_pod_logs", []):
        if r.ok:
            tail = r.data["logs"].strip().splitlines()[-3:]
            facts.append(f"Last log lines of {r.data['pod']}: {' | '.join(tail)}")
    svc = _first_ok(by_tool, "k8s_list_services")
    if svc:
        for s in svc:
            facts.append(f"Service {s['name']} ({s['type']}) has {s['ready_endpoints']} ready endpoint(s).")
            if s["ready_endpoints"] == 0:
                actions.append(f"Service {s['name']} has no ready endpoints: traffic cannot be served (check readiness probes).")

    runs = _first_ok(by_tool, "airflow_dag_runs")
    if runs is not None:
        if runs:
            facts.append("Recent training DAG runs: " + ", ".join(f"{r['dag_run_id']}={r['state']}" for r in runs[:5]) + ".")
            if runs[0]["state"] == "failed":
                quality_failed = True
        else:
            unknown.append("Airflow reports no runs for the training DAG.")
    ft = _first_ok(by_tool, "airflow_failed_tasks")
    if ft is not None:
        if ft["all_ok"]:
            facts.append(f"No failed tasks in run {ft['dag_run_id']}.")
        else:
            facts.append(f"Failed tasks in run {ft['dag_run_id']}: " + ", ".join(f"{t['task_id']}({t['state']})" for t in ft["failed_tasks"]) + ".")
            quality_failed = True
    trig = _first_ok(by_tool, "airflow_trigger_dag")
    if trig:
        facts.append(f"Triggered DAG {trig['dag_id']}: run {trig['dag_run_id']} state={trig['state']}.")

    lst = _first_ok(by_tool, "mlflow_list_runs")
    if lst is not None:
        for r in lst:
            facts.append(f"Run {r['run_id']} [{r['status']}] metrics={r['metrics']}")

    dep_list = _first_ok(by_tool, "k8s_list_deployments")
    if dep_list is not None:
        for d in dep_list:
            facts.append(f"Deployment {d['name']}: {d['ready']}/{d['desired']} ready, {d['available']} available, image {d['image']}.")
    pod_list = _first_ok(by_tool, "k8s_list_pods")
    if pod_list is not None and "k8s_unhealthy" not in by_tool:
        facts.append(
            "Pods: "
            + "; ".join(f"{p['name']} ({p['phase']}, ready={p['ready']}, restarts={p['restarts']}"
                        + (f", {p['waiting_reason']}" if p.get("waiting_reason") else "") + ")" for p in pod_list)
            + "."
        )
    _covered = {
        "mlflow_best_run", "mlflow_list_runs", "dvc_dataset_for_run", "dvc_list_datasets", "dvc_current_version",
        "dvc_detect_changes", "dvc_explain_reproduce", "k8s_app_version", "k8s_unhealthy", "k8s_pod_logs",
        "k8s_list_services", "k8s_list_deployments", "k8s_list_pods", "airflow_dag_runs", "airflow_failed_tasks",
        "airflow_trigger_dag",
    }
    for r in results:  # never silently drop evidence from a tool that succeeded
        if r.ok and r.tool not in _covered:
            facts.append(f"{r.tool}: {_short(r.data, 400)}")

    # retraining policy (evidence-gated)
    wants_retrain = any(r.tool.startswith("airflow_") or r.tool == "mlflow_best_run" for r in results) and best_data is not None
    if wants_retrain:
        if quality_failed:
            actions.append("Retraining is justified by evidence above (failed quality gate/run or metric below policy). Not started automatically; confirm to trigger 'training_pipeline'.")
        else:
            actions.append("Retraining is NOT indicated: no failed quality gate, failed run, or supplied degradation evidence. Note: no freshness check is possible without model timestamps/policy breach data.")
    return facts, unknown, actions


def _first_ok(by_tool: dict[str, list[ToolResult]], name: str):
    for r in by_tool.get(name, []):
        if r.ok:
            return r.data
    return None


def _short(d: Any, n: int = 300) -> str:
    s = str(d)
    return s if len(s) <= n else s[:n] + "..."


def render_answer(facts: list[str], unknown: list[str], actions: list[str]) -> str:
    def block(title: str, items: list[str]) -> str:
        return f"{title}:\n" + ("\n".join(f"- {i}" for i in items) if items else "- (none)")

    return "\n\n".join([block("Known facts", facts), block("Uncertainties", unknown), block("Recommended actions", actions)])