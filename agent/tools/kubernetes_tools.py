"""Kubernetes operations tools, restricted to approved namespaces.

Read tools are always allowed (inside approved namespaces). State-changing tools are
marked `mutating`, so the registry requires explicit user confirmation, and they re-check
the namespace allowlist themselves (defence in depth)."""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..config import Settings
from ..registry import Tool, ToolDenied, ToolRegistry, ToolUnavailable


class NsInput(BaseModel):
    namespace: Optional[str] = Field(default=None, description="Defaults to configured namespace")


class PodInput(BaseModel):
    namespace: Optional[str] = None
    pod_name: str = Field(pattern=r"^[a-z0-9][a-z0-9.\-]{0,252}$")
    tail_lines: int = Field(default=50, ge=1, le=500)


class RestartInput(BaseModel):
    namespace: Optional[str] = None
    deployment: str = Field(pattern=r"^[a-z0-9][a-z0-9.\-]{0,252}$")


class ScaleInput(BaseModel):
    namespace: Optional[str] = None
    deployment: str = Field(pattern=r"^[a-z0-9][a-z0-9.\-]{0,252}$")
    replicas: int = Field(ge=0, le=10)


class K8sBackend:
    """Wraps the official client. Injected/mocked in tests."""

    def __init__(self):
        try:
            from kubernetes import client, config

            try:
                config.load_incluster_config()
            except Exception:  # noqa: BLE001
                config.load_kube_config()
            self.core = client.CoreV1Api()
            self.apps = client.AppsV1Api()
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"Kubernetes not reachable: {exc}") from exc


def _age(ts) -> Optional[str]:
    if not ts:
        return None
    return str(dt.datetime.now(dt.timezone.utc) - ts).split(".")[0]


def build_kubernetes_tools(settings: Settings, registry: ToolRegistry, backend_factory=K8sBackend) -> None:
    cache: dict[str, Any] = {}

    def be():
        if "b" not in cache:
            cache["b"] = backend_factory()
        return cache["b"]

    def ns_of(value: Optional[str]) -> str:
        ns = value or settings.k8s_namespace
        if ns not in settings.k8s_allowed_namespaces:
            raise ToolDenied(f"namespace '{ns}' is not approved; allowed: {list(settings.k8s_allowed_namespaces)}")
        return ns

    def list_deployments(inp: NsInput):
        ns = ns_of(inp.namespace)
        deps = be().apps.list_namespaced_deployment(ns).items
        return [
            {
                "name": d.metadata.name,
                "desired": d.spec.replicas,
                "ready": d.status.ready_replicas or 0,
                "available": d.status.available_replicas or 0,
                "image": d.spec.template.spec.containers[0].image,
                "labels": d.metadata.labels,
            }
            for d in deps
        ]

    def list_pods(inp: NsInput):
        ns = ns_of(inp.namespace)
        pods = be().core.list_namespaced_pod(ns, label_selector=settings.k8s_app_label).items
        out = []
        for p in pods:
            cs = (p.status.container_statuses or [None])[0]
            waiting = cs.state.waiting.reason if cs and cs.state and cs.state.waiting else None
            terminated = cs.last_state.terminated.reason if cs and cs.last_state and cs.last_state.terminated else None
            out.append(
                {
                    "name": p.metadata.name,
                    "phase": p.status.phase,
                    "ready": bool(cs.ready) if cs else False,
                    "restarts": cs.restart_count if cs else 0,
                    "waiting_reason": waiting,
                    "last_terminated_reason": terminated,
                    "image": p.spec.containers[0].image,
                    "age": _age(p.metadata.creation_timestamp),
                }
            )
        return out

    def pod_logs(inp: PodInput):
        ns = ns_of(inp.namespace)
        text = be().core.read_namespaced_pod_log(inp.pod_name, ns, tail_lines=inp.tail_lines)
        return {"pod": inp.pod_name, "logs": text[-4000:]}

    def list_services(inp: NsInput):
        ns = ns_of(inp.namespace)
        svcs = be().core.list_namespaced_service(ns).items
        out = []
        for s in svcs:
            try:
                eps = be().core.read_namespaced_endpoints(s.metadata.name, ns)
                n_ep = sum(len(sub.addresses or []) for sub in (eps.subsets or []))
            except Exception:  # noqa: BLE001
                n_ep = None
            out.append(
                {
                    "name": s.metadata.name,
                    "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "ports": [{"port": p.port, "target": str(p.target_port)} for p in (s.spec.ports or [])],
                    "ready_endpoints": n_ep,
                }
            )
        return out

    def unhealthy(inp: NsInput):
        ns = ns_of(inp.namespace)
        pods = list_pods(NsInput(namespace=ns))
        bad = [p for p in pods if p["phase"] != "Running" or not p["ready"] or p["restarts"] > 0 or p["waiting_reason"]]
        events = be().core.list_namespaced_event(ns).items
        warn = [
            {"object": e.involved_object.name, "reason": e.reason, "message": (e.message or "")[:200], "count": e.count}
            for e in events
            if e.type == "Warning"
        ][-15:]
        return {"unhealthy_pods": bad, "warning_events": warn, "healthy": not bad}

    def app_version(inp: NsInput):
        ns = ns_of(inp.namespace)
        deps = be().apps.list_namespaced_deployment(ns, label_selector=settings.k8s_app_label).items
        out = []
        for d in deps:
            c = d.spec.template.spec.containers[0]
            ann = d.spec.template.metadata.annotations or {}
            out.append(
                {
                    "deployment": d.metadata.name,
                    "image": c.image,
                    "model_version_annotation": ann.get("model-version"),
                    "dataset_version_annotation": ann.get("dataset-version"),
                    "mlflow_run_annotation": ann.get("mlflow-run-id"),
                }
            )
        return out

    def restart(inp: RestartInput):
        ns = ns_of(inp.namespace)
        body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": dt.datetime.now(dt.timezone.utc).isoformat()}}}}}
        be().apps.patch_namespaced_deployment(inp.deployment, ns, body)
        return {"restarted": inp.deployment, "namespace": ns}

    def scale(inp: ScaleInput):
        ns = ns_of(inp.namespace)
        be().apps.patch_namespaced_deployment_scale(inp.deployment, ns, {"spec": {"replicas": inp.replicas}})
        return {"scaled": inp.deployment, "replicas": inp.replicas, "namespace": ns}

    reg = registry.register
    reg(Tool("k8s_list_deployments", "List Deployments with desired/ready replicas.", NsInput, list_deployments, "kubernetes"))
    reg(Tool("k8s_list_pods", "List application Pods with status, readiness and restart counts.", NsInput, list_pods, "kubernetes"))
    reg(Tool("k8s_pod_logs", "Get recent logs for a Pod.", PodInput, pod_logs, "kubernetes"))
    reg(Tool("k8s_list_services", "Inspect Services and ready endpoints.", NsInput, list_services, "kubernetes"))
    reg(Tool("k8s_unhealthy", "Identify unhealthy Pods, restarts and Warning events.", NsInput, unhealthy, "kubernetes"))
    reg(Tool("k8s_app_version", "Report deployed application image and model/dataset version annotations.", NsInput, app_version, "kubernetes"))
    reg(Tool("k8s_restart_deployment", "Rolling-restart a Deployment.", RestartInput, restart, "kubernetes", mutating=True))
    reg(Tool("k8s_scale_deployment", "Scale a Deployment.", ScaleInput, scale, "kubernetes", mutating=True))
