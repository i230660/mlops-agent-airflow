"""Central configuration. All values come from environment variables; no secrets in code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # General
    repo_path: str = field(default_factory=lambda: _env("REPO_PATH", "."))
    tool_timeout_s: float = field(default_factory=lambda: float(_env("TOOL_TIMEOUT_S", "20")))

    # DVC
    dvc_remote_name: str = field(default_factory=lambda: _env("DVC_REMOTE", ""))
    dvc_bin: str = field(default_factory=lambda: _env("DVC_BIN", "dvc"))
    git_bin: str = field(default_factory=lambda: _env("GIT_BIN", "git"))

    # MLflow
    mlflow_tracking_uri: str = field(default_factory=lambda: _env("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    dataset_tag_key: str = "dvc.dataset_version"
    git_commit_tag_key: str = "dvc.git_commit"
    dataset_hash_tag_key: str = "dvc.data_md5"

    # Airflow 3
    airflow_base_url: str = field(default_factory=lambda: _env("AIRFLOW_BASE_URL", "http://localhost:8080"))
    airflow_username: str = field(default_factory=lambda: _env("AIRFLOW_USERNAME", ""))
    airflow_password: str = field(default_factory=lambda: _env("AIRFLOW_PASSWORD", ""))
    airflow_allowed_dags: tuple = field(
        default_factory=lambda: tuple(x.strip() for x in _env("AIRFLOW_ALLOWED_DAGS", "training_pipeline").split(",") if x.strip())
    )

    # Kubernetes
    k8s_namespace: str = field(default_factory=lambda: _env("K8S_NAMESPACE", "mlops-demo"))
    k8s_allowed_namespaces: tuple = field(
        default_factory=lambda: tuple(x.strip() for x in _env("K8S_ALLOWED_NAMESPACES", "mlops-demo").split(",") if x.strip())
    )
    k8s_app_label: str = field(default_factory=lambda: _env("K8S_APP_LABEL", "app=ml-inference"))

    # LLM
    llm_model: str = field(default_factory=lambda: _env("AGENT_LLM_MODEL", "gemini-3.8-flash"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
    max_agent_steps: int = field(default_factory=lambda: int(_env("MAX_AGENT_STEPS", "8")))

    # Policy
    freshness_policy_days: int = field(default_factory=lambda: int(_env("MODEL_FRESHNESS_DAYS", "30")))
    min_quality_f1: float = field(default_factory=lambda: float(_env("MIN_QUALITY_F1", "0.80")))


def get_settings() -> Settings:
    return Settings()
