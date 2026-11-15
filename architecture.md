# Architecture

## Flow
1. **FastAPI `/chat`** validates the request (`ChatRequest`: 1–4000 chars, optional `confirm_actions`).
2. **Orchestrator** plans tool calls. Two planners share one execution path:
   * `GeminiLLM` — Gemini function-calling loop (bounded by `MAX_AGENT_STEPS`), used when `GEMINI_API_KEY` is set. Tool schemas are converted from pydantic to Gemini's OpenAPI subset; automatic function calling is disabled so every call goes through the registry.
   * `RulePlanner` — deterministic keyword planner for offline use and tests. Dependent steps (best run → its dataset; unhealthy pods → their logs) are chained from earlier results.
3. **ToolRegistry** is the only way to reach a backend. For every call it: validates arguments (pydantic) → enforces confirmation for mutating tools → runs with a timeout → maps failures to a status (`ok`, `error`, `timeout`, `invalid_request`, `unavailable`, `denied`, `confirmation_required`) → writes a structured log line. Tools never raise into the agent.
4. **Evidence composer** (`analyse`) builds *Known facts*, *Uncertainties*, *Recommended actions* strictly from `ToolResult` objects. A failed tool becomes an uncertainty, never a fact.

## Provenance design (DVC ↔ MLflow)
Training logs these run tags: `dvc.dataset_version` (git tag on HEAD, else `md5-<8>`), `dvc.git_commit`, `dvc.data_md5`, `quality_gate`.
"Which dataset was used for run X?" reads those tags — the answer comes from recorded provenance, not inference.
The Kubernetes Deployment carries `model-version`, `dataset-version`, `mlflow-run-id` pod-template annotations so deployed ↔ trained can be correlated.

## Tools and permissions
| Module | Tool | Mutating | Guard |
|---|---|---|---|
| DVC | `dvc_list_datasets`, `dvc_current_version`, `dvc_detect_changes`, `dvc_compare_versions`, `dvc_dataset_for_run`, `dvc_explain_reproduce` | no | command allowlist, rev regex |
| DVC | `dvc_checkout_version` | restores tracked data in working dir | allowlist only (`dvc checkout`) |
| MLflow | `mlflow_list_experiments`, `_list_runs`, `_get_run`, `_best_run`, `_compare_experiments`, `_model_registry` | no | `MlflowClient` only |
| Airflow | `airflow_list_dags`, `_dag_runs`, `_task_instances`, `_failed_tasks`, `_task_logs` | no | JWT |
| Airflow | `airflow_trigger_dag` | **yes** | confirmation + approved DAG list |
| Kubernetes | `k8s_list_deployments`, `_list_pods`, `_pod_logs`, `_list_services`, `_unhealthy`, `_app_version` | no | approved namespaces |
| Kubernetes | `k8s_restart_deployment`, `k8s_scale_deployment` | **yes** | confirmation + approved namespace + RBAC |

## Retraining policy
Retraining is never started from an analysis question. It is *recommended* only when evidence exists: failed quality-gate tag, metric below `MIN_QUALITY_F1`, or a failed latest DAG run / failed tasks. The agent explicitly states when no freshness check is possible (no timestamp/policy evidence).

## DAG
`validate_dataset → preprocess_data → train_and_evaluate → quality_gate`, retries=2 with 30 s delay, task logging. Validation or gate failures raise, so nothing downstream (promotion) runs.
Note: the figure's "evaluate and log to MLflow" is implemented inside `train_and_evaluate` so one process owns the MLflow run id; the gate task then reads the result.

## Known limitations
* Rule planner is keyword-based; the LLM planner handles free-form phrasing better.
* "Freshness" needs a model-age signal; none is recorded by default.
* `dvc_checkout_version` restores data for the *current* checkout; it does not switch git revisions (deliberate safety choice).
* Integration with live services is not exercised in CI (mocked); run the demo scenarios manually for evidence.
