# Technical Report — MLOps AI Agent

> Fill the **[EVIDENCE]** slots with your own screenshots/outputs from a live run. Nothing below is pre-recorded evidence;
> the sample values are illustrative only.

## 1. System architecture
See `architecture.md` (flow, provenance design, retraining policy). Include an exported diagram here.

## 2. Tool definitions and permissions
See the permissions table in `architecture.md`. Mutating tools: `airflow_trigger_dag`, `k8s_restart_deployment`, `k8s_scale_deployment`, `dvc_checkout_version`.

## 3. Evidence from each system  **[EVIDENCE]**
* DVC: `dvc status`, `.dvc` file, `git tag`, `dvc checkout` output → `screenshots/dvc.png`
* MLflow: experiment runs table with `dvc.*` tags → `screenshots/mlflow.png`
* Airflow: DAG graph + a run with task states → `screenshots/airflow.png`
* Kubernetes: `kubectl -n mlops-demo get pods,svc,deploy` and `describe deploy` → `screenshots/k8s.png`

## 4. Five tested conversations  **[EVIDENCE — paste the real JSON `evidence` from `/chat`]**
| # | Scenario | Prompt | Expected tools |
|---|---|---|---|
| 1 | Dataset provenance | "Which dataset version was used for run `<run_id>`?" | `dvc_dataset_for_run` (+ `dvc_explain_reproduce`) |
| 2 | Model selection | "Compare the last five training runs and recommend the best candidate by F1." | `mlflow_list_runs`, `mlflow_best_run` |
| 3 | Pipeline execution | "Trigger the model training pipeline and report whether it succeeded." (send with `confirm_actions:true`, then ask "show dag run status") | `airflow_trigger_dag`, `airflow_dag_runs`, `airflow_failed_tasks` |
| 4 | Deployment diagnosis | "Why is my prediction API unavailable?" | `k8s_unhealthy`, `k8s_list_pods`, `k8s_list_services`, `k8s_pod_logs` |
| 5 | Multi-tool | "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary." | MLflow + DVC + Kubernetes + Airflow |

## 5. Failure investigations

### Failure A — Missing model file (infrastructure layer)
**Inject:** build the image without `models/model.joblib` (or `kubectl -n mlops-demo set env deploy/ml-inference MODEL_PATH=/nope`).
**Symptom:** pods Running but `READY 0/1`; Service has 0 endpoints; clients get connection errors.
**Agent diagnosis:** ask "Why is my prediction API unavailable?" — evidence: `k8s_unhealthy` (not ready, Warning event *Readiness probe failed: 503*), `k8s_list_services` (0 ready endpoints), `k8s_pod_logs` (*model file not found*).
**Root cause:** readiness probe (`/ready`) correctly fails because the model cannot be loaded; liveness stays green so the pod is not restarted.
**Resolution:** restore the model path / rebuild image; `kubectl rollout restart` (or ask the agent with confirmation).

### Failure B — Data-quality regression blocks promotion (data/model layer)
**Inject:** `python scripts/make_dataset.py 0.45 && dvc add data/dataset.csv`, commit, trigger the DAG with `MIN_QUALITY_F1=0.80`.
**Symptom:** `quality_gate` task fails; run state `failed`; MLflow run tagged `quality_gate=failed`.
**Agent diagnosis:** "Which airflow tasks failed?" then the multi-tool question — evidence: `airflow_failed_tasks` (quality_gate), `airflow_task_logs` (*quality gate FAILED: f1=… < 0.80; promotion blocked*), `mlflow_best_run` caveat, `dvc_dataset_for_run` showing the new dataset hash.
**Root cause:** label noise in the new dataset version lowered F1 below the policy.
**Resolution:** `git checkout v1.0 && dvc checkout` to reproduce the good dataset, fix the data, re-run the pipeline.

(Other injectable failures: stop MLflow server → agent reports `unavailable` and refuses to guess; wrong Airflow password → `unavailable: authentication failed`.)

## 6. Known limitations and improvements
* Keyword rule planner limits offline phrasing; LLM planner recommended for the demo.
* No model-age/freshness signal is stored; add a `trained_at` tag and policy check.
* Add a human-approval workflow (signed confirmation tokens) instead of a boolean flag.
* Integration tests against kind + MLflow + Airflow in a dedicated CI job.
