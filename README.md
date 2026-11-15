# Assignment 03 — Production-Ready MLOps AI Agent

A conversational agent that answers MLOps questions by calling **real tools** for DVC, MLflow, Airflow 3 and Kubernetes,
and answers only from tool evidence ("Known facts / Uncertainties / Recommended actions").

```
User -> FastAPI (/chat) -> Orchestrator (LLM or rule planner) -> ToolRegistry -> DVC | MLflow | Airflow | Kubernetes
                                                      \-> evidence-based response
```

See `architecture.md` for the design and `docs/technical_report.md` for the report template to complete with your evidence.

## Layout
| Path | Purpose |
|---|---|
| `agent/` | config, schemas, registry (validation/timeouts/confirmation/logging), orchestrator, LLM loop, tools |
| `agent/tools/` | `dvc_tools.py`, `mlflow_tools.py`, `airflow_tools.py`, `kubernetes_tools.py` (3+ tools each) |
| `api/main.py` | FastAPI **inference** service deployed on Kubernetes |
| `dags/training_pipeline.py` | Airflow 3 DAG: validate → preprocess → train/evaluate/log → quality gate |
| `training/train.py` | trains, logs to MLflow with DVC provenance tags and a quality-gate tag |
| `kubernetes/` | Deployment (2 replicas, probes, limits), ClusterIP Service, ConfigMap, namespace, agent RBAC |
| `tests/` | 29 pytest tests, all external services mocked |
| `.github/workflows/ci.yml` | ruff + pytest on pull requests |

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env     # fill in values; never commit .env
set -a; source .env; set +a
```

## 2. Initialise Git + DVC and create a dataset version
```bash
git init && dvc init
python scripts/make_dataset.py            # writes data/dataset.csv
dvc add data/dataset.csv
git add data/dataset.csv.dvc data/.gitignore dvc.yaml .dvc && git commit -m "dataset v1"
git tag v1.0
# optional remote:  dvc remote add -d storage /path/or/s3://bucket && dvc push
```
Reproduce a previous version: `git checkout v1.0 && dvc checkout` (the agent's `dvc_explain_reproduce` returns these steps).

## 3. MLflow
```bash
mlflow server --host 0.0.0.0 --port 5000        # separate terminal; MLFLOW_TRACKING_URI=http://localhost:5000
python -m training.train --data data/dataset.csv   # logs run + tags dvc.dataset_version / dvc.git_commit / dvc.data_md5 / quality_gate
```
Run training 3–5 times (vary `--n-estimators`) so there are runs to compare.

## 4. Airflow 3
Install per the official docs (`pip install apache-airflow` with the constraints file for your Python version), then:
```bash
export AIRFLOW_HOME=~/airflow
export PROJECT_DIR=$(pwd)                         # DAG reads data/ and imports training.train
export PYTHONPATH=$(pwd)
airflow standalone                                # prints the admin password; UI/API on :8080
# point DAGs at this repo:  airflow config get-value core dags_folder  -> symlink or copy dags/training_pipeline.py there
```
Set `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` in `.env`. The agent obtains a JWT from `/auth/token` and calls `/api/v2`.
Only DAG ids in `AIRFLOW_ALLOWED_DAGS` can be triggered.

## 5. Kubernetes (minikube / kind / Docker Desktop)
```bash
python -m training.train          # produces models/model.joblib (baked into the image)
docker build -t mlops-inference:1.0.0 .
minikube image load mlops-inference:1.0.0         # or: kind load docker-image mlops-inference:1.0.0
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl -n mlops-demo get pods,svc
# update deployment annotations model-version / dataset-version / mlflow-run-id with the real values, then re-apply
```
Optional least-privilege identity for the agent when running in-cluster: `kubernetes/agent-rbac.yaml`.
Locally the agent uses your kubeconfig, but **all tools are restricted to `K8S_ALLOWED_NAMESPACES`**.

## 6. Launch the agent
```bash
uvicorn agent.main:app --port 8001
curl -s localhost:8001/tools | python -m json.tool
curl -s localhost:8001/chat -H 'content-type: application/json' -d '{"message":"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."}' | python -m json.tool
# CLI:   python -m agent.main "Why is my prediction API unavailable?"
```
* With `GEMINI_API_KEY` set (free key from Google AI Studio) the agent uses Gemini function calling (`AGENT_LLM_MODEL`, default `gemini-3.8-flash`). Without it a deterministic **rule planner** is used (same tools, same evidence rules) — handy for offline demos and tests.
* State-changing requests (trigger DAG, restart/scale) are **refused unless** the request body has `"confirm_actions": true`.

## 7. Tests and lint
```bash
pytest                    # 29 tests, no external services needed
ruff check .
```

## Safety summary
* DVC: argv-only subprocess (no shell), command allowlist (`dvc status/diff/checkout/list`, `git rev-parse/log/show/status`), revision strings validated.
* Airflow: JWT auth, trigger only for approved DAGs and only with confirmation.
* Kubernetes: approved-namespace allowlist on every tool, mutating tools need explicit confirmation, RBAC manifest provided.
* Logs: JSON with tool, status, duration, request id; secret-looking keys/values are redacted.
* No secrets in the repo; everything comes from environment variables.
