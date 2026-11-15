"""Airflow 3 training DAG: validate -> preprocess -> train -> evaluate/log -> quality gate.

Retries and logging are configured via default_args/task logger. A failed validation or
quality gate raises, so downstream promotion never runs (tasks are linear dependencies).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

log = logging.getLogger("airflow.task")

REPO = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
DATA = os.path.join(REPO, "data", "dataset.csv")
CLEAN = os.path.join(REPO, "data", "clean.csv")
MODEL = os.path.join(REPO, "models", "model.joblib")
MIN_F1 = float(os.environ.get("MIN_QUALITY_F1", "0.80"))

DEFAULT_ARGS = {"owner": "mlops", "retries": 2, "retry_delay": timedelta(seconds=30)}


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training"],
)
def training_pipeline():
    @task
    def validate_dataset() -> str:
        import pandas as pd

        if not os.path.exists(DATA):
            raise FileNotFoundError(f"dataset missing: {DATA} (run `dvc pull`)")
        df = pd.read_csv(DATA)
        problems = []
        if "target" not in df.columns:
            problems.append("missing 'target' column")
        if len(df) < 100:
            problems.append(f"too few rows: {len(df)}")
        if df.isna().mean().max() > 0.2:
            problems.append("a column has >20% nulls")
        if problems:
            raise ValueError("dataset validation failed: " + "; ".join(problems))
        log.info("validation passed rows=%d", len(df))
        return DATA

    @task
    def preprocess_data(path: str) -> str:
        import pandas as pd

        df = pd.read_csv(path).dropna().drop_duplicates()
        df.to_csv(CLEAN, index=False)
        log.info("preprocessed rows=%d", len(df))
        return CLEAN

    @task
    def train_and_evaluate(path: str) -> str:
        """Trains, evaluates and logs to MLflow (stages 3 and 4 of the figure share one process so the run id is shared)."""
        from training.train import train

        result = train(path, MIN_F1, MODEL)
        log.info("trained run=%s metrics=%s", result["run_id"], result["metrics"])
        return json.dumps(result)

    @task
    def quality_gate(result_json: str) -> str:
        result = json.loads(result_json)
        if result["quality_gate"] != "passed":
            raise ValueError(f"quality gate FAILED: f1={result['metrics']['f1']:.3f} < {MIN_F1}; promotion blocked")
        log.info("quality gate passed run=%s", result["run_id"])
        return result["run_id"]

    quality_gate(train_and_evaluate(preprocess_data(validate_dataset())))


training_pipeline()
