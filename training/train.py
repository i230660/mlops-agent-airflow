"""Training script: trains a classifier, evaluates, logs to MLflow with DVC provenance, applies a quality gate.

Usage:
    python -m training.train --data data/dataset.csv --min-f1 0.80
Environment: MLFLOW_TRACKING_URI
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def md5_of(path: str) -> str:
    h = hashlib.md5()  # noqa: S324 - matches DVC's content hash, not used for security
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dataset_version(data_path: str) -> str:
    """Prefer a git tag on HEAD (e.g. v2.1); fall back to short md5."""
    tag = git("describe", "--tags", "--exact-match", "HEAD")
    return tag if tag and tag != "unknown" else f"md5-{md5_of(data_path)[:8]}"


def train(data_path: str, min_f1: float, model_out: str, n_estimators: int = 100, seed: int = 42) -> dict:
    import mlflow

    df = pd.read_csv(data_path)
    X, y = df.drop(columns=["target"]), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed).fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {"f1": float(f1_score(yte, pred, average="weighted")), "accuracy": float(accuracy_score(yte, pred))}
    gate = "passed" if metrics["f1"] >= min_f1 else "failed"

    mlflow.set_experiment("mlops-agent-training")
    with mlflow.start_run() as run:
        mlflow.log_params({"n_estimators": n_estimators, "seed": seed, "data_path": data_path})
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "dvc.dataset_version": dataset_version(data_path),
                "dvc.git_commit": git("rev-parse", "HEAD"),
                "dvc.data_md5": md5_of(data_path),
                "quality_gate": gate,
            }
        )
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": list(X.columns)}, model_out)
        mlflow.log_artifact(model_out)
        result = {"run_id": run.info.run_id, "metrics": metrics, "quality_gate": gate}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--min-f1", type=float, default=float(os.environ.get("MIN_QUALITY_F1", "0.80")))
    ap.add_argument("--model-out", default="models/model.joblib")
    ap.add_argument("--n-estimators", type=int, default=100)
    a = ap.parse_args()
    train(a.data, a.min_f1, a.model_out, a.n_estimators)
