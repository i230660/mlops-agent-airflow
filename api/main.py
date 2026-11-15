"""FastAPI inference service deployed on Kubernetes."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from agent.schemas import PredictRequest, PredictResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

app = FastAPI(title="ML Inference Service", version=os.environ.get("APP_VERSION", "1.0.0"))
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"model file not found at {MODEL_PATH}")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/healthz")
def healthz():
    """Liveness: process is up."""
    return {"status": "alive"}


@app.get("/ready")
def ready():
    """Readiness: model must be loadable, otherwise pod receives no traffic."""
    try:
        _load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model not ready: {exc}") from exc
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/version")
def version():
    return {"app_version": app.version, "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        b = _load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if len(req.features) != len(b["features"]):
        raise HTTPException(status_code=422, detail=f"expected {len(b['features'])} features")
    proba = b["model"].predict_proba([req.features])[0]
    cls = int(proba.argmax())
    return PredictResponse(prediction=cls, probability=float(proba[cls]), model_version=MODEL_VERSION)
