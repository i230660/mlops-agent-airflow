"""Inference service: readiness must fail without a model (this is what Kubernetes probes rely on)."""
import importlib

import joblib
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier


def _app(monkeypatch, path):
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("MODEL_VERSION", "vtest")
    import api.main as m

    importlib.reload(m)
    return m.app


def test_ready_fails_when_model_missing(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path / "nope.joblib"))
    assert c.get("/healthz").status_code == 200  # liveness unaffected
    assert c.get("/ready").status_code == 503  # readiness fails -> no traffic


def test_predict_with_model(monkeypatch, tmp_path):
    X = [[0, 1], [1, 0], [0, 0], [1, 1]] * 5
    y = [1, 0, 0, 1] * 5
    p = tmp_path / "m.joblib"
    joblib.dump({"model": RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y), "features": ["a", "b"]}, p)
    c = TestClient(_app(monkeypatch, p))
    assert c.get("/ready").json()["model_version"] == "vtest"
    r = c.post("/predict", json={"features": [0, 1]})
    assert r.status_code == 200 and r.json()["model_version"] == "vtest"
    assert c.post("/predict", json={"features": [1]}).status_code == 422
