FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Inference service only needs a slim set of deps (agent schemas are shared)
RUN pip install --no-cache-dir fastapi "uvicorn>=0.29" "pydantic>=2.6" scikit-learn pandas joblib

COPY agent/__init__.py agent/schemas.py ./agent/
COPY api ./api
# Model is baked in after `python -m training.train` (models/model.joblib). Missing model => readiness fails (503).
COPY models ./models

RUN useradd -r -u 10001 appuser && chown -R appuser /app
USER 10001

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
