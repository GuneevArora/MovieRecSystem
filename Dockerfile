# API image for the movie recommender.
#
# Build (needs artifacts/ from `python -m scripts.export_serving_artifacts`):
#   docker build -t movie-recommender-api .
# Run:
#   docker run -p 8080:8080 movie-recommender-api

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only torch: the catalog is small enough that inference takes a few
# milliseconds on CPU, and the CUDA wheels would add several GB.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/__init__.py src/model.py ./src/
COPY src/api ./src/api
COPY artifacts ./artifacts

RUN useradd --create-home --uid 1000 app
USER app

ENV ARTIFACT_DIR=/app/artifacts \
    DEVICE=cpu \
    PORT=8080

EXPOSE 8080

# Cloud Run provides PORT. One worker: the model is held in memory once and
# concurrent requests are micro-batched inside the process.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
