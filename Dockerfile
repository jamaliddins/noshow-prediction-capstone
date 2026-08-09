# Serves the no-show prediction API.
#
# The image contains code only — the dataset and the trained model are mounted
# or built at run time, so no data or large binaries are baked in.
#
#   docker build -t noshow-api .
#   docker run -p 8000:8000 -v "$(pwd)/models:/app/models" noshow-api
#
# If models/ is empty, train first (needs data/ mounted too):
#   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
#     noshow-api python -m src.train

FROM python:3.12-slim

# Keeps the image small and logs unbuffered so `docker logs` is live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# libgomp1 is required by XGBoost's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so code edits do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "uvicorn[standard]"

COPY src/ ./src/
COPY tests/ ./tests/
COPY pytest.ini README.md ./

# Mount points for artifacts that are deliberately not in the image.
RUN mkdir -p /app/data /app/models /app/reports/figures

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
