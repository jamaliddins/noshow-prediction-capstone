#!/usr/bin/env bash
# Builds the image, starts the API, and checks the endpoints respond.
#
#   bash scripts/docker_smoke_test.sh
#
# Requires Docker, and a trained model in models/ (python -m src.train).

set -euo pipefail

IMAGE="noshow-api:smoke"
CONTAINER="noshow-api-smoke"
PORT=8099

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Building image"
docker build -t "$IMAGE" .

if [ ! -f models/model.joblib ]; then
  echo "==> No model found; training inside the container"
  docker run --rm \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/models:/app/models" \
    "$IMAGE" python -m src.train
fi

echo "==> Starting container"
cleanup
docker run -d --name "$CONTAINER" -p "${PORT}:8000" \
  -v "$(pwd)/models:/app/models" "$IMAGE" >/dev/null

echo "==> Waiting for /health"
for i in $(seq 1 40); do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if [ "$i" -eq 40 ]; then
    echo "FAILED: service did not become healthy"
    docker logs "$CONTAINER"
    exit 1
  fi
  sleep 1
done

echo "==> /health";  curl -fsS "http://localhost:${PORT}/health"; echo
echo "==> /model";   curl -fsS "http://localhost:${PORT}/model";  echo

echo "==> POST /predict"
curl -fsS -X POST "http://localhost:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d '{"scheduled_day":"2016-05-02T09:00:00","appointment_day":"2016-05-30",
       "age":22,"gender":"F","neighbourhood":"JARDIM CAMBURI"}'
echo

echo "==> Invalid input should return 422"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d '{"scheduled_day":"2016-05-01","appointment_day":"2016-05-10",
       "age":999,"gender":"F","neighbourhood":"CENTRO"}')
[ "$code" = "422" ] && echo "OK (422)" || { echo "FAILED: got $code"; exit 1; }

echo
echo "All checks passed."
