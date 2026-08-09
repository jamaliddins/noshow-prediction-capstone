"""Tests for the FastAPI service."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import MODELS_DIR

model_required = pytest.mark.skipif(
    not (MODELS_DIR / "model.joblib").exists(),
    reason="no trained model — run `python -m src.train` first",
)

VALID = {
    "scheduled_day": "2016-05-02T09:00:00",
    "appointment_day": "2016-05-30",
    "age": 22,
    "gender": "F",
    "neighbourhood": "JARDIM CAMBURI",
}


@pytest.fixture(scope="module")
def client():
    # The context manager triggers the lifespan hook that loads the model.
    with TestClient(app) as c:
        yield c


class TestStatusEndpoints:
    def test_health_is_always_available(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in {"ok", "degraded"}

    def test_root_lists_endpoints(self, client):
        assert "/predict" in client.get("/").json()["endpoints"]

    @model_required
    def test_model_info_reports_metrics(self, client):
        body = client.get("/model").json()
        assert body["model"] == "xgboost"
        assert 0 < body["decision_threshold"] < 1
        assert "f1" in body["test_metrics"]


@model_required
class TestPredict:
    def test_valid_request_returns_a_scored_prediction(self, client):
        body = client.post("/predict", json=VALID).json()
        assert 0.0 <= body["no_show_probability"] <= 1.0
        assert body["risk_tier"] in {"Low", "Medium", "High"}
        assert body["lead_time_days"] == 28

    def test_longer_lead_time_scores_higher(self, client):
        same_day = client.post("/predict", json={
            **VALID, "scheduled_day": "2016-05-30T08:00:00",
            "appointment_day": "2016-05-30"}).json()
        month = client.post("/predict", json=VALID).json()
        assert month["no_show_probability"] > same_day["no_show_probability"]

    def test_patient_history_is_accepted(self, client):
        body = client.post("/predict", json={
            **VALID, "prior_appointments": 10, "prior_noshows": 8}).json()
        assert body["no_show_probability"] > 0

    @pytest.mark.parametrize("field", ["scheduled_day", "appointment_day",
                                       "age", "gender", "neighbourhood"])
    def test_missing_required_field_is_422(self, client, field):
        payload = {k: v for k, v in VALID.items() if k != field}
        assert client.post("/predict", json=payload).status_code == 422

    @pytest.mark.parametrize("patch", [
        {"age": -5}, {"age": 200}, {"gender": "X"},
        {"handicap": 9}, {"scholarship": 2},
    ])
    def test_out_of_range_values_are_422(self, client, patch):
        assert client.post("/predict", json={**VALID, **patch}).status_code == 422

    def test_reversed_dates_are_422(self, client):
        r = client.post("/predict", json={
            **VALID, "scheduled_day": "2016-06-01", "appointment_day": "2016-05-01"})
        assert r.status_code == 422

    def test_noshows_exceeding_appointments_is_422(self, client):
        r = client.post("/predict", json={
            **VALID, "prior_appointments": 2, "prior_noshows": 5})
        assert r.status_code == 422


@model_required
class TestBatch:
    def test_batch_is_sorted_by_descending_risk(self, client):
        body = client.post("/predict/batch", json={"appointments": [
            {**VALID, "scheduled_day": "2016-05-30T08:00", "age": 70},
            VALID,
        ]}).json()
        assert body["count"] == 2
        pcts = [r["no_show_percentage"] for r in body["results"]]
        assert pcts == sorted(pcts, reverse=True)

    def test_tier_counts_sum_to_result_count(self, client):
        body = client.post("/predict/batch",
                           json={"appointments": [VALID] * 5}).json()
        assert sum(body["tier_counts"].values()) == body["count"]

    def test_empty_batch_is_rejected(self, client):
        assert client.post("/predict/batch",
                           json={"appointments": []}).status_code == 422

    def test_oversized_batch_is_rejected(self, client):
        r = client.post("/predict/batch", json={"appointments": [VALID] * 1001})
        assert r.status_code == 422
