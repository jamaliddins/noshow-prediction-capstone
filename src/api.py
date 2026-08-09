"""FastAPI service exposing the no-show model.

Endpoints:
  GET  /health     liveness plus whether the model is loaded
  GET  /model      model name, metrics and the thresholds in use
  POST /predict    score one appointment
  POST /predict/batch  score up to 1000 appointments, highest risk first

Run locally:  uvicorn src.api:app --reload
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.config import MODELS_DIR
from src.predict import (
    InvalidAppointmentError,
    load_model,
    predict_batch,
    predict_one,
)

MAX_BATCH_SIZE = 1000

_state: dict = {"model_loaded": False, "metadata": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than on the first request."""
    try:
        _, metadata = load_model()
        _state["model_loaded"] = True
        _state["metadata"] = metadata
        print(f"Model loaded: {metadata['model_name']} "
              f"(threshold {metadata['threshold']:.3f})")
    except FileNotFoundError as exc:
        # Serve /health so the failure is visible rather than crash-looping.
        print(f"WARNING: {exc}")
    yield


app = FastAPI(
    title="No-Show Prediction API",
    description=(
        "Estimates the probability that a patient will miss a scheduled "
        "appointment, for prioritising reminder calls. A decision-support "
        "prototype — not a clinical or automated decision-maker."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # prototype; restrict before any real deployment
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------- schemas
class AppointmentRequest(BaseModel):
    """One scheduled appointment, using only pre-appointment information."""

    scheduled_day: str = Field(..., description="When the appointment was booked (ISO 8601)",
                               json_schema_extra={"example": "2016-05-02T09:00:00"})
    appointment_day: str = Field(..., description="When the appointment is due (ISO 8601)",
                                 json_schema_extra={"example": "2016-05-30"})
    age: int = Field(..., ge=0, le=110, json_schema_extra={"example": 22})
    gender: Literal["F", "M", "f", "m"] = Field(..., json_schema_extra={"example": "F"})
    neighbourhood: str = Field(..., min_length=1,
                               json_schema_extra={"example": "JARDIM CAMBURI"})

    scholarship: int = Field(0, ge=0, le=1, description="Welfare (Bolsa Família) recipient")
    hypertension: int = Field(0, ge=0, le=1)
    diabetes: int = Field(0, ge=0, le=1)
    alcoholism: int = Field(0, ge=0, le=1)
    handicap: int = Field(0, ge=0, le=4, description="Source encodes 0-4, not a flag")
    sms_received: int = Field(0, ge=0, le=1)

    prior_appointments: int = Field(0, ge=0, description="Patient's past appointments")
    prior_noshows: int = Field(0, ge=0, description="How many of those were missed")

    @field_validator("prior_noshows")
    @classmethod
    def noshows_cannot_exceed_appointments(cls, v, info):
        prior = info.data.get("prior_appointments", 0)
        if v > prior:
            raise ValueError(
                f"prior_noshows ({v}) cannot exceed prior_appointments ({prior})"
            )
        return v

    def to_record(self) -> dict:
        """Convert to the dict shape src.predict expects, deriving history."""
        record = self.model_dump()
        prior = record["prior_appointments"]
        noshows = record["prior_noshows"]
        record["prior_noshow_rate"] = noshows / prior if prior > 0 else -1.0
        record["is_first_appointment"] = int(prior == 0)
        record["days_since_last_appointment"] = -1.0
        return record


class BatchRequest(BaseModel):
    appointments: list[AppointmentRequest] = Field(..., min_length=1,
                                                   max_length=MAX_BATCH_SIZE)
    skip_invalid: bool = Field(
        False, description="Report bad records instead of rejecting the batch"
    )


class PredictionResponse(BaseModel):
    no_show_probability: float = Field(..., description="Calibrated, 0-1")
    no_show_percentage: float
    risk_tier: Literal["Low", "Medium", "High"]
    recommendation: str
    lead_time_days: int
    threshold_used: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


def _require_model() -> None:
    if not _state["model_loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run `python -m src.train` and restart.",
        )


# -------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse, tags=["status"])
def health():
    """Liveness check, including whether the model artifact is present."""
    return HealthResponse(
        status="ok" if _state["model_loaded"] else "degraded",
        model_loaded=_state["model_loaded"],
        version=app.version,
    )


@app.get("/model", tags=["status"])
def model_info():
    """Model name, held-out test metrics and the thresholds in use."""
    _require_model()
    metadata = _state["metadata"]
    return {
        "model": metadata["model_name"],
        "calibration": metadata.get("calibration"),
        "decision_threshold": metadata["threshold"],
        "risk_tiers": metadata["risk_tiers"],
        "test_metrics": {
            k: round(v, 4) for k, v in metadata["test_metrics"].items()
            if isinstance(v, float)
        },
        "n_features": (len(metadata["features"]["numeric"])
                       + len(metadata["features"]["categorical"])),
        "trained_on_rows": metadata["train_rows"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(appointment: AppointmentRequest):
    """Score a single appointment."""
    _require_model()
    try:
        return PredictionResponse(**predict_one(appointment.to_record()))
    except InvalidAppointmentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@app.post("/predict/batch", tags=["prediction"])
def predict_many(request: BatchRequest):
    """Score a batch, returned highest-risk first."""
    _require_model()
    records = [a.to_record() for a in request.appointments]
    try:
        frame = predict_batch(records, skip_invalid=request.skip_invalid)
    except InvalidAppointmentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    results = json.loads(frame.to_json(orient="records", date_format="iso"))
    tiers = frame["risk_tier"].value_counts().to_dict() if "risk_tier" in frame else {}
    return {
        "count": len(results),
        "tier_counts": {t: int(tiers.get(t, 0)) for t in ("High", "Medium", "Low")},
        "results": results,
    }


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/health", "/model", "/predict", "/predict/batch"],
    }
