"""Inference: no-show probability plus a Low/Medium/High risk tier.

Handles one appointment or a batch. Input validation is deliberately strict —
the brief requires invalid or incomplete inputs to be handled, not guessed at.
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.preprocessing import MAX_PLAUSIBLE_AGE

MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Supplied per appointment; everything else is derived.
REQUIRED_FIELDS = ["scheduled_day", "appointment_day", "age", "gender", "neighbourhood"]

OPTIONAL_DEFAULTS = {
    "scholarship": 0, "hypertension": 0, "diabetes": 0, "alcoholism": 0,
    "handicap": 0, "sms_received": 0,
    # A new patient with no known history: mirrors the training sentinel.
    "prior_appointments": 0, "prior_noshows": 0, "prior_noshow_rate": -1.0,
    "is_first_appointment": 1, "days_since_last_appointment": -1.0,
}


class InvalidAppointmentError(ValueError):
    """Raised when an input record cannot be scored as given."""


@lru_cache(maxsize=1)
def load_model():
    """Load the trained pipeline and its metadata (cached across calls)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Run: python -m src.train"
        )
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, metadata


def _parse_date(value, field: str) -> pd.Timestamp:
    if value in (None, ""):
        raise InvalidAppointmentError(f"'{field}' is required")
    try:
        ts = pd.to_datetime(value, utc=True)
    except (ValueError, TypeError) as exc:
        raise InvalidAppointmentError(
            f"'{field}' is not a valid date/time: {value!r}"
        ) from exc
    if pd.isna(ts):
        raise InvalidAppointmentError(f"'{field}' is not a valid date/time: {value!r}")
    return ts


def validate_appointment(record: dict) -> dict:
    """Check one record and fill optional fields with their defaults."""
    if not isinstance(record, dict):
        raise InvalidAppointmentError(f"Expected a dict, got {type(record).__name__}")

    missing = [f for f in REQUIRED_FIELDS if record.get(f) in (None, "")]
    if missing:
        raise InvalidAppointmentError(f"Missing required field(s): {missing}")

    clean = {**OPTIONAL_DEFAULTS, **record}

    scheduled = _parse_date(clean["scheduled_day"], "scheduled_day")
    appointment = _parse_date(clean["appointment_day"], "appointment_day")
    if scheduled.normalize() > appointment.normalize():
        raise InvalidAppointmentError(
            f"scheduled_day ({scheduled.date()}) is after "
            f"appointment_day ({appointment.date()})"
        )
    clean["scheduled_day"], clean["appointment_day"] = scheduled, appointment

    try:
        age = int(clean["age"])
    except (TypeError, ValueError) as exc:
        raise InvalidAppointmentError(f"'age' must be a number, got {clean['age']!r}") from exc
    if not 0 <= age <= MAX_PLAUSIBLE_AGE:
        raise InvalidAppointmentError(
            f"'age' must be between 0 and {MAX_PLAUSIBLE_AGE}, got {age}"
        )
    clean["age"] = age

    gender = str(clean["gender"]).strip().upper()
    if gender not in {"F", "M"}:
        raise InvalidAppointmentError(f"'gender' must be 'F' or 'M', got {clean['gender']!r}")
    clean["gender"] = gender

    for flag in ("scholarship", "hypertension", "diabetes", "alcoholism", "sms_received"):
        clean[flag] = int(bool(int(clean[flag])))
    clean["handicap"] = max(0, int(clean["handicap"]))
    clean["neighbourhood"] = str(clean["neighbourhood"]).strip().upper()
    return clean


def _derive_features(records: list[dict]) -> pd.DataFrame:
    """Rebuild the engineered columns the pipeline expects, per record."""
    df = pd.DataFrame(records)

    lead = (df["appointment_day"].dt.normalize()
            - df["scheduled_day"].dt.normalize()).dt.days
    df["lead_time_days"] = lead.clip(lower=0)
    df["is_same_day"] = (df["lead_time_days"] == 0).astype(int)
    df["appt_dayofweek"] = df["appointment_day"].dt.dayofweek
    df["appt_month"] = df["appointment_day"].dt.month
    df["sched_dayofweek"] = df["scheduled_day"].dt.dayofweek
    df["sched_hour"] = df["scheduled_day"].dt.hour
    df["is_female"] = (df["gender"] == "F").astype(int)
    df["has_handicap"] = (df["handicap"] > 0).astype(int)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[-1, 2, 12, 18, 35, 55, 75, MAX_PLAUSIBLE_AGE],
        labels=["infant", "child", "teen", "young_adult", "adult", "senior", "elderly"],
    ).astype(str)
    df["n_conditions"] = df[
        ["hypertension", "diabetes", "alcoholism", "has_handicap"]
    ].sum(axis=1)
    return df


def assign_risk_tier(probability: float, tiers: dict) -> str:
    if probability < tiers["low_max"]:
        return "Low"
    if probability < tiers["medium_max"]:
        return "Medium"
    return "High"


def _recommendation(tier: str) -> str:
    return {
        "High": "Call to confirm — highest priority for staff follow-up.",
        "Medium": "Send a reminder; call if capacity allows.",
        "Low": "Standard automated reminder is sufficient.",
    }[tier]


def predict_one(record: dict) -> dict:
    """Score a single appointment.

    Returns probability, percentage, risk tier and a suggested action.
    """
    pipeline, metadata = load_model()
    validated = validate_appointment(record)
    features = _derive_features([validated])

    columns = metadata["features"]["numeric"] + metadata["features"]["categorical"]
    probability = float(pipeline.predict_proba(features[columns])[0, 1])
    tier = assign_risk_tier(probability, metadata["risk_tiers"])

    return {
        "no_show_probability": round(probability, 4),
        "no_show_percentage": round(probability * 100, 1),
        "risk_tier": tier,
        "recommendation": _recommendation(tier),
        "threshold_used": metadata["threshold"],
        "model": metadata["model_name"],
        "lead_time_days": int(features.loc[0, "lead_time_days"]),
    }


def predict_batch(records: list[dict], skip_invalid: bool = False) -> pd.DataFrame:
    """Score many appointments, sorted highest-risk first.

    With skip_invalid=True, bad records are reported in an `error` column
    rather than aborting the whole batch.
    """
    pipeline, metadata = load_model()

    validated, errors = [], []
    for i, record in enumerate(records):
        try:
            validated.append({"_row": i, **validate_appointment(record)})
        except InvalidAppointmentError as exc:
            if not skip_invalid:
                raise InvalidAppointmentError(f"Record {i}: {exc}") from exc
            errors.append({"_row": i, "error": str(exc)})

    if not validated:
        return pd.DataFrame(errors)

    features = _derive_features(validated)
    columns = metadata["features"]["numeric"] + metadata["features"]["categorical"]
    probabilities = pipeline.predict_proba(features[columns])[:, 1]

    out = pd.DataFrame({
        "row": [v["_row"] for v in validated],
        "age": features["age"],
        "gender": features["gender"],
        "neighbourhood": features["neighbourhood"],
        "appointment_day": features["appointment_day"].dt.date,
        "lead_time_days": features["lead_time_days"],
        "no_show_percentage": (probabilities * 100).round(1),
        "risk_tier": [assign_risk_tier(p, metadata["risk_tiers"]) for p in probabilities],
    })
    out = out.sort_values("no_show_percentage", ascending=False).reset_index(drop=True)

    if errors:
        out = pd.concat([out, pd.DataFrame(errors).rename(columns={"_row": "row"})],
                        ignore_index=True)
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("SINGLE APPOINTMENT")
    print("=" * 70)
    example = {
        "scheduled_day": "2016-05-02T09:00:00",
        "appointment_day": "2016-05-30",
        "age": 22,
        "gender": "F",
        "neighbourhood": "JARDIM CAMBURI",
        "scholarship": 1,
        "sms_received": 1,
    }
    for key, value in predict_one(example).items():
        print(f"  {key:<22} {value}")

    print("\n" + "=" * 70)
    print("BATCH")
    print("=" * 70)
    batch = [
        example,
        {"scheduled_day": "2016-05-30T08:00:00", "appointment_day": "2016-05-30",
         "age": 65, "gender": "M", "neighbourhood": "CENTRO", "hypertension": 1},
        {"scheduled_day": "2016-05-10", "appointment_day": "2016-05-24",
         "age": 15, "gender": "M", "neighbourhood": "SANTOS DUMONT"},
        {"scheduled_day": "2016-05-20", "appointment_day": "2016-05-23",
         "age": 45, "gender": "F", "neighbourhood": "MARIA ORTIZ"},
    ]
    print(predict_batch(batch).to_string(index=False))

    print("\n" + "=" * 70)
    print("INVALID INPUT HANDLING")
    print("=" * 70)
    for bad in (
        {"age": 30, "gender": "F"},
        {"scheduled_day": "2016-06-01", "appointment_day": "2016-05-01",
         "age": 30, "gender": "F", "neighbourhood": "CENTRO"},
        {"scheduled_day": "2016-05-01", "appointment_day": "2016-05-10",
         "age": 999, "gender": "F", "neighbourhood": "CENTRO"},
    ):
        try:
            predict_one(bad)
        except InvalidAppointmentError as exc:
            print(f"  rejected: {exc}")
