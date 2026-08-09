"""Tests for inference and input validation in src/predict.py."""
from __future__ import annotations

import pytest

from src.predict import (
    InvalidAppointmentError,
    assign_risk_tier,
    predict_batch,
    predict_one,
    validate_appointment,
)

VALID = {
    "scheduled_day": "2016-05-02T09:00:00",
    "appointment_day": "2016-05-30",
    "age": 30,
    "gender": "F",
    "neighbourhood": "CENTRO",
}

# The trained artifacts must exist; skip cleanly rather than fail confusingly.
model_required = pytest.mark.skipif(
    not (__import__("src.config", fromlist=["MODELS_DIR"]).MODELS_DIR
         / "model.joblib").exists(),
    reason="no trained model — run `python -m src.train` first",
)


class TestValidation:
    def test_accepts_a_valid_record(self):
        out = validate_appointment(VALID)
        assert out["age"] == 30 and out["gender"] == "F"

    def test_fills_optional_defaults(self):
        out = validate_appointment(VALID)
        assert out["sms_received"] == 0
        assert out["is_first_appointment"] == 1
        assert out["prior_noshow_rate"] == -1.0

    @pytest.mark.parametrize("field", ["scheduled_day", "appointment_day", "age",
                                       "gender", "neighbourhood"])
    def test_rejects_each_missing_required_field(self, field):
        record = {k: v for k, v in VALID.items() if k != field}
        with pytest.raises(InvalidAppointmentError, match="Missing required"):
            validate_appointment(record)

    def test_rejects_appointment_before_scheduling(self):
        bad = {**VALID, "scheduled_day": "2016-06-01", "appointment_day": "2016-05-01"}
        with pytest.raises(InvalidAppointmentError, match="after"):
            validate_appointment(bad)

    def test_allows_same_day_booking(self):
        ok = {**VALID, "scheduled_day": "2016-05-30T18:00:00",
              "appointment_day": "2016-05-30"}
        assert validate_appointment(ok) is not None

    @pytest.mark.parametrize("age", [-1, 999, "abc", None])
    def test_rejects_impossible_age(self, age):
        with pytest.raises(InvalidAppointmentError):
            validate_appointment({**VALID, "age": age})

    def test_rejects_unknown_gender(self):
        with pytest.raises(InvalidAppointmentError, match="gender"):
            validate_appointment({**VALID, "gender": "X"})

    def test_rejects_unparseable_date(self):
        with pytest.raises(InvalidAppointmentError, match="valid date"):
            validate_appointment({**VALID, "appointment_day": "not-a-date"})

    def test_rejects_non_dict(self):
        with pytest.raises(InvalidAppointmentError, match="dict"):
            validate_appointment(["not", "a", "dict"])


class TestRiskTiers:
    TIERS = {"low_max": 0.25, "medium_max": 0.35}

    @pytest.mark.parametrize("prob,expected", [
        (0.05, "Low"), (0.249, "Low"),
        (0.25, "Medium"), (0.349, "Medium"),
        (0.35, "High"), (0.95, "High"),
    ])
    def test_boundaries(self, prob, expected):
        assert assign_risk_tier(prob, self.TIERS) == expected


@model_required
class TestPrediction:
    def test_single_prediction_shape(self):
        out = predict_one(VALID)
        assert 0.0 <= out["no_show_probability"] <= 1.0
        assert out["risk_tier"] in {"Low", "Medium", "High"}
        assert out["no_show_percentage"] == pytest.approx(
            out["no_show_probability"] * 100, abs=0.1
        )

    def test_probabilities_are_calibrated_not_inflated(self):
        """A typical low-risk appointment must not score near 50%."""
        low_risk = {**VALID, "scheduled_day": "2016-05-30T08:00:00",
                    "appointment_day": "2016-05-30", "age": 65}
        assert predict_one(low_risk)["no_show_probability"] < 0.25

    def test_longer_lead_time_raises_risk(self):
        """The strongest signal in EDA must hold in the model's output."""
        same_day = predict_one({**VALID, "scheduled_day": "2016-05-30T08:00:00",
                                "appointment_day": "2016-05-30"})
        month_out = predict_one({**VALID, "scheduled_day": "2016-04-30",
                                 "appointment_day": "2016-05-30"})
        assert month_out["no_show_probability"] > same_day["no_show_probability"]

    def test_batch_is_sorted_by_descending_risk(self):
        out = predict_batch([
            {**VALID, "scheduled_day": "2016-05-30T08:00:00",
             "appointment_day": "2016-05-30", "age": 70},
            {**VALID, "age": 20},
        ])
        assert len(out) == 2
        assert out["no_show_percentage"].is_monotonic_decreasing

    def test_batch_rejects_invalid_by_default(self):
        with pytest.raises(InvalidAppointmentError, match="Record 1"):
            predict_batch([VALID, {"age": 30}])

    def test_batch_can_skip_invalid_records(self):
        out = predict_batch([VALID, {"age": 30}], skip_invalid=True)
        assert "error" in out.columns
        assert out["error"].notna().sum() == 1
