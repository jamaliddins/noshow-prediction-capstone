"""Tests for the leakage-safety guarantees in src/preprocessing.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import GROUP_COL, TARGET
from src.preprocessing import (
    add_patient_history_features,
    build_features,
    clean,
    get_feature_columns,
    load_raw,
    split_by_patient,
)


def _make_row(pid, apt_id, sched, appt, no_show, age=30):
    return {
        "patient_id": pid,
        "appointment_id": apt_id,
        "gender": "F",
        "scheduled_day": pd.Timestamp(sched, tz="UTC"),
        "appointment_day": pd.Timestamp(appt, tz="UTC"),
        "age": age,
        "neighbourhood": "CENTRO",
        "scholarship": 0,
        "hypertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 0,
        TARGET: no_show,
    }


@pytest.fixture
def toy_df():
    """One patient with a known outcome sequence: no-show, show, no-show."""
    return pd.DataFrame([
        _make_row(1, 101, "2016-01-01", "2016-01-10", 1),
        _make_row(1, 102, "2016-02-01", "2016-02-10", 0),
        _make_row(1, 103, "2016-03-01", "2016-03-10", 1),
        _make_row(2, 201, "2016-01-05", "2016-01-15", 0),
    ])


class TestHistoryFeaturesUseOnlyThePast:
    def test_first_appointment_has_no_history(self, toy_df):
        out = add_patient_history_features(toy_df)
        first = out[out["appointment_id"] == 101].iloc[0]
        assert first["prior_appointments"] == 0
        assert first["prior_noshows"] == 0
        assert first["is_first_appointment"] == 1
        # Sentinel, not a real rate — a first-timer has no rate.
        assert first["prior_noshow_rate"] == -1.0

    def test_counts_accumulate_strictly_from_prior_rows(self, toy_df):
        out = add_patient_history_features(toy_df).set_index("appointment_id")
        # 2nd appointment: one prior, which was a no-show.
        assert out.loc[102, "prior_appointments"] == 1
        assert out.loc[102, "prior_noshows"] == 1
        assert out.loc[102, "prior_noshow_rate"] == pytest.approx(1.0)
        # 3rd: two priors, one of which was a no-show.
        assert out.loc[103, "prior_appointments"] == 2
        assert out.loc[103, "prior_noshows"] == 1
        assert out.loc[103, "prior_noshow_rate"] == pytest.approx(0.5)

    def test_current_outcome_never_leaks_into_its_own_features(self, toy_df):
        """Flipping a row's target must not change that same row's features."""
        base = add_patient_history_features(toy_df).set_index("appointment_id")
        flipped_src = toy_df.copy()
        flipped_src.loc[flipped_src["appointment_id"] == 103, TARGET] = 0
        flipped = add_patient_history_features(flipped_src).set_index("appointment_id")

        for col in ("prior_appointments", "prior_noshows", "prior_noshow_rate"):
            assert base.loc[103, col] == flipped.loc[103, col], (
                f"{col} changed when the row's own outcome changed — leakage"
            )

    def test_patients_do_not_contaminate_each_other(self, toy_df):
        out = add_patient_history_features(toy_df).set_index("appointment_id")
        assert out.loc[201, "prior_appointments"] == 0
        assert out.loc[201, "is_first_appointment"] == 1

    def test_days_since_last_appointment(self, toy_df):
        out = add_patient_history_features(toy_df).set_index("appointment_id")
        assert out.loc[101, "days_since_last_appointment"] == -1  # no previous
        assert out.loc[102, "days_since_last_appointment"] == 31  # Jan 1 -> Feb 1


class TestCleaning:
    def test_drops_scheduled_after_appointment(self):
        df = pd.DataFrame([
            _make_row(1, 1, "2016-05-10", "2016-05-01", 0),  # invalid order
            _make_row(2, 2, "2016-05-01", "2016-05-10", 0),
        ])
        assert len(clean(df, verbose=False)) == 1

    def test_keeps_same_day_appointments(self):
        """Scheduled 18:00 for the same calendar day is valid, not invalid."""
        df = pd.DataFrame([
            _make_row(1, 1, "2016-05-01T18:00:00", "2016-05-01T00:00:00", 0),
        ])
        assert len(clean(df, verbose=False)) == 1

    def test_drops_impossible_ages(self):
        df = pd.DataFrame([
            _make_row(1, 1, "2016-05-01", "2016-05-10", 0, age=-1),
            _make_row(2, 2, "2016-05-01", "2016-05-10", 0, age=200),
            _make_row(3, 3, "2016-05-01", "2016-05-10", 0, age=45),
        ])
        assert len(clean(df, verbose=False)) == 1


class TestSplitSafety:
    def test_no_patient_spans_two_splits(self):
        rng = np.random.default_rng(0)
        rows = []
        for pid in range(400):
            for k in range(rng.integers(1, 4)):
                rows.append(_make_row(
                    pid, pid * 10 + k, "2016-01-01", "2016-01-10",
                    int(rng.random() < 0.2),
                ))
        feats = build_features(pd.DataFrame(rows))
        train, val, test = split_by_patient(feats)

        tr, va, te = (set(d[GROUP_COL]) for d in (train, val, test))
        assert not tr & va and not tr & te and not va & te
        assert len(tr | va | te) == 400

    def test_split_is_deterministic(self):
        rows = [
            _make_row(pid, pid, "2016-01-01", "2016-01-10", pid % 5 == 0)
            for pid in range(300)
        ]
        feats = build_features(pd.DataFrame(rows))
        a = split_by_patient(feats, random_state=42)[0][GROUP_COL].tolist()
        b = split_by_patient(feats, random_state=42)[0][GROUP_COL].tolist()
        assert a == b


class TestFeatureSelection:
    def test_identifiers_are_never_features(self, toy_df):
        feats = build_features(toy_df)
        numeric, categorical = get_feature_columns(feats)
        for banned in ("patient_id", "appointment_id", TARGET, "no_show"):
            assert banned not in numeric and banned not in categorical


@pytest.mark.integration
class TestRealDataset:
    """Guards the properties the written proposal claims about the raw file."""

    def test_dataset_matches_documented_shape(self):
        raw = load_raw()
        assert len(raw) == 110_527
        assert raw["patient_id"].nunique() == 62_299
        assert raw[TARGET].mean() == pytest.approx(0.2019, abs=1e-3)

    def test_pipeline_preserves_no_patient_leakage(self):
        train, val, test = split_by_patient(build_features(clean(load_raw(), verbose=False)))
        tr, va, te = (set(d[GROUP_COL]) for d in (train, val, test))
        assert not tr & va and not tr & te and not va & te
