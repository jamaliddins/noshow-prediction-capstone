"""Loading, cleaning, feature engineering and leakage-safe splitting.

The two rules this module exists to enforce:

1. Patient no-show history is computed from *strictly prior* appointments only.
2. Train/val/test are split by patient, so one patient never spans two sets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    GROUP_COL,
    RANDOM_STATE,
    RAW_CSV,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)

# Kaggle ships several misspelled headers; normalise them once, here.
COLUMN_RENAMES = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",   # misspelled in source
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",            # misspelled in source
    "SMS_received": "sms_received",
    "No-show": "no_show",
}

MAX_PLAUSIBLE_AGE = 110


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV and standardise column names and dtypes."""
    df = pd.read_csv(path)
    missing = set(COLUMN_RENAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema; missing columns: {sorted(missing)}")

    df = df.rename(columns=COLUMN_RENAMES)
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"], utc=True)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True)
    # "Yes" means the patient did NOT attend -> positive class.
    df[TARGET] = (df["no_show"].str.strip().str.lower() == "yes").astype(int)
    df["patient_id"] = df["patient_id"].astype("int64")
    return df


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop impossible rows and repair out-of-range values."""
    n0 = len(df)
    report: dict[str, int] = {}

    # AppointmentDay has no time component, so compare on calendar dates:
    # a same-day appointment scheduled in the afternoon is legitimate.
    bad_order = df["scheduled_day"].dt.normalize() > df["appointment_day"].dt.normalize()
    report["scheduled_after_appointment"] = int(bad_order.sum())
    df = df.loc[~bad_order].copy()

    bad_age = (df["age"] < 0) | (df["age"] > MAX_PLAUSIBLE_AGE)
    report["impossible_age"] = int(bad_age.sum())
    df = df.loc[~bad_age].copy()

    dupes = df.duplicated(subset=["appointment_id"])
    report["duplicate_appointment_id"] = int(dupes.sum())
    df = df.loc[~dupes].copy()

    if verbose:
        for reason, count in report.items():
            print(f"  dropped {count:>5} rows: {reason}")
        print(f"  {n0} -> {len(df)} rows retained")
    return df.reset_index(drop=True)


def add_appointment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivable from a single appointment, known before it happens."""
    df = df.copy()

    lead = (df["appointment_day"].dt.normalize() - df["scheduled_day"].dt.normalize()).dt.days
    df["lead_time_days"] = lead.clip(lower=0)
    df["is_same_day"] = (df["lead_time_days"] == 0).astype(int)

    df["appt_dayofweek"] = df["appointment_day"].dt.dayofweek
    df["appt_month"] = df["appointment_day"].dt.month
    df["sched_dayofweek"] = df["scheduled_day"].dt.dayofweek
    df["sched_hour"] = df["scheduled_day"].dt.hour

    df["is_female"] = (df["gender"].str.upper() == "F").astype(int)
    # Source encodes handicap 0-4 (a count, not a flag); keep both views.
    df["has_handicap"] = (df["handicap"] > 0).astype(int)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[-1, 2, 12, 18, 35, 55, 75, MAX_PLAUSIBLE_AGE],
        labels=["infant", "child", "teen", "young_adult", "adult", "senior", "elderly"],
    ).astype(str)

    n_conditions = ["hypertension", "diabetes", "alcoholism", "has_handicap"]
    df["n_conditions"] = df[n_conditions].sum(axis=1)
    return df


def add_patient_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient history using strictly prior appointments only.

    For each row we use that patient's appointments scheduled strictly earlier.
    Implemented with shifted expanding aggregates over a sorted frame, so the
    current row's own outcome can never enter its own features.
    """
    df = df.sort_values(["patient_id", "scheduled_day", "appointment_id"]).copy()
    grp = df.groupby("patient_id", sort=False)

    # cumcount is already exclusive of the current row.
    df["prior_appointments"] = grp.cumcount()

    prior_noshows = grp[TARGET].apply(
        lambda s: s.shift(1).expanding().sum()
    ).reset_index(level=0, drop=True)
    df["prior_noshows"] = prior_noshows.fillna(0.0)

    # Undefined for a first-ever appointment; -1 is a sentinel trees can split on.
    df["prior_noshow_rate"] = np.where(
        df["prior_appointments"] > 0,
        df["prior_noshows"] / df["prior_appointments"].replace(0, np.nan),
        -1.0,
    )
    df["prior_noshow_rate"] = df["prior_noshow_rate"].fillna(-1.0)
    df["is_first_appointment"] = (df["prior_appointments"] == 0).astype(int)

    days_since = grp["scheduled_day"].diff().dt.days
    df["days_since_last_appointment"] = days_since.fillna(-1.0)

    return df.sort_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: clean -> appointment features -> history features."""
    return add_patient_history_features(add_appointment_features(df))


def split_by_patient(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split 70/15/15 grouped by patient, keeping class balance similar.

    GroupShuffleSplit cannot stratify directly, so we group patients by their
    dominant outcome and shuffle within those strata — this keeps the positive
    rate close across splits while guaranteeing no patient spans two sets.
    """
    patient_label = df.groupby(GROUP_COL)[TARGET].max()
    patients = patient_label.index.to_numpy()
    labels = patient_label.to_numpy()

    rng = np.random.default_rng(random_state)

    def _take(pool: np.ndarray, pool_labels: np.ndarray, frac: float):
        """Take `frac` of patients from each label stratum."""
        picked: list[np.ndarray] = []
        for lab in np.unique(pool_labels):
            members = pool[pool_labels == lab]
            members = rng.permutation(members)
            k = int(round(len(members) * frac))
            picked.append(members[:k])
        return np.concatenate(picked)

    test_patients = _take(patients, labels, test_size)
    remaining_mask = ~np.isin(patients, test_patients)
    rem, rem_labels = patients[remaining_mask], labels[remaining_mask]

    # val_size is a fraction of the whole, so rescale against what's left.
    val_patients = _take(rem, rem_labels, val_size / (1.0 - test_size))
    train_patients = rem[~np.isin(rem, val_patients)]

    parts = tuple(
        df[df[GROUP_COL].isin(p)].reset_index(drop=True)
        for p in (train_patients, val_patients, test_patients)
    )
    assert_no_patient_leakage(*parts)
    return parts


def assert_no_patient_leakage(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Hard guarantee that no patient appears in more than one split."""
    tr, va, te = (set(d[GROUP_COL]) for d in (train, val, test))
    for a, b, name in ((tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")):
        overlap = a & b
        if overlap:
            raise AssertionError(
                f"Patient leakage across {name}: {len(overlap)} shared patients"
            )


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) model feature names, excluding leaky columns."""
    numeric = [
        "age", "lead_time_days", "is_same_day", "appt_dayofweek", "appt_month",
        "sched_dayofweek", "sched_hour", "is_female", "scholarship", "hypertension",
        "diabetes", "alcoholism", "handicap", "has_handicap", "n_conditions",
        "sms_received", "prior_appointments", "prior_noshows", "prior_noshow_rate",
        "is_first_appointment", "days_since_last_appointment",
    ]
    categorical = ["neighbourhood", "age_group"]
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


if __name__ == "__main__":
    print("Loading raw data...")
    raw = load_raw()
    print(f"  {len(raw):,} rows, {raw[TARGET].mean():.2%} no-show")

    print("Cleaning...")
    cleaned = clean(raw)

    print("Building features...")
    feats = build_features(cleaned)

    print("Splitting by patient (70/15/15)...")
    train, val, test = split_by_patient(feats)
    total = len(feats)
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(
            f"  {name:<5} {len(part):>7,} rows "
            f"({len(part)/total:.1%})  {part[GROUP_COL].nunique():>6,} patients  "
            f"no-show {part[TARGET].mean():.2%}"
        )
    print("No patient leakage across splits.")
