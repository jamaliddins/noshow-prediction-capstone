"""Shared configuration: paths, constants, random seed."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_CSV = DATA_DIR / "KaggleV2-May-2016.csv"

RANDOM_STATE = 42

# 70/15/15 train/val/test, split by patient (never by row).
TEST_SIZE = 0.15
VAL_SIZE = 0.15

TARGET = "no_show"
GROUP_COL = "patient_id"

# Never usable as features: identifiers, or known only after the appointment.
LEAKY_COLS = ["patient_id", "appointment_id", "no_show"]

RISK_TIERS = {"low": "Low", "medium": "Medium", "high": "High"}
