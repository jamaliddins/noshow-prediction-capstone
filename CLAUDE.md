# CLAUDE.md — No-Show Prediction Capstone

## Project
Predict whether a patient will miss a scheduled medical appointment.
Dataset: Kaggle "Medical Appointment No Shows" (~110,527 rows, ~62k unique patients, Brazil 2016).
Target: `No-show` → 1 = did not attend, 0 = attended.
This is a **prototype decision-support tool**, not a clinical or punitive system.

## Non-negotiable ML rules
- **Leakage-safe split is mandatory.** Split by patient, not by row: use `GroupShuffleSplit` on `PatientId` so the same patient never appears in both train and test. Also stratify on the target. 70/15/15 train/val/test. Hold the test set out until final evaluation.
- **Patient no-show history features must use only past appointments** (strictly before the row being predicted). Never compute them across the whole dataset.
- **Never use as features:** `AppointmentID`, `PatientId`, or anything only known after the appointment.
- **Verify `SMS_received` timing** before trusting it — flag if it could be post-hoc.
- **Primary metric: F1** on the no-show class. Report PR-AUC, ROC-AUC, and a confusion matrix too. Do not optimize or report accuracy alone (classes are ~20/80 imbalanced).
- Tune the decision threshold on validation; do not assume 0.5.

## Data quality to handle
- Drop/repair impossible `Age` (negative, absurdly high).
- Remove rows where `ScheduledDay` is after `AppointmentDay`.
- Standardize the misspelled columns: `Hipertension`, `Handcap`, `No-show`, `SMS_received`.

## Modeling
- Baselines first: majority-class, then Logistic Regression. Every later model must beat LR's F1.
- Then Random Forest and XGBoost. Test `class_weight='balanced'` / resampling for imbalance.
- Log every experiment to MLflow (params + metrics).

## Output
- Inference returns a no-show **probability (0–100%)** plus a **risk tier (Low/Medium/High)**.
- Demo notebook visualizes it: gauge/bar for one appointment, sorted risk table/bar chart for a batch.

## Conventions
- Python 3.10+. Reusable logic in `src/`, exploration in `notebooks/`.
- `src/preprocessing.py`, `src/train.py`, `src/evaluate.py`, `src/predict.py`.
- Save model artifacts with `joblib` to `models/`.
- Figures to `reports/figures/`.
- Keep code reproducible: fixed `random_state`, everything runnable top-to-bottom on a fresh runtime.
- **The final demo must run in a clean Google Colab pulling from this GitHub repo.**

## Workflow rule
- **After each step, run the code and confirm it works before moving on** (execute the notebook/script, check the assertion or metric, don't just write code).
- Never commit secrets, tokens, the raw dataset, or large binaries. `.gitignore` must cover `data/`, `*.pkl`, `.env`, and credentials.

## Git
- Commit in small, described steps. Push to the project repo after each working step.
