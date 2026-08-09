# No-Show Prediction — MED-01 Capstone

Predicts whether a patient will miss a scheduled medical appointment, using the
Kaggle ["Medical Appointment No Shows"](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
dataset (~110,527 appointments, ~62k patients, public clinics in Vitória, Espírito
Santo, Brazil, 2016).

This is a **prototype decision-support tool** for clinic operations staff to
prioritize reminder/confirmation calls. It is not a clinical or punitive system.

## Status

Work in progress. See `CLAUDE.md` for project rules and
`Jamoliddin_Solikhov_MED-01.docx` for the full technical proposal.

## Setup

```bash
pip install -r requirements.txt
```

Place `KaggleV2-May-2016.csv` in `data/` (not committed — see `.gitignore`).

## Project structure

```
src/
  preprocessing.py   # cleaning, feature engineering, patient-grouped split
  train.py           # baselines + models, MLflow logging
  evaluate.py         # metrics, confusion matrix, threshold tuning
  predict.py          # inference: probability + risk tier
notebooks/            # EDA and demo notebooks
models/                # saved joblib artifacts (not committed)
reports/figures/       # evaluation plots
```

## Key methodology decisions

- **Patient-grouped split** (`GroupShuffleSplit` on `PatientId`, stratified on
  target, 70/15/15) — no patient appears in both train and test.
- **No-show history features use only strictly-prior appointments** for each
  patient.
- **Primary metric: F1** on the no-show class (not accuracy — classes are
  ~20/80 imbalanced). PR-AUC, ROC-AUC, and confusion matrix reported alongside.
  Decision threshold tuned on validation, not assumed at 0.5.
- `AppointmentID` and `PatientId` are never used as model features.

## Run instructions

(To be filled in as `src/` scripts land.)

## Limitations

(To be filled in — see technical proposal doc for the current draft: single
city/year, SMS timing verification, socioeconomic proxy fairness concerns.)
