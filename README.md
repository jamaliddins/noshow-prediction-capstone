# No-Show Prediction — MED-01 Capstone

Predicts whether a patient will miss a scheduled medical appointment, using the
Kaggle ["Medical Appointment No Shows"](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
dataset (110,527 appointments, 62,299 patients, public clinics in Vitória, Espírito
Santo, Brazil, 2016).

This is a **prototype decision-support tool** for clinic operations staff to
prioritize reminder and confirmation calls. It is not a clinical or punitive system.

## Results

Selected model: **XGBoost with isotonic calibration**, scored once on a held-out
test set of 16,751 appointments.

| Metric | Value |
|---|---|
| **F1 (no-show class)** | **0.459** |
| Precision | 0.338 |
| Recall | 0.717 |
| PR-AUC | 0.411 |
| ROC-AUC | 0.754 |
| Decision threshold | 0.217 (tuned on validation) |

Model progression — every model after the baseline had to beat Logistic Regression:

| Model | Validation F1 |
|---|---|
| Majority-class baseline | 0.000 |
| Logistic Regression | 0.450 |
| Random Forest | 0.454 |
| **XGBoost** | **0.457** |

The validation → test gap is **−0.001**, indicating the patient-grouped split held.

### What this means operationally

| Risk tier | Share of appointments | Actual no-show rate |
|---|---|---|
| Low | 60.7% | 10.6% |
| Medium | 28.9% | 30.2% |
| **High** | **10.4%** | **47.3%** |

High-risk patients miss **4.5× more often** than low-risk ones. Calling the top
10% of appointments reaches 25% of all no-shows.

**On the modest F1:** no feature correlates above 0.28 with the target, so this is
close to the ceiling for this dataset. The model is useful for *ranking* who to
call, not for confident individual judgements.

## Quickstart

```bash
git clone https://github.com/jamaliddins/noshow-prediction-capstone.git
cd noshow-prediction-capstone
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Place `KaggleV2-May-2016.csv` in `data/` (not committed — see `.gitignore`), then:

```bash
python -m src.preprocessing       # clean, engineer features, split
python -m src.visualize           # EDA figures        -> reports/figures/
python -m src.train               # train, log to MLflow, save model
python -m src.evaluation_plots    # result figures     -> reports/figures/
python -m src.predict             # inference demo
pytest                            # 41 tests
```

View experiments: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

### Demo notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) runs top-to-bottom on a clean
Google Colab runtime — it clones this repo, fetches the dataset, trains the model
and demonstrates inference. Verified end to end: 18 code cells, zero errors.

## Project structure

```
src/
  config.py             # paths, random seed, constants
  preprocessing.py      # cleaning, feature engineering, patient-grouped split
  train.py              # baselines + models, calibration, MLflow logging
  evaluate.py           # metrics and threshold selection
  evaluation_plots.py   # post-training result figures
  predict.py            # inference: probability + risk tier
  visualize.py          # pre-modelling EDA figures
tests/                  # 41 tests, focused on the leakage guarantees
notebooks/demo.ipynb    # Colab-runnable demo
reports/figures/        # 13 figures
models/                 # saved artifacts (not committed)
```

## Methodology

**Leakage safety** — the two risks that would inflate results here, and how each
is prevented:

1. **Repeated patients.** 62,299 patients generate 110,527 appointments, so a
   random row split would place the same patient in both train and test. The
   split is grouped on `PatientId` (70/15/15), stratified by each patient's
   dominant outcome, with an assertion that no patient spans two splits.
2. **History features.** `prior_noshow_rate` and related features are computed
   from appointments scheduled *strictly before* the row being predicted, using
   shifted expanding aggregates. A test flips a row's own outcome and asserts
   that row's features do not change.

`AppointmentID` and `PatientId` are never used as features.

**Metric choice** — classes are ~20/80, so accuracy is misleading: always
predicting "attended" scores 79.8% while catching zero no-shows. Primary metric
is **F1 on the no-show class**, with PR-AUC, ROC-AUC and a confusion matrix
reported alongside. The decision threshold is tuned on validation (0.217), not
assumed at 0.5 — at the default it would score F1 0.107.

**Calibration** — `scale_pos_weight` corrects the imbalance but inflates
probabilities (mean 0.431 against a true rate of 0.202). Since the deliverable
shows staff a percentage, isotonic calibration is fitted on validation: mean
predicted becomes 0.202 and Brier improves from 0.204 to 0.140, with F1 unchanged.

**Data cleaning** — 11 rows removed from 110,527:
- 5 where `ScheduledDay` fell after `AppointmentDay`
- 6 with impossible ages (one negative, five above 110)

Dates are compared as **calendar dates**, not timestamps: an appointment booked
at 18:00 for that same day is legitimate, and a naive timestamp comparison would
discard ~38,000 valid same-day rows.

## Key findings

- **Lead time dominates.** Same-day appointments have a 5% no-show rate; those
  booked 31+ days ahead, 33%. It is the top feature by permutation importance.
- **The SMS paradox.** Patients who received an SMS miss *more* appointments
  (27.6% vs 16.7%) — but SMS is only sent for longer waits, and conditioning on
  lead time reverses the effect. A confounder, not a causal signal.
- **Sicker patients attend better.** Hypertension 17.3% vs 20.9% without.
- **Prior behaviour predicts future behaviour.** Patients who never missed before
  are at 15%; those who missed over half their prior appointments, 34%.

## Limitations

1. **One city, one year** (Vitória, 2016). Generalization elsewhere is untested.
2. **Modest discrimination** — F1 0.459, ROC-AUC 0.754. Useful for ranking, not
   for confident individual predictions.
3. **Precision ~34%** — about two in three flagged patients would have attended.
   Acceptable when the action is a phone call; not for anything punitive.
4. **Fairness** — `Neighbourhood` and `Scholarship` are socioeconomic proxies and
   no-show rates vary by neighbourhood. Must not be used to deprioritise or deny
   care.
5. **SMS timing is undocumented** in the source data. The confounding analysis is
   consistent with it being pre-appointment, but this could not be verified.
6. **No specialty or appointment type**, which likely matter and are absent.

**Appropriate use:** a prioritisation aid for reminder calls with a human in the
loop — never an automated clinical or administrative decision.

## Next steps

- Validate on another clinic or year before any deployment
- Add specialty, appointment type and distance-to-clinic if obtainable
- Run a controlled trial: does calling the High tier actually reduce no-shows?
- Monitor per-neighbourhood performance for disparities in production

## Reproducibility

Fixed `random_state=42` throughout. Python 3.10+. All scripts run top-to-bottom
on a fresh runtime, and the demo notebook is verified against a clean Colab.
