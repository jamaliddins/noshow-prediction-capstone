# No-Show Prediction — MED-01 Capstone

**Author:** Jamoliddin Solikhov
**Track:** MED-01 — Healthcare / Medical
**ML task:** Binary classification (supervised), on imbalanced tabular data

Predicts whether a patient will miss a scheduled medical appointment, using the
Kaggle ["Medical Appointment No Shows"](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
dataset (110,527 appointments, 62,299 patients, public clinics in Vitória, Espírito
Santo, Brazil, 2016).

This is a **prototype decision-support tool** for clinic operations staff to
prioritize reminder and confirmation calls. It is not a clinical or punitive system.

## Dataset sample

The first 10 rows of the raw Kaggle CSV, unmodified. The dataset itself is not
committed (see [Quickstart](#quickstart) for the download step) - this is here so
the schema is readable without obtaining the file.

| PatientId | ApptID | Gen | ScheduledDay | ApptDay | Age | Neighbourhood | Schol | Hiper | Diab | Alco | Handcap | SMS | No-show |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 29872499824296 | 5642903 | F | 2016-04-29 18:38:08 | 2016-04-29 | 62 | JARDIM DA PENHA | 0 | 1 | 0 | 0 | 0 | 0 | No |
| 558997776694438 | 5642503 | M | 2016-04-29 16:08:27 | 2016-04-29 | 56 | JARDIM DA PENHA | 0 | 0 | 0 | 0 | 0 | 0 | No |
| 4262962299951 | 5642549 | F | 2016-04-29 16:19:04 | 2016-04-29 | 62 | MATA DA PRAIA | 0 | 0 | 0 | 0 | 0 | 0 | No |
| 867951213174 | 5642828 | F | 2016-04-29 17:29:31 | 2016-04-29 | 8 | PONTAL DE CAMBURI | 0 | 0 | 0 | 0 | 0 | 0 | No |
| 8841186448183 | 5642494 | F | 2016-04-29 16:07:23 | 2016-04-29 | 56 | JARDIM DA PENHA | 0 | 1 | 1 | 0 | 0 | 0 | No |
| 95985133231274 | 5626772 | F | 2016-04-27 08:36:51 | 2016-04-29 | 76 | REPÚBLICA | 0 | 1 | 0 | 0 | 0 | 0 | No |
| 733688164476661 | 5630279 | F | 2016-04-27 15:05:12 | 2016-04-29 | 23 | GOIABEIRAS | 0 | 0 | 0 | 0 | 0 | 0 | Yes |
| 3449833394123 | 5630575 | F | 2016-04-27 15:39:58 | 2016-04-29 | 39 | GOIABEIRAS | 0 | 0 | 0 | 0 | 0 | 0 | Yes |
| 56394729949972 | 5638447 | F | 2016-04-29 08:02:16 | 2016-04-29 | 21 | ANDORINHAS | 0 | 0 | 0 | 0 | 0 | 0 | No |
| 78124564369297 | 5629123 | F | 2016-04-27 12:48:25 | 2016-04-29 | 19 | CONQUISTA | 0 | 0 | 0 | 0 | 0 | 0 | No |

Notes on the raw schema, all handled in `src/preprocessing.py`:

- **`No-show` is inverted.** `Yes` means the patient *did not* attend, so it becomes
  the positive class (`1`). Rows 7 and 8 above are the missed appointments.
- **Three headers are misspelled** in the source: `Hipertension`, `Handcap`, and the
  hyphenated `No-show`. All are renamed on load.
- **`Handcap` is a 0-4 count, not a flag** - it is kept as both the raw count and a
  binary `has_handicap`.
- **`ScheduledDay` carries a timestamp, `AppointmentDay` does not** (always
  `00:00:00`). Dates are therefore compared as calendar dates, so a same-day booking
  at 18:00 is legitimate rather than a data error.
- **`PatientId` repeats** - 62,299 patients across 110,527 appointments, which is
  why the split is grouped by patient rather than random.

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
pytest                            # 63 tests
```

View experiments: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## API

```bash
uvicorn src.api:app --reload      # interactive docs at http://localhost:8000/docs
```

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness, and whether the model is loaded |
| GET | `/model` | Model name, test metrics, thresholds in use |
| POST | `/predict` | Score one appointment |
| POST | `/predict/batch` | Score up to 1000, highest risk first |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"scheduled_day":"2016-05-02T09:00:00","appointment_day":"2016-05-30",
       "age":22,"gender":"F","neighbourhood":"JARDIM CAMBURI"}'
```

```json
{
  "no_show_probability": 0.4044,
  "no_show_percentage": 40.4,
  "risk_tier": "High",
  "recommendation": "Call to confirm — highest priority for staff follow-up.",
  "lead_time_days": 28,
  "threshold_used": 0.2173,
  "model": "xgboost"
}
```

Invalid input returns **422** with a specific message — missing fields, an
appointment date before the booking date, impossible ages, unknown gender
values, or `prior_noshows` exceeding `prior_appointments`.

## Docker

```bash
docker compose up --build                  # API on :8000
docker compose run --rm train              # train into ./models
bash scripts/docker_smoke_test.sh          # build + verify endpoints
```

The image contains code only. `data/` and `models/` are bind-mounted, so neither
the dataset nor the model artifact is baked in. Runs as a non-root user with a
healthcheck on `/health`.

> **Note:** the API is verified end to end (live uvicorn server, 22 passing
> tests). The Docker setup is written and reviewed but **has not been built on
> this machine** — Docker was not available in the development environment. Run
> `scripts/docker_smoke_test.sh` to verify it in yours.

### Demo notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) runs top-to-bottom on a clean
Google Colab runtime — it clones this repo, fetches the dataset, trains the model
and demonstrates inference. Verified end to end: 18 code cells, zero errors.

## Pipeline / architecture

```
data/KaggleV2-May-2016.csv          raw Kaggle export (not committed)
        │
        ▼
src/preprocessing.py                clean 11 bad rows · engineer 23 features
        │                           patient-grouped 70/15/15 split
        │                           history features from strictly prior rows
        ▼
src/train.py                        ColumnTransformer (scale + one-hot)
        │                           4 models compared · isotonic calibration
        │                           every run logged to MLflow
        ▼
models/*.joblib                     calibrated pipeline + threshold + tiers
        │
        ├──► src/evaluate.py + evaluation_plots.py   metrics · 13 figures
        ├──► src/predict.py                          probability + risk tier
        └──► src/api.py                              FastAPI /predict endpoints
```

Preprocessing, training and inference all share one `Pipeline` object, so the
transformations applied at inference are exactly those fitted during training.

## Example input / output

**Input** — only fields knowable *before* the appointment:

```json
{
  "scheduled_day": "2016-05-02T09:00:00",
  "appointment_day": "2016-05-30",
  "age": 22,
  "gender": "F",
  "neighbourhood": "JARDIM CAMBURI"
}
```

**Output** — a calibrated probability, a tier, and the action to take:

```json
{
  "no_show_probability": 0.4044,
  "no_show_percentage": 40.4,
  "risk_tier": "High",
  "recommendation": "Call to confirm — highest priority for staff follow-up.",
  "lead_time_days": 28,
  "threshold_used": 0.2173,
  "model": "xgboost"
}
```

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
  api.py                # FastAPI service
tests/                  # 63 tests, focused on the leakage guarantees
notebooks/demo.ipynb    # Colab-runnable demo
reports/figures/        # 13 figures
models/                 # saved artifacts (not committed)
scripts/                # Docker smoke test
Dockerfile              # API image (code only; data/models mounted)
docker-compose.yml      # serve, or run a one-shot training job
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

## Error analysis

`python -m src.error_analysis` → [`reports/figures/14_error_analysis.png`](reports/figures/14_error_analysis.png)

Aggregate F1 hides *which* appointments the model gets wrong. Slicing the 16,751
test appointments by error type gives four findings:

**1. Same-day appointments are the blind spot.** Recall collapses to 18.4% there
against 88.4% at 31+ days:

| Lead time | n | Actual rate | Recall | Precision |
|---|---|---|---|---|
| Same day | 5,950 | 4.7% | **18.4%** | 38.8% |
| 1–3 | 2,112 | 23.5% | 50.6% | 27.5% |
| 4–7 | 2,606 | 25.9% | 65.8% | 31.9% |
| 8–14 | 1,800 | 30.3% | 84.0% | 34.6% |
| 15–30 | 2,667 | 31.8% | 88.1% | 35.1% |
| 31+ | 1,616 | 32.1% | **88.4%** | 37.0% |

This is the model behaving correctly, not a bug: same-day appointments genuinely
miss only 4.7% of the time, so the calibrated probability rarely clears the
threshold. It matters operationally because same-day bookings are 36% of all
appointments — the call list will systematically under-cover them.

**2. Most errors are borderline, not confident blunders.** 61.8% of false
negatives fall within 0.05 of the 0.217 threshold. The model is uncertain where
it is wrong, which is the desired failure mode — it is not confidently wrong.

**3. Errors track the base rate by age.** Recall is 32.8% for seniors (14.5%
actual no-show rate) and 89.1% for teens (26.2%). The model is weakest exactly
where no-shows are rarest, which follows from a single global threshold.

**4. Fairness — the welfare subgroup is flagged more often.** This is the finding
that most constrains deployment:

| Group | n | Actual rate | Flagged | Recall | Precision |
|---|---|---|---|---|---|
| No welfare | 15,080 | 19.6% | 41.2% | 69.9% | 33.4% |
| Scholarship (welfare) | 1,671 | 24.1% | **55.2%** | 84.3% | 36.8% |

Welfare recipients are flagged at 55.2% versus 41.2% — a **14-point gap**.
Precision is comparable across groups (36.8% vs 33.4%), so the model is not
*less accurate* for this group; it flags them more because they genuinely miss
more (24.1% vs 19.6%). The disparity is in exposure, not in error rate.

Since the intervention is a reminder phone call, higher exposure is defensible
and arguably beneficial. It would **not** be defensible for any rationing,
penalty, or deprioritisation use — see Responsible AI below.

## Responsible AI

This model predicts human behaviour from administrative records about patients,
in a healthcare setting. The limitations below are therefore part of the
deliverable, not a footnote to it.

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

## License and acknowledgements

### Dataset

**"Medical Appointment No Shows"** by **JoniHoppen** ([Aquarela Analytics](http://bit.ly/2Ulwskh)),
published on Kaggle: https://www.kaggle.com/datasets/joniarroba/noshowappointments

Licensed **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**
(Attribution — NonCommercial — ShareAlike). The terms carry three consequences
for this project:

- **Attribution** — the author is credited here and in the demo notebook.
- **NonCommercial** — this is coursework. The dataset, and any model trained on
  it, may not be used commercially. A real clinic deployment would require
  retraining on data the operator is licensed to use.
- **ShareAlike** — derivatives of the dataset must carry the same licence.

The data is de-identified: patients appear as numeric ids, with no names,
addresses or contact details. It is not re-identifiable from the fields provided.

The raw CSV is **not redistributed in this repository** (`.gitignore` excludes
`data/`). The demo notebook downloads it at run time and verifies the schema and
row count before use.

### Code

Project code is released under the MIT Licence — see [LICENSE](LICENSE).
This covers the code only, not the dataset, which remains under CC BY-NC-SA 4.0.

### Libraries

Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/),
[pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/),
[MLflow](https://mlflow.org/), [matplotlib](https://matplotlib.org/),
[seaborn](https://seaborn.pydata.org/), [FastAPI](https://fastapi.tiangolo.com/)
and [pytest](https://pytest.org/) — all open source under permissive licences.

### AI assistance

Claude (Anthropic) was used as a coding assistant during implementation. All
methodology decisions, results and their interpretation are the author's own, and
every component is understood and defensible as required by the course policy.
