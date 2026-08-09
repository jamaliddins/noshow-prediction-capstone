"""Train and compare models, logging every run to MLflow.

Progression: majority-class baseline -> Logistic Regression (the bar every
later model must clear) -> Random Forest -> XGBoost.

Model selection uses validation F1 only. The test set is scored exactly once,
at the end, with the threshold already fixed on validation.
"""
from __future__ import annotations

import json
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PROJECT_ROOT, RANDOM_STATE, TARGET
from src.evaluate import (
    compute_metrics,
    confusion_summary,
    find_best_threshold,
    format_metrics,
)
from src.preprocessing import (
    build_features,
    clean,
    get_feature_columns,
    load_raw,
    split_by_patient,
)

warnings.filterwarnings("ignore", category=FutureWarning)

MLFLOW_EXPERIMENT = "noshow-prediction"


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Scale numerics, one-hot the categoricals.

    Scaling matters for Logistic Regression; it is harmless for the tree models,
    so one shared preprocessor keeps the comparison honest.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                              sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def get_models(scale_pos_weight: float) -> dict[str, object]:
    """Candidate estimators, each handling imbalance in its own idiom."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def _predict_proba(pipe: Pipeline, X) -> np.ndarray:
    """Positive-class probability, tolerating the degenerate baseline."""
    proba = pipe.predict_proba(X)
    if proba.shape[1] == 1:  # DummyClassifier saw a single class
        return np.zeros(len(X)) if pipe.classes_[0] == 0 else np.ones(len(X))
    return proba[:, 1]


def train_all(log_to_mlflow: bool = True) -> dict:
    print("=" * 78)
    print("LOADING AND PREPARING DATA")
    print("=" * 78)
    df = build_features(clean(load_raw(), verbose=True))
    train, val, test = split_by_patient(df)

    numeric, categorical = get_feature_columns(df)
    features = numeric + categorical
    print(f"\n  {len(features)} features ({len(numeric)} numeric, "
          f"{len(categorical)} categorical)")
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:<5} {len(part):>7,} rows  no-show {part[TARGET].mean():.2%}")

    X_train, y_train = train[features], train[TARGET].to_numpy()
    X_val, y_val = val[features], val[TARGET].to_numpy()
    X_test, y_test = test[features], test[TARGET].to_numpy()

    # XGBoost's imbalance knob: ratio of negatives to positives.
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    if log_to_mlflow:
        # MLflow 3 deprecated the file store; SQLite is the recommended local
        # backend. View with: mlflow ui --backend-store-uri sqlite:///mlflow.db
        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("\n" + "=" * 78)
    print("TRAINING (validation F1 selects the model and its threshold)")
    print("=" * 78)

    results: dict[str, dict] = {}
    for name, estimator in get_models(scale_pos_weight).items():
        pipe = Pipeline([
            ("prep", build_preprocessor(numeric, categorical)),
            ("clf", estimator),
        ])
        pipe.fit(X_train, y_train)

        val_prob = _predict_proba(pipe, X_val)

        # The baseline has no meaningful threshold to tune; leave it at 0.5.
        if name == "majority_baseline":
            best_t, _ = 0.5, None
        else:
            best_t, _ = find_best_threshold(y_val, val_prob, metric="f1")

        val_metrics = compute_metrics(y_val, val_prob, best_t)
        val_at_half = compute_metrics(y_val, val_prob, 0.5)

        results[name] = {
            "pipeline": pipe,
            "threshold": best_t,
            "val": val_metrics,
            "val_at_0.5": val_at_half,
        }

        print(f"\n  {format_metrics(name, val_metrics)}")
        if name != "majority_baseline":
            print(f"    (at default t=0.5: F1={val_at_half['f1']:.4f} "
                  f"-> tuning gains {val_metrics['f1'] - val_at_half['f1']:+.4f})")

        if log_to_mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_param("n_features", len(features))
                mlflow.log_param("n_train_rows", len(X_train))
                mlflow.log_param("split", "GroupShuffleSplit on patient_id 70/15/15")
                mlflow.log_param("random_state", RANDOM_STATE)
                if hasattr(estimator, "get_params"):
                    for k, v in estimator.get_params().items():
                        if isinstance(v, (int, float, str, bool, type(None))):
                            mlflow.log_param(f"clf__{k}", v)
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", v)

    # --- Model selection, on validation only -------------------------------
    ranked = sorted(
        ((n, r) for n, r in results.items() if n != "majority_baseline"),
        key=lambda kv: kv[1]["val"]["f1"],
        reverse=True,
    )
    best_name, best = ranked[0]
    lr_f1 = results["logistic_regression"]["val"]["f1"]

    print("\n" + "=" * 78)
    print("MODEL COMPARISON (validation)")
    print("=" * 78)
    print(f"  {'model':<24}{'F1':>9}{'PR-AUC':>10}{'ROC-AUC':>10}   vs LR")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["val"]["f1"],
                          reverse=True):
        m = r["val"]
        delta = "" if name in ("logistic_regression", "majority_baseline") \
            else f"{m['f1'] - lr_f1:+.4f}"
        mark = " <-- best" if name == best_name else ""
        print(f"  {name:<24}{m['f1']:>9.4f}{m['pr_auc']:>10.4f}"
              f"{m['roc_auc']:>10.4f}   {delta:<9}{mark}")

    beat_lr = [n for n, r in ranked if r["val"]["f1"] > lr_f1 and
               n != "logistic_regression"]
    print(f"\n  Models beating Logistic Regression: "
          f"{', '.join(beat_lr) if beat_lr else 'none'}")

    # --- Calibration --------------------------------------------------------
    # class_weight / scale_pos_weight deliberately distort probabilities to
    # correct the imbalance, which is good for ranking but makes the output
    # unreadable as a probability (mean ~0.43 against a true rate of ~0.20).
    # The brief asks staff to read a percentage, so refit the best model on
    # train and calibrate it against validation.
    print("\n" + "=" * 78)
    print("CALIBRATION")
    print("=" * 78)
    raw_val_prob = _predict_proba(best["pipeline"], X_val)
    print(f"  before: mean predicted {raw_val_prob.mean():.3f} "
          f"vs actual {y_val.mean():.3f}  (Brier {best['val']['brier']:.4f})")

    # FrozenEstimator keeps the fitted pipeline as-is; only the calibrator
    # is fitted here (sklearn >=1.6 replacement for cv="prefit").
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(best["pipeline"]), method="isotonic"
    )
    calibrated.fit(X_val, y_val)
    cal_val_prob = calibrated.predict_proba(X_val)[:, 1]

    # Re-tune the threshold: calibration rescales the probability axis.
    cal_threshold, _ = find_best_threshold(y_val, cal_val_prob, metric="f1")
    cal_val_metrics = compute_metrics(y_val, cal_val_prob, cal_threshold)
    print(f"  after:  mean predicted {cal_val_prob.mean():.3f} "
          f"vs actual {y_val.mean():.3f}  (Brier {cal_val_metrics['brier']:.4f})")
    print(f"  F1 {best['val']['f1']:.4f} -> {cal_val_metrics['f1']:.4f} "
          f"@ t={cal_threshold:.3f}")

    # Calibration is fitted on validation, so val F1 is now optimistic; the
    # held-out test set below is what the reported number comes from.
    final_model, final_threshold = calibrated, cal_threshold

    # Risk tiers from validation quantiles, so each tier is a usable share of
    # the workload rather than an almost-empty bucket.
    low_max = float(np.percentile(cal_val_prob, 60))
    medium_max = float(np.percentile(cal_val_prob, 90))
    print(f"\n  risk tiers (validation quantiles): "
          f"Low <{low_max:.3f}  Medium <{medium_max:.3f}  High >=")
    for tier, lo, hi in (("Low", 0.0, low_max), ("Medium", low_max, medium_max),
                         ("High", medium_max, 1.01)):
        mask = (cal_val_prob >= lo) & (cal_val_prob < hi)
        rate = y_val[mask].mean() if mask.sum() else 0.0
        print(f"    {tier:<7}{mask.sum():>7,} ({mask.mean():>5.1%})  "
              f"actual no-show {rate:>6.1%}")

    # --- Final: score the held-out test set exactly once --------------------
    print("\n" + "=" * 78)
    print(f"FINAL TEST EVALUATION — {best_name} (calibrated) @ t={final_threshold:.3f}")
    print("=" * 78)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob, final_threshold)
    print(f"\n  {format_metrics(best_name, test_metrics)}")
    print(confusion_summary(test_metrics))
    print(f"    Mean predicted {test_prob.mean():.3f} vs actual "
          f"{y_test.mean():.3f} — calibrated.")

    gap = cal_val_metrics["f1"] - test_metrics["f1"]
    print(f"\n  val F1 {cal_val_metrics['f1']:.4f} -> test F1 "
          f"{test_metrics['f1']:.4f} (gap {gap:+.4f})")

    if log_to_mlflow:
        with mlflow.start_run(run_name=f"{best_name}_calibrated_TEST"):
            mlflow.log_param("model", best_name)
            mlflow.log_param("selected_on", "validation F1")
            mlflow.log_param("calibration", "isotonic (prefit, on validation)")
            mlflow.log_param("threshold", final_threshold)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

    # --- Persist the artifacts predict.py will load -------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "model.joblib")

    metadata = {
        "model_name": best_name,
        "calibration": "isotonic",
        "threshold": final_threshold,
        "features": {"numeric": numeric, "categorical": categorical},
        "validation_metrics": {k: v for k, v in cal_val_metrics.items()},
        "test_metrics": test_metrics,
        "train_rows": len(X_train),
        "random_state": RANDOM_STATE,
        # Tier cuts from validation quantiles: top 10% High, next 30% Medium.
        "risk_tiers": {"low_max": low_max, "medium_max": medium_max},
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\n  saved models/model.joblib ({best_name})")
    print(f"  saved models/model_metadata.json")

    results["_best"] = {"name": best_name, "test": test_metrics,
                        "threshold": final_threshold, "model": final_model}
    results["_data"] = {"test": test, "test_prob": test_prob,
                        "features": features, "y_test": y_test}
    return results


if __name__ == "__main__":
    train_all()
