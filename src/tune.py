"""Systematic hyperparameter search for the XGBoost candidate.

Randomised search over a defined grid, scored by F1 on the no-show class with
patient-grouped cross-validation folds, so no patient spans a fold boundary.
Every trial is logged to MLflow alongside the runs from src/train.py.

    python -m src.tune                 # 40 trials (default)
    python -m src.tune --n-iter 10     # quicker
    python -m src.tune --no-mlflow     # skip logging

The search runs on train+validation only; the test set is never touched here.
"""
from __future__ import annotations

import argparse
import json
import warnings

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import GROUP_COL, MODELS_DIR, PROJECT_ROOT, RANDOM_STATE, TARGET
from src.preprocessing import (
    build_features,
    clean,
    get_feature_columns,
    load_raw,
    split_by_patient,
)
from src.train import MLFLOW_EXPERIMENT, build_preprocessor

warnings.filterwarnings("ignore", category=FutureWarning)

# Ranges chosen around the hand-picked values in src/train.py, wide enough to
# tell whether those defaults were actually a good choice.
SEARCH_SPACE = {
    "clf__n_estimators": randint(200, 800),
    "clf__max_depth": randint(3, 9),
    "clf__learning_rate": loguniform(0.01, 0.3),
    "clf__subsample": uniform(0.6, 0.4),          # 0.6 - 1.0
    "clf__colsample_bytree": uniform(0.6, 0.4),   # 0.6 - 1.0
    "clf__min_child_weight": randint(1, 20),
    "clf__gamma": uniform(0.0, 5.0),
    "clf__reg_lambda": loguniform(0.1, 10.0),
}


def run_search(n_iter: int = 40, log_to_mlflow: bool = True) -> dict:
    print("=" * 78)
    print("HYPERPARAMETER SEARCH — XGBoost")
    print("=" * 78)

    df = build_features(clean(load_raw(), verbose=False))
    train, val, test = split_by_patient(df)
    numeric, categorical = get_feature_columns(df)
    features = numeric + categorical

    # Search on train+val; the test set stays sealed until src/train.py scores it.
    search_df = pd.concat([train, val], ignore_index=True)
    X = search_df[features]
    y = search_df[TARGET].to_numpy()
    groups = search_df[GROUP_COL].to_numpy()
    print(f"  search set: {len(X):,} rows, {search_df[GROUP_COL].nunique():,} patients")

    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    pipe = Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("clf", XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])

    # Grouped folds: a patient's rows stay together, mirroring the real split.
    cv = GroupKFold(n_splits=4)
    search = RandomizedSearchCV(
        pipe, SEARCH_SPACE, n_iter=n_iter, scoring="f1", cv=cv,
        random_state=RANDOM_STATE, n_jobs=1, verbose=1, refit=False,
    )
    print(f"  {n_iter} trials x {cv.get_n_splits()} folds, scoring = F1 (no-show)\n")
    search.fit(X, y, groups=groups)

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    print("\n" + "=" * 78)
    print("TOP 5 TRIALS")
    print("=" * 78)
    show = ["rank_test_score", "mean_test_score", "std_test_score"]
    param_cols = [c for c in results.columns if c.startswith("param_")]
    for _, row in results.head(5).iterrows():
        print(f"  #{int(row['rank_test_score'])}  F1 {row['mean_test_score']:.4f} "
              f"+/- {row['std_test_score']:.4f}")
        for c in param_cols:
            print(f"        {c.replace('param_clf__',''):<20}{row[c]}")

    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"\n  best CV F1: {search.best_score_:.4f}")

    # How the hand-picked defaults in src/train.py compare.
    baseline = {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
                "subsample": 0.9, "colsample_bytree": 0.9}
    print(f"  hand-picked defaults were: {baseline}")

    if log_to_mlflow:
        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        for _, row in results.iterrows():
            with mlflow.start_run(run_name=f"tune_trial_{int(row['rank_test_score']):03d}"):
                mlflow.log_param("search", "RandomizedSearchCV")
                mlflow.log_param("cv", "GroupKFold(4) on patient_id")
                for c in param_cols:
                    mlflow.log_param(c.replace("param_", ""), row[c])
                mlflow.log_metric("cv_f1_mean", float(row["mean_test_score"]))
                mlflow.log_metric("cv_f1_std", float(row["std_test_score"]))
                mlflow.log_metric("rank", int(row["rank_test_score"]))
        print(f"  logged {len(results)} trials to MLflow")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in best_params.items()},
        "best_cv_f1": float(search.best_score_),
        "n_trials": int(n_iter),
        "cv": "GroupKFold(4) on patient_id",
        "scoring": "f1 (no-show class)",
    }
    (MODELS_DIR / "best_params.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("  saved models/best_params.json")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for XGBoost.")
    parser.add_argument("--n-iter", type=int, default=40, help="trials (default 40)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run_search(n_iter=args.n_iter, log_to_mlflow=not args.no_mlflow)
