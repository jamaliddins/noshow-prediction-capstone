"""Error analysis: where the model fails, and why.

Aggregate metrics say how well the model does overall; this module asks *which
appointments it gets wrong*. It slices the held-out test set by the features that
matter operationally, separates the two error types, and checks whether the
mistakes are concentrated in identifiable groups — including the socioeconomic
groups flagged as a fairness risk in the README.

    python -m src.error_analysis

Writes reports/figures/14_error_analysis.png and prints a text summary.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, MODELS_DIR, TARGET
from src.evaluate import compute_metrics
from src.preprocessing import (
    build_features,
    clean,
    get_feature_columns,
    load_raw,
    split_by_patient,
)

SHOW_COLOR = "#4C8CBF"
NOSHOW_COLOR = "#D1495B"
ACCENT = "#2A9D8F"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 14,
    "axes.labelsize": 12, "font.size": 11,
})

# Slices worth reporting: each is a question a reviewer would actually ask.
LEAD_TIME_BINS = [-1, 0, 3, 7, 14, 30, 10_000]
LEAD_TIME_LABELS = ["Same day", "1-3", "4-7", "8-14", "15-30", "31+"]


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  saved reports/figures/{name}.png")


def build_error_frame(test: pd.DataFrame, y_prob: np.ndarray,
                      threshold: float) -> pd.DataFrame:
    """Attach predictions and an error-type label to each test appointment."""
    df = test.copy()
    df["y_prob"] = y_prob
    df["y_pred"] = (y_prob >= threshold).astype(int)

    actual, pred = df[TARGET] == 1, df["y_pred"] == 1
    df["error_type"] = np.select(
        [actual & pred, ~actual & ~pred, ~actual & pred, actual & ~pred],
        ["TP", "TN", "FP (false alarm)", "FN (missed no-show)"],
        default="?",
    )
    df["correct"] = df[TARGET] == df["y_pred"]
    df["lead_bin"] = pd.cut(df["lead_time_days"], bins=LEAD_TIME_BINS,
                            labels=LEAD_TIME_LABELS)
    return df


def slice_metrics(df: pd.DataFrame, by: str, threshold: float,
                  min_n: int = 100) -> pd.DataFrame:
    """Per-group recall, precision and F1 on the no-show class.

    Groups smaller than `min_n` are dropped: with a 20% positive rate their
    rates are too noisy to interpret, and reporting them invites false stories.
    """
    rows = []
    for value, group in df.groupby(by, observed=True):
        if len(group) < min_n:
            continue
        m = compute_metrics(group[TARGET].to_numpy(),
                            group["y_prob"].to_numpy(), threshold)
        rows.append({
            by: str(value),
            "n": len(group),
            "actual_rate": group[TARGET].mean(),
            "flagged_rate": group["y_pred"].mean(),
            "recall": m["recall"],
            "precision": m["precision"],
            "f1": m["f1"],
        })
    return pd.DataFrame(rows)


def plot_error_analysis(df: pd.DataFrame, threshold: float) -> None:
    """Four panels: where errors concentrate, and how confident they are."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 1. Recall and precision by lead time — the dominant feature.
    ax = axes[0, 0]
    by_lead = slice_metrics(df, "lead_bin", threshold)
    x = np.arange(len(by_lead))
    ax.bar(x - 0.2, by_lead["recall"] * 100, 0.4, label="Recall",
           color=NOSHOW_COLOR)
    ax.bar(x + 0.2, by_lead["precision"] * 100, 0.4, label="Precision",
           color=SHOW_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(by_lead["lead_bin"])
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("%")
    ax.set_title("Same-day appointments are the blind spot")
    ax.legend(fontsize=10)

    # 2. Where the two error types actually live.
    ax = axes[0, 1]
    err = df[df["error_type"].str.startswith(("FP", "FN"))]
    counts = (err.groupby(["lead_bin", "error_type"], observed=True)
                 .size().unstack(fill_value=0))
    counts.plot(kind="bar", stacked=True, ax=ax,
                color={"FP (false alarm)": SHOW_COLOR,
                       "FN (missed no-show)": NOSHOW_COLOR}, width=0.7)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Appointments")
    ax.set_title("Error volume by lead time")
    ax.legend(fontsize=9, title=None)
    ax.tick_params(axis="x", rotation=0)

    # 3. Confidence of the mistakes: are they near the threshold or confident?
    ax = axes[1, 0]
    for label, color in (("FN (missed no-show)", NOSHOW_COLOR),
                         ("FP (false alarm)", SHOW_COLOR)):
        subset = df.loc[df["error_type"] == label, "y_prob"]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=label)
    ax.axvline(threshold, ls="--", lw=2, color="#333",
               label=f"threshold {threshold:.3f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Most errors sit just either side of the threshold")
    ax.legend(fontsize=9)

    # 4. Fairness check across the socioeconomic slice.
    ax = axes[1, 1]
    by_sch = slice_metrics(df, "scholarship", threshold)
    by_sch["label"] = by_sch["scholarship"].map({"0": "No welfare", "1": "Welfare"})
    x = np.arange(len(by_sch))
    ax.bar(x - 0.2, by_sch["recall"] * 100, 0.4, label="Recall",
           color=NOSHOW_COLOR)
    ax.bar(x + 0.2, by_sch["precision"] * 100, 0.4, label="Precision",
           color=SHOW_COLOR)
    for i, row in by_sch.reset_index(drop=True).iterrows():
        ax.text(i, 3, f"flagged {row['flagged_rate']:.0%}", ha="center",
                fontsize=10, color="#222", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(by_sch["label"])
    ax.set_ylabel("%")
    ax.set_title("Fairness check: Scholarship (welfare) subgroup")
    ax.legend(fontsize=10)

    fig.suptitle("Error analysis — where the model fails on the test set",
                 fontsize=17, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "14_error_analysis")


def plot_error_analysis_slide(df: pd.DataFrame, threshold: float) -> None:
    """Two-panel version for the defense deck.

    The four-panel figure is right for the report but unreadable projected;
    this keeps only the two findings that carry the slide.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    by_lead = slice_metrics(df, "lead_bin", threshold)
    x = np.arange(len(by_lead))
    ax.bar(x - 0.2, by_lead["recall"] * 100, 0.4, label="Recall",
           color=NOSHOW_COLOR)
    ax.bar(x + 0.2, by_lead["precision"] * 100, 0.4, label="Precision",
           color=SHOW_COLOR)
    for i, r in by_lead.reset_index(drop=True).iterrows():
        ax.text(i - 0.2, r["recall"] * 100 + 2, f"{r['recall']:.0%}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(by_lead["lead_bin"], fontsize=12)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("%")
    ax.set_title("Recall collapses on same-day bookings")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)

    ax = axes[1]
    for label, color in (("FN (missed no-show)", NOSHOW_COLOR),
                         ("FP (false alarm)", SHOW_COLOR)):
        ax.hist(df.loc[df["error_type"] == label, "y_prob"], bins=40,
                alpha=0.72, color=color, label=label)
    ax.axvline(threshold, ls="--", lw=2.5, color=ACCENT,
               label=f"threshold {threshold:.3f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Errors cluster at the threshold, not at the extremes")
    ax.legend(fontsize=11)

    fig.suptitle("Where the model fails — 16,751 held-out appointments",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "15_error_analysis_slide")


def print_summary(df: pd.DataFrame, threshold: float) -> None:
    """Text findings — the part that goes into the README and the defense."""
    n = len(df)
    fn = df[df["error_type"] == "FN (missed no-show)"]
    fp = df[df["error_type"] == "FP (false alarm)"]

    print("\n" + "=" * 68)
    print("ERROR ANALYSIS")
    print("=" * 68)
    print(f"Test appointments: {n:,}   threshold: {threshold:.4f}")
    print(f"  False negatives (missed no-shows): {len(fn):,} "
          f"({len(fn) / n:.1%} of all appointments)")
    print(f"  False positives (false alarms):    {len(fp):,} "
          f"({len(fp) / n:.1%})")

    # How close are the errors to the decision boundary?
    near = 0.05
    fn_near = (fn["y_prob"] >= threshold - near).mean() if len(fn) else 0.0
    fp_near = (fp["y_prob"] <= threshold + near).mean() if len(fp) else 0.0
    print(f"\nErrors within {near} of the threshold:")
    print(f"  {fn_near:.1%} of false negatives, {fp_near:.1%} of false alarms")
    print("  -> most mistakes are borderline, not confident blunders.")

    print("\nBy lead time:")
    by_lead = slice_metrics(df, "lead_bin", threshold)
    for _, r in by_lead.iterrows():
        print(f"  {r['lead_bin']:<9} n={r['n']:>6,}  actual={r['actual_rate']:.1%}"
              f"  recall={r['recall']:.1%}  precision={r['precision']:.1%}")

    worst = by_lead.loc[by_lead["recall"].idxmin()]
    print(f"  -> weakest recall: '{worst['lead_bin']}' at {worst['recall']:.1%}")

    print("\nBy age group:")
    by_age = slice_metrics(df, "age_group", threshold)
    for _, r in by_age.sort_values("recall").iterrows():
        print(f"  {r['age_group']:<13} n={r['n']:>6,}  actual={r['actual_rate']:.1%}"
              f"  recall={r['recall']:.1%}  precision={r['precision']:.1%}")

    print("\nBy patient history:")
    by_first = slice_metrics(df, "is_first_appointment", threshold)
    for _, r in by_first.iterrows():
        label = "First visit" if r["is_first_appointment"] == "1" else "Returning"
        print(f"  {label:<13} n={r['n']:>6,}  actual={r['actual_rate']:.1%}"
              f"  recall={r['recall']:.1%}  precision={r['precision']:.1%}")

    print("\nFairness check — Scholarship (welfare):")
    by_sch = slice_metrics(df, "scholarship", threshold)
    for _, r in by_sch.iterrows():
        label = "Welfare" if r["scholarship"] == "1" else "No welfare"
        print(f"  {label:<13} n={r['n']:>6,}  actual={r['actual_rate']:.1%}"
              f"  flagged={r['flagged_rate']:.1%}  recall={r['recall']:.1%}"
              f"  precision={r['precision']:.1%}")
    if len(by_sch) == 2:
        gap = abs(by_sch["flagged_rate"].iloc[0] - by_sch["flagged_rate"].iloc[1])
        print(f"  -> flag-rate gap between groups: {gap:.1%}")

    print("=" * 68 + "\n")


def generate_all() -> None:
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("No trained model. Run: python -m src.train")

    print("Loading model and test split...")
    model = joblib.load(model_path)
    metadata = json.loads(
        (MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    threshold = metadata["threshold"]

    df = build_features(clean(load_raw(), verbose=False))
    _, _, test = split_by_patient(df)
    numeric, categorical = get_feature_columns(df)

    y_prob = model.predict_proba(test[numeric + categorical])[:, 1]
    errors = build_error_frame(test, y_prob, threshold)

    # Slice columns are compared as strings so 0/1 flags print predictably.
    for col in ("scholarship", "is_first_appointment"):
        errors[col] = errors[col].astype(int).astype(str)

    print("Generating figures...")
    plot_error_analysis(errors, threshold)
    plot_error_analysis_slide(errors, threshold)
    print_summary(errors, threshold)


if __name__ == "__main__":
    generate_all()
