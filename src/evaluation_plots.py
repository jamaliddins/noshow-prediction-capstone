"""Result figures produced after training, from held-out test predictions.

Separate from visualize.py (which is pre-modelling EDA) because these need a
trained model. Run after src.train.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TARGET
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


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved reports/figures/{name}.png")


def plot_confusion(y_true, y_prob, threshold: float) -> None:
    """Raw counts beside row-normalised rates."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Attended", "No-show"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format=","
    )
    axes[0].set_title(f"Counts (threshold = {threshold:.3f})")
    axes[0].grid(False)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".1%", cmap="Blues", cbar=False,
        xticklabels=labels, yticklabels=labels, ax=axes[1],
        annot_kws={"size": 15, "weight": "bold"}, vmin=0, vmax=1,
    )
    axes[1].set_title("Row-normalised (recall per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    tn, fp, fn, tp = cm.ravel()
    fig.suptitle(
        f"Confusion matrix — caught {tp:,} of {tp + fn:,} no-shows "
        f"({tp / (tp + fn):.1%}), flagged {tp + fp:,} appointments",
        fontsize=14, fontweight="bold", y=1.02,
    )
    _save(fig, "08_confusion_matrix")


def plot_curves(y_true, y_prob, threshold: float, metrics: dict) -> None:
    """PR curve (primary under imbalance) and ROC side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()

    ax = axes[0]
    ax.plot(recall, precision, color=NOSHOW_COLOR, lw=2.5,
            label=f"Model (PR-AUC = {metrics['pr_auc']:.3f})")
    ax.axhline(baseline, ls="--", color="grey", lw=1.5,
               label=f"Random ({baseline:.3f})")
    idx = np.argmin(np.abs(pr_thresholds - threshold))
    ax.plot(recall[idx], precision[idx], "o", ms=12, color=ACCENT, zorder=5,
            label=f"Operating point (t={threshold:.2f})")
    ax.set_xlabel("Recall — share of no-shows caught")
    ax.set_ylabel("Precision — share of flags that were right")
    ax.set_title("Precision-Recall (the honest curve under imbalance)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, color=SHOW_COLOR, lw=2.5,
            label=f"Model (ROC-AUC = {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1.5, label="Random (0.500)")
    idx = np.argmin(np.abs(roc_thresholds - threshold))
    ax.plot(fpr[idx], tpr[idx], "o", ms=12, color=ACCENT, zorder=5,
            label=f"Operating point (t={threshold:.2f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (optimistic when classes are imbalanced)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Test-set performance", fontsize=15, fontweight="bold", y=1.02)
    _save(fig, "09_pr_roc_curves")


def plot_threshold_tuning(y_true, y_prob, chosen: float) -> None:
    """Why 0.5 was not assumed."""
    thresholds = np.linspace(0.02, 0.95, 200)
    rows = [compute_metrics(y_true, y_prob, t) for t in thresholds]
    f1 = [r["f1"] for r in rows]
    precision = [r["precision"] for r in rows]
    recall = [r["recall"] for r in rows]

    best_idx = int(np.argmax(f1))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, f1, lw=3, color=NOSHOW_COLOR, label="F1 (primary)")
    ax.plot(thresholds, precision, lw=2, ls="--", color=SHOW_COLOR, label="Precision")
    ax.plot(thresholds, recall, lw=2, ls="--", color=ACCENT, label="Recall")

    ax.axvline(thresholds[best_idx], color=NOSHOW_COLOR, ls=":", lw=2)
    ax.plot(thresholds[best_idx], f1[best_idx], "o", ms=13,
            color=NOSHOW_COLOR, zorder=5)
    ax.annotate(
        f"Tuned t = {thresholds[best_idx]:.3f}\nF1 = {f1[best_idx]:.3f}",
        xy=(thresholds[best_idx], f1[best_idx]),
        xytext=(thresholds[best_idx] + 0.13, f1[best_idx] + 0.10),
        fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3CD", ec="#E0C97F"),
    )

    f1_at_half = compute_metrics(y_true, y_prob, 0.5)["f1"]
    ax.axvline(0.5, color="grey", ls=":", lw=2)
    ax.plot(0.5, f1_at_half, "s", ms=11, color="grey", zorder=5)
    ax.annotate(
        f"Default t = 0.5\nF1 = {f1_at_half:.3f}",
        xy=(0.5, f1_at_half), xytext=(0.60, f1_at_half + 0.13),
        fontsize=10, color="#555",
        arrowprops=dict(arrowstyle="->", lw=1.3, color="grey"),
        bbox=dict(boxstyle="round,pad=0.35", fc="#EEEEEE", ec="#BBBBBB"),
    )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold tuning: the default 0.5 is not optimal")
    ax.legend(fontsize=10, loc="center right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _save(fig, "10_threshold_tuning")


def plot_calibration(y_true, y_prob_raw, y_prob_cal) -> None:
    """Before/after isotonic calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1.5, label="Perfect calibration")
    for probs, color, name in ((y_prob_raw, NOSHOW_COLOR, "Before (raw XGBoost)"),
                               (y_prob_cal, ACCENT, "After (isotonic)")):
        true_frac, pred_frac = calibration_curve(y_true, probs, n_bins=10,
                                                 strategy="quantile")
        ax.plot(pred_frac, true_frac, "o-", lw=2.5, ms=8, color=color, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed no-show rate")
    ax.set_title("Reliability curve")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    bins = np.linspace(0, 1, 41)
    ax.hist(y_prob_raw, bins=bins, alpha=0.6, color=NOSHOW_COLOR,
            label=f"Before (mean {y_prob_raw.mean():.3f})")
    ax.hist(y_prob_cal, bins=bins, alpha=0.6, color=ACCENT,
            label=f"After (mean {y_prob_cal.mean():.3f})")
    ax.axvline(y_true.mean(), color="black", ls="--", lw=2,
               label=f"Actual rate ({y_true.mean():.3f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Appointments")
    ax.set_title("Predicted probability distribution")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Calibration: raw probabilities were inflated by class weighting",
        fontsize=15, fontweight="bold", y=1.02,
    )
    _save(fig, "11_calibration")


def plot_feature_importance(model, X_test, y_test, top_n: int = 15) -> None:
    """Permutation importance — model-agnostic and measured on test data."""
    print("  computing permutation importance (this takes a minute)...")
    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE,
        scoring="average_precision", n_jobs=-1,
    )
    order = result.importances_mean.argsort()[::-1][:top_n]
    names = [X_test.columns[i] for i in order]
    means = result.importances_mean[order]
    stds = result.importances_std[order]

    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = np.arange(len(names))[::-1]
    colors = plt.cm.Reds(np.linspace(0.85, 0.35, len(names)))
    ax.barh(y_pos, means, xerr=stds, color=colors, error_kw={"lw": 1.2})
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=10)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title(f"Top {top_n} features by permutation importance (test set)")
    for y, v in zip(y_pos, means):
        ax.text(v + max(means) * 0.015, y, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, max(means + stds) * 1.18)
    _save(fig, "12_feature_importance")
    return list(zip(names, means))


def plot_risk_tiers(y_true, y_prob, tiers: dict) -> None:
    """The operational payoff: what each tier means for staff."""
    labels = np.where(
        y_prob < tiers["low_max"], "Low",
        np.where(y_prob < tiers["medium_max"], "Medium", "High"),
    )
    order = ["Low", "Medium", "High"]
    colors = {"Low": "#52B788", "Medium": "#F4A261", "High": "#D1495B"}

    share = [np.mean(labels == t) for t in order]
    rate = [y_true[labels == t].mean() if (labels == t).sum() else 0 for t in order]
    counts = [int((labels == t).sum()) for t in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax = axes[0]
    bars = ax.bar(order, [s * 100 for s in share],
                  color=[colors[t] for t in order], width=0.6)
    for bar, s, n in zip(bars, share, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, s * 100 + 1,
                f"{s:.1%}\n({n:,})", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Share of appointments (%)")
    ax.set_title("How the workload splits")
    ax.set_ylim(0, max(share) * 100 * 1.25)

    ax = axes[1]
    bars = ax.bar(order, [r * 100 for r in rate],
                  color=[colors[t] for t in order], width=0.6)
    for bar, r in zip(bars, rate):
        ax.text(bar.get_x() + bar.get_width() / 2, r * 100 + 1,
                f"{r:.1%}", ha="center", fontweight="bold", fontsize=13)
    ax.axhline(y_true.mean() * 100, ls="--", color="#333", lw=1.5,
               label=f"Overall {y_true.mean():.1%}")
    ax.set_ylabel("Actual no-show rate (%)")
    ax.set_title("Actual no-show rate per tier")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(rate) * 100 * 1.25)

    lift = rate[2] / rate[0] if rate[0] else float("nan")
    fig.suptitle(
        f"Risk tiers on the test set — High-risk patients miss "
        f"{lift:.1f}x more often than Low-risk",
        fontsize=15, fontweight="bold", y=1.02,
    )
    _save(fig, "13_risk_tiers")


def generate_all() -> None:
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("No trained model. Run: python -m src.train")

    print("Loading model and test split...")
    model = joblib.load(model_path)
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    threshold = metadata["threshold"]

    df = build_features(clean(load_raw(), verbose=False))
    _, _, test = split_by_patient(df)
    numeric, categorical = get_feature_columns(df)
    features = numeric + categorical

    X_test, y_test = test[features], test[TARGET].to_numpy()
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob, threshold)

    print(f"  test F1={metrics['f1']:.4f}  PR-AUC={metrics['pr_auc']:.4f}\n")
    print("Generating figures...")

    plot_confusion(y_test, y_prob, threshold)
    plot_curves(y_test, y_prob, threshold, metrics)
    plot_threshold_tuning(y_test, y_prob, threshold)

    # The uncalibrated pipeline sits inside the calibrated wrapper.
    raw_pipeline = model.calibrated_classifiers_[0].estimator
    y_prob_raw = raw_pipeline.predict_proba(X_test)[:, 1]
    plot_calibration(y_test, y_prob_raw, y_prob)

    plot_feature_importance(model, X_test, y_test)
    plot_risk_tiers(y_test, y_prob, metadata["risk_tiers"])

    print(f"\nAll evaluation figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all()
