"""Metrics and threshold selection.

Kept separate from training so the same scoring code runs during experiments
and in the final held-out test evaluation.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Score predictions at a given decision threshold.

    F1 on the no-show (positive) class is primary; the rest are reported
    alongside because F1 alone hides the precision/recall trade-off.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        # Threshold-independent, so safe to compare across models.
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
        # Reported for completeness only — never used to choose a model.
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_best_threshold(
    y_true, y_prob, metric: str = "f1", n_steps: int = 200
) -> tuple[float, float]:
    """Sweep thresholds and return the (threshold, score) maximising `metric`.

    Must only ever be called on validation data — tuning on test would leak.
    """
    lo, hi = np.percentile(y_prob, [1, 99])
    candidates = np.linspace(max(lo, 0.01), min(hi, 0.99), n_steps)

    best_t, best_score = 0.5, -np.inf
    for t in candidates:
        score = compute_metrics(y_true, y_prob, t)[metric]
        if score > best_score:
            best_t, best_score = float(t), float(score)
    return best_t, best_score


def format_metrics(name: str, m: dict[str, float]) -> str:
    return (
        f"{name:<26} F1={m['f1']:.4f}  P={m['precision']:.4f}  "
        f"R={m['recall']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
        f"ROC-AUC={m['roc_auc']:.4f}  @t={m['threshold']:.3f}"
    )


def confusion_summary(m: dict[str, float]) -> str:
    """Plain-language read of the confusion matrix, in clinic terms."""
    caught = m["tp"] / max(m["tp"] + m["fn"], 1)
    workload = m["tp"] + m["fp"]
    precision = m["tp"] / max(workload, 1)
    return (
        f"    Of {m['tp'] + m['fn']:,} actual no-shows, caught {m['tp']:,} ({caught:.1%}).\n"
        f"    Flagging {workload:,} appointments; {precision:.1%} of those were real no-shows."
    )
