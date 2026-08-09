"""Exploratory figures for the no-show dataset.

Every figure answers one question a reviewer is likely to ask, and each is
saved to reports/figures/ at presentation resolution.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TARGET
from src.preprocessing import build_features, clean, load_raw

# One consistent visual language across every figure.
SHOW_COLOR = "#4C8CBF"
NOSHOW_COLOR = "#D1495B"
PALETTE = [SHOW_COLOR, NOSHOW_COLOR]

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "font.size": 11,
})


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(FIGURES_DIR.parents[1])}")


def _rate_labels(ax, rates, fmt="{:.1%}", pad=0.4):
    """Annotate bars with their no-show rate."""
    for patch, rate in zip(ax.patches, rates):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + pad,
            fmt.format(rate),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )


def plot_class_balance(df: pd.DataFrame) -> None:
    """The imbalance that justifies F1 over accuracy."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Attended", "No-show"]
    bars = ax.bar(labels, counts.values, color=PALETTE, width=0.6)

    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + total * 0.012,
            f"{value:,}\n({value/total:.1%})",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )

    majority = counts.max() / total
    ax.set_title("Class imbalance: ~1 in 5 appointments is missed")
    ax.set_ylabel("Appointments")
    # Headroom above the tallest bar so the callout never overlaps its label.
    ax.set_ylim(0, counts.max() * 1.42)
    ax.text(
        0.5, 0.99,
        f"Always predicting “Attended” scores {majority:.1%} accuracy\n"
        f"while catching zero no-shows — which is why F1 on the\n"
        f"no-show class is the primary metric.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
        style="italic", color="#444",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3CD", ec="#E0C97F", alpha=0.95),
    )
    _save(fig, "01_class_balance")


def plot_lead_time(df: pd.DataFrame) -> None:
    """Lead time is the strongest single signal in the data."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    bins = [-0.5, 0.5, 1.5, 3.5, 7.5, 14.5, 30.5, np.inf]
    labels = ["Same\nday", "1 day", "2-3", "4-7", "8-14", "15-30", "31+"]
    binned = pd.cut(df["lead_time_days"], bins=bins, labels=labels)
    rates = df.groupby(binned, observed=True)[TARGET].mean()
    volume = df.groupby(binned, observed=True).size()

    ax = axes[0]
    colors = plt.cm.Reds(np.linspace(0.35, 0.85, len(rates)))
    ax.bar(rates.index.astype(str), rates.values * 100, color=colors, width=0.7)
    overall = df[TARGET].mean() * 100
    ax.axhline(overall, ls="--", color="#333", lw=1.5,
               label=f"Overall {overall:.1f}%")
    for i, v in enumerate(rates.values * 100):
        ax.text(i, v + 0.6, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("No-show rate rises sharply with waiting time")
    ax.set_xlabel("Days between booking and appointment")
    ax.set_ylabel("No-show rate (%)")
    ax.set_ylim(0, max(rates.values * 100) * 1.2)
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.bar(volume.index.astype(str), volume.values, color=SHOW_COLOR, width=0.7)
    ax.set_title("Most appointments are booked same-day")
    ax.set_xlabel("Days between booking and appointment")
    ax.set_ylabel("Appointments")
    for i, v in enumerate(volume.values):
        ax.text(i, v + volume.max() * 0.015, f"{v/1000:.0f}k",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, volume.max() * 1.15)

    fig.suptitle("Lead time is the strongest single predictor",
                 fontsize=16, fontweight="bold", y=1.02)
    _save(fig, "02_lead_time")


def plot_demographics(df: pd.DataFrame) -> None:
    """Age and the health/welfare flags."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    order = ["infant", "child", "teen", "young_adult", "adult", "senior", "elderly"]
    order = [g for g in order if g in df["age_group"].unique()]
    rates = df.groupby("age_group", observed=True)[TARGET].mean().reindex(order)
    ax.bar(range(len(rates)), rates.values * 100,
           color=plt.cm.viridis(np.linspace(0.2, 0.8, len(rates))), width=0.7)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([g.replace("_", "\n") for g in rates.index], fontsize=10)
    ax.axhline(df[TARGET].mean() * 100, ls="--", color="#333", lw=1.5)
    for i, v in enumerate(rates.values * 100):
        ax.text(i, v + 0.4, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("Teens and young adults miss most")
    ax.set_ylabel("No-show rate (%)")
    ax.set_ylim(0, max(rates.values * 100) * 1.2)

    ax = axes[1]
    flags = {
        "Scholarship\n(welfare)": "scholarship",
        "Hypertension": "hypertension",
        "Diabetes": "diabetes",
        "Alcoholism": "alcoholism",
        "Handicap": "has_handicap",
        "SMS received": "sms_received",
    }
    with_flag = [df.loc[df[c] == 1, TARGET].mean() * 100 for c in flags.values()]
    without = [df.loc[df[c] == 0, TARGET].mean() * 100 for c in flags.values()]

    x = np.arange(len(flags))
    ax.bar(x - 0.2, without, 0.4, label="No", color=SHOW_COLOR)
    ax.bar(x + 0.2, with_flag, 0.4, label="Yes", color=NOSHOW_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(flags.keys(), fontsize=9)
    ax.set_title("No-show rate by patient flag")
    ax.set_ylabel("No-show rate (%)")
    ax.legend(fontsize=10)
    _save(fig, "03_demographics")


def plot_sms_paradox(df: pd.DataFrame) -> None:
    """SMS looks harmful until you condition on lead time — a confounder."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    rates = df.groupby("sms_received")[TARGET].mean() * 100
    bars = ax.bar(["No SMS", "SMS sent"], rates.values,
                  color=[SHOW_COLOR, NOSHOW_COLOR], width=0.55)
    for bar, v in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.1f}%",
                ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Apparent paradox:\npatients who got an SMS miss MORE")
    ax.set_ylabel("No-show rate (%)")
    ax.set_ylim(0, max(rates.values) * 1.25)

    ax = axes[1]
    bins = [-0.5, 0.5, 3.5, 7.5, 14.5, 30.5, np.inf]
    labels = ["Same day", "1-3", "4-7", "8-14", "15-30", "31+"]
    binned = pd.cut(df["lead_time_days"], bins=bins, labels=labels)
    grouped = df.groupby([binned, "sms_received"], observed=True)[TARGET].mean() * 100
    grouped = grouped.unstack()

    x = np.arange(len(grouped))
    for offset, col, color, name in ((-0.2, 0, SHOW_COLOR, "No SMS"),
                                     (0.2, 1, NOSHOW_COLOR, "SMS sent")):
        if col in grouped.columns:
            ax.bar(x + offset, grouped[col].values, 0.4, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index.astype(str), fontsize=10)
    ax.set_title("Resolved: SMS is only sent for\nlonger waits, which drive no-shows")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("No-show rate (%)")
    ax.legend(fontsize=10)

    fig.suptitle("Confounding: SMS does not cause no-shows — lead time does",
                 fontsize=16, fontweight="bold", y=1.03)
    _save(fig, "04_sms_confounding")


def plot_patient_history(df: pd.DataFrame) -> None:
    """Prior behaviour is the second-strongest signal."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    first = df[df["is_first_appointment"] == 1][TARGET].mean() * 100
    returning = df[df["is_first_appointment"] == 0][TARGET].mean() * 100
    bars = ax.bar(["First-ever\nappointment", "Has prior\nhistory"],
                  [first, returning], color=[SHOW_COLOR, NOSHOW_COLOR], width=0.55)
    for bar, v in zip(bars, [first, returning]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%",
                ha="center", fontweight="bold", fontsize=12)
    ax.set_title("First-timers vs returning patients")
    ax.set_ylabel("No-show rate (%)")
    ax.set_ylim(0, max(first, returning) * 1.25)

    ax = axes[1]
    hist = df[df["prior_appointments"] > 0].copy()
    bins = [-0.01, 0.001, 0.25, 0.5, 0.75, 1.01]
    labels = ["0%\n(never missed)", "1-25%", "26-50%", "51-75%", "76-100%"]
    binned = pd.cut(hist["prior_noshow_rate"], bins=bins, labels=labels)
    rates = hist.groupby(binned, observed=True)[TARGET].mean() * 100

    ax.bar(range(len(rates)), rates.values,
           color=plt.cm.Reds(np.linspace(0.3, 0.9, len(rates))), width=0.7)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(rates.index, fontsize=9)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("Past behaviour strongly predicts future behaviour")
    ax.set_xlabel("Patient's prior no-show rate")
    ax.set_ylabel("No-show rate (%)")
    ax.set_ylim(0, max(rates.values) * 1.2)

    fig.suptitle("Patient history (computed from prior appointments only)",
                 fontsize=16, fontweight="bold", y=1.03)
    _save(fig, "05_patient_history")


def plot_correlation(df: pd.DataFrame) -> None:
    """Correlation heatmap of the numeric feature set."""
    cols = [
        "age", "lead_time_days", "is_same_day", "scholarship", "hypertension",
        "diabetes", "alcoholism", "has_handicap", "sms_received",
        "prior_appointments", "prior_noshow_rate", "is_first_appointment",
        TARGET,
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        annot_kws={"size": 8}, ax=ax, vmin=-1, vmax=1,
    )
    ax.set_title("Feature correlations\n(no single feature dominates the target)",
                 pad=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    _save(fig, "06_correlation_heatmap")


def plot_weekday_and_neighbourhood(df: pd.DataFrame) -> None:
    """Temporal and geographic variation."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rates = df.groupby("appt_dayofweek")[TARGET].mean() * 100
    counts = df.groupby("appt_dayofweek").size()
    present = [(d, rates.get(i, np.nan), counts.get(i, 0))
               for i, d in enumerate(days) if i in rates.index]

    # Days with too little volume for a stable rate are greyed out, not hidden.
    MIN_N = 1000
    palette = plt.cm.Blues(np.linspace(0.4, 0.85, len(present)))
    colors = [c if p[2] >= MIN_N else "#BFBFBF" for c, p in zip(palette, present)]
    ax.bar([p[0] for p in present], [p[1] for p in present],
           color=colors, width=0.65)
    ax.axhline(df[TARGET].mean() * 100, ls="--", color="#333", lw=1.5)
    for i, p in enumerate(present):
        ax.text(i, p[1] + 0.3, f"{p[1]:.0f}%", ha="center",
                fontweight="bold", fontsize=10,
                color="#777" if p[2] < MIN_N else "black")
        label = f"n={p[2]:,}"
        inside = p[2] >= MIN_N
        ax.text(i, 0.6 if inside else p[1] + 1.7, label, ha="center",
                fontsize=7.5, fontweight="bold",
                color="white" if inside else "#777")
    low = [p[0] for p in present if p[2] < MIN_N]
    note = f"  (grey = n<{MIN_N:,}, too few to read into)" if low else ""
    ax.set_title(f"No-show rate by appointment weekday{note}", fontsize=13)
    ax.set_ylabel("No-show rate (%)")
    ax.set_ylim(0, max(p[1] for p in present) * 1.25)

    ax = axes[1]
    # Restrict to neighbourhoods with enough volume for a stable rate.
    stats = df.groupby("neighbourhood").agg(
        rate=(TARGET, "mean"), n=(TARGET, "size")
    )
    stats = stats[stats["n"] >= 300].sort_values("rate", ascending=False)
    top = pd.concat([stats.head(8), stats.tail(8)])
    colors = [NOSHOW_COLOR] * 8 + [SHOW_COLOR] * 8
    ax.barh(range(len(top)), top["rate"].values * 100, color=colors[:len(top)])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([n[:18].title() for n in top.index], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(df[TARGET].mean() * 100, ls="--", color="#333", lw=1.5)
    ax.set_title("Highest vs lowest neighbourhoods\n(≥300 appointments)")
    ax.set_xlabel("No-show rate (%)")

    fig.suptitle("Geographic variation motivates the fairness check",
                 fontsize=16, fontweight="bold", y=1.03)
    _save(fig, "07_weekday_neighbourhood")


def generate_all() -> pd.DataFrame:
    print("Loading and preparing data...")
    df = build_features(clean(load_raw(), verbose=False))
    print(f"  {len(df):,} appointments, {df[TARGET].mean():.2%} no-show\n")

    print("Generating figures...")
    plot_class_balance(df)
    plot_lead_time(df)
    plot_demographics(df)
    plot_sms_paradox(df)
    plot_patient_history(df)
    plot_correlation(df)
    plot_weekday_and_neighbourhood(df)
    print(f"\nAll figures written to {FIGURES_DIR}")
    return df


if __name__ == "__main__":
    generate_all()
