"""Standard plots for the paper."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_depeg(history: pd.DataFrame, title: str = "", ax=None) -> plt.Axes:
    """Plot depeg time series with shock markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.plot(history["step"], history["depeg"], color="steelblue", linewidth=1.2)
    ax.fill_between(history["step"], history["depeg"], 0, alpha=0.15, color="steelblue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Depeg (price − 1)")
    if title:
        ax.set_title(title)
    return ax


def plot_welfare(welfare_dict: dict[str, float], title: str = "", ax=None) -> plt.Axes:
    """Bar chart of cumulative P&L by agent type."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    labels = list(welfare_dict.keys())
    values = list(welfare_dict.values())
    colors = ["green" if v >= 0 else "red" for v in values]
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative P&L (USD)")
    if title:
        ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return ax


def plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    metric: str = "peg_recovery_half_life",
    scenarios: list[str] | None = None,
    ax=None,
) -> plt.Axes:
    """Heatmap of metric values across interventions × scenarios."""
    import seaborn as sns

    pivot = sweep_df.groupby(["intervention", "scenario"])[metric].mean().unstack("scenario")
    if scenarios:
        pivot = pivot[[s for s in scenarios if s in pivot.columns]]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{metric} by intervention × scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Intervention")
    plt.tight_layout()
    return ax
