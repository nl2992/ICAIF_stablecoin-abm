"""Tests for analysis metrics."""

import numpy as np
import pandas as pd
from stablesim.analysis.metrics import compute_ou_half_life, compute_metrics


def test_ou_half_life_ar1():
    """AR(1) with known kappa should give correct half-life."""
    rng = np.random.default_rng(0)
    kappa = 0.10  # half-life = ln(2)/0.1 ≈ 6.93 steps
    x = np.zeros(500)
    for t in range(1, 500):
        x[t] = x[t - 1] - kappa * x[t - 1] + rng.normal(0, 0.01)
    hl = compute_ou_half_life(x)
    assert 4.0 < hl < 12.0, f"Expected ~6.9, got {hl:.2f}"


def test_ou_half_life_nonstationary():
    # A random walk should produce a longer half-life than a stationary AR(1).
    rng = np.random.default_rng(1)
    x_rw = np.cumsum(rng.standard_normal(500))
    x_stat = np.zeros(500)
    for t in range(1, 500):
        x_stat[t] = x_stat[t - 1] * 0.80 + rng.normal(0, 0.01)
    hl_rw = compute_ou_half_life(x_rw)
    hl_stat = compute_ou_half_life(x_stat)
    assert hl_rw > hl_stat


def test_compute_metrics_runs():
    df = pd.DataFrame({
        "step": range(50),
        "mid_price": 1.0 + 0.01 * np.sin(np.linspace(0, 4, 50)),
        "depeg": 0.01 * np.sin(np.linspace(0, 4, 50)),
        "reserve_ratio": np.ones(50),
        "queue_depth": np.zeros(50),
    })
    metrics = compute_metrics(df, agents=[])
    assert "contagion_magnitude" in metrics
    assert "peg_recovery_half_life" in metrics
    assert metrics["contagion_magnitude"] > 0
