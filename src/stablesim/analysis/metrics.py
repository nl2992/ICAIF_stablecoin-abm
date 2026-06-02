"""Outcome metrics for intervention analysis.

Primary metrics (match the paper framing):
  - contagion_magnitude : peak cross-venue depeg spread during shock episode
  - peg_recovery_half_life : OU half-life of depeg process post-shock
  - lp_total_il : total impermanent loss across LP agents
  - welfare_by_type : dict of cumulative P&L by agent class
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ou_half_life(series: np.ndarray, dt: float = 1.0) -> float:
    """Estimate OU mean-reversion half-life via OLS on AR(1).

    For X_t = kappa*(theta - X_{t-1})*dt + sigma*dW, OLS gives:
        X_t - X_{t-1} = a + b*X_{t-1}  =>  kappa = -b/dt

    Returns ln(2)/kappa (half-life in steps).  Returns inf if non-stationary.
    """
    if len(series) < 10:
        return float("inf")
    x = series[:-1]
    dx = np.diff(series)
    X = np.column_stack([np.ones_like(x), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, dx, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    b = coeffs[1]
    kappa = -b / dt
    if kappa <= 0:
        return float("inf")
    return float(np.log(2) / kappa)


def compute_contagion_magnitude(df: pd.DataFrame) -> float:
    """Peak absolute depeg observed during the episode."""
    if "depeg" not in df.columns:
        return 0.0
    return float(df["depeg"].abs().max())


def compute_metrics(df: pd.DataFrame, agents: list) -> dict:
    """Compute all outcome metrics from a completed episode.

    Parameters
    ----------
    df : pd.DataFrame
        history_df from MultiVenueMarket.
    agents : list[BaseAgent]
        All agents that participated in the episode.
    """
    metrics: dict = {}

    # Peg metrics
    depeg = df["depeg"].values if "depeg" in df.columns else np.zeros(len(df))
    metrics["contagion_magnitude"] = float(np.abs(depeg).max())
    metrics["mean_abs_depeg"] = float(np.abs(depeg).mean())
    metrics["peg_recovery_half_life"] = compute_ou_half_life(depeg)

    # Reserve
    if "reserve_ratio" in df.columns:
        metrics["min_reserve_ratio"] = float(df["reserve_ratio"].min())
        metrics["mean_reserve_ratio"] = float(df["reserve_ratio"].mean())

    # Redemption queue
    if "queue_depth" in df.columns:
        metrics["max_queue_depth"] = float(df["queue_depth"].max())
        metrics["mean_queue_depth"] = float(df["queue_depth"].mean())

    # Welfare by agent type
    from ..agents.arbitrageur import Arbitrageur
    from ..agents.redeemer import Redeemer
    from ..agents.lp import LPAgent
    from ..agents.issuer import IssuerAgent
    from ..agents.noise import NoiseTrader

    type_map = {
        Arbitrageur: "arbitrageur",
        Redeemer: "redeemer",
        LPAgent: "lp",
        IssuerAgent: "issuer",
        NoiseTrader: "noise",
    }
    welfare: dict[str, float] = {}
    for agent in agents:
        key = type_map.get(type(agent), "other")
        welfare[key] = welfare.get(key, 0.0) + agent.cumulative_pnl

    for k, v in welfare.items():
        metrics[f"welfare_{k}"] = v

    metrics["total_welfare"] = sum(welfare.values())

    return metrics
