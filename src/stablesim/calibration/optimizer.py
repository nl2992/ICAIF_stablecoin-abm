"""Calibration optimizer.

Tunes market/agent parameters so that no-shock baseline runs reproduce the
empirical OU half-life and price volatility from stablecoin-contagion-network.

Uses scipy.optimize.differential_evolution (global optimizer; no gradients needed).

Usage:
    python -m stablesim.calibration.optimizer --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from ..analysis.metrics import compute_ou_half_life
from ..experiments.runner import run_episode
from ..scenarios.schedule import ShockSchedule
from ..experiments.interventions import BASELINE
from .targets import EmpiricalTargets


def _simulate_moments(params: np.ndarray, n_steps: int = 200, n_seeds: int = 10) -> dict:
    """Run no-shock baseline with given params and return simulated moments.

    params = [reserve_speed, reserve_vol, arb_min_spread, noise_trade_prob]
    """
    reserve_speed, reserve_vol, arb_min_spread, noise_trade_prob = params

    half_lives = []
    vols = []
    for seed in range(n_seeds):
        result = run_episode(
            scenario=ShockSchedule(name="baseline"),
            intervention=BASELINE,
            n_steps=n_steps,
            rng_seed=seed,
        )
        df = result["history"]
        prices = df["mid_price"].values
        vols.append(float(np.std(np.diff(prices))))
        hl = compute_ou_half_life(prices - 1.0)
        half_lives.append(hl)

    return {
        "ou_half_life": float(np.median(half_lives)),
        "baseline_price_vol": float(np.median(vols)),
    }


def calibrate(
    targets: EmpiricalTargets | None = None,
    n_steps: int = 200,
    n_seeds: int = 10,
    maxiter: int = 50,
    popsize: int = 8,
) -> dict[str, float]:
    """Run differential evolution to find market params matching empirical targets.

    Returns best-fit parameter dict.
    """
    if targets is None:
        targets = EmpiricalTargets.from_contagion_results()

    bounds = [
        (0.01, 0.30),   # reserve_speed (OU kappa)
        (0.005, 0.05),  # reserve_vol
        (0.0005, 0.01), # arb_min_spread
        (0.10, 0.60),   # noise_trade_prob
    ]

    def objective(params):
        moments = _simulate_moments(params, n_steps, n_seeds)
        return targets.loss(moments)

    result = differential_evolution(
        objective, bounds, maxiter=maxiter, popsize=popsize,
        tol=1e-4, seed=42, disp=True,
    )

    best = result.x
    param_names = ["reserve_speed", "reserve_vol", "arb_min_spread", "noise_trade_prob"]
    best_params = dict(zip(param_names, best.tolist()))
    print(f"Calibration complete. Best params: {best_params}")
    print(f"Final loss: {result.fun:.6f}")
    return best_params


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    calibrate(
        n_steps=cfg.get("calibration", {}).get("n_steps", 200),
        n_seeds=cfg.get("calibration", {}).get("n_seeds", 10),
    )


if __name__ == "__main__":
    _cli()
