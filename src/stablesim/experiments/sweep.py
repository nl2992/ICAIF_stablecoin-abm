"""Sweep runner: scenarios × interventions → results table.

Usage:
    python -m stablesim.experiments.sweep --config configs/interventions.yaml
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..scenarios.loader import load_stressbench_scenarios
from .interventions import DEFAULT_SWEEP, InterventionConfig
from .runner import run_episode


def run_sweep(
    scenarios=None,
    interventions=None,
    n_steps: int = 100,
    n_seeds: int = 5,
    output_path: str = "experiments/results/sweep_results.csv",
) -> pd.DataFrame:
    """Run all scenario × intervention × seed combinations.

    Returns
    -------
    pd.DataFrame with one row per (scenario, intervention, seed, metric).
    """
    if scenarios is None:
        scenarios = load_stressbench_scenarios()
    if interventions is None:
        interventions = DEFAULT_SWEEP

    rows = []
    combos = list(product(scenarios, interventions, range(n_seeds)))
    for scenario, intervention, seed in tqdm(combos, desc="sweep"):
        result = run_episode(scenario, intervention, n_steps=n_steps, rng_seed=seed)
        row = {
            "scenario": result["scenario"],
            "intervention": result["intervention"],
            "seed": seed,
            **result["metrics"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved sweep results to {output_path}  ({len(df)} rows)")
    return df


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/interventions.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_sweep(
        n_steps=cfg.get("n_steps", 100),
        n_seeds=cfg.get("n_seeds", 5),
        output_path=cfg.get("output_path", "experiments/results/sweep_results.csv"),
    )


if __name__ == "__main__":
    _cli()
