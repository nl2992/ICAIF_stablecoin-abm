"""Empirical calibration targets derived from stablecoin-contagion-network.

The ABM must reproduce these stylized facts in no-shock baseline runs before
intervention sweeps are trusted (same validation strategy as Gu et al.).

Values below are illustrative placeholders; populate from the actual IAQF
analysis results in stablecoin-contagion-network/results/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EmpiricalTargets:
    """Calibration targets from the contagion-network empirical analysis.

    Parameters
    ----------
    ou_half_life : float
        Target OU half-life (steps) for peg recovery: ln(2)/kappa.
    max_contagion_rho : float
        Maximum cross-venue correlation during stress (ρ̂ from the IAQF metrics).
    baseline_price_vol : float
        Expected price standard deviation in no-shock baseline.
    propagation_lag : float
        Median steps for shock to propagate from pool 0 to pool 1.
    """

    ou_half_life: float = 12.0
    max_contagion_rho: float = 0.75
    baseline_price_vol: float = 0.003
    propagation_lag: float = 2.0

    @classmethod
    def from_contagion_results(cls, results_dir: str | Path | None = None) -> "EmpiricalTargets":
        """Load targets from stablecoin-contagion-network results if available."""
        root = Path(results_dir or Path(__file__).parents[6] / "stablecoin-contagion-network" / "results")
        summary = root / "calibration_targets.json"
        if summary.exists():
            import json
            with open(summary) as f:
                data = json.load(f)
            return cls(
                ou_half_life=data.get("ou_half_life", 12.0),
                max_contagion_rho=data.get("max_contagion_rho", 0.75),
                baseline_price_vol=data.get("baseline_price_vol", 0.003),
                propagation_lag=data.get("propagation_lag", 2.0),
            )
        return cls()

    def loss(self, simulated: dict) -> float:
        """Weighted MSE between simulated and target moments."""
        targets = {
            "ou_half_life": self.ou_half_life,
            "baseline_price_vol": self.baseline_price_vol,
        }
        total = 0.0
        for key, target in targets.items():
            sim_val = simulated.get(key, 0.0)
            total += ((sim_val - target) / max(abs(target), 1e-9)) ** 2
        return total
