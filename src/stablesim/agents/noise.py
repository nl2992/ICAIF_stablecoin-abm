"""Background noise traders.

Provide realistic order flow: random buy/sell pressure each step
drawn from a calibrated distribution so that the no-shock baseline
exhibits empirically observed price variance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import BaseAgent

if TYPE_CHECKING:
    from ..engine.market import MultiVenueMarket


class NoiseTrader(BaseAgent):
    """Random buy/sell agent with configurable activity rate.

    Parameters
    ----------
    trade_prob : float
        Probability of trading each step.
    trade_size_mean : float
        Mean trade size (USD).
    trade_size_std : float
        Std dev of trade size.
    rng : np.random.Generator | None
    """

    def __init__(
        self,
        agent_id: str,
        wealth: float = 50_000.0,
        trade_prob: float = 0.30,
        trade_size_mean: float = 1_000.0,
        trade_size_std: float = 500.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(agent_id, wealth)
        self.trade_prob = trade_prob
        self.trade_size_mean = trade_size_mean
        self.trade_size_std = trade_size_std
        self.rng = rng or np.random.default_rng()

    def act(self, market: "MultiVenueMarket", obs: dict) -> None:
        if self.rng.random() > self.trade_prob:
            return

        size = abs(self.rng.normal(self.trade_size_mean, self.trade_size_std))
        size = min(size, self.wealth * 0.05)
        if size < 1.0:
            return

        pool_idx = self.rng.integers(0, len(market.pools))
        pool = market.pools[pool_idx]

        buy = self.rng.random() < 0.5
        try:
            if buy:
                pool.swap_y_for_x(size)
                self.wealth -= size
            else:
                pool.swap_x_for_y(size)
                self.wealth += size
        except (ValueError, ZeroDivisionError):
            pass
