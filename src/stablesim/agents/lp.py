"""Liquidity provider agent.

Provides and withdraws liquidity from the stableswap pool based on
observed implied volatility (pool imbalance) and LP incentive subsidy.
Tracks impermanent loss vs. a hold benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseAgent

if TYPE_CHECKING:
    from ..engine.market import MultiVenueMarket


class LPAgent(BaseAgent):
    """Liquidity provider in the stableswap AMM.

    Parameters
    ----------
    pool_idx : int
        Which pool to provide liquidity to.
    add_threshold : float
        Add liquidity when |depeg| < add_threshold (pool near equilibrium).
    remove_threshold : float
        Remove liquidity when |depeg| > remove_threshold (pool stressed).
    lp_fraction : float
        Fraction of wealth to deploy / withdraw each action.
    subsidy_rate : float
        Per-step LP incentive subsidy (fraction of deployed capital).
    """

    def __init__(
        self,
        agent_id: str,
        wealth: float = 200_000.0,
        pool_idx: int = 0,
        add_threshold: float = 0.003,
        remove_threshold: float = 0.015,
        lp_fraction: float = 0.10,
        subsidy_rate: float = 0.0,
    ) -> None:
        super().__init__(agent_id, wealth)
        self.pool_idx = pool_idx
        self.add_threshold = add_threshold
        self.remove_threshold = remove_threshold
        self.lp_fraction = lp_fraction
        self.subsidy_rate = subsidy_rate
        self._lp_tokens: float = 0.0
        self._entry_price: float = 1.0

    def act(self, market: "MultiVenueMarket", obs: dict) -> None:
        depeg = abs(obs.get("depeg", market.depeg()))
        pool = market.pools[self.pool_idx]

        # Collect subsidy
        if self._lp_tokens > 0 and self.subsidy_rate > 0:
            subsidy = self._lp_tokens * self.subsidy_rate
            self.record_pnl(subsidy)

        if depeg < self.add_threshold and self.wealth > 0:
            deploy = self.wealth * self.lp_fraction / 2
            lp = pool.add_liquidity(deploy, deploy)
            self._lp_tokens += lp
            self.wealth -= deploy * 2
            self._entry_price = pool.price()

        elif depeg > self.remove_threshold and self._lp_tokens > 0:
            frac = self.lp_fraction
            total_D = pool._D
            remove_frac = min(self._lp_tokens / max(total_D, 1e-9) * frac, 0.99)
            if remove_frac > 1e-6:
                dx, dy = pool.remove_liquidity(remove_frac)
                self._lp_tokens *= (1 - frac)
                # Approximate IL vs. hold at entry price
                hold_value = (dx + dy) * self._entry_price
                actual_value = dx * pool.price() + dy
                il = actual_value - hold_value
                self.wealth += dx * pool.price() + dy
                self.record_pnl(il)  # records IL (negative = loss)

    @property
    def lp_position(self) -> float:
        return self._lp_tokens
