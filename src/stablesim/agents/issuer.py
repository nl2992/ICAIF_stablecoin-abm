"""Issuer / reserve manager agent.

Passive by default (processes redemptions via the RedemptionChannel).
Can optionally perform open-market operations to defend the peg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseAgent

if TYPE_CHECKING:
    from ..engine.market import MultiVenueMarket


class IssuerAgent(BaseAgent):
    """Stablecoin issuer that manages reserves and can intervene in pools.

    Parameters
    ----------
    intervention_threshold : float
        |depeg| level at which the issuer performs open-market operations.
    intervention_size : float
        USD deployed per open-market operation (buy-back to defend peg).
    """

    def __init__(
        self,
        agent_id: str = "issuer",
        wealth: float = 10_000_000.0,
        intervention_threshold: float = 0.05,
        intervention_size: float = 500_000.0,
    ) -> None:
        super().__init__(agent_id, wealth)
        self.intervention_threshold = intervention_threshold
        self.intervention_size = intervention_size

    def act(self, market: "MultiVenueMarket", obs: dict) -> None:
        depeg = obs.get("depeg", market.depeg())

        # If stablecoin is trading below peg by more than threshold, buy back
        if depeg < -self.intervention_threshold and self.wealth > self.intervention_size:
            pool = market.pools[0]
            try:
                dx = pool.swap_y_for_x(self.intervention_size)
                self.wealth -= self.intervention_size
                # Issuer holds dx stablecoins (can burn them — here just tracked)
                self.record_pnl(-self.intervention_size)
            except (ValueError, ZeroDivisionError):
                pass
