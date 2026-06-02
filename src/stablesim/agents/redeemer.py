"""Primary redemption agent.

Heuristic: submits redemption when AMM price is sufficiently below par
(expects issuer to return 1.0 USD per stablecoin, net of fee/delay).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseAgent

if TYPE_CHECKING:
    from ..engine.market import MultiVenueMarket


class Redeemer(BaseAgent):
    """Redeemer who arbitrages the AMM vs. primary redemption channel.

    Parameters
    ----------
    trigger_depeg : float
        Submit redemption when |depeg| exceeds this threshold.
    redemption_frac : float
        Fraction of stablecoin holdings to redeem each trigger.
    stablecoin_holdings : float
        Initial stablecoin balance (denominated in stablecoins).
    """

    def __init__(
        self,
        agent_id: str,
        wealth: float = 100_000.0,
        trigger_depeg: float = 0.005,
        redemption_frac: float = 0.20,
        stablecoin_holdings: float | None = None,
        policy=None,
    ) -> None:
        super().__init__(agent_id, wealth)
        self.trigger_depeg = trigger_depeg
        self.redemption_frac = redemption_frac
        self.stablecoin = stablecoin_holdings if stablecoin_holdings is not None else wealth
        self.policy = policy

    def act(self, market: "MultiVenueMarket", obs: dict) -> None:
        depeg = obs.get("depeg", market.depeg())
        step = obs.get("step", market.step_count)

        if self.policy is not None:
            action = self.policy(obs)
            self._execute_rl_action(market, action, step)
            return

        # Only redeem when stablecoin trades at a discount (depeg < 0)
        if depeg >= -self.trigger_depeg or self.stablecoin <= 0:
            return

        amount = self.stablecoin * self.redemption_frac
        accepted = market.redemption.submit(self.agent_id, amount, step)
        if accepted:
            self.stablecoin -= amount

    def receive_settlement(self, net_usd: float) -> None:
        """Called by market when a redemption order is settled."""
        self.record_pnl(net_usd)

    def _execute_rl_action(self, market, action, step: int) -> None:
        frac = float(action) if hasattr(action, "__float__") else 0.0
        frac = max(0.0, min(1.0, frac))
        if frac == 0 or self.stablecoin <= 0:
            return
        amount = self.stablecoin * frac
        if market.redemption.submit(self.agent_id, amount, step):
            self.stablecoin -= amount
