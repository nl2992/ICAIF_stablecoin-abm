"""Abstract base class for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine.market import MultiVenueMarket


class BaseAgent(ABC):
    """All agents expose a single act() method called each market step."""

    def __init__(self, agent_id: str, wealth: float = 100_000.0) -> None:
        self.agent_id = agent_id
        self.wealth = float(wealth)
        self._pnl: float = 0.0
        self._step: int = 0

    @abstractmethod
    def act(self, market: "MultiVenueMarket", obs: dict) -> None:
        """Observe market state and take action(s)."""

    def record_pnl(self, delta: float) -> None:
        self._pnl += delta
        self.wealth += delta
        self._step += 1

    @property
    def cumulative_pnl(self) -> float:
        return self._pnl

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, wealth={self.wealth:.0f})"
