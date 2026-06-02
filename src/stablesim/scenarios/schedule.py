"""Shock event schedule: exogenous shocks imported from StressBench or defined inline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ShockEvent:
    """Single exogenous shock applied to the market at a specific step.

    Parameters
    ----------
    step : int
        Market step at which the shock fires.
    kind : str
        One of: sell_pressure | buy_pressure | liquidity_removal | reserve_drop
    magnitude : float
        Shock intensity (fraction of affected pool reserve or reserve ratio).
    pool_idx : int
        Target pool index (0 = primary AMM).
    label : str
        Human-readable label (e.g. "UST depeg May-22").
    """

    step: int
    kind: str
    magnitude: float
    pool_idx: int = 0
    label: str = ""


@dataclass
class ShockSchedule:
    """Ordered sequence of ShockEvents for a single simulation episode."""

    events: list[ShockEvent] = field(default_factory=list)
    name: str = "unnamed"

    def events_at(self, step: int) -> list[ShockEvent]:
        return [e for e in self.events if e.step == step]

    def __iter__(self) -> Iterator[ShockEvent]:
        return iter(sorted(self.events, key=lambda e: e.step))

    def __len__(self) -> int:
        return len(self.events)
