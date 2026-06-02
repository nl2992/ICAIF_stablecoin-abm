"""Primary redemption channel — the issuer's on-demand mint/redeem facility.

Supports:
- flat fee on redemption
- queue with maximum length and processing delay
- circuit breaker that halts redemptions when depeg exceeds threshold
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class RedemptionOrder:
    agent_id: str
    amount: float
    submitted_step: int


@dataclass
class RedemptionChannel:
    """Issuer redemption facility with configurable gating.

    Parameters
    ----------
    fee_bps : float
        Flat redemption fee in basis points.
    max_queue : int
        Maximum pending orders (0 = unlimited).
    delay_steps : int
        Steps between submission and settlement.
    cb_threshold : float
        Depeg magnitude (|price - 1|) that triggers circuit breaker.
    cb_duration : int
        Steps the circuit breaker stays active once triggered.
    """

    fee_bps: float = 0.0
    max_queue: int = 0
    delay_steps: int = 0
    cb_threshold: float = 0.10
    cb_duration: int = 12

    _queue: deque[RedemptionOrder] = field(default_factory=deque, init=False, repr=False)
    _cb_active_until: int = field(default=0, init=False, repr=False)
    _settled: list[dict] = field(default_factory=list, init=False, repr=False)

    def submit(self, agent_id: str, amount: float, current_step: int) -> bool:
        """Attempt to submit a redemption order.  Returns True if accepted."""
        if self.max_queue > 0 and len(self._queue) >= self.max_queue:
            return False
        if current_step < self._cb_active_until:
            return False
        self._queue.append(RedemptionOrder(agent_id, amount, current_step))
        return True

    def settle(self, current_step: int) -> list[dict]:
        """Process orders whose delay has elapsed.  Returns settled records."""
        fee = self.fee_bps / 10_000.0
        ready = []
        remaining = deque()
        for order in self._queue:
            if current_step - order.submitted_step >= self.delay_steps:
                ready.append(order)
            else:
                remaining.append(order)
        self._queue = remaining

        settled = []
        for order in ready:
            net = order.amount * (1 - fee)
            record = {
                "agent_id": order.agent_id,
                "gross": order.amount,
                "fee": order.amount * fee,
                "net": net,
                "settled_step": current_step,
            }
            settled.append(record)
        self._settled.extend(settled)
        return settled

    def trigger_circuit_breaker(self, current_step: int) -> None:
        self._cb_active_until = current_step + self.cb_duration

    def check_and_trigger(self, price: float, current_step: int) -> bool:
        """Trigger circuit breaker if depeg exceeds threshold.  Returns True if triggered."""
        if abs(price - 1.0) >= self.cb_threshold:
            self.trigger_circuit_breaker(current_step)
            return True
        return False

    def is_halted(self, current_step: int) -> bool:
        return current_step < self._cb_active_until

    def queue_depth(self) -> int:
        return len(self._queue)

    def state(self) -> dict:
        return {
            "queue_depth": self.queue_depth(),
            "fee_bps": self.fee_bps,
            "cb_active_until": self._cb_active_until,
        }
