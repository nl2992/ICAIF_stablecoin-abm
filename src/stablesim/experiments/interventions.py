"""Intervention configuration knobs.

Each InterventionConfig is one point in the design space.
The sweep runner creates a MultiVenueMarket for each config × scenario pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterventionConfig:
    """Full specification of intervention parameters.

    Reserve transparency
    --------------------
    transparency_freq : int
        Steps between reserve disclosures (0 = opaque).
    transparency_noise : float
        Std dev of Gaussian noise on disclosed ratio.

    Redemption gating
    -----------------
    gate_fee_bps : float
        Flat redemption fee in basis points.
    gate_queue_len : int
        Max pending redemptions (0 = unlimited).
    gate_delay_steps : int
        Steps between submission and settlement.

    Circuit breaker
    ---------------
    cb_threshold : float
        |depeg| that triggers halt.
    cb_duration : int
        Steps the circuit breaker stays active.

    LP incentives
    -------------
    lp_subsidy_rate : float
        Per-step subsidy to LPs as fraction of deployed capital.

    label : str
        Human-readable identifier for this configuration.
    """

    transparency_freq: int = 0
    transparency_noise: float = 0.0
    gate_fee_bps: float = 0.0
    gate_queue_len: int = 0
    gate_delay_steps: int = 0
    cb_threshold: float = 0.10
    cb_duration: int = 0
    lp_subsidy_rate: float = 0.0
    label: str = "baseline"

    def to_market_kwargs(self) -> dict:
        """Convert to keyword dicts for MultiVenueMarket constructors."""
        return {
            "reserve": {
                "transparency_freq": self.transparency_freq,
                "transparency_noise": self.transparency_noise,
            },
            "redemption": {
                "fee_bps": self.gate_fee_bps,
                "max_queue": self.gate_queue_len,
                "delay_steps": self.gate_delay_steps,
                "cb_threshold": self.cb_threshold,
                "cb_duration": self.cb_duration,
            },
            "lp_subsidy_rate": self.lp_subsidy_rate,
        }


# --- Pre-defined sweep grid ---

BASELINE = InterventionConfig(label="baseline")

TRANSPARENCY_DAILY = InterventionConfig(transparency_freq=1, label="transparency_daily")
TRANSPARENCY_WEEKLY = InterventionConfig(transparency_freq=7, label="transparency_weekly")
TRANSPARENCY_NOISY = InterventionConfig(
    transparency_freq=1, transparency_noise=0.02, label="transparency_noisy"
)

GATE_FEE_1PCT = InterventionConfig(gate_fee_bps=100, label="gate_fee_1pct")
GATE_FEE_2PCT = InterventionConfig(gate_fee_bps=200, label="gate_fee_2pct")
GATE_QUEUE = InterventionConfig(gate_queue_len=50, gate_delay_steps=6, label="gate_queue")

CIRCUIT_BREAKER_5PCT = InterventionConfig(
    cb_threshold=0.05, cb_duration=12, label="cb_5pct_12steps"
)
CIRCUIT_BREAKER_10PCT = InterventionConfig(
    cb_threshold=0.10, cb_duration=24, label="cb_10pct_24steps"
)

LP_SUBSIDY = InterventionConfig(lp_subsidy_rate=0.0001, label="lp_subsidy_1bp")

COMBINED = InterventionConfig(
    transparency_freq=1,
    gate_fee_bps=50,
    cb_threshold=0.05,
    cb_duration=12,
    lp_subsidy_rate=0.0001,
    label="combined",
)

DEFAULT_SWEEP = [
    BASELINE,
    TRANSPARENCY_DAILY,
    TRANSPARENCY_WEEKLY,
    GATE_FEE_1PCT,
    GATE_FEE_2PCT,
    GATE_QUEUE,
    CIRCUIT_BREAKER_5PCT,
    CIRCUIT_BREAKER_10PCT,
    LP_SUBSIDY,
    COMBINED,
]
