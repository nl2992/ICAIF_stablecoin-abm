"""Load StressBench scenarios from the stablecoin-contagion-network repo.

The contagion repo exports scenario configs (YAML) or event CSVs.
This loader converts them into ShockSchedule objects usable by the simulator.
Fallback: built-in synthetic scenarios for standalone use.
"""

from __future__ import annotations

import os
from pathlib import Path

from .schedule import ShockEvent, ShockSchedule

# Default path to the sibling stablecoin-contagion-network repo
_DEFAULT_CONTAGION_ROOT = Path(__file__).parents[6] / "stablecoin-contagion-network"


def load_stressbench_scenarios(
    contagion_root: str | Path | None = None,
) -> list[ShockSchedule]:
    """Load StressBench scenarios from stablecoin-contagion-network configs.

    Falls back to built-in synthetic scenarios if the repo is not found.
    """
    root = Path(contagion_root or _DEFAULT_CONTAGION_ROOT)
    config_dir = root / "configs"

    if config_dir.exists():
        try:
            return _load_from_contagion_configs(config_dir)
        except Exception:
            pass

    return _synthetic_scenarios()


def _load_from_contagion_configs(config_dir: Path) -> list[ShockSchedule]:
    import yaml

    schedules = []
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            cfg = yaml.safe_load(f)
        shocks = cfg.get("shocks", [])
        events = [
            ShockEvent(
                step=s["step"],
                kind=s["kind"],
                magnitude=s["magnitude"],
                pool_idx=s.get("pool_idx", 0),
                label=s.get("label", yaml_file.stem),
            )
            for s in shocks
        ]
        schedules.append(ShockSchedule(events=events, name=yaml_file.stem))
    return schedules


def _synthetic_scenarios() -> list[ShockSchedule]:
    """Built-in scenarios covering depeg archetypes from empirical episodes."""
    return [
        ShockSchedule(
            name="ust_style_bank_run",
            events=[
                ShockEvent(step=10, kind="sell_pressure", magnitude=0.05, label="initial panic"),
                ShockEvent(step=12, kind="sell_pressure", magnitude=0.10, label="cascade"),
                ShockEvent(step=14, kind="liquidity_removal", magnitude=0.30, label="LP exit"),
                ShockEvent(step=16, kind="reserve_drop", magnitude=0.20, label="reserve impairment"),
            ],
        ),
        ShockSchedule(
            name="usdc_circuit_breaker",
            events=[
                ShockEvent(step=5, kind="sell_pressure", magnitude=0.08, label="SVB news"),
                ShockEvent(step=7, kind="liquidity_removal", magnitude=0.15, label="pool drain"),
            ],
        ),
        ShockSchedule(
            name="slow_depeg",
            events=[
                ShockEvent(step=20, kind="sell_pressure", magnitude=0.03, label="slow bleed"),
                ShockEvent(step=30, kind="sell_pressure", magnitude=0.03, label="slow bleed"),
                ShockEvent(step=40, kind="reserve_drop", magnitude=0.05, label="reserve concern"),
            ],
        ),
        ShockSchedule(
            name="no_shock_baseline",
            events=[],
        ),
    ]
