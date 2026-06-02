"""Tests for the stableswap AMM engine."""

import pytest
import numpy as np
from stablesim.engine.amm import StableswapAMM


def test_initial_price_near_one():
    amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=100)
    assert abs(amm.price() - 1.0) < 1e-4


def test_swap_x_for_y_reduces_x_increases_y():
    amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=100)
    x0, y0 = amm.x, amm.y
    dy = amm.swap_x_for_y(10_000)
    assert amm.x > x0
    assert amm.y < y0
    assert dy > 0


def test_swap_y_for_x_reduces_y_increases_x():
    amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=100)
    x0, y0 = amm.x, amm.y
    dx = amm.swap_y_for_x(10_000)
    assert amm.y > y0
    assert amm.x < x0
    assert dx > 0


def test_large_swap_moves_price():
    amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=100)
    p0 = amm.price()
    amm.swap_x_for_y(200_000)
    p1 = amm.price()
    assert p1 < p0  # selling x depresses x price


def test_add_and_remove_liquidity():
    amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=100)
    lp = amm.add_liquidity(100_000, 100_000)
    assert lp > 0
    dx, dy = amm.remove_liquidity(0.10)
    assert dx > 0 and dy > 0


def test_higher_amp_tighter_price_impact():
    """Higher A should produce less slippage on the same trade."""
    amm_low = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=10)
    amm_high = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=500)
    dy_low = amm_low.swap_x_for_y(50_000)
    dy_high = amm_high.swap_x_for_y(50_000)
    assert dy_high > dy_low  # higher A → more output (less slippage)
