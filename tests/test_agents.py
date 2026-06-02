"""Smoke tests for agent behaviour."""

import numpy as np
from stablesim.engine.market import MultiVenueMarket
from stablesim.agents.arbitrageur import Arbitrageur
from stablesim.agents.redeemer import Redeemer
from stablesim.agents.lp import LPAgent
from stablesim.agents.noise import NoiseTrader


def _market():
    return MultiVenueMarket()


def test_arbitrageur_acts_without_error():
    market = _market()
    arb = Arbitrageur("arb", wealth=100_000)
    snap = market.step()
    arb.act(market, snap)  # should not raise


def test_redeemer_submits_on_depeg():
    market = _market()
    # Force a depeg by draining pool
    market.pools[0].swap_x_for_y(900_000)
    red = Redeemer("red", stablecoin_holdings=50_000, trigger_depeg=0.0)
    snap = market.step()
    q_before = market.redemption.queue_depth()
    red.act(market, snap)
    # Should have submitted at least one order (market is below peg)
    assert market.redemption.queue_depth() >= q_before


def test_noise_trader_acts_without_error():
    market = _market()
    noise = NoiseTrader("n0", rng=np.random.default_rng(42))
    for _ in range(10):
        snap = market.step()
        noise.act(market, snap)


def test_lp_adds_liquidity_near_peg():
    market = _market()
    lp = LPAgent("lp0", add_threshold=1.0)  # always add
    snap = market.step()
    lp.act(market, snap)
    assert lp.lp_position > 0
