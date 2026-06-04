"""Tests for the networked-contagion engine + causal knockout semantics."""
from __future__ import annotations

import numpy as np

from stablesim.netcontagion.model import ContagionNetwork, estimate_transmission_matrix


def _line_network():
    # A -> B -> C transmission chain (A leads B leads C); D isolated sink.
    nodes = ["A", "B", "C", "D"]
    W = np.zeros((4, 4))
    W[1, 0] = 1.0  # B receives from A
    W[2, 1] = 1.0  # C receives from B
    return ContagionNetwork(nodes=nodes, W=W, coupling=0.02, kappa=0.006,
                            common=0.0, sigma=0.0)


def test_simulation_is_bounded():
    net = _line_network()
    d = net.simulate("A", 0.15, shock_step=20, n_steps=300, seed=0)
    assert np.isfinite(d).all()
    assert np.abs(d).max() <= 0.6  # clamp holds


def test_shock_propagates_along_chain():
    net = _line_network()
    d = net.simulate("A", 0.15, shock_step=20, n_steps=400, seed=0, noise=False)
    # B and C should depeg (downstream of A); A is the origin
    assert np.abs(d[:, 1]).max() > 0.001
    assert np.abs(d[:, 2]).max() > 0.0005


def test_isolated_sink_has_zero_causal_effect():
    """Protecting an isolated/sink node must not change contagion to others."""
    net = _line_network()
    delta_D = net.causal_delta("A", 0.15, "D")   # D transmits to no one
    delta_B = net.causal_delta("A", 0.15, "B")   # B is on the transmission path
    assert abs(delta_D) < 1e-6
    assert delta_B > abs(delta_D)                # B is causally important, D is not


def test_protecting_origin_removes_contagion():
    net = _line_network()
    measure = ["B", "C", "D"]
    base = net.contagion_over("A", 0.15, measure, protect=None)
    prot = net.contagion_over("A", 0.15, measure, protect="A")
    assert prot < 0.1 * base + 1e-9   # protecting the source kills the cascade


def test_transmission_matrix_is_directional():
    # A leads B by a fixed lag -> W[B,A] > 0 and W[A,B] == 0 after net-directionalisation
    rng = np.random.default_rng(0)
    a = np.zeros(2000)
    a[500:520] = -200.0  # A depegs (bps)
    b = np.zeros(2000)
    b[510:530] = -180.0  # B follows 10 steps later
    W = estimate_transmission_matrix({"A": a, "B": b}, ["A", "B"], max_lag=30, stress_bps=25.0)
    assert W[1, 0] > 0.0       # B receives from A
    assert W[0, 1] == 0.0      # A does not receive from B (net-directional)


def test_moments_keys_present():
    net = _line_network()
    m = net.moments("A", 0.12, n_seeds=4)
    for k in ("contagion_magnitude", "crisis_half_life", "baseline_price_vol", "cross_venue_rho"):
        assert k in m
    assert np.isfinite(m["crisis_half_life"])  # analytic ln2/kappa, never NaN


def test_interventions_reduce_contagion():
    net = _line_network()
    measure = ["B", "C"]
    base = net.contagion_over("A", 0.15, measure)
    # circuit breaker (tight cap) and reserve strengthening on the source both help
    cb = net.contagion_over("A", 0.15, measure, cb_threshold=0.02)
    rs = net.contagion_over("A", 0.15, measure, kappa_scale={"A": 10.0})
    assert cb < base
    assert rs < base


def test_rl_env_step_contract():
    import numpy as np
    from stablesim.netcontagion.rl_env import RegulatorEnv
    net = _line_network()
    env = RegulatorEnv(net, "A", 0.15)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (net.N * 3,)
    obs, reward, term, trunc, info = env.step(np.zeros(net.N))
    assert term and "alloc" in info and np.isfinite(reward)
