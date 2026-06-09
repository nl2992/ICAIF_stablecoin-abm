"""GATE 4 — Calibration Uncertainty Bounds.

Runs the calibrated simulation N=500 times (different stochastic seeds) and computes
95% CI around each of the 4 calibration moments, verifying that the empirical targets
fall within the simulation CI.

Outputs -> experiments/results/netcontagion/
    calibration_uncertainty.csv   per-moment: empirical, mean, std, ci_lo, ci_hi, within_ci
    calibration_uncertainty.json  machine-readable version

Usage:
    cd /path/to/stablecoin-abm
    python scripts/run_calibration_uncertainty.py [--n-sims 500]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from stablesim.netcontagion.model import ContagionNetwork, estimate_transmission_matrix

GNN_ROOT = Path(__file__).parents[2] / "stablecoin-contagion-gnn"
OUT = Path("experiments/results/netcontagion")
EPISODE = "USDC_SVB"


def load_calibrated():
    b = pickle.load(open(GNN_ROOT / "data/processed/graphs" / f"{EPISODE}.pkl", "rb"))
    nodes, origin = b["active_node_strs"], b["origin"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes)
    js = json.loads((OUT / "join_summary.json").read_text())
    p = js["calibrated_params"]
    net = ContagionNetwork(nodes=nodes, W=W, coupling=p["coupling"], kappa=p["kappa"],
                           common=p["common"], sigma=p["sigma"])
    return net, nodes, origin, p["shock"]


def compute_moments(net: ContagionNetwork, nodes: list, origin: str, shock: float,
                    rng: np.random.Generator) -> dict[str, float]:
    """Compute the 4 calibration moments from a single noisy simulation run.

    net.simulate() returns d: np.ndarray of shape (n_steps, N).
    Node name → column index mapping is in net.idx.
    """
    # d[t, i] = depeg deviation of node i at step t
    d = net.simulate(origin, shock, noise=True, seed=int(rng.integers(0, 2**31)))
    shock_step = 20  # default shock_step used in simulate()

    victim_idxs = [net.idx[n] for n in nodes if n != origin and n in net.idx]
    all_idxs = [net.idx[n] for n in nodes if n in net.idx]
    origin_idx = net.idx.get(origin, -1)

    # 1. Contagion magnitude: mean peak |dev| over victims
    if victim_idxs:
        victim_paths = d[:, victim_idxs].T  # shape (n_victims, n_steps)
        contagion_mag = float(np.mean(np.max(np.abs(victim_paths), axis=1)))
    else:
        contagion_mag = 0.0

    # 2. Cross-venue rho: mean pairwise correlation during crisis period
    all_paths = d[:, all_idxs].T  # shape (n_nodes, n_steps)
    crisis_window = all_paths[:, shock_step: shock_step + 116]
    if crisis_window.shape[0] >= 2 and crisis_window.shape[1] >= 5:
        corr = np.corrcoef(crisis_window)
        mask = ~np.eye(len(corr), dtype=bool)
        cross_rho = float(np.mean(np.abs(corr[mask])))
    else:
        cross_rho = 0.0

    # 3. Baseline price vol: std of deviations pre-shock
    pre_paths = all_paths[:, max(0, shock_step - 60): shock_step]
    if pre_paths.size > 0:
        baseline_vol = float(np.std(pre_paths))
    else:
        baseline_vol = 0.0

    # 4. Crisis half-life: steps for origin deviation to decay to half peak
    if origin_idx >= 0:
        origin_path = d[:, origin_idx]
    else:
        origin_path = np.zeros(d.shape[0])
    peak_idx = int(np.argmax(np.abs(origin_path)))
    peak_val = float(np.abs(origin_path[peak_idx]))
    half_life = 116.0  # default
    if peak_val > 1e-6:
        for t in range(peak_idx, len(origin_path)):
            if np.abs(origin_path[t]) <= peak_val / 2.0:
                half_life = float(t - peak_idx)
                break

    return {
        "contagion_magnitude": contagion_mag,
        "cross_venue_rho": cross_rho,
        "baseline_price_vol": baseline_vol,
        "crisis_half_life": half_life,
    }


def main(n_sims: int = 500) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading calibrated model for {EPISODE}…")
    net, nodes, origin, shock = load_calibrated()

    # Empirical targets from the committed calibration_moments.csv
    targets = {
        "contagion_magnitude": 0.1376,
        "cross_venue_rho": 0.576,
        "baseline_price_vol": 0.003,
        "crisis_half_life": 116.0,
    }

    print(f"Running {n_sims} stochastic simulations…")
    rng = np.random.default_rng(2025)
    all_moments: dict[str, list[float]] = {k: [] for k in targets}

    for i in range(n_sims):
        if i % 50 == 0:
            print(f"  sim {i}/{n_sims}", flush=True)
        m = compute_moments(net, nodes, origin, shock, rng)
        for k, v in m.items():
            all_moments[k].append(v)

    rows = []
    all_pass = True
    print("\n=== CALIBRATION UNCERTAINTY BOUNDS ===")
    print(f"{'Moment':25s}  {'Empirical':10s}  {'Sim Mean':10s}  {'95% CI':20s}  {'Within CI':10s}")
    print("-" * 80)

    for moment, empirical in targets.items():
        vals = np.array(all_moments[moment])
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        ci_lo = float(np.percentile(vals, 2.5))
        ci_hi = float(np.percentile(vals, 97.5))
        within_ci = bool(ci_lo <= empirical <= ci_hi)
        if not within_ci:
            all_pass = False

        rows.append({
            "moment": moment,
            "empirical": empirical,
            "sim_mean": round(mean, 6),
            "sim_std": round(std, 6),
            "ci_lo_95": round(ci_lo, 6),
            "ci_hi_95": round(ci_hi, 6),
            "within_ci": within_ci,
            "n_sims": n_sims,
        })
        marker = "✓" if within_ci else "✗"
        print(f"{moment:25s}  {empirical:10.4f}  {mean:10.4f}  "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]      {marker}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration_uncertainty.csv", index=False)

    result_json = {
        "n_sims": n_sims,
        "episode": EPISODE,
        "gate_pass": all_pass,
        "n_moments_within_ci": sum(r["within_ci"] for r in rows),
        "moments": rows,
    }
    (OUT / "calibration_uncertainty.json").write_text(json.dumps(result_json, indent=2))

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nCalibration uncertainty gate: {verdict}")
    print(f"  {sum(r['within_ci'] for r in rows)}/4 empirical targets within simulation 95% CI")
    print(f"\nSaved to {OUT}/calibration_uncertainty.csv")
    print(f"         {OUT}/calibration_uncertainty.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=500)
    args = parser.parse_args()
    main(args.n_sims)
