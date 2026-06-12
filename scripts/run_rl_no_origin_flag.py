"""
Ablation: PPO regulator WITHOUT the origin indicator in its observation.

The headline RL experiment (scripts/run_rl_regulator.py) gives the agent three structural
features per venue: out-transmission, in-transmission, and an origin flag. The flag
identifies the shock source, so a referee can argue the agent is handed the origin rather
than discovering it. This ablation removes the flag (observation = in/out-transmission
only, shape N*2) and retrains PPO from scratch over multiple seeds, holding everything
else fixed (same calibrated engine, same reward, same hyperparameters and 12,000 steps).

Question: without being told which venue is the origin, does the agent still fund the
causal venues and allocate ~0 to the spurious hub (BUSD)?

Outputs -> experiments/results/netcontagion/rl_no_origin_flag.json
(The headline artifact rl_regulator.json is left untouched.)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from stablesim.netcontagion.model import ContagionNetwork, estimate_transmission_matrix  # noqa: E402
from stablesim.netcontagion.rl_env import RegulatorEnv  # noqa: E402

GNN_ROOT = Path(__file__).parents[2] / "stablecoin-contagion-gnn"
OUT = Path("experiments/results/netcontagion")
EPISODE = "USDC_SVB"
SEEDS = [0, 1, 2, 3, 4]


def build():
    b = pickle.load(open(GNN_ROOT / "data/processed/graphs" / f"{EPISODE}.pkl", "rb"))
    nodes, origin = b["active_node_strs"], b["origin"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes)
    p = json.loads((OUT / "join_summary.json").read_text())["calibrated_params"]
    net = ContagionNetwork(nodes=nodes, W=W, coupling=p["coupling"], kappa=p["kappa"],
                           common=p["common"], sigma=p["sigma"])
    return net, origin, float(p["shock"])


def main(timesteps: int = 12000):
    OUT.mkdir(parents=True, exist_ok=True)
    net, origin, shock = build()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    runs = []
    for seed in SEEDS:
        venv = DummyVecEnv([lambda: RegulatorEnv(net, origin, shock, include_origin_flag=False)])
        model = PPO("MlpPolicy", venv, verbose=0, seed=seed, n_steps=256, batch_size=64,
                    gae_lambda=0.95, gamma=0.99, ent_coef=0.01, learning_rate=3e-4)
        model.learn(total_timesteps=timesteps)
        env = RegulatorEnv(net, origin, shock, include_origin_flag=False)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        runs.append({"seed": seed,
                     "allocation": info["alloc"],
                     "reduction_pct": round(100 * info["reduction"], 1)})
        print(f"seed {seed}: reduction {runs[-1]['reduction_pct']}% alloc {info['alloc']}")

    reds = [r["reduction_pct"] for r in runs]
    busd = [r["allocation"].get("BUSD", 0.0) for r in runs]
    usdc = [r["allocation"].get("USDC", 0.0) for r in runs]
    dai = [r["allocation"].get("DAI", 0.0) for r in runs]
    res = {
        "episode": EPISODE,
        "ppo_timesteps": timesteps,
        "seeds": SEEDS,
        "observation": "in/out-transmission only (origin flag REMOVED)",
        "per_seed": runs,
        "reduction_pct_mean": round(float(np.mean(reds)), 1),
        "reduction_pct_std": round(float(np.std(reds)), 2),
        "budget_on_spurious_BUSD_max": round(max(busd), 3),
        "budget_on_origin_USDC_min": round(min(usdc), 3),
        "budget_on_relay_DAI_min": round(min(dai), 3),
    }
    (OUT / "rl_no_origin_flag.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_seed"}, indent=2))
    print(f"\nWrote {OUT/'rl_no_origin_flag.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=12000)
    main(ap.parse_args().timesteps)
