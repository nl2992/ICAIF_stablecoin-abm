"""RL regulator on the USDT/Curve May-2022 episode (cross-crisis generalization of rl_regulator.py).

Same engine and RegulatorEnv as run_rl_regulator.py, but built on the May-2022 USDT network with the
same default-params + calibrated-shock regime used by run_multi_episode_join.py (so it is consistent
with that episode's committed knockout deltas: TUSD is the causal relay, the GNN's top hub USDC is
near-inert). Shows whether a PPO regulator, given only structural wiring and no causal labels, funds
the causal relay rather than the GNN's correlational pick on a SECOND crisis.

Outputs -> experiments/results/netcontagion/rl_regulator_usdt_may.json
"""
from __future__ import annotations
import importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stablesim.netcontagion.model import ContagionNetwork, estimate_transmission_matrix  # noqa: E402
from stablesim.netcontagion.rl_env import RegulatorEnv  # noqa: E402

# reuse calibrate + episode_targets from the multi-episode script (same regime)
_spec = importlib.util.spec_from_file_location("mej", ROOT / "scripts" / "run_multi_episode_join.py")
mej = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mej)

GNN_ROOT = ROOT.parent / "stablecoin-contagion-gnn"
OUT = ROOT / "experiments/results/netcontagion"
EPISODE = "USDT_May2022"


def build():
    b = pickle.load(open(GNN_ROOT / "data/processed/graphs" / f"{EPISODE}.pkl", "rb"))
    nodes, origin = b["active_node_strs"], b["origin"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes)
    net = ContagionNetwork(nodes=nodes, W=W)                    # default params, as in multi-episode
    shock = mej.calibrate(net, origin, mej.episode_targets(EPISODE))
    return net, origin, float(shock)


def short(n):  # "USDC/binance" -> "USDC"
    return n.split("/")[0]


def main(timesteps: int = 12000, seeds=(0, 1, 2)):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    net, origin, shock = build()
    gnn_pick, relay = "USDC", "TUSD"      # GNN top hub vs ABM causal relay (committed knockout)
    allocs = []
    for s in seeds:
        venv = DummyVecEnv([lambda: RegulatorEnv(net, origin, shock)])
        model = PPO("MlpPolicy", venv, verbose=0, seed=s, n_steps=256, batch_size=64,
                    gae_lambda=0.95, gamma=0.99, ent_coef=0.01, learning_rate=3e-4)
        model.learn(total_timesteps=timesteps)
        env = RegulatorEnv(net, origin, shock)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        a = {short(k): v for k, v in info["alloc"].items()}
        allocs.append({"seed": s, "alloc": a, "reduction_pct": round(100 * info["reduction"], 1),
                       "on_gnn_pick_USDC": round(a.get(gnn_pick, 0.0), 3),
                       "on_relay_TUSD": round(a.get(relay, 0.0), 3)})
        print(allocs[-1])
    import statistics as st
    out = {
        "episode": EPISODE, "origin": short(origin), "ppo_timesteps": timesteps, "seeds": list(seeds),
        "gnn_top_hub": gnn_pick, "abm_causal_relay": relay,
        "mean_budget_on_gnn_pick_USDC": round(st.mean(a["on_gnn_pick_USDC"] for a in allocs), 3),
        "mean_budget_on_relay_TUSD": round(st.mean(a["on_relay_TUSD"] for a in allocs), 3),
        "mean_reduction_pct": round(st.mean(a["reduction_pct"] for a in allocs), 1),
        "per_seed": allocs,
    }
    (OUT / "rl_regulator_usdt_may.json").write_text(json.dumps(out, indent=2))
    print("\nSUMMARY:", json.dumps({k: out[k] for k in
          ["mean_budget_on_relay_TUSD", "mean_budget_on_gnn_pick_USDC", "mean_reduction_pct"]}))


if __name__ == "__main__":
    main()
