"""
Plan D — RL Training Convergence Evidence.

Re-runs PPO training with 5 random seeds and records how the USDC/BUSD allocation
and contagion reduction evolve over 12,000 timesteps. Shows the policy reliably
converges to the causal allocation, not a single-seed artefact.

Outputs -> experiments/results/netcontagion/
    rl_convergence.csv          timestep × seed: usdc_alloc, busd_alloc, reduction_pct
    fig_rl_convergence.png      3-panel: USDC alloc, BUSD alloc, reduction vs timestep
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from stablesim.netcontagion.model import ContagionNetwork, estimate_transmission_matrix  # noqa: E402
from stablesim.netcontagion.rl_env import RegulatorEnv  # noqa: E402

GNN_ROOT = Path(__file__).parents[2] / "stablecoin-contagion-gnn"
OUT = Path("experiments/results/netcontagion")
EPISODE = "USDC_SVB"
N_SEEDS = 5
TOTAL_TIMESTEPS = 12000
EVAL_FREQ = 500   # evaluate policy every N timesteps


def build():
    b = pickle.load(open(GNN_ROOT / "data/processed/graphs" / f"{EPISODE}.pkl", "rb"))
    nodes, origin = b["active_node_strs"], b["origin"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes)
    p = json.loads((OUT / "join_summary.json").read_text())["calibrated_params"]
    net = ContagionNetwork(nodes=nodes, W=W, coupling=p["coupling"], kappa=p["kappa"],
                           common=p["common"], sigma=p["sigma"])
    return net, origin, float(p["shock"])


class AllocTracker:
    """Lightweight callback compatible with stable_baselines3."""

    def __init__(self, env: RegulatorEnv, eval_freq: int, seed: int):
        self.env = env
        self.eval_freq = eval_freq
        self.seed = seed
        self.records: list[dict] = []
        self._calls = 0

    def on_step(self, model) -> bool:
        self._calls += 1
        if self._calls % self.eval_freq == 0:
            obs, _ = self.env.reset()
            action, _ = model.predict(obs, deterministic=True)
            _, _, _, _, info = self.env.step(action)
            alloc = info["alloc"]
            self.records.append({
                "seed": self.seed,
                "timestep": self._calls,
                "usdc_alloc": round(alloc.get("USDC", 0.0), 3),
                "busd_alloc": round(alloc.get("BUSD", 0.0), 3),
                "reduction_pct": round(100.0 * info["reduction"], 1),
            })
        return True


def _train_with_tracking(net, origin, shock, seed: int) -> list[dict]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    env = RegulatorEnv(net, origin, shock)
    tracker = AllocTracker(env, EVAL_FREQ, seed)

    class _Callback(BaseCallback):
        def __init__(self, tracker):
            super().__init__()
            self._t = tracker

        def _on_step(self) -> bool:
            return self._t.on_step(self.model)

    venv = DummyVecEnv([lambda: RegulatorEnv(net, origin, shock)])
    model = PPO("MlpPolicy", venv, verbose=0, seed=seed, n_steps=256, batch_size=64,
                gae_lambda=0.95, gamma=0.99, ent_coef=0.01, learning_rate=3e-4)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=_Callback(tracker))

    # final evaluation
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    alloc = info["alloc"]
    tracker.records.append({
        "seed": seed,
        "timestep": TOTAL_TIMESTEPS,
        "usdc_alloc": round(alloc.get("USDC", 0.0), 3),
        "busd_alloc": round(alloc.get("BUSD", 0.0), 3),
        "reduction_pct": round(100.0 * info["reduction"], 1),
    })
    return tracker.records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net, origin, shock = build()

    all_records = []
    for seed in range(N_SEEDS):
        print(f"--- seed {seed} ---")
        records = _train_with_tracking(net, origin, shock, seed)
        all_records.extend(records)
        final = records[-1]
        print(f"  final: USDC={final['usdc_alloc']:.3f}, "
              f"BUSD={final['busd_alloc']:.3f}, "
              f"reduction={final['reduction_pct']:.1f}%")

    df = pd.DataFrame(all_records)
    df.to_csv(OUT / "rl_convergence.csv", index=False)

    # summary stats across seeds at final timestep
    finals = df[df["timestep"] == TOTAL_TIMESTEPS]
    usdc_mean = finals["usdc_alloc"].mean()
    usdc_std = finals["usdc_alloc"].std()
    busd_mean = finals["busd_alloc"].mean()
    busd_std = finals["busd_alloc"].std()
    red_mean = finals["reduction_pct"].mean()
    red_std = finals["reduction_pct"].std()
    print(f"\n=== CONVERGENCE SUMMARY (N={N_SEEDS} seeds) ===")
    print(f"USDC allocation: {usdc_mean:.3f} ± {usdc_std:.3f}")
    print(f"BUSD allocation: {busd_mean:.3f} ± {busd_std:.3f}")
    print(f"Contagion reduction: {red_mean:.1f} ± {red_std:.1f}%")

    summary = {
        "n_seeds": N_SEEDS, "total_timesteps": TOTAL_TIMESTEPS,
        "usdc_alloc_mean": round(usdc_mean, 3), "usdc_alloc_std": round(usdc_std, 3),
        "busd_alloc_mean": round(busd_mean, 3), "busd_alloc_std": round(busd_std, 3),
        "reduction_pct_mean": round(red_mean, 1), "reduction_pct_std": round(red_std, 1),
        "all_seeds_usdc_ge_0.9": bool((finals["usdc_alloc"] >= 0.9).all()),
        "all_seeds_busd_le_0.1": bool((finals["busd_alloc"] <= 0.1).all()),
        "all_seeds_reduction_ge_90": bool((finals["reduction_pct"] >= 90.0).all()),
    }
    (OUT / "rl_convergence_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    _plot(df, OUT / "fig_rl_convergence.png")
    print(f"\n=> rl_convergence.csv written to {OUT / 'rl_convergence.csv'}")


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.apply()

    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDE[0] * 1.2, ps.WIDE[1]))
    panels = [
        ("usdc_alloc", "USDC allocation", ps.GREEN, (0, 1.05), 0.9, "≥0.9 target"),
        ("busd_alloc", "BUSD allocation (spurious hub)", ps.RED, (-0.05, 1.05), 0.1, "≤0.1 target"),
        ("reduction_pct", "Contagion reduction (%)", ps.BLUE, (-5, 110), 90.0, "≥90% target"),
    ]
    for ax, (col, ylabel, color, ylim, thr, thr_lbl) in zip(axes, panels):
        for seed, grp in df.groupby("seed"):
            ax.plot(grp["timestep"], grp[col], alpha=0.6, lw=1.2, color=color)
        # mean across seeds at each checkpoint
        mean_curve = df.groupby("timestep")[col].mean()
        ax.plot(mean_curve.index, mean_curve.values, color="k", lw=2, label="mean")
        ax.axhline(thr, color="gray", ls="--", lw=1, label=thr_lbl)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle(f"RL convergence across {df['seed'].nunique()} seeds (PPO, {df['timestep'].max()} timesteps)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("figure ->", path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = ap.parse_args()
    N_SEEDS = args.seeds
    TOTAL_TIMESTEPS = args.timesteps
    main()
