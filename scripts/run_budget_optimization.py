"""
Plan B — Budget-Constrained Optimal Allocation.

Given a budget covering only K venues (K=1,2,3), compares three protection strategies:
  - Greedy optimal  : enumerate all C(n,K) subsets, pick max contagion reduction
  - GNN-guided      : protect top K by correlational (GNN predicted) hub score
  - ABM-guided      : protect top K by ABM causal Δ-contagion
  - RL regulator    : allocate to top K venues by PPO learned_allocation scores

Outputs -> experiments/results/netcontagion/
    budget_allocation.csv      per strategy per K: contagion_reduction_pct
    fig_budget_allocation.png  grouped bar chart
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from stablesim.netcontagion.model import ContagionNetwork, estimate_transmission_matrix  # noqa: E402

GNN_ROOT = Path(__file__).parents[2] / "stablecoin-contagion-gnn"
OUT = Path("experiments/results/netcontagion")
EPISODE = "USDC_SVB"
SHOCK_STEP = 40


def load_calibrated():
    b = pickle.load(open(GNN_ROOT / "data/processed/graphs" / f"{EPISODE}.pkl", "rb"))
    nodes, origin = b["active_node_strs"], b["origin"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes)
    p = json.loads((OUT / "join_summary.json").read_text())["calibrated_params"]
    net = ContagionNetwork(nodes=nodes, W=W, coupling=p["coupling"], kappa=p["kappa"],
                           common=p["common"], sigma=p["sigma"])
    return net, nodes, origin, float(p["shock"])


def protect_k(net, origin, shock, nodes, subset, victims):
    """Contagion when protecting all nodes in subset simultaneously."""
    return net.contagion_over(origin, shock, victims, protect_nodes=list(subset))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net, nodes, origin, shock = load_calibrated()
    victims = [n for n in nodes if n != origin]
    base = net.contagion_over(origin, shock, victims)

    # --- per-node reductions (from existing sweep or re-compute) ---
    sweep = pd.read_csv(OUT / "intervention_sweep.csv")
    prot = sweep[sweep["intervention"] == "targeted_protection"].copy()
    prot["node_short"] = prot["target"]
    node_map = {nd.split("/")[0]: nd for nd in nodes}

    # ABM causal ordering: rank non-origin nodes by protection reduction
    abm_single = {}
    for _, r in prot.iterrows():
        full_node = node_map.get(r["node_short"])
        if full_node and full_node != origin:
            abm_single[full_node] = float(r["pct_reduction"])
    # include origin (protecting origin = 100% by definition)
    abm_single[origin] = 100.0
    abm_order = sorted(nodes, key=lambda n: abm_single.get(n, 0.0), reverse=True)

    # GNN hub ordering from hub_ranking file
    hub_path = GNN_ROOT / "exports" / f"hub_ranking_v1_{EPISODE}.csv"
    hub_df = pd.read_csv(hub_path)
    pred_col = "hub_score_full" if "hub_score_full" in hub_df.columns else "hub_score"
    hub_score = {row["node"]: float(row[pred_col]) for _, row in hub_df.iterrows()}
    gnn_order = sorted(nodes, key=lambda n: hub_score.get(n, 0.0), reverse=True)

    # RL regulator ordering from learned allocation
    rl = json.loads((OUT / "rl_regulator.json").read_text())["learned_allocation"]
    rl_full = {node_map.get(k, k): v for k, v in rl.items()}
    rl_order = sorted(nodes, key=lambda n: rl_full.get(n, rl.get(n.split("/")[0], 0.0)), reverse=True)

    print(f"ABM order: {[n.split('/')[0] for n in abm_order]}")
    print(f"GNN order: {[n.split('/')[0] for n in gnn_order]}")
    print(f"RL order:  {[n.split('/')[0] for n in rl_order]}")

    rows = []
    for K in [1, 2, 3]:
        # greedy optimal: all C(n,K) subsets
        best_red = 0.0
        best_subset = None
        for subset in itertools.combinations(nodes, K):
            c = protect_k(net, origin, shock, nodes, subset, victims)
            red = 100.0 * (base - c) / base if base > 0 else 0.0
            if red > best_red:
                best_red = red
                best_subset = subset
        rows.append({"K": K, "strategy": "greedy_optimal",
                     "protected": "+".join(n.split("/")[0] for n in (best_subset or [])),
                     "contagion_reduction_pct": round(best_red, 1)})

        # GNN-guided: top K by GNN hub score
        gnn_k = gnn_order[:K]
        c_gnn = protect_k(net, origin, shock, nodes, gnn_k, victims)
        rows.append({"K": K, "strategy": "gnn_guided",
                     "protected": "+".join(n.split("/")[0] for n in gnn_k),
                     "contagion_reduction_pct": round(100.0 * (base - c_gnn) / base, 1)})

        # ABM-guided: top K by causal ranking
        abm_k = abm_order[:K]
        c_abm = protect_k(net, origin, shock, nodes, abm_k, victims)
        rows.append({"K": K, "strategy": "abm_guided",
                     "protected": "+".join(n.split("/")[0] for n in abm_k),
                     "contagion_reduction_pct": round(100.0 * (base - c_abm) / base, 1)})

        # RL-guided: top K by learned allocation
        rl_k = rl_order[:K]
        c_rl = protect_k(net, origin, shock, nodes, rl_k, victims)
        rows.append({"K": K, "strategy": "rl_regulator",
                     "protected": "+".join(n.split("/")[0] for n in rl_k),
                     "contagion_reduction_pct": round(100.0 * (base - c_rl) / base, 1)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "budget_allocation.csv", index=False)
    print("\n=== BUDGET ALLOCATION ===")
    print(df.to_string(index=False))

    _plot(df, OUT / "fig_budget_allocation.png")
    print(f"\n=> budget_allocation.csv written to {OUT / 'budget_allocation.csv'}")


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.apply()

    strategies = ["greedy_optimal", "gnn_guided", "abm_guided", "rl_regulator"]
    labels = ["Greedy\noptimal", "GNN-guided\n(correlational)", "ABM-guided\n(causal)", "RL\nregulator"]
    colors = [ps.BLUE, ps.RED, ps.GREEN, "#8B6914"]
    ks = [1, 2, 3]
    x = np.arange(len(ks))
    width = 0.2

    fig, ax = plt.subplots(figsize=ps.WIDE)
    for i, (strat, lbl, col) in enumerate(zip(strategies, labels, colors)):
        vals = [float(df[(df["K"] == k) & (df["strategy"] == strat)]["contagion_reduction_pct"].iloc[0])
                for k in ks]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=lbl, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in ks])
    ax.set_ylabel("Contagion reduction (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Budget-constrained allocation: correlational vs causal strategies")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("figure ->", path)


if __name__ == "__main__":
    main()
