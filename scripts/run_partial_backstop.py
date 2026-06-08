"""
Plan F — Optimal Partial Backstop Analysis.

For USDC (causal origin) and BUSD (spurious hub), sweeps the kappa multiplier
(reserve-strengthening intensity) and finds the cost-effectiveness frontier:
minimum multiplier to achieve 50%, 80%, 95% contagion reduction.

USDC: costs are justified (steep positive slope on reduction)
BUSD: any intensity is wasted (flat near-zero regardless of spending)

Outputs -> experiments/results/netcontagion/
    partial_backstop.csv       kappa_mult × target → contagion, pct_reduction, cost_proxy
    fig_partial_backstop.png   cost-effectiveness curves for USDC vs BUSD
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

GNN_ROOT = Path(__file__).parents[2] / "stablecoin-contagion-gnn"
OUT = Path("experiments/results/netcontagion")
EPISODE = "USDC_SVB"

# approximate relative market cap at SVB time (USDC >> BUSD)
MCAP_PROXY = {"USDC": 43.5, "BUSD": 16.1}   # billion USD (rough order-of-magnitude)


def load_calibrated():
    b = pickle.load(open(GNN_ROOT / "data/processed/graphs" / f"{EPISODE}.pkl", "rb"))
    nodes, origin = b["active_node_strs"], b["origin"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes)
    p = json.loads((OUT / "join_summary.json").read_text())["calibrated_params"]
    net = ContagionNetwork(nodes=nodes, W=W, coupling=p["coupling"], kappa=p["kappa"],
                           common=p["common"], sigma=p["sigma"])
    return net, nodes, origin, float(p["shock"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net, nodes, origin, shock = load_calibrated()
    victims = [n for n in nodes if n != origin]
    base = net.contagion_over(origin, shock, victims)

    # kappa multipliers: from 1.0 (no intervention) to full backstop
    kappa_mults = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1e6]
    targets = ["USDC/binance", "BUSD/binance"]
    rows = []

    for target in targets:
        short = target.split("/")[0]
        mcap = MCAP_PROXY.get(short, 1.0)
        for mult in kappa_mults:
            if mult >= 1e5:
                # approximate full backstop with protect=
                c = net.contagion_over(origin, shock, victims, protect=target)
                label = "full_backstop"
            else:
                c = net.contagion_over(origin, shock, victims,
                                       kappa_scale={target: mult})
                label = f"kappa_x{mult:.1f}".rstrip("0").rstrip(".")
            red = 100.0 * (base - c) / base if base > 0 else 0.0
            cost_proxy = (mult - 1.0) * mcap if mult < 1e5 else (50.0 * mcap)
            rows.append({
                "target": short,
                "kappa_mult": mult if mult < 1e5 else float("inf"),
                "intensity_label": label,
                "contagion": round(c, 6),
                "pct_reduction": round(red, 1),
                "cost_proxy_bUSD": round(cost_proxy, 1),
            })
        print(f"\n{short}:")
        for r in rows[-len(kappa_mults):]:
            print(f"  {r['intensity_label']:20s}  reduction={r['pct_reduction']:6.1f}%  "
                  f"cost~{r['cost_proxy_bUSD']:.0f}B")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "partial_backstop.csv", index=False)

    # efficient frontier: minimum multiplier per threshold per target
    thresholds = [50.0, 80.0, 95.0]
    frontier = {}
    for short in ["USDC", "BUSD"]:
        sub = df[df["target"] == short].sort_values("kappa_mult")
        frontier[short] = {}
        for thr in thresholds:
            above = sub[sub["pct_reduction"] >= thr]
            if len(above):
                row = above.iloc[0]
                frontier[short][f"pct_{int(thr)}"] = {
                    "min_kappa_mult": row["kappa_mult"],
                    "label": row["intensity_label"],
                    "cost_proxy_bUSD": row["cost_proxy_bUSD"],
                }
            else:
                frontier[short][f"pct_{int(thr)}"] = None

    print("\n=== EFFICIENT FRONTIER ===")
    for node, f in frontier.items():
        print(f"\n{node}:")
        for thr, info in f.items():
            if info:
                print(f"  {thr}: kappa×{info['min_kappa_mult']}, cost~${info['cost_proxy_bUSD']:.0f}B")
            else:
                print(f"  {thr}: never achieved")

    (OUT / "partial_backstop_frontier.json").write_text(json.dumps(frontier, indent=2))
    _plot(df, OUT / "fig_partial_backstop.png")
    print(f"\n=> partial_backstop.csv written to {OUT / 'partial_backstop.csv'}")


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.apply()

    fig, ax = plt.subplots(figsize=ps.SINGLE)

    color_map = {"USDC": ps.GREEN, "BUSD": ps.RED}
    for short, grp in df.groupby("target"):
        finite = grp[np.isfinite(grp["kappa_mult"])]
        mults = list(finite["kappa_mult"]) + [finite["kappa_mult"].max() * 2.5]
        reds = list(finite["pct_reduction"]) + [
            float(df[(df["target"] == short) & (df["kappa_mult"] > 1e5 - 1)]["pct_reduction"].iloc[-1])
        ]
        ax.plot(mults[:-1], reds[:-1], "o-", color=color_map[short],
                linewidth=2, markersize=5, label=f"{short} ({'causal origin' if short == 'USDC' else 'spurious hub'})")
        ax.scatter([mults[-1]], [reds[-1]], marker="*", s=120, color=color_map[short], zorder=5)

    for thr in [50, 80, 95]:
        ax.axhline(thr, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.text(51, thr + 1.5, f"{thr}%", color="gray", fontsize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Reserve-strengthening multiplier (kappa ×)")
    ax.set_ylabel("Contagion reduction (%)")
    ax.set_title("Partial backstop cost-effectiveness: causal origin vs spurious hub")
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("figure ->", path)


if __name__ == "__main__":
    main()
