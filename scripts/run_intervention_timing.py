"""
Plan C — Intervention Timing Sensitivity.

Shows how the benefit of a USDC backstop decays as the regulatory response is delayed
past the shock onset (step 40). Each step ≈ 5 minutes of real time.

Delay = 0  → backstop applied at the same step as the shock (ideal)
Delay = D  → backstop applied D steps (= D×5 min) after the shock

Outputs -> experiments/results/netcontagion/
    intervention_timing.csv    delay_steps, delay_min, contagion, pct_reduction
    fig_intervention_timing.png effectiveness curve with 50%-threshold marked
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
SHOCK_STEP = 40
MIN_PER_STEP = 5   # each model step ≈ 5 minutes


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

    # baseline: no intervention
    base = net.contagion_over(origin, shock, victims, shock_step=SHOCK_STEP)

    delay_grid = [0, 5, 10, 20, 40, 80, 120]
    rows = []
    for delay in delay_grid:
        backstop_step = SHOCK_STEP + delay
        # protect USDC starting from backstop_step (protect_from_step)
        c = net.contagion_over(origin, shock, victims,
                               protect=origin,
                               shock_step=SHOCK_STEP,
                               protect_from_step=backstop_step)
        red = 100.0 * (base - c) / base if base > 0 else 0.0
        rows.append({
            "delay_steps": delay,
            "delay_min": delay * MIN_PER_STEP,
            "backstop_step": backstop_step,
            "contagion": round(c, 6),
            "pct_reduction": round(red, 1),
        })
        print(f"  delay={delay:3d} steps ({delay*MIN_PER_STEP:4d} min): "
              f"backstop@{backstop_step}, contagion={c:.4f}, reduction={red:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "intervention_timing.csv", index=False)
    print("\n=== INTERVENTION TIMING ===")
    print(df.to_string(index=False))

    # find the D (steps/minutes) at which effectiveness first falls below 50%
    threshold_50 = df[df["pct_reduction"] >= 50.0]
    if len(threshold_50):
        d50_steps = int(threshold_50["delay_steps"].max())
        d50_min = int(threshold_50["delay_min"].max())
        print(f"\n50%-effectiveness threshold: delay ≤ {d50_steps} steps = {d50_min} minutes")
        companion_lead_min = 24 * 60  # 1440 minutes
        margin_min = companion_lead_min - d50_min
        print(f"Companion GNN 24h horizon leaves {margin_min} min margin "
              f"before the {d50_min}-min effectiveness threshold.")

    _plot(df, OUT / "fig_intervention_timing.png")
    print(f"\n=> intervention_timing.csv written to {OUT / 'intervention_timing.csv'}")


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.apply()

    fig, ax = plt.subplots(figsize=ps.SINGLE)
    ax.plot(df["delay_min"], df["pct_reduction"], "o-", color=ps.BLUE,
            linewidth=2, markersize=6, label="USDC backstop")
    ax.axhline(50, color=ps.RED, lw=1.2, ls="--", label="50% effectiveness")
    ax.axhline(0, color="k", lw=0.5, ls=":")

    # mark 50%-threshold
    above50 = df[df["pct_reduction"] >= 50.0]
    if len(above50):
        d50 = int(above50["delay_min"].max())
        ax.axvline(d50, color=ps.RED, lw=0.8, ls=":", alpha=0.7)
        ax.text(d50 + 5, 52, f"{d50} min", color=ps.RED, fontsize=8)

    # mark companion paper's 24h early warning
    companion_min = 1440
    ax.axvline(companion_min, color="gray", lw=0.8, ls="--", alpha=0.5, label="GNN 24h horizon")

    ax.set_xlabel("Response delay after shock onset (minutes)")
    ax.set_ylabel("Contagion reduction (%)")
    ax.set_title("Intervention timing: backstop effectiveness vs delay")
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("figure ->", path)


if __name__ == "__main__":
    main()
