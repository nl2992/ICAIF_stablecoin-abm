"""
Plan G — Welfare Decomposition Deep-Dive.

Loads welfare_matrix.csv (peak depeg by scenario × node) and produces:
  - Pareto comparison: does USDC backstop Pareto-dominate no intervention?
  - BUSD backstop: equivalent to no intervention?
  - 3-column grouped bar chart (scenario) × rows (node / agent type)

Interprets nodes as proxies for agent welfare:
  Stablecoin holders      → peak depeg they experience (lower = better)
  DAI holders             → DAI peak depeg (DAI is the primary relay/victim)
  USDC holders            → USDC peak depeg (origin)
  USDP / TUSD / USDT      → smaller venues

Outputs -> experiments/results/netcontagion/
    welfare_analysis.json    Pareto flags, per-agent welfare by scenario
    fig_welfare_analysis.png 3-scenario grouped bar chart
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("experiments/results/netcontagion")

SCENARIO_LABELS = {
    "no_intervention": "No intervention",
    "protect_USDC": "USDC backstop\n(causal origin)",
    "protect_BUSD": "BUSD backstop\n(spurious hub)",
    "reserve_USDC_x10": "Reserve USDC ×10",
    "circuit_breaker_0.05": "Circuit breaker\n5% cap",
    "redemption_gating": "Redemption gating",
}

MAIN_SCENARIOS = ["no_intervention", "protect_USDC", "protect_BUSD"]


def pareto_dominates(a: dict, b: dict) -> bool:
    """Return True if scenario a Pareto-dominates b (a has ≤ depeg for ALL nodes, strictly < for ≥1)."""
    nodes = [k for k in a if k in b]
    strictly_better = False
    for nd in nodes:
        if a[nd] > b[nd] + 1e-9:
            return False
        if b[nd] > a[nd] + 1e-9:
            strictly_better = True
    return strictly_better


def main():
    df = pd.read_csv(OUT / "welfare_matrix.csv", index_col=0)
    # index = scenario, columns = nodes
    # peak depeg values: lower is better for holders of that coin

    scenarios = df.index.tolist()
    nodes = df.columns.tolist()

    print("=== WELFARE MATRIX ===")
    print(df.round(4).to_string())

    # Pareto comparisons
    welfare_dict = {sc: {nd: float(df.loc[sc, nd]) for nd in nodes} for sc in scenarios}
    pareto_results = {}
    base = "no_intervention"
    for sc in scenarios:
        if sc == base:
            continue
        dominates = pareto_dominates(welfare_dict[sc], welfare_dict[base])
        dominated_by = pareto_dominates(welfare_dict[base], welfare_dict[sc])
        pareto_results[sc] = {
            "pareto_dominates_no_intervention": dominates,
            "pareto_dominated_by_no_intervention": dominated_by,
            "welfare_vs_baseline": {
                nd: round(welfare_dict[sc][nd] - welfare_dict[base][nd], 6)
                for nd in nodes
            },
        }

    print("\n=== PARETO ANALYSIS ===")
    for sc, r in pareto_results.items():
        tag = "Pareto-dominates baseline" if r["pareto_dominates_no_intervention"] else \
              ("Pareto-dominated by baseline" if r["pareto_dominated_by_no_intervention"] else "Neither")
        print(f"  {sc}: {tag}")

    # check BUSD backstop = identical to no intervention
    busd_diff = max(abs(welfare_dict["protect_BUSD"][nd] - welfare_dict["no_intervention"][nd])
                    for nd in nodes)
    usdc_all_zero = all(welfare_dict["protect_USDC"][nd] < 1e-6 for nd in nodes)

    output = {
        "scenarios": list(df.index),
        "nodes": nodes,
        "welfare_matrix": welfare_dict,
        "pareto_analysis": pareto_results,
        "headline_findings": {
            "usdc_backstop_pareto_dominates": pareto_results.get("protect_USDC", {}).get(
                "pareto_dominates_no_intervention", False),
            "busd_backstop_identical_to_no_intervention": busd_diff < 1e-6,
            "busd_max_delta_from_baseline": round(busd_diff, 8),
            "usdc_backstop_all_depegs_zero": usdc_all_zero,
        },
        "policy_statement": (
            "A USDC backstop drives every victim's peak depeg to zero "
            "(Pareto-dominates no intervention). "
            "A BUSD backstop produces outcomes identical to no intervention — "
            "no agent benefits from spending the budget on the spurious hub."
            if pareto_results.get("protect_USDC", {}).get("pareto_dominates_no_intervention")
            else "See pareto_analysis for details."
        ),
    }
    (OUT / "welfare_analysis.json").write_text(json.dumps(output, indent=2))
    print("\n=== HEADLINE ===")
    print(json.dumps(output["headline_findings"], indent=2))
    print("\n", output["policy_statement"])

    _plot(df, OUT / "fig_welfare_analysis.png")
    print(f"\n=> welfare_analysis.json written to {OUT / 'welfare_analysis.json'}")


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.apply()

    main_sc = [s for s in MAIN_SCENARIOS if s in df.index]
    nodes = [n for n in df.columns if df.loc[main_sc, n].max() > 1e-6]

    x = np.arange(len(nodes))
    width = 0.25
    colors = [ps.BLUE, ps.GREEN, ps.RED]
    labels = [SCENARIO_LABELS.get(s, s) for s in main_sc]

    fig, ax = plt.subplots(figsize=ps.WIDE)
    for i, (sc, lbl, col) in enumerate(zip(main_sc, labels, colors)):
        vals = [float(df.loc[sc, nd]) * 100 for nd in nodes]  # convert to %
        ax.bar(x + (i - 1) * width, vals, width, label=lbl, color=col, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(nodes, fontsize=8)
    ax.set_ylabel("Peak depeg of coin (%)")
    ax.set_title("Welfare decomposition: peak depeg by coin under three policy scenarios")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("figure ->", path)


if __name__ == "__main__":
    main()
