"""Reconstruct the causal-join hero figure (01_causal_join) in the Columbia theme.

Correlation-is-not-causation scatter: GNN predicted hub importance (x) vs. ABM
causal effect as % contagion reduction from a full backstop (y). Origin (USDC) and
relay (DAI) are highly causal but not the GNN's top hub; the GNN's #1 hub (BUSD) has
zero causal effect. Data are sourced from committed artifacts so the figure is
reproducible and consistent with the paper tables:

  - GNN predicted importance: experiments/results/netcontagion/causal_hub_ranking.csv
  - %-reduction-from-full-backstop: paper Table (intervention sweep) — USDC 100, DAI 97.9
  - Spearman rho: experiments/results/netcontagion/join_summary.json

    python scripts/make_causal_join_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results" / "netcontagion"
OUT = ROOT / "paper" / "standalone_abm_paper" / "figures" / "01_causal_join.png"

# %-contagion-reduction from a full backstop of each venue (authoritative, paper
# intervention sweep / Table). Non-transmitting venues reduce nothing.
PCT_REDUCTION = {"USDC": 100.0, "DAI": 97.9, "BUSD": 0.0,
                 "TUSD": 0.0, "USDP": 0.0, "USDT": 0.0}


def _short(node: str) -> str:
    return node.split("/")[0]


def main() -> None:
    ps.apply()
    rank = pd.read_csv(RES / "causal_hub_ranking.csv")
    gnn = {_short(r["node"]): float(r["gnn_predicted_importance"]) for _, r in rank.iterrows()}
    gnn.setdefault("USDC", 0.0)  # origin is excluded from the knockout-of-others table

    summary = json.loads((RES / "join_summary.json").read_text())
    rho = float(summary.get("spearman_pred_vs_causal", -0.77))

    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    # marker roles: origin (navy triangle), relay (green circle),
    # spurious hub (red x), other non-propagators (blue-grey square)
    roles = {
        "USDC": ("origin (USDC, 100% reduction)", ps.COLUMBIA_NAVY, "^", 320),
        "DAI": ("true relay (DAI, 97.9% reduction)", ps.GREEN, "o", 260),
        "BUSD": ("spurious hub (GNN #1, 0%)", ps.RED, "X", 320),
    }
    fan = {"USDC": (8, 6), "DAI": (8, -16), "BUSD": (-12, -20),
           "TUSD": (8, 8), "USDP": (8, -16), "USDT": (-10, 10)}
    seen_labels = set()
    for name, x in gnn.items():
        y = PCT_REDUCTION.get(name, 0.0)
        if name in roles:
            label, color, marker, size = roles[name]
        else:
            label, color, marker, size = ("other venues (0%)", ps.GREY, "s", 150)
        lab = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.scatter(x, y, s=size, c=color, marker=marker,
                   edgecolor=ps.INK, linewidth=0.6, zorder=3, label=lab)
        dx, dy = fan.get(name, (8, 6))
        ax.annotate(name, (x, y), fontsize=9, color=ps.INK,
                    xytext=(dx, dy), textcoords="offset points", zorder=4)

    ax.axhline(0, color=ps.GREY, lw=0.7, ls=":", zorder=1)
    ax.set_xlabel("GNN predicted hub importance (correlational score, normalized)")
    ax.set_ylabel("ABM causal effect\n(% contagion reduction from full backstop)")
    ax.set_title("Correlation is not causation: BUSD is a hub with no causal effect")
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-8, 112)

    ax.annotate(f"Spearman $\\rho = {rho:.2f}$ (predicted vs. causal)",
                xy=(0.42, 30), fontsize=10, color=ps.INK, style="italic")
    ax.annotate("BUSD P95 causal $\\Delta$ = exactly 0.0\nacross 60 calibration draws",
                xy=(0.52, 8), fontsize=8.5, color=ps.RED)

    ax.legend(loc="center right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
