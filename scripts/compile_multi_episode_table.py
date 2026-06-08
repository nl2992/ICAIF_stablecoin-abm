"""
Plan A — Multi-Episode Agreement Table.

Loads multi_episode_join.csv (5 episodes), classifies each as high/low contagion,
and produces a LaTeX table comparing GNN predicted hub vs ABM causal driver.

Outputs -> experiments/results/netcontagion/
    multi_episode_table.tex    LaTeX table for paper §4
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("experiments/results/netcontagion")


def classify(row: pd.Series) -> str:
    if row["low_contagion"]:
        return "Low"
    return "High"


def agreement_str(row: pd.Series) -> str:
    if row["low_contagion"]:
        return "—"
    v = row["gnn_top_is_causal_top"]
    if v is True or str(v).lower() == "true":
        return r"\checkmark"
    return r"\ding{55}"


def spurious_str(row: pd.Series) -> str:
    v = row.get("spurious_hub", "")
    if pd.isna(v) or str(v).strip() in ("", "nan"):
        return "—"
    return str(v)


def rho_str(row: pd.Series) -> str:
    v = row.get("spearman_pred_vs_causal", float("nan"))
    try:
        f = float(v)
        if np.isfinite(f):
            return f"{f:.2f}"
    except (TypeError, ValueError):
        pass
    return "—"


def format_episode(ep: str) -> str:
    mapping = {
        "USDC_SVB": r"\textsc{USDC/SVB}",
        "UST_Terra": r"\textsc{UST/Terra}",
        "USDT_May2022": r"\textsc{USDT/May-2022}",
        "DAI_FTX": r"\textsc{DAI/FTX}",
        "BUSD_winddown": r"\textsc{BUSD/Winddown}",
    }
    return mapping.get(ep, ep)


def main():
    df = pd.read_csv(OUT / "multi_episode_join.csv")

    rows_tex = []
    for _, row in df.iterrows():
        ep = format_episode(row["episode"])
        cls = classify(row)
        gnn_hub = row["gnn_top_hub"] if not row["low_contagion"] else "—"
        abm_top = str(row.get("abm_causal_top", "—"))
        if pd.isna(abm_top) or abm_top in ("nan", ""):
            abm_top = "—"
        agree = agreement_str(row)
        spurious = spurious_str(row)
        rho = rho_str(row)
        note = ""
        if row["episode"] == "UST_Terra":
            note = r"\textsuperscript{$\dagger$}"
        rows_tex.append(
            f"    {ep} & {cls} & {gnn_hub} & {abm_top} & ${agree}${note} & {spurious} & {rho} \\\\"
        )

    table = r"""\begin{table}[t]
\centering
\caption{Multi-episode GNN vs ABM hub agreement.
  \checkmark\ = GNN top hub equals ABM causal driver (non-origin node);
  \ding{55}\ = divergence (spurious hub).
  $^\dagger$For UST/Terra all non-origin nodes have causal~$\Delta=0$ (no relay structure);
  the agreement is a tie-breaking artefact. Cross-crisis test (§\ref{sec:rl}) confirms
  protecting the GNN's USDC pick achieves $0\%$ contagion reduction — the origin (UST)
  is the only effective intervention target.}
\label{tab:multi_episode}
\begin{tabular}{lllllcc}
\toprule
Episode & Contagion & GNN hub & ABM causal & Agree & Spurious & Spearman~$\rho$ \\
\midrule
""" + "\n".join(rows_tex) + r"""
\midrule
\multicolumn{7}{l}{\small Low-contagion episodes excluded from the causal test (model not applicable when peak depeg $<2\%$).} \\
\bottomrule
\end{tabular}
\end{table}
"""

    (OUT / "multi_episode_table.tex").write_text(table)
    print(table)

    # summary stats
    high = df[~df["low_contagion"]]
    n_agree = int((high["gnn_top_is_causal_top"].astype(str).str.lower() == "true").sum())
    n_diverge = int((high["gnn_top_is_causal_top"].astype(str).str.lower() == "false").sum())
    n_spurious = int(high["spurious_hub"].notna().sum())
    print(f"\nSummary: {len(high)} high-contagion episodes, "
          f"{n_agree} agree, {n_diverge} diverge, {n_spurious} with spurious hub identified.")
    print(f"=> multi_episode_table.tex written to {OUT / 'multi_episode_table.tex'}")


if __name__ == "__main__":
    main()
