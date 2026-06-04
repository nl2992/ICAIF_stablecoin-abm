"""
Documented balance-sheet exposure network.

The lead-lag transmission matrix (``estimate_transmission_matrix``) is *estimated* from
observational price data, so a skeptic can call the resulting causal knockout circular. This
module instead encodes the DIRECTED transmission network from PUBLICLY DOCUMENTED reserve /
collateral linkages: a depeg in asset j mechanically propagates to asset i if and only if i
holds j as reserve backing. That is a balance-sheet fact, not a statistical inference, so the
counterfactual built on it is mechanically grounded rather than correlational.

Exposures (asset i -> holds fraction of j as backing), early-2023 / SVB window:

  DAI   : MakerDAO's Peg-Stability-Module was ~50%+ USDC-collateralised — the documented
          reason DAI depegged with USDC during SVB.                      DAI  <- USDC ~0.50
  FRAX  : fractional-algorithmic, ~90% USDC-collateralised.             FRAX <- USDC ~0.90
  USDP  : Paxos reserves = cash + US T-bills, no stablecoin collateral.  (no incoming)
  TUSD  : independent attested reserves, no stablecoin collateral.       (no incoming)
  BUSD  : Paxos-issued, independent reserves; critically, NO other       (no incoming,
          stablecoin holds BUSD as backing.                              no outgoing)
  USDT  : independent reserves (cash/T-bills/CP); not backing for the    (negligible)
          fiat-backed coins here.
  USDC  : the dominant reserve asset others hold -> high OUTGOING exposure; itself shocked
          exogenously by the SVB cash exposure (the episode origin).

Key consequence: **BUSD has zero outgoing exposure** — no stablecoin's peg depends on it — so
protecting/strengthening BUSD cannot reduce contagion, regardless of how central it looks in
the correlation graph. That is the mechanical reason the GNN's top hub is spurious.

Sources: MakerDAO PSM composition disclosures; Frax Finance protocol docs; Paxos/Circle
reserve attestations (2022-2023). Fractions are order-of-magnitude documented values; the
qualitative network (who is exposed to whom) is what the causal result depends on, and the
result is shown to be robust to the exact magnitudes.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# asset i -> {asset j: fraction of i's backing held in j}
DOCUMENTED_EXPOSURES: Dict[str, Dict[str, float]] = {
    "DAI": {"USDC": 0.50},
    "FRAX": {"USDC": 0.90},
    # all others: independent reserves (no stablecoin collateral) -> empty
    "USDP": {}, "TUSD": {}, "BUSD": {}, "USDT": {}, "USDC": {}, "FDUSD": {}, "UST": {},
}


def exposure_matrix(nodes: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Directed transmission W[i,j] = fraction of receiver i's backing held in sender j.

    `nodes` are "ASSET/venue" strings. Returns (W, out_exposure) where out_exposure[asset]
    is the total fraction of OTHER coins' reserves held in that asset (its systemic
    importance as collateral). Rows are left UN-normalised: the exposure fraction is itself
    the mechanical pass-through gain (a 1% USDC depeg moves a 50%-USDC-backed coin ~0.5%).
    """
    n = len(nodes)
    assets = [s.split("/")[0] for s in nodes]
    W = np.zeros((n, n))
    for i, ai in enumerate(assets):
        for j, aj in enumerate(assets):
            if i == j:
                continue
            frac = DOCUMENTED_EXPOSURES.get(ai, {}).get(aj, 0.0)
            if frac > 0:
                W[i, j] = frac
    out_exposure = {a: float(W[:, k].sum()) for k, a in enumerate(assets)}
    return W, out_exposure
