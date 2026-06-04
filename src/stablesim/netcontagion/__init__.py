"""Reduced-form networked-contagion engine for per-asset stablecoin depeg dynamics.

Replaces the AMM-only market for the causal-counterfactual analysis: an AMM is
*bimodal* (it either resists a depeg or collapses), so depeg magnitude is not
smoothly controllable and the SMM cannot match the empirical moments.  A networked
mean-reverting (OU) peg process per asset, with a DIRECTED transmission network
estimated from real lead-lag structure, is smoothly calibratable AND supports clean
per-node knockout counterfactuals — exactly what the GNN-vs-ABM hub-agreement join
needs.
"""

from .model import ContagionNetwork, estimate_transmission_matrix, episode_moments

__all__ = ["ContagionNetwork", "estimate_transmission_matrix", "episode_moments"]
