# Claims and Evidence

What the paper argues, and for every headline number, the committed file it comes from. Artifacts live
under `experiments/results/netcontagion/`.

## The narrative

A correlational model — network centrality, or a graph neural network's attention — tells you that
during the March-2023 SVB crisis BUSD was the most central contagion hub. A supervisor with a limited
backstop budget would protect BUSD first. That reading is wrong, and wrong in a way that wastes the
budget: centrality rewards a venue for *moving with* a crisis, which is not the same as *causing* it to
spread. A coin can plunge in lockstep with a panic while transmitting nothing to anyone.

We build a calibrated, intervenable model of stablecoin contagion and ask the counterfactual directly:
if we had held venue X at its peg, how much contagion would the other venues have been spared? The
knockout is unambiguous. BUSD's causal effect is exactly zero — no stablecoin held BUSD as backing, so
it was mechanically incapable of transmitting stress — while protecting the crisis origin (USDC) removes
all of it, and protecting the true relay DAI removes 98%. The predicted hub ranking is *negatively*
correlated with the true causal ranking: the most correlationally central venue is the least causally
relevant.

The result survives four robustness checks (±30% calibration perturbation, strategic two-agent markets,
a switch from the price-estimated network to a documented reserve-exposure network, and a synthetic
placebo with known ground truth), and an independent DebtRank computation agrees. A reinforcement-learning
regulator, given only structural wiring and no causal labels, learns to fund the causal venues and
allocate nothing to the spurious hub.

The contribution is a transferable audit protocol: any correlational hub-ranking method can be tested
against a calibrated, intervenable model the same way. The scope is single-crisis depth for the headline
BUSD refutation (honestly disclosed), and the calibration is a five-parameter reduced form, not a
microfounded market.

## Where each number lives

| Claim | Number | File | Field / row |
|---|---|---|---|
| Calibration: 3 of 4 moments within tolerance | contagion mag 0.1376, half-life 116, ρ 0.576 (vol-floor caveat) | `calibration_moments.csv`, targets in `../../configs/calibration_targets.json` | per-moment empirical vs simulated |
| BUSD causal Δ=0, USDC 100%, DAI 98% | 0% / 100% / 97.9% | `intervention_sweep.csv` | `targeted_protection` rows (USDC, DAI, BUSD) |
| Spearman predicted vs causal | −0.54 harmonized, −0.77 SVB-specific (n=4, p=0.23) | `multi_episode_join.csv` (−0.544), `join_summary.json` (`spearman_pred_vs_causal`=−0.7746, `spearman_p`=0.2254) | SVB row |
| ±30% robustness | USDC top-causal 100% of draws, BUSD inert 100%; USDC Δ mean 0.033 (≈ full baseline) | `calibration_uncertainty.json`, `robustness_summary.json` | per-draw rankings |
| Welfare matrix: protect USDC → all victims 0; protect BUSD identical | DAI 0.138→0, USDC 0.103→0; BUSD max-Δ 0 | `welfare_matrix.csv`, `welfare_analysis.json` | `protect_USDC` / `protect_BUSD` rows |
| DebtRank agrees | USDC 0.076 (#1), BUSD 0.000 (#6) | `debtrank_validation.json` | `debtrank_scores`, `debtrank_ranking` |
| Intervention sweep | USDC 100, gating 100, reserve×20 89, breaker 85, BUSD 0 | `intervention_sweep.csv` | `pct_reduction` |
| Dose-response (reserve strengthening) | 5× $174M/64%, 20× $827M/89%, 50× $2.1B/95% | `partial_backstop.csv`, `partial_backstop_frontier.json` | `cost_proxy_bUSD`, `pct_reduction`; BUSD 0 at every cost |
| K=1 / K=2 budget allocation | GNN-pick BUSD 0%, ABM-pick USDC 100%; K=2 both 100% (different sets) | `budget_allocation.csv` | `gnn_guided` / `abm_guided` / `rl_regulator` |
| RL regulator | 93.7% (headline run), 93.6±0.1% (5-seed); origin flag not load-bearing | `rl_regulator.json`, `rl_no_origin_flag.json` | reduction + per-seed allocations |
| Two-agent robustness; redeemer 7.7×, breaker 63→95% | baseline C per config | `two_agent_robustness.csv`, `adaptive_robustness.csv` | per-config C; circuit-breaker flip |
| Placebo / negative control | upstream A 0.148, B 0.131 (large); terminal C and spurious SPUR both 0 | `placebo_control.csv`, `placebo_control.json` | `causal_delta_true_W`, `is_true_transmitter` / `is_planted_spurious` |
| Near-criticality (exact, sub-critical) | netted W acyclic ⇒ spectral radius 1−κ=0.994, half-life 116 | `near_criticality.json` | acyclicity + spectral radius |
| Cross-crisis generalization | top hub zero-out-exposure in 3 of 5 episodes | `multi_episode_join.csv` | `gnn_top_hub`, `spurious_hub` per episode |
| Terra is algorithmic (contrast) | all knockouts Δ≈0 | `terra_case_study.json` | `all_deltas_near_zero` |
| Balance-sheet variant calibrated separately, same verdict | both versions BUSD=0, USDC #1 | `exposure_calibration.csv`, `exposure_join.json` | side-by-side params |

The contagion-magnitude target (0.1376) is the companion GNN repo's measurement of the real USDC/SVB peak
depeg; the other moments are statistics of the real price data. All numbers regenerate from `scripts/`.
