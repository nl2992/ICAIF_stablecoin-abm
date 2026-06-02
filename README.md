# stablecoin-abm

Agent-based stablecoin market simulator with reinforcement-learning policies and intervention analysis.

## Research question

Which policy interventions — reserve transparency, redemption gating, circuit breakers, LP incentives — reduce peg-depeg contagion and at what cost to which agents?

## Lineage

| Upstream repo | Role here |
|---|---|
| `stablecoin-contagion-network` (StressBench + IAQF metrics) | Shock schedule + calibration targets (OU half-lives, propagation ρ̂) |
| Gu et al. (PyMarketSim) | Market-mechanism reference; PPO spoofing agent design |
| JaxMARL-HFT / JAX-LOB | Optional GPU backend for large scenario sweeps |

## Architecture

```
src/stablesim/
  engine/       market mechanism — stableswap AMM, order book, reserve model
  agents/       arbitrageurs, redeemers, LPs, issuer/reserve, noise traders
  scenarios/    StressBench shock loader + exogenous event schedule
  rl/           Gymnasium env wrapper + PPO training (stable-baselines3)
  experiments/  intervention knobs + scenario × intervention sweep runner
  calibration/  match OU half-lives / propagation ρ̂ from empirical data
  analysis/     metrics (half-life, contagion magnitude, welfare by agent)
```

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# 1. Calibrate to empirical stylized facts
make calibrate

# 2. Train RL arbitrageur/redeemer policies
make train

# 3. Sweep interventions × StressBench scenarios
make sweep
```

## Intervention knobs

| Knob | Parameter | Range |
|---|---|---|
| Reserve transparency | `transparency_freq`, `transparency_noise` | {daily, weekly, none} × σ |
| Redemption gating | `gate_fee`, `gate_queue_len`, `gate_delay` | [0, 5%] × [0, ∞) × [0, 72h] |
| Circuit breaker | `cb_threshold`, `cb_duration` | depeg % × minutes |
| LP incentives | `lp_subsidy_rate` | [0, 1%] per block |

## Outcome metrics

- **Contagion magnitude**: peak cross-venue depeg spread during shock
- **Peg-recovery half-life** (OU): calibration target and post-intervention comparison
- **LP impermanent loss**: per-episode Δ vs. hold
- **Welfare by agent type**: net P&L decomposition across arbitrageurs, LPs, redeemers

## Validation strategy

No-intervention runs must reproduce the empirical half-lives and ρ̂ propagation patterns from `stablecoin-contagion-network` before intervention sweeps are trusted. See `calibration/`.
