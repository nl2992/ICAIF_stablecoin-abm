# TODO — stablecoin-abm

Goal: a calibrated agent-based stablecoin market that **causally validates** the GNN's hub
predictions and measures which interventions reduce depeg contagion. Template paper: Gu, Wang,
Wellman et al. — *The Effect of Liquidity on the Spoofability of Financial Markets* (ICAIF'24).
Killer framing: **"From Correlation to Causation."**

---

## Current State *(as of June 2026 — update on next run)*

All core experiments are **complete** and committed to `experiments/results/netcontagion/`:

| Artifact | Status | Key number |
|---|---|---|
| `calibration_moments.csv` | ✅ done | 4/4 moments within tolerance |
| `causal_hub_ranking.csv` | ✅ done | BUSD causal effect = 0.0%, USDC = 100% |
| `intervention_sweep.csv` | ✅ done | USDC full backstop = 100% reduction; BUSD = 0% |
| `rl_regulator.json` | ✅ done | PPO puts 1.0 on USDC, 0.0 on BUSD → 93.7% reduction |
| `multi_episode_join.csv` | ✅ done | 5 episodes: SVB spurious hub confirmed; Terra GNN correct |
| `two_agent_robustness.json` | ✅ done | Strategic agents do not reverse the causal verdict |
| `robustness_summary.json` | ✅ done | ±30% calibration uncertainty preserves BUSD=0 finding |
| `placebo_control.csv` | ✅ done | Synthetic placebo hub also has ~0 causal effect |
| `welfare_matrix.csv` | ✅ done | Welfare decomposition by agent type |

The paper (`paper/standalone_abm_paper/main.tex`) incorporates all of the above. The BUSD=0
causal result is the headline. The RL regulator independently learns to protect USDC and
ignore BUSD. The four robustness checks all pass.

**What is missing** is (a) richer multi-episode analysis now that 5 episodes are in
`multi_episode_join.csv`, (b) budget-constrained policy optimization, (c) timing sensitivity
of intervention, (d) RL training convergence evidence, and (e) a stronger policy section tied
to GENIUS Act / MiCA. Plans below are ordered by expected reviewer impact.

---

## Plan A — Multi-Episode Agreement Table  *(generalize beyond SVB)*

**Goal**: The paper's headline is SVB-specific (BUSD=0). `multi_episode_join.csv` has 5
episodes. Two show spurious hubs (SVB: BUSD spurious; USDT_May2022: USDC spurious). One shows
GNN is correct (UST_Terra: GNN top = USDC = ABM causal top). Two are low-contagion events
(DAI_FTX, BUSD_winddown). Turn this into a formal multi-episode agreement table.

**Code to write**: `scripts/compile_multi_episode_table.py`
```
Load: experiments/results/netcontagion/multi_episode_join.csv
For each episode:
  - classify: high_contagion/low_contagion
  - record: GNN top hub, ABM causal top, agreement (True/False), Spearman ρ if available
  - note: spurious_hub if present
Print: LaTeX table \label{tab:multi_episode}
Save: experiments/results/netcontagion/multi_episode_table.tex
```

**Execute**:
```
python scripts/compile_multi_episode_table.py
```

**Target result**: Table with 5 rows showing: 2 high-contagion episodes with spurious hubs
confirmed; 1 episode where GNN IS correct (and explain why: UST origin is USDC, so GNN
correctly flagged USDC); 2 low-contagion episodes skipped (model not applicable when
contagion < threshold). The mixed finding is honest and actually STRONGER than always finding
spurious hubs — it validates the ABM as a real test, not one that always says "GNN is wrong."

**Write into paper**: New Table in §4 (Multi-Episode Validation), replacing the current
single-episode framing. Key sentence: "Of the three high-contagion episodes, two contain a
spurious GNN hub; in the third (UST/Terra), the GNN hub \emph{is} the origin — a concordance
that validates the ABM rather than contradicting it." This makes the paper more honest and
more robust.

---

## Plan B — Budget-Constrained Optimal Allocation  *(the policy-design result)*

**Goal**: A regulator cannot always protect every venue. Given a budget covering only K venues
(K=1, 2, 3), what is the optimal protection allocation? Compare: GNN-guided (protect top K
correlational hubs) vs ABM-guided (protect top K causal venues) vs RL regulator's implicit
allocation.

**Code to write**: `scripts/run_budget_optimization.py`
```
From intervention_sweep.csv, for K=1,2,3:
  Greedy optimal K-subset: enumerate all C(n,K) combinations, pick max contagion reduction
  GNN-guided K-subset: protect top K by predicted (correlational) importance
  ABM-guided K-subset: protect top K by causal Delta_X
  RL regulator: allocate to top K venues by rl_regulator.json learned_allocation scores

Compute: contagion_reduction_pct for each strategy at each K
Save: experiments/results/netcontagion/budget_allocation.csv

For K=1:  ABM=100%, GNN=0%, RL=~94%
For K=2:  ABM≈100% (USDC+DAI=99.7%), GNN=?%, RL=?%
```

**Execute**:
```
python scripts/run_budget_optimization.py
```

**Target result**: At K=1: ABM-guided = 100%, GNN-guided = 0%, RL ≈ 94%. At K=2: ABM-guided
≈ 100% (USDC+DAI), GNN-guided still < 50% (protecting correlational hubs). This becomes the
quantitative policy recommendation: "a regulator following the correlational ranking wastes
its entire first-venue budget."

**Write into paper**: New Table in §5 (Policy Implications), titled "Contagion reduction (%)
under three allocation strategies, by budget K." Add sentence: "A regulator following the GNN
ranking achieves $X\%$ reduction with K=1 and $Y\%$ with K=2; the ABM-guided regulator
achieves $100\%$ with K=1 alone."

---

## Plan C — Intervention Timing Sensitivity  *(justify early-warning → early-intervention)*

**Goal**: Show that intervention benefit decays as the response is delayed past crisis onset.
A practical regulator can't act at t=0 (shock); they need to detect first, then act. How many
steps of delay before the intervention becomes ineffective?

**Code to write**: `scripts/run_intervention_timing.py`
```
Using the calibrated SVB model:
  For delay_steps in {0, 5, 10, 20, 40, 80} (each step ≈ 5 min):
    Apply USDC backstop at step (40 + delay_steps) instead of at step 40 (shock onset)
    Measure: contagion at victims M, pct_reduction vs no-intervention baseline
  Save: experiments/results/netcontagion/intervention_timing.csv
  Plot: pct_reduction vs delay (minutes), mark the "50% effectiveness" threshold
```

**Execute**:
```
python scripts/run_intervention_timing.py
```

**Target result**: Intervention is still ≥50% effective up to D steps of delay (target: D ≥ 30
steps = 2.5 hours). If effectiveness holds for 2+ hours, then the GNN/HMM companion papers'
"1440-minute horizon" early-warning gives a 23-hour head start — a directly actionable
connection between the two companion papers.

**Write into paper**: New Figure in §5 (Policy Implications). Caption: "Contagion reduction
falls as intervention is delayed past shock onset. The $50\%$-effectiveness threshold at
$D$ minutes of delay means the companion paper's $24$-hour early warning leaves ample response
time." This is the explicit bridge between the ABM causal results and the GNN predictive results.

---

## Plan D — RL Training Convergence Evidence  *(show learning, not luck)*

**Goal**: The rl_regulator.json shows the *final* allocation after 12,000 PPO timesteps, but
does not show the *learning curve*. A reviewer will ask: "Does the RL regulator reliably
converge to the causal allocation, or is this one lucky seed?" Add training curves and
multi-seed confirmation.

**Code to write**: `scripts/run_rl_convergence.py`
```
Re-run PPO training (or load checkpoints if available) with 5 random seeds:
  Record per-episode: cumulative reward, BUSD allocation, USDC allocation, contagion_reduction
  Save training curves as: experiments/results/netcontagion/rl_convergence.csv
  Plot: (a) USDC allocation vs timestep (should rise from ~1/N to ~1.0)
         (b) BUSD allocation vs timestep (should fall from ~1/N to ~0.0)
         (c) Contagion reduction % vs timestep (should converge to ~93%)
  Report: mean ± std across 5 seeds for final contagion_reduction_pct
```

**Execute**:
```
python scripts/run_rl_convergence.py
```

**Target result**: All 5 seeds converge to USDC allocation ≥ 0.9 and BUSD allocation ≤ 0.1.
Mean contagion reduction across seeds ≥ 90%. This makes the "RL independently learns to be
causal" claim bullet-proof.

**Write into paper**: Add training convergence figure to §5.3 (RL Regulator). Key sentence:
"Across 5 random seeds, PPO converges to $\geq 0.9$ allocation to the causal origin within
$X$ timesteps, with mean contagion reduction $Y \pm Z\%$, confirming that the causal discovery
is reproducible and not a single-seed artefact."

---

## Plan E — Cross-Crisis Policy Transfer (RL generalization test)  *(Lucas-critique robustness)*

**Goal**: Does the RL regulator trained on SVB transfer to a different episode? Specifically,
train PPO on USDC_SVB parameters, then test its allocation on UST_Terra parameters
(different origin, different network topology). If it still allocates to the causal origin,
the policy generalizes across crisis types.

**Code to write**: `scripts/run_rl_cross_crisis.py`
```
Step 1: Load rl_regulator.json (SVB-trained PPO weights)
Step 2: Initialize the contagion simulator with UST_Terra calibrated parameters
        (load from multi_episode_join.csv: UST_Terra row has calibration_pass 4/4)
Step 3: Run the SVB-trained policy on the Terra simulator for 1000 episodes
        Record: contagion_reduction_pct, USDC allocation, UST allocation
Step 4: Compare to: (a) Terra-trained policy, (b) GNN-guided allocation (USDC is GNN top)
        (Note: For Terra, GNN is correct — so this tests whether RL also finds what GNN got right)
Save: experiments/results/netcontagion/rl_cross_crisis.json
```

**Execute**:
```
python scripts/run_rl_cross_crisis.py
```

**Target result**: SVB-trained RL achieves ≥ 70% contagion reduction on Terra episode. Since
the Terra GNN top is also USDC (the actual causal origin), and our RL learned to protect USDC
from the SVB run, the policy *should* transfer. If it does, the paper gains a cross-crisis
generalization claim. If it doesn't, this becomes the Lucas-critique robustness negative result
(honest and publishable).

**Write into paper**: New paragraph in §5.3: "To test generalization, we run the SVB-trained
regulator on the UST/Terra calibration. It achieves $X\%$ contagion reduction, compared to a
Terra-trained regulator at $Y\%$, because both crises share a USDC-origin causal structure.
This cross-crisis transfer validates the policy as mechanism-robust, not event-specific."

---

## Plan F — Optimal Partial Backstop Analysis  *(reserve strength vs full protection)*

**Goal**: A full backstop ("hold coin X at peg") is the strongest intervention. Regulators
often apply partial interventions (strengthen reserves by 2×, 5×). The intervention_sweep.csv
has `reserve_strengthen kappa_x2 → 32.1%` and `kappa_x5 → 64.3%`. Find the partial backstop
level that achieves a given contagion reduction at minimum cost (budget = intervention intensity
× protected venue's capitalization).

**Code to write**: `scripts/run_partial_backstop.py`
```
For USDC (causal origin):
  Sweep kappa multiplier: {1.5, 2.0, 3.0, 5.0, 10.0, full}
  Record: pct_reduction, intervention_cost_proxy (multiplier × USDC_mcap)
  Find: "efficient frontier" — minimum multiplier for 50%, 80%, 95% reduction
For BUSD (spurious hub):
  Same sweep → shows that ANY level of BUSD intervention yields ~0% reduction
Save: experiments/results/netcontagion/partial_backstop.csv
```

**Execute**:
```
python scripts/run_partial_backstop.py
```

**Target result**: For USDC: 5× reserve strengthening achieves 64.3% reduction at fraction of
full-backstop cost. For BUSD: even 10× strengthening achieves < 5% reduction. This gives
regulators a practical cost curve, not just a binary "backstop or not" result.

**Write into paper**: New Figure in §5 (Policy Implications), "Intervention cost-effectiveness
curve." X-axis: intervention intensity (kappa multiplier); Y-axis: contagion reduction. Two
lines: USDC (causal origin, steep positive slope) vs BUSD (spurious hub, flat near-zero).
Caption: "Partial backstop of the causal origin is strictly cost-effective; any investment in
the spurious hub is wasted regardless of intensity."

---

## Plan G — Welfare Decomposition Deep-Dive  *(the "who wins and who loses" policy table)*

**Goal**: The welfare_matrix.csv has welfare by agent type. Turn this into the Gu et al.
analogue: a decomposition showing who benefits from USDC protection vs no intervention vs
BUSD protection. This is the "policy has distributional consequences" result.

**Code to write**: `scripts/run_welfare_analysis.py`
```
Load: experiments/results/netcontagion/welfare_matrix.csv
For each intervention scenario (no-intervention, USDC backstop, BUSD backstop):
  Compute welfare by agent type: stablecoin holders, arbitrageurs, liquidity providers
  Compute: Pareto comparison (does USDC backstop Pareto-dominate no-intervention?)
  Compute: BUSD backstop welfare vs no-intervention for each agent type
Save: experiments/results/netcontagion/welfare_analysis.json
Plot: 3-column bar chart (scenarios) × rows (agent types)
```

**Execute**:
```
python scripts/run_welfare_analysis.py
```

**Target result**: USDC backstop Pareto-dominates no-intervention for all agent types (every
group is at least as well off). BUSD backstop: holders of non-BUSD coins are no better off
(confirms wasted budget), while BUSD holders benefit at the expense of the intervention budget.

**Write into paper**: New Table in §5 (Policy Implications), welfare decomposition matrix.
Caption: "A USDC backstop Pareto-dominates no intervention; a BUSD backstop benefits BUSD
holders but transfers no contagion reduction to the system." This is the strongest policy
statement in the paper, and it maps directly to regulatory cost-benefit analysis.

---

## Plan H — GENIUS Act / MiCA Policy Translation  *(make the paper relevant to regulators)*

**Goal**: Translate simulation results into specific, citable regulatory recommendations.
The US GENIUS Act (2025) and EU MiCA (2023) are explicitly cited in the paper. Use our causal
results to make three concrete recommendations that go beyond the current paper's broad
"correlation vs causation" framing.

**Code to write**: `scripts/compile_policy_recommendations.py`
```
No new experiments. Compile from existing results:
  Result 1 (causal knockout): "Reserve transparency disclosures should be in documented
    balance-sheet form, not inferred from price correlation, because the two can disagree
    on which venue is the contagion hub (BUSD example)."
  Result 2 (budget allocation): "With K=1 intervention budget, regulators using correlational
    rankings waste the entire budget (0% reduction); causal ranking achieves 100%."
  Result 3 (timing): "Intervention D minutes after crisis onset achieves X% reduction —
    the GNN companion's 24-hour lead time leaves ample response time under GENIUS Act
    emergency authorization procedures."
  Tie each to specific GENIUS Act provision and MiCA Article
Save: paper/policy_supplement.md
```

**Execute**:
```
python scripts/compile_policy_recommendations.py
```

**Target result**: Three crisp, numbered policy recommendations with direct numerical backing
from our simulation results, each paired with the relevant GENIUS Act / MiCA provision.

**Write into paper**: Expand current §6 (Policy) from 1 paragraph to 3 numbered
recommendations, each ≤ 3 sentences with explicit citation to the GENIUS Act or MiCA article.
End the section with: "These recommendations follow directly from the causal knockout
experiments above and do not require any model assumptions beyond calibration to the observed
crisis." This makes the policy section reviewer-proof: every recommendation has a specific
numerical backing.

---

## Credibility checklist (reviewer-facing, non-negotiable)
1. Calibration passes 4/4 moments for SVB (done ✅). Add Terra episode (Plan A).
2. GNN vs ABM hub agreement for ≥3 episodes (Plan A) — do NOT claim "GNN is always wrong."
3. Causal knockout effect sizes with standard errors across noise seeds (partially done ✅).
4. RL contagion reduction with 5-seed ± std (Plan D).
5. Budget-constrained comparison: GNN vs ABM at K=1,2 (Plan B).
6. At least one intervention mechanism explained (why USDC backstop works) — already in paper ✅.
7. No-intervention baseline anchors all comparisons — already in paper ✅.
