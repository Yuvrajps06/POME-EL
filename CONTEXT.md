# PricePilot: Project Context (read this first every session)

## Goal
Real-data-driven pricing decision support. Research question: does using
ML-predicted competitor responses inside a Stackelberg pricing decision improve
profit versus baseline pricing, and under what conditions?

## Pipeline
M5 data -> cleaning/features -> demand model + elasticity -> competitor response
module -> Stackelberg optimizer -> strategy comparison -> sensitivity analysis
-> Streamlit dashboard.

## Non-negotiable methodological rules
1. M5 has NO external competitor prices. Never claim competitor behavior was
   learned from M5. The competitor module is either (A) real competitor data if
   obtained, (B) M5 cross-store proxy (exploratory only), or (C) calibrated
   simulation. Always label which one is in use.
2. M5 has NO unit cost. Cost is a user input / scenario parameter. Never use
   selling price as cost.
3. Use chronological train/validation/test splits, never random splits.
4. Elasticity from regression is observational, not causal. Say so in outputs.
5. Constrain optimization to a reasonable price range around observed prices.
6. Do not assume ML-aware pricing wins. Experiments must report wins, losses,
   and negligible differences.
7. Accuracy alone is not enough: evaluate decision-level profit impact.

## Three strategies
A. Traditional: maximize profit using own demand only, competitor ignored.
B. Game-theoretic: Stackelberg with an assumed/rule-based follower response.
C. ML-aware Stackelberg: follower response predicted by ML model.

## Key formulas
- Profit: pi_L = (P_L - C_L) * Q_L
- Demand (game engine): Q_L = a - b*P_L + c*P_F + d*X
- Follower: P_F = R(P_L, X)
- Stackelberg: P_L* = argmax (P_L - C_L) * Q_L(P_L, R(P_L), X)
- Log-log elasticity: ln(Q) = b0 + b1*ln(P) + b2*X  (b1 ~ elasticity)
- Profit improvement % = (pi_ML - pi_base) / pi_base * 100

## Working conventions
- Modular code in src/; no giant scripts. Type hints and docstrings.
- Every phase ends with: tests passing, PROGRESS.md updated (decisions +
  function signatures others will call), and a short summary.
- Seed = 42 everywhere. Config in src/utils/config.py.
