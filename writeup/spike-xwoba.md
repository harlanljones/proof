# Spike: xwOBAcon as a blend input (phase-2 candidate)

> Throwaway experiment, 2026-09. Code: `spikes/xwoba-con/spike.py`
> (untracked, disposable). Question from spec-audit §8 and the §4b note
> that the pitch-level store "pre-positions phase-2 xwOBA features."

## Question

Does year-to-date contact quality (xwOBAcon, mean estimated wOBA on batted
balls from Statcast) add rest-of-season FIP predictive signal beyond what
PROOF's existing event-rate blend (SO / uBB+HBP / HR, shrunk toward the
Marcel prior) already captures?

## Method (leakage-safe)

- YTD xwOBAcon per pitcher per checkpoint (May–Sep grid, 2023–2025) from
  the pinned pitch-level snapshots; BBE floor 25, league mean regressed by
  k BBE (grid 100–800).
- Residual target: ROS FIP actual minus the existing `proof` projection
  (from the pinned backtest parquet — no model re-run).
- Shrink k and correction weight b tuned jointly on 2023–24 ONLY;
  2025 evaluated once as holdout. pred' = pred_proof + b·(xw_shrunk − lg_xw).

## Results

- Per-cell residual correlations hover on zero: mean ≈ 0.00, max 0.088
  (2023-08), one negative outlier −0.154 (2024-05). No consistent sign.
- Training RMSE improvement from the best correction: 1.15051 → 1.15043
  (~0.007%, noise-scale). The fitted b is unstable across k (b scales
  inversely with the shrink denominator — the fit is absorbing its own
  standardization, not signal).
- Holdout 2025: RMSE 1.15111 → 1.15118 (−0.006%, i.e. slightly worse);
  residual corr +0.006.

## Verdict: INVALIDATED

Contact quality carries no incremental ROS-FIP signal once the event-rate
blend is in place — plausibly because SO/uBB/HR rates are already the
dominant carrier, and xwOBAcon's remaining variance (park, defense,
sampling at ~150 BBE/season) is exactly what the FIP formulation strips
out on purpose.

### Recommendation for the real build

Do not wire xwOBAcon into the blend. The pitch-level store keeps its value
as the validated BR substitute (corr ≥ 0.9999) and for future targets that
are *defined* on contact quality (e.g. phase-3 pitcher xwOBA-contra
evaluation), not as a predictor input here.
