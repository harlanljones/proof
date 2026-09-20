# Spike: empirical pitcher aging curves (phase-2 candidate)

> Throwaway experiment, 2026-09. Code: `spikes/aging/spike.py` (untracked).
> Prompted by the models/pitchers.py note: "crude — empirical aging curves
> are phase 2".

## Question

Is Marcel's crude aging adjustment (K scaled by f, uBB/HR by 1/f, f the
standard Marcel age curve) leaving RMSE on the table versus an empirical
pitcher-specific curve — or is aging not worth modeling at all?

## Method (leakage-safe)

- Parameterize the aging factor as f(age)**m: m=1 is the current Marcel
  adjustment, m=0 disables aging, m<1/m>1 interpolates toward/away from it.
- Walk-forward FIP backtest (pinned Savant snapshots, k_mult=2.0) for
  m ∈ {0, 0.5, 1, 1.5} on 2023–24; select m on training only; evaluate
  m* and m=1 once on the 2025 holdout.
- Implementation note: the first run of this spike had a capture bug
  (each patch re-captured the previous patch, silently pinning m=0) —
  all four rows returned identical RMSE. Fixed by capturing the original
  factor once; a suspiciously flat sweep is the tell to watch for.

## Results

| m | train 2023–24 wRMSE (proof) |
|-----|--------|
| 0.0 | 1.00327 |
| 0.5 | 1.00146 |
| 1.0 | **1.00074** |
| 1.5 | 1.00115 |

Holdout 2025: m=1 → **0.98436**; m=0 → 0.99019 (0.6% worse).

The curve is U-shaped with its minimum exactly at m=1: Marcel's aging
constants are already the empirical optimum at this resolution, and no
aging at all costs ~0.6% on holdout.

## Verdict: INVALIDATED

An empirical aging curve has no headroom over Marcel's constants here —
the tuning knob that interpolates between "no aging" and "Marcel aging"
selects Marcel aging on training data and confirms on holdout.

### Recommendation for the real build

Keep the existing aging adjustment unchanged and drop "empirical aging
curves" from the phase-2 list. The comment in models/pitchers.py is
updated to cite this result.
