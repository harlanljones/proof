# PROOF: Probabilistic Rest-Of-season Objective Forecasts

An open, reproducible MLB rest-of-season projection system — a mini
Steamer/ZiPS built on empirical-Bayes shrinkage, with a walk-forward
backtest harness that treats honest evaluation as the main artifact.

## Headline results (walk-forward, 2023–2025, monthly checkpoints, n=4,892)

PA-weighted RMSE on rest-of-season wOBA:

| system | what it is | RMSE (PA-wtd) |
|---|---|---|
| naive | ROS = year-to-date, no shrinkage | .0572 |
| eb_league | shrink YTD toward league mean | .0437 |
| prior | preseason Marcel projection alone | .0411 |
| **proof** | **shrink YTD toward player prior, per-component k** | **.0405** |

PROOF beats its own prior in all three seasons individually — the margin is
small because it should be: rest-of-season wOBA is one of the hardest
public forecasting targets, and honest gains here are measured in the third
decimal place.

Calibration of the probabilistic forecast: nominal 50/80/95% intervals
covered **49.3 / 80.6 / 95.5%**. The uncertainty model is the strongest
part of the system.

**How the shrinkage constants were chosen** (no test-set tuning): a global
multiplier on the borrowed Carleton constants was selected on 2023–24 and
validated on the 2025 holdout the tuning loop never saw. The backtest said
trust the prior ~4× more than the borrowed constants (train 0.0429→0.0411;
holdout proof 0.0393 vs prior 0.0400). Calibration was then restored by a
training-estimated "players change" variance term (σ≈0.015 wOBA of
irreducible in-season talent/role/injury drift). Full trail:
[`writeup/spec-audit.md`](writeup/spec-audit.md).

## Pitchers (M3)

BF-weighted RMSE, rest-of-season (2023–2025; k_mult=2.0 tuned on 2023–24):

| target | n | naive | prior | **proof** | eb_league |
|---|---|---|---|---|---|
| **FIP** | 3,473 | 1.218 | 1.028 | **0.990** | 0.971 |
| **ERA** | 3,400 | 1.910 | 1.467 | **1.431** | 1.378 |

FIP calibration 52.2/80.6/94.0; ERA calibration 51.0/79.6/93.1 after an
ERA-specific drift term (defense/park/sequencing variance, ~1.0 ERA of sd,
estimated on 2023-24 training residuals only — the 2025 holdout
independently implied 1.03; before it the ERA interval, which reused the
FIP sd, covered just 37.1/64.0/82.5%). Two honest findings: (1) the
player-specific
prior beats naive and prior-only clearly on both targets, but *loses to
league-mean shrinkage* — three-year pitcher history decays fast (injuries,
arsenal changes, role conversions); the pitcher "players change" drift is
σ≈0.23 FIP vs ≈.015 wOBA for hitters, ~15× in scale-adjusted terms.
(2) When Baseball-Reference 429-banned the build IP mid-milestone, the
pitcher pipeline moved to pitch-level Statcast — reconstructed lines
validated against pinned BR data to corr ≥ 0.9999 before use
(`python -m ingest.validate_statcast`) — and after a ~2.5 h cool-down the
polite prefetcher (30 s spacing) completed the BR set anyway, so the ERA
evaluation above covers all three seasons.

## Dashboard

```bash
make dashboard   # -> http://localhost:8501
```

Sortable live leaderboards (as of yesterday): ROS wOBA for hitters, ROS
FIP/ERA for pitchers, 80% predictive intervals, and risers/fallers vs. the
preseason prior. Built from the same validated machinery as the backtest —
the live run uses the 2023–25 pinned priors + a daily-pinned YTD pull. No
WAR column, by design (audit item 11).

## Reproduce

```bash
uv sync
make backtest   # pulls any missing snapshots once (~5 min), then fully cached
make test
```

Every external pull is pinned to a write-once parquet snapshot under
`data/snapshots/` (tracked in git, ~1.4 MB), so the backtest re-runs
offline, byte-identical.

## Architecture

```
ingest/      baseball_ref.py — BR date ranges, polite+backoff, write-once snapshots
             statcast.py    — pitch-level Savant -> local pitcher-line reconstruction
             identity.py    — names/birthdates from MLB Stats API
             guts.py        — FanGraphs wOBA weights + cFIP per season
             prefetch.py    — polite (30s) snapshot warmer; safe to re-run
features/    components.py — per-PA event rates {uBB,HBP,SO,1B,2B,3B,HR} + wOBA
             pitching.py    — per-BF rates {SO,uBBHBP,HR}, BF/IP, FIP, SP/RP roles
models/      priors.py   — hitter Marcel prior (5/4/3, 1200-PA regression, aging)
             blend.py    — per-component EB shrinkage + predictive variance
             pitchers.py — pitcher prior/blend, role-specific league means, gap
backtest/    pre-registered monthly checkpoints; walkforward.py (hitters),
             walkforward_pitchers.py (FIP via Savant + ERA via BR);
             tune.py — k_mult tuned on 2023-24, validated on 2025 holdout;
             metrics.py — RMSE/MAE (weighted + raw), calibration, plots
app/         projections.py — live as-of-today projection builder
dashboard.py (root)         — Streamlit UI: leaderboards, intervals, risers/fallers
writeup/     spec-audit.md (pre-build + build log), results/ (make backtest)
```

Deliberate deviations from the original spec are documented in
`writeup/spec-audit.md` — the big ones: the model lives in event-rate space
because BB%/K%/ISO/BABIP don't uniquely determine wOBA; benchmarks are
computable baselines because historical Steamer/ZiPS snapshots don't exist;
WAR is deferred because the spec under-specified it.

## Roadmap (phase-gated)

- [x] **M1** ingestion with immutable snapshot versioning
- [x] **M2** hitter projections, 3-season walk-forward backtest
- [x] **M2.5** shrinkage constants tuned on train/validated on holdout;
      talent-drift variance term; Statcast pitch-level ingest (3 seasons)
- [x] **M3** pitchers: SP/RP split, per-component FIP, role-specific league
      means, ERA-FIP gap; FIP backtest 2023-25 (Savant) + ERA 2023-25 (BR)
- [ ] **M3.5** daily FG projection snapshots -> first real public benchmark
      for 2027 (historical ones don't exist — audit §1)
- [x] **M4a** dashboard live: `make dashboard` (hitters/pitchers, risers &
      fallers, 80% intervals; Streamlit, verified headless via AppTest)
- [x] **M4b** the write-up prose (writeup/methodology.md — full draft,
      numbers quoted from writeup/results/)

## Leakage discipline

- Prior uses only seasons *before* the target season; league regression
  target is the trailing 3-season mean; projections use prior-season wOBA
  weights.
- Checkpoints fixed monthly (May 1 – Sep 1) before seeing results.
- Snapshots are write-once: re-pulling restated data is impossible by
  construction.

## Known limitations (the short list)

No-history players project as league average; injuries, trades and role
changes are invisible to the model (a persistent +0.004–0.007 wOBA bias is
likely injury-related); 2020's 60-game season enters priors with correctly
reduced weight. Long version: `writeup/spec-audit.md` §15–16. Phase-2 xwOBA blend input
was spiked and rejected: YTD contact quality adds no ROS-FIP signal beyond
the event-rate blend (`writeup/spike-xwoba.md`).
