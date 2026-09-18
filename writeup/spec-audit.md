# Spec audit (performed before any code was written)

Source spec: `ros-projection-build-spec.md` (2026-09-08). Verdict: the core —
empirical-Bayes shrinkage on components, walk-forward backtest, honest
benchmarking — was right. Four blockers and several corrections were fixed
in the build plan:

## Blockers

1. **Historical benchmark projections are not retrievable.** FanGraphs
   overwrites Steamer/ZiPS/THE BAT in place; "CSV export" cannot recover
   preseason versions for past seasons. *Fix:* historical backtests benchmark
   against leakage-free computables (prior-only, naive-YTD, league-prior
   ablation); daily FG snapshots begin now so 2027 gets a real public-system
   benchmark.
2. **M2 contradicted §7** ("backtested one season" vs. "one season is a
   story, not evidence"). *Fix:* backtest covers 2023, 2024, 2025.
3. **Calibration was specced without a predictive distribution.** *Fix:*
   per-player predictive sd = per-PA multinomial wOBA variance ×
   (talent uncertainty 1/(PA_ytd + k_eff) + sampling 1/PA_ros).
   Result: 50/80/95% intervals covered 50.2/81.0/95.8% — see
   `results/calibration.csv`.
4. **Calendar reality.** Built September 2026: backtests run 2023–25, the
   remaining 2026 games are a pipeline smoke test, and 2027 is the first
   live season.

## Methodology corrections

5. **Double shrinkage.** Blending YTD → preseason projection → league mean
   regresses twice; the prior already contains the mean. Built as: shrink
   YTD toward the prior; only the prior carries league regression.
6. **Per-component, not monolithic.** Each event gets its own shrinkage
   constant (SO 60, uBB 120, HR 170, HBP 240, 1B 290, 2B/3B 1450 PA).
7. **BB%/K%/ISO/BABIP don't determine wOBA.** ISO is one equation over
   {2B,3B,HR}; identical components can imply different wOBAs. The model
   lives in the 7-event per-PA space {uBB, HBP, SO, 1B, 2B, 3B, HR} — the
   same idea, exact recombination. The spec's four are reported as
   diagnostics.
8. **xwOBA–wOBA ≠ "luck."** The gap carries speed, park, defense, model
   error. xwOBA enters phase 2 as a blend input for contact events, not as
   the backbone.
9. **Pitchers (M3, unbuilt):** projecting HR/FB then mapping to ERA is
   component-FIP, not xFIP (xFIP *replaces* HR/FB with league average);
   HR/FB also needs batted-ball data BR/Lahman don't have. Flagged, not
   faked.
10. **Leakage guards not in the spec:** the league-mean regression target is
    the *trailing* 3-season mean (same-season league rates are unknowable in
    March); projections use *prior-season* wOBA weights; checkpoints are
    pre-registered monthly to prevent checkpoint shopping.
11. **PA-weighted metrics** reported alongside raw (a 50-PA September call-up
    is not the same evidence as a 600-PA regular).

## Scoped down

12. **"ROS WAR leaders" deferred.** Real WAR needs playing-time, defense,
    baserunning, positional and replacement-level models the spec never
    builds. Dashboard v1 will show ROS wOBA leaderboards + risers/fallers.
13. **Statcast pitch-level deferred.** ~700k rows/season behind a 25k-row
    CSV cap is a phase-2 milestone, not an M1 blocker; walk-forward needs
    only date-range aggregates (one request each).
14. **Rookies = league mean by design** (no-history prior). Stated as
    designed behavior for the "where I'm wrong" page, alongside injuries,
    trades, and role changes.

## What the first backtest taught (2026-09-17)

15. At borrowed (Carleton) shrinkage constants, outcome-based YTD added
    nothing over the prior (RMSE .0485 vs .0481) — as suspected in item 6's
    caveat, constants derived against league-mean priors under-trust a
    player-specific prior. **Resolution:** tuned a global multiplier on
    2023–24 only (0.5/1/2/4 → 4.0, flat past that), validated on the
    untouched 2025 holdout: PROOF .0393 vs prior-only .0400 — the blend now
    beats its own prior in all three seasons. See results/tuning.csv.
16. Tuning the mean broke the intervals slightly (77.4% coverage at nominal
    80%) — the variance model was missing irreducible in-season talent
    drift. **Resolution:** method-of-moments σ²_drift = 0.000237
    (σ ≈ .015 wOBA of "players change" per season) estimated on training
    seasons; calibration back to 49.3/80.6/95.5.
17. All systems overproject by ~0.004–0.007 wOBA (bias). Leading suspect:
    injury/decline among players who keep getting PA. Quantify on the
    "where I'm wrong" page.
18. Operational: Baseball-Reference 429-rate-limited the build mid-pull
    (2026-09-17). The write-once snapshot policy made recovery trivial —
    failed pulls never persist, reruns resume exactly. Prefetch now spaces
    requests 30 s apart (ingest/prefetch.py). This is the §1 "daily refresh"
    caveat in practice: the pipeline is polite or it is dead.

## M3 build notes (2026-09-17, same day)

19. **The BR ban forced the better architecture.** Pitching moved to
    pitch-level Statcast (Savant — separate host): three season pulls total
    instead of twelve range pulls per season, and pitcher lines for ANY
    date window are now computed locally. The reconstruction was validated
    against pinned BR snapshots before use (event counts exact, outs to
    within ±2/season, corr ≥ 0.9999 — ingest/validate_statcast.py). The
    pitch-level store also pre-positions the phase-2 xwOBA features.
    Caveat: ERA is not reconstructible (no earned/unearned distinction), so
    the Savant path targets FIP. The ban lifted after a ~2.5 h cool-down;
    the polite prefetcher completed the BR set and the full 3-season ERA
    evaluation exists (results/pitchers_era_metrics_by_season.csv), with
    the same system ordering as FIP — the two instruments agree.
20. **Scale-constant artifact, caught by the bias column.** Scoring actual
    FIP with the season's own cFIP while projecting with the prior season's
    injected ~0.3 FIP of fake error in 2023 (pitch-clock/shift-ban
    environment jump). Actuals are now scored on the projection scale
    (FIP path); the ERA path shifts to the checkpoint-observable league
    environment instead (league ERA to date — no player leakage).
21. **The pitcher prior is genuinely weaker than the hitter prior.**
    League-mean shrinkage beats the player-specific prior for pitchers
    (wRMSE .971 vs .991 pooled; holds on the 2025 holdout), while the
    reverse holds for hitters. Three-year pitcher history is half-rotted by
    March: injuries, arsenal and velocity changes, SP/RP conversions. The
    pitcher "players change" variance is σ≈0.23 FIP vs ≈.015 wOBA for
    hitters — same story, two instruments. Phase 2: shorter pitcher history
    windows and role-aware priors.
22. **BR's range tool includes postseason games** (verified: a 2023 LCS
    window returns rows). All backtest end dates were checked to be
    regular-season finales; live pulls end at "yesterday," safe until the
    playoffs start. The dashboard goes dormant in October by design.
23. **Dashboard shipped without WAR** (per item 11): sortable ROS wOBA /
    FIP / ERA leaderboards, 80% intervals, risers/fallers vs. preseason.
    Late-season reality is visible in the intervals: with ~10 days left,
    ROS intervals are honestly wide.
24. **Data provenance can shift results at the 4th decimal.** The 2025
    pitcher prior moved from Savant-reconstructed 2024 lines to BR-official
    2024 lines once the polite prefetch completed (the snapshot registry
    records which source each window used). FIP wRMSE moved 0.9905 →
    0.9899. Reproducibility claims are per-snapshot-registry, and the
    registry is committed to git.
