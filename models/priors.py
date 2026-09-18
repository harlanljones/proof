"""Marcel-style preseason prior from the three previous seasons.

The method (Tango's Marcel, generalized to our event-rate space):
  1. Weight the past 3 seasons 5/4/3 (most recent heaviest).
  2. Regress toward the league mean by adding `regression_pa` PA of
     league-average performance — small samples shrink hard.
  3. Apply a simple age adjustment.

Two leakage guards, both load-bearing:
  * The league-mean regression target is the *trailing* 3-season league
    mean — the target season's actual league rates are unknowable in March.
  * wOBA weights used to score the prior are the most recent *completed*
    season's, passed in by the caller.
"""

from __future__ import annotations

import pandas as pd

from features.components import EVENTS, league_rates

SEASON_WEIGHTS = (5, 4, 3)  # most-recent-first
REGRESSION_PA = 1200  # Marcel's hitter value

# Marcel aging: +0.3%/yr toward peak below 29, -0.6%/yr past it.
# Phase 2 replaces this with empirical aging curves from Lahman.
AGE_PEAK = 29
AGE_UP = 0.003
AGE_DOWN = 0.006


def aging_factor(age: pd.Series) -> pd.Series:
    return (
        1
        + (AGE_PEAK - age).clip(lower=0) * AGE_UP
        - (age - AGE_PEAK).clip(lower=0) * AGE_DOWN
    )


def _weighted_seasons(prev: list[pd.DataFrame], seasons: list[int]) -> pd.DataFrame:
    """Outer-join prior seasons on mlbID with 5/4/3 weighted counts and PAs."""
    frames = []
    for w, season, df in zip(SEASON_WEIGHTS, seasons, prev):
        f = df[["mlbID", "Name", "Age", "PA", "IBB", "SH"] + EVENTS].copy()
        f["w"] = w
        f["seen_season"] = season
        frames.append(f)
    long = pd.concat(frames, ignore_index=True)

    # Identity from the most recent season the player appeared in.
    long = long.sort_values("seen_season", ascending=False)
    ident = long.groupby("mlbID").first()[["Name", "Age", "seen_season"]]

    out = ident.copy()
    out["PA_hist"] = (long["PA"] * long["w"]).groupby(long["mlbID"]).sum()
    for ev in EVENTS + ["IBB", "SH"]:
        out[f"wcount_{ev}"] = (long[ev] * long["w"]).groupby(long["mlbID"]).sum()
    return out.reset_index()


def trailing_league_rates(prev: list[pd.DataFrame]) -> dict:
    """5/4/3-weighted trailing league mean — the leakage-free regression target."""
    lg = {}
    for ev in EVENTS + ["IBBSH"]:
        num = sum(
            w * league_rates(df)[ev] * df["PA"].sum()
            for w, df in zip(SEASON_WEIGHTS, prev)
        )
        den = sum(w * df["PA"].sum() for w, df in zip(SEASON_WEIGHTS, prev))
        lg[ev] = num / den
    return lg


def marcel_prior(
    prev: list[pd.DataFrame], seasons: list[int], target_season: int, woba_weights: dict
) -> pd.DataFrame:
    """Preseason projection for `target_season` from the 3 prior seasons.

    `prev` / `seasons` must be ordered most-recent-first and the frames must
    already have event columns attached (features.components.add_events).
    """
    ws = _weighted_seasons(prev, seasons)
    lg = trailing_league_rates(prev)

    for ev in EVENTS:
        ws[f"r_{ev}_prior"] = (ws[f"wcount_{ev}"] + REGRESSION_PA * lg[ev]) / (
            ws["PA_hist"] + REGRESSION_PA
        )
    ws["r_IBBSH_prior"] = (
        ws["wcount_IBB"] + ws["wcount_SH"] + REGRESSION_PA * lg["IBBSH"]
    ) / (ws["PA_hist"] + REGRESSION_PA)

    # Age in the target season, from the most recent appearance.
    ws["Age"] = ws["Age"] + (target_season - ws["seen_season"])

    # Aging: scale positive-value events proportionally (shares constant,
    # wOBA scales ~multiplicatively, simplex stays valid). SO untouched.
    factor = aging_factor(ws["Age"])
    for ev in ["uBB", "HBP", "1B", "2B", "3B", "HR"]:
        ws[f"r_{ev}_prior"] = ws[f"r_{ev}_prior"] * factor

    keep = ["mlbID", "Name", "Age", "PA_hist", "r_IBBSH_prior"] + [
        f"r_{ev}_prior" for ev in EVENTS
    ]
    return ws[keep].reset_index(drop=True)
