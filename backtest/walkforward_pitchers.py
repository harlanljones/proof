"""Walk-forward harness for pitchers.

Two evaluation paths, kept strictly separate by source:

  * FIP path (2023-2025): everything sourced from local Savant pitch-level
    reconstructions (uniform definitions, validated against BR in
    ingest/validate_statcast). Runs fully offline once pitch-level
    snapshots exist. This is the M3 headline.
  * ERA path (2023 only): the original BR-sourced harness; 2023 inputs were
    all pinned before the rate-limit ban, so it runs offline. ERA can't be
    reconstructed from pitch-level data (no earned/unearned distinction),
    so full ERA evaluation awaits BR access; 2023 is the honest subsample.

Projection-side cFIP is the prior season's (published); actuals are scored
on that same scale — cFIP is an additive constant and mixing scales injects
run-environment drift into the error term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.checkpoints import grid
from features.pitching import add_pitch_events, fip_from_rates
from ingest import snapshots
from ingest.baseball_ref import (
    SEASON_DATES,
    fetch_pitching_range,
    fetch_pitching_season,
)
from ingest.guts import season_weights
from ingest.identity import with_identity
from ingest.statcast import pitching_lines
from models.pitchers import (
    _trailing_role_rates,
    attach_pitcher_prior,
    marcel_pitcher_prior,
    project_pitchers,
)

MIN_BF_YTD = 80  # ~20 IP
MIN_BF_ROS = 80

SYSTEMS_P = ["proof", "naive", "prior", "eb_league"]


def _with_gap(df: pd.DataFrame, cfip: float) -> pd.DataFrame:
    out = df.copy()
    out["gap"] = out["ERA"] - fip_from_rates(out, cfip, "")
    return out


# --------------------------------------------------------------- FIP (Savant)

IDENTITY_POOLS = [2023, 2025]  # covers ~everyone active 2023-25


def _sc_lines(season: int, start: str, end: str) -> pd.DataFrame:
    lines = with_identity(pitching_lines(season, start, end), season, IDENTITY_POOLS)
    return add_pitch_events(lines)


def _season_frames_sc(season: int) -> tuple[list[pd.DataFrame], list[int]]:
    """Prior-season frames: pinned BR where it exists (2020-2023), else the
    validated Savant reconstruction. Equivalence established in
    ingest/validate_statcast; priors are heavily regressed regardless."""
    seasons = [season - 1, season - 2, season - 3]
    frames = []
    for s in seasons:
        start, end = SEASON_DATES[s]
        if snapshots.exists(f"pitching_{start}_{end}"):
            df = add_pitch_events(snapshots.load(f"pitching_{start}_{end}"))
            df = _with_gap(df, season_weights(s)["cFIP"])
        else:
            # Savant reconstruction has no ERA (no earned/unearned in
            # pitch-level). The FIP path never uses `gap`; zero it.
            df = _sc_lines(s, start, end).assign(gap=0.0)
        frames.append(df)
    return frames, seasons


def run_cell_fip(
    season: int,
    cp: dict,
    prev: list[pd.DataFrame],
    prev_seasons: list[int],
    k_mult: float = 1.0,
) -> pd.DataFrame:
    cfip_prior = season_weights(season - 1)["cFIP"]  # published pre-season

    prior = marcel_pitcher_prior(prev, prev_seasons, season)
    league_by_role = _trailing_role_rates(prev)

    ytd = _sc_lines(season, cp["ytd_start"], cp["ytd_end"])
    ytd = ytd[ytd["BF"] >= MIN_BF_YTD]

    ros = _sc_lines(season, cp["ros_start"], cp["ros_end"])
    ros = ros[ros["BF"] >= MIN_BF_ROS].copy()
    # Actuals are scored with the SAME (prior-season) cFIP as projections:
    # cFIP is a pure additive scale constant, and scoring on the projection
    # scale keeps season-to-season run-environment drift out of the error
    # term (it swamped everything: ~0.3 FIP of fake bias in 2023).
    ros["fip_actual_ros"] = fip_from_rates(ros, cfip_prior, "")
    ros = ros[["mlbID", "BF", "fip_actual_ros"]].rename(columns={"BF": "BF_ros"})

    m = attach_pitcher_prior(ytd, prior, league_by_role)
    m = m.merge(ros, on="mlbID", how="inner")
    # Savant path has no earned runs; ERA outputs are unused for FIP scoring.
    m["ERA"] = np.nan
    m["gap_ytd"] = 0.0

    preds = project_pitchers(
        m, cfip_prior, league_by_role, bf_ros=m["BF_ros"], k_mult=k_mult
    )
    out = preds.merge(m[["mlbID", "BF_ros", "fip_actual_ros"]], on="mlbID")
    out["season"] = season
    out["checkpoint"] = cp["checkpoint"]
    return out


def run_backtest_pitchers_fip(seasons: list[int], k_mult: float = 1.0) -> pd.DataFrame:
    cells = []
    for season in seasons:
        prev, prev_seasons = _season_frames_sc(season)
        for cp in grid(season):
            print(f"  cell (P/fip): {season} @ {cp['checkpoint']}")
            cells.append(run_cell_fip(season, cp, prev, prev_seasons, k_mult))
    return pd.concat(cells, ignore_index=True)


# ---------------------------------------------------------------- ERA (BR)


def _season_frames_br(season: int) -> tuple[list[pd.DataFrame], list[int]]:
    seasons = [season - 1, season - 2, season - 3]
    frames = []
    for s in seasons:
        df = add_pitch_events(fetch_pitching_season(s))
        frames.append(_with_gap(df, season_weights(s)["cFIP"]))
    return frames, seasons


def run_cell(
    season: int,
    cp: dict,
    prev: list[pd.DataFrame],
    prev_seasons: list[int],
    k_mult: float = 1.0,
) -> pd.DataFrame:
    prior = marcel_pitcher_prior(prev, prev_seasons, season)
    league_by_role = _trailing_role_rates(prev)

    ytd_raw = add_pitch_events(fetch_pitching_range(cp["ytd_start"], cp["ytd_end"]))

    # ERA is an absolute scale, so the projection must track the *current*
    # run environment. League ERA/FIP to date are observable at checkpoint T
    # (league-wide info, no player leakage): effective cFIP = lg ERA - lg raw
    # FIP from the YTD window. Using last season's cFIP injected ~0.35 ERA of
    # fake bias in 2023 (pitch-clock/shift-ban environment jump).
    bf, outs = ytd_raw["BF"].sum(), ytd_raw["outs"].sum()
    lg_raw_fip = (
        (
            13 * ytd_raw["HR"].sum()
            + 3 * ytd_raw["uBBHBP"].sum()
            - 2 * ytd_raw["SO"].sum()
        )
        / bf
        * (3 * bf / outs)
    )
    cfip_eff = 27 * ytd_raw["ER"].sum() / outs - lg_raw_fip

    ytd = _with_gap(ytd_raw, cfip_eff).rename(columns={"gap": "gap_ytd"})
    ytd = ytd[ytd["BF"] >= MIN_BF_YTD]

    ros = fetch_pitching_range(cp["ros_start"], cp["ros_end"])
    ros = ros[["mlbID", "BF", "ERA"]].rename(
        columns={"BF": "BF_ros", "ERA": "era_actual_ros"}
    )
    ros = ros[ros["BF_ros"] >= MIN_BF_ROS]

    m = attach_pitcher_prior(ytd, prior, league_by_role)
    m = m.merge(ros, on="mlbID", how="inner")

    preds = project_pitchers(
        m, cfip_eff, league_by_role, bf_ros=m["BF_ros"], k_mult=k_mult
    )
    out = preds.merge(m[["mlbID", "BF_ros", "era_actual_ros"]], on="mlbID")
    out["season"] = season
    out["checkpoint"] = cp["checkpoint"]
    return out


def run_backtest_pitchers(seasons: list[int], k_mult: float = 1.0) -> pd.DataFrame:
    cells = []
    for season in seasons:
        prev, prev_seasons = _season_frames_br(season)
        for cp in grid(season):
            print(f"  cell (P/era): {season} @ {cp['checkpoint']}")
            cells.append(run_cell(season, cp, prev, prev_seasons, k_mult))
    return pd.concat(cells, ignore_index=True)
