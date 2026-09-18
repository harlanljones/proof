"""Walk-forward harness: train through T, project rest-of-season, score vs actuals.

Information discipline at a (season, checkpoint) cell:
  * prior   — the 3 seasons *before* `season`, Marcel-weighted, regressed to
              the trailing league mean. Computable in March: no in-season info.
  * YTD     — [season start, checkpoint)
  * actuals — [checkpoint, season end]
  * wOBA weights for projections: prior season's (published). For scoring
    actuals: the season's own weights.
"""

from __future__ import annotations

import pandas as pd

from backtest.checkpoints import grid
from features.components import add_events, add_woba
from ingest.baseball_ref import fetch_range, fetch_season
from ingest.guts import season_weights
from models.blend import attach_prior, project_all
from models.priors import marcel_prior, trailing_league_rates

MIN_PA_YTD = 50  # need a meaningful YTD signal to update
MIN_PA_ROS = 50  # need meaningful actuals to score against

SYSTEMS = ["proof", "naive", "prior", "eb_league"]


def _season_frames(season: int) -> tuple[list[pd.DataFrame], list[int]]:
    seasons = [season - 1, season - 2, season - 3]
    return [add_events(fetch_season(s)) for s in seasons], seasons


def run_cell(
    season: int,
    cp: dict,
    prev: list[pd.DataFrame],
    prev_seasons: list[int],
    k_mult: float = 1.0,
) -> pd.DataFrame:
    w_prior = season_weights(season - 1)  # published before the season
    w_eval = season_weights(season)  # scoring actuals

    prior = marcel_prior(prev, prev_seasons, season, w_prior)
    league = trailing_league_rates(prev)

    ytd = add_events(fetch_range(cp["ytd_start"], cp["ytd_end"]))
    ytd = ytd[ytd["PA"] >= MIN_PA_YTD]

    ros = fetch_range(cp["ros_start"], cp["ros_end"])
    ros = add_woba(add_events(ros), w_eval)
    ros = ros[["mlbID", "PA", "wOBA"]].rename(
        columns={"PA": "PA_ros", "wOBA": "woba_actual_ros"}
    )
    ros = ros[ros["PA_ros"] >= MIN_PA_ROS]

    m = attach_prior(ytd, prior, league)
    m = m.merge(ros, on="mlbID", how="inner")

    preds = project_all(m, w_prior, league, pa_ros=m["PA_ros"], k_mult=k_mult)
    out = preds.merge(m[["mlbID", "PA_ros", "woba_actual_ros"]], on="mlbID")
    out["season"] = season
    out["checkpoint"] = cp["checkpoint"]
    return out


def run_backtest(seasons: list[int], k_mult: float = 1.0) -> pd.DataFrame:
    cells = []
    for season in seasons:
        prev, prev_seasons = _season_frames(season)
        for cp in grid(season):
            print(f"  cell: {season} @ {cp['checkpoint']}")
            cells.append(run_cell(season, cp, prev, prev_seasons, k_mult))
    return pd.concat(cells, ignore_index=True)
