"""Live rest-of-season projections as of a date — the dashboard's data layer.

Uses exactly the backtest-validated machinery: Marcel prior from 2023-2025
(pinned), YTD from Baseball-Reference through yesterday (one polite pull,
pinned per day), per-component EB blend with the tuned constants
(k_mult 4.0 hitters / 2.0 pitchers, from the 2023-24 -> 2025 holdout
protocol).

Rate projections don't depend on remaining schedule; only the predictive
interval does, via an estimated remaining playing time (days-left scaled
from the player's own YTD rate). Documented, user-visible assumption.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from backtest.run import K_MULT, K_MULT_P
from features.components import add_events
from features.pitching import add_pitch_events, fip_from_rates
from ingest.baseball_ref import (
    SEASON_DATES,
    fetch_pitching_range,
    fetch_pitching_season,
    fetch_range,
    fetch_season,
)
from ingest.guts import season_weights
from models.blend import attach_prior, project_all
from models.pitchers import (
    K_GAP,
    _trailing_role_rates,
    attach_pitcher_prior,
    marcel_pitcher_prior,
    project_pitchers,
)
from models.priors import marcel_prior, trailing_league_rates

LIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "live"
PRIOR_SEASONS = [2025, 2024, 2023]  # most-recent-first
Z80 = 1.2816


def _season(as_of: dt.date) -> int:
    for s, (start, end) in SEASON_DATES.items():
        if pd.Timestamp(start) <= pd.Timestamp(as_of) <= pd.Timestamp(end):
            return s
    raise ValueError(f"{as_of} is outside every known season window")


def _remaining_pt(pa_or_bf: pd.Series, as_of: dt.date, season: int) -> pd.Series:
    start = pd.Timestamp(SEASON_DATES[season][0]).date()
    end = pd.Timestamp(SEASON_DATES[season][1]).date()
    days_elapsed = max((as_of - start).days, 1)
    days_left = max((end - as_of).days, 1)
    return (pa_or_bf / days_elapsed * days_left * 0.95).clip(lower=5)


def build_hitters(as_of: dt.date) -> pd.DataFrame:
    season = _season(as_of)
    w = season_weights(season - 1)  # latest published weights

    prev = [add_events(fetch_season(s)) for s in PRIOR_SEASONS]
    prior = marcel_prior(prev, PRIOR_SEASONS, season, w)
    league = trailing_league_rates(prev)

    ytd = add_events(fetch_range(SEASON_DATES[season][0], as_of.isoformat()))
    ytd = ytd[ytd["PA"] >= 1]

    m = attach_prior(ytd, prior, league)
    pa_ros = _remaining_pt(m["PA"], as_of, season)
    preds = project_all(m, w, league, pa_ros=pa_ros, k_mult=K_MULT)

    out = preds.merge(m[["mlbID", "Tm", "BB%", "K%", "ISO", "BABIP"]], on="mlbID")
    out["lo80"] = out["pred_woba_proof"] - Z80 * out["sd_proof"]
    out["hi80"] = out["pred_woba_proof"] + Z80 * out["sd_proof"]
    out["delta"] = out["pred_woba_proof"] - out["pred_woba_prior"]
    return out[
        [
            "mlbID",
            "Name",
            "Tm",
            "Age",
            "PA_ytd",
            "pred_woba_naive",
            "pred_woba_prior",
            "pred_woba_proof",
            "lo80",
            "hi80",
            "delta",
            "BB%",
            "K%",
            "ISO",
            "BABIP",
        ]
    ].sort_values("pred_woba_proof", ascending=False)


def build_pitchers(as_of: dt.date) -> pd.DataFrame:
    season = _season(as_of)

    prev = [add_pitch_events(fetch_pitching_season(s)) for s in PRIOR_SEASONS]
    for df, s in zip(prev, PRIOR_SEASONS):
        df["gap"] = df["ERA"] - fip_from_rates(df, season_weights(s)["cFIP"], "")
    prior = marcel_pitcher_prior(prev, PRIOR_SEASONS, season)
    league_by_role = _trailing_role_rates(prev)

    ytd = add_pitch_events(
        fetch_pitching_range(SEASON_DATES[season][0], as_of.isoformat())
    )
    ytd = ytd[ytd["BF"] >= 1]

    # current run environment from YTD league totals (same trick as the
    # ERA backtest path): effective cFIP = lg ERA - lg raw FIP
    bf, outs = ytd["BF"].sum(), ytd["outs"].sum()
    lg_raw_fip = (
        (13 * ytd["HR"].sum() + 3 * ytd["uBBHBP"].sum() - 2 * ytd["SO"].sum())
        / bf
        * (3 * bf / outs)
    )
    cfip_eff = 27 * ytd["ER"].sum() / outs - lg_raw_fip

    ytd["gap_ytd"] = ytd["ERA"] - fip_from_rates(ytd, cfip_eff, "")
    m = attach_pitcher_prior(ytd, prior, league_by_role)
    bf_ros = _remaining_pt(m["BF"], as_of, season)
    preds = project_pitchers(m, cfip_eff, league_by_role, bf_ros, k_mult=K_MULT_P)

    m["FIP_ytd"] = fip_from_rates(m, cfip_eff, "")
    m["gap_ros"] = (m["BF"] * m["gap_ytd"] + K_GAP * m["gap_prior"]) / (m["BF"] + K_GAP)
    out = preds.merge(
        m[["mlbID", "Tm", "G", "GS", "outs", "ERA", "FIP_ytd", "gap_ros"]],
        on="mlbID",
    )
    out = out.rename(columns={"ERA": "ERA_ytd"})
    out["IP"] = out["outs"] / 3
    out["era_ros"] = out["pred_era_proof"]
    out["delta"] = out["pred_fip_proof"] - out["pred_fip_prior"]
    out["lo80"] = out["pred_fip_proof"] - Z80 * out["sd_proof"]
    out["hi80"] = out["pred_fip_proof"] + Z80 * out["sd_proof"]
    return out[
        [
            "mlbID",
            "Name",
            "Tm",
            "Age",
            "role",
            "IP",
            "BF_ytd",
            "ERA_ytd",
            "FIP_ytd",
            "pred_fip_prior",
            "pred_fip_proof",
            "lo80",
            "hi80",
            "era_ros",
            "delta",
        ]
    ].sort_values("pred_fip_proof")


def build_all(as_of: dt.date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = as_of or (dt.date.today() - dt.timedelta(days=1))
    hitters, pitchers = build_hitters(as_of), build_pitchers(as_of)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    hitters.to_parquet(LIVE_DIR / f"hitters_{as_of}.parquet", index=False)
    pitchers.to_parquet(LIVE_DIR / f"pitchers_{as_of}.parquet", index=False)
    return hitters, pitchers


if __name__ == "__main__":
    h, p = build_all()
    print(f"hitters: {len(h)} | pitchers: {len(p)}")
    print(h.head(10).to_string(index=False))
