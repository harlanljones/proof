"""Pitch-level Statcast pulls (Baseball Savant) -> local pitcher-line aggregates.

Why this exists: Baseball-Reference 429-banned the build IP mid-pull
(2026-09-17). Savant is a different host with its own tolerance, and
pitch-level data lets us reconstruct pitching lines for ANY date window
locally — no per-range HTTP at all. The reconstruction is validated against
the BR snapshots already pinned (see `python -m ingest.validate_statcast`).

ERA is NOT reconstructible from pitch-level data (no earned/unearned
distinction), so the 2023-25 pitcher backtest targets FIP; ERA evaluation
runs on 2023 BR actuals only. Kept columns include contact quality for the
phase-2 xwOBA features — this pull does double duty.
"""

from __future__ import annotations

import time

import pandas as pd
from pybaseball import statcast

from ingest import snapshots
from ingest.baseball_ref import SEASON_DATES

KEEP_COLS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "pitcher",
    "events",
    # phase-2 contact quality (xwOBA features); cheap to keep now
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
]

# Outs credited to the pitcher for the PA-ending event itself.
EVENT_OUTS = {
    "strikeout": 1,
    "field_out": 1,
    "force_out": 1,
    "fielders_choice_out": 1,
    "batter_interference": 1,
    "sac_fly": 1,
    "sac_bunt": 1,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "sac_fly_double_play": 2,
    "sac_bunt_double_play": 2,
    "strikeout_double_play": 2,
    "triple_play": 3,
}


def _month_windows(season: int) -> list[tuple[str, str]]:
    start = pd.Timestamp(SEASON_DATES[season][0])
    end = pd.Timestamp(SEASON_DATES[season][1])
    out = []
    cur = start
    while cur <= end:
        nxt = min(cur + pd.offsets.MonthEnd(0), end)
        out.append((cur.date().isoformat(), nxt.date().isoformat()))
        cur = nxt + pd.Timedelta(days=1)
    return out


def fetch_pitch_level(season: int) -> pd.DataFrame:
    """Whole-season pitch-level data, monthly chunks, one pinned snapshot."""
    name = f"pitchlevel_{season}"

    def _fetch() -> pd.DataFrame:
        chunks = []
        for s, e in _month_windows(season):
            print(f"    statcast {s}..{e}", flush=True)
            df = statcast(s, e, verbose=False)
            df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
            chunks.append(df)
            time.sleep(8)  # Savant is tolerant; stay that way
        return pd.concat(chunks, ignore_index=True)

    return snapshots.get_or_fetch(name, _fetch)


def _attach_outs(pa: pd.DataFrame) -> pd.Series:
    """Outs recorded by the pitcher on each completed PA.

    Vectorized delta of `outs_when_up` within a half-inning (captures runner
    outs between PAs). A half-inning's last PA gets 3 - outs_when_up when the
    half-inning completed, else just the event's own outs (walk-off/shortened
    final half-inning; the rare inning-ending caught-stealing is missed —
    documented, negligible).
    """
    pa = pa.sort_values(["game_pk", "at_bat_number"])
    half = [pa["game_pk"], pa["inning"], pa["inning_topbot"]]

    nxt_outs = pa.groupby(half)["outs_when_up"].shift(-1)
    same_half = nxt_outs.notna()
    delta = (nxt_outs - pa["outs_when_up"]).where(same_half)

    # Final half-inning of each game: (max inning, last topbot seen there).
    last_rows = pa.groupby("game_pk").tail(1)[["game_pk", "inning", "inning_topbot"]]
    final_half = set(
        zip(last_rows["game_pk"], last_rows["inning"], last_rows["inning_topbot"])
    )
    is_final_half = pd.Series(
        zip(pa["game_pk"], pa["inning"], pa["inning_topbot"]), index=pa.index
    ).isin(final_half)

    ev_outs = pa["events"].map(EVENT_OUTS).fillna(0)
    completes = (~is_final_half) | (pa["outs_when_up"] + ev_outs >= 3)
    last_delta = pd.Series(
        (3 - pa["outs_when_up"]).where(completes, ev_outs), index=pa.index
    )
    return delta.fillna(last_delta).clip(lower=0)


def pitching_lines(season: int, start: str, end: str) -> pd.DataFrame:
    """Reconstruct per-pitcher lines for a date window from pitch-level data.

    Columns match the BR pitching ingest contract (outs, BF, SO, BB, IBB,
    HBP, HR, G, GS) so features/models consume either source unchanged.
    """
    df = snapshots.load(f"pitchlevel_{season}")
    df = df[(df["game_date"] >= start) & (df["game_date"] <= end)]
    pa = df[df["events"].notna() & (df["events"] != "truncated_pa")].copy()
    # index-aligned assignment: _attach_outs sorts internally
    pa["outs_rec"] = _attach_outs(pa)

    pa["uBB"] = (pa["events"] == "walk").astype(int)
    pa["IBB"] = (pa["events"] == "intent_walk").astype(int)
    pa["SO"] = pa["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    pa["HBP"] = (pa["events"] == "hit_by_pitch").astype(int)
    pa["HR"] = (pa["events"] == "home_run").astype(int)

    agg = pa.groupby("pitcher").agg(
        BF=("events", "size"),
        SO=("SO", "sum"),
        uBB=("uBB", "sum"),
        IBB=("IBB", "sum"),
        HBP=("HBP", "sum"),
        HR=("HR", "sum"),
        G=("game_pk", "nunique"),
        outs=("outs_rec", "sum"),
    )

    # GS: pitcher of the game's opening PA on each side is the starter.
    first_pa = (
        pa.sort_values("at_bat_number").groupby(["game_pk", "inning_topbot"]).first()
    )
    gs = first_pa.groupby("pitcher").size().rename("GS")
    agg = agg.join(gs).fillna({"GS": 0})

    agg["BB"] = agg["uBB"] + agg["IBB"]
    agg = agg.reset_index().rename(columns={"pitcher": "mlbID"})
    for col in [
        "mlbID",
        "BF",
        "SO",
        "BB",
        "uBB",
        "IBB",
        "HBP",
        "HR",
        "G",
        "GS",
        "outs",
    ]:
        agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0).astype("int64")
    return agg
