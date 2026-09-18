"""Baseball-Reference date-range batting pulls via pybaseball, pinned to snapshots.

One HTTP request covers an arbitrary date range for all of MLB, which is all
the walk-forward backtest needs — pitch-level Statcast is a phase-2 concern
(see writeup/spec-audit.md, item 14).

Rate limits: Baseball-Reference 429s aggressive scrapers. Every request is
serialized through a module-level polite delay, and results are pinned to
immutable snapshots so each range is ever fetched once.
"""

from __future__ import annotations

import time

import pandas as pd
from pybaseball import batting_stats_range, pitching_stats_range

from ingest import snapshots

_POLITE_DELAY_S = 8.0
_BACKOFF_S = 45.0
_MAX_ATTEMPTS = 5
_last_request_at = 0.0

# Regular-season boundaries (inclusive). Hardcoded so date windows never
# depend on a postseason-inclusive default upstream. NOTE: the BR range tool
# DOES include postseason games (verified 2026-09-17: a 2023 LCS window
# returns rows), so end dates are exactly the regular-season finale.
SEASON_DATES = {
    2020: ("2020-07-23", "2020-09-27"),  # 60-game COVID season
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-18", "2025-09-28"),
    # 2026: live dashboard pulls only; start is loose (harmless — no games
    # before opening day), end = presumed regular-season finale.
    2026: ("2026-03-01", "2026-09-27"),
}

COUNTING_COLS = [
    "G",
    "PA",
    "AB",
    "R",
    "H",
    "2B",
    "3B",
    "HR",
    "RBI",
    "BB",
    "IBB",
    "SO",
    "HBP",
    "SH",
    "SF",
    "GDP",
    "SB",
    "CS",
]
IDENTITY_COLS = ["Name", "Age", "Tm"]


def _polite_get(fetch, start: str, end: str) -> pd.DataFrame:
    """Serialize requests with a fixed delay; on a block (BR serves a page with
    no stats table, which pybaseball surfaces as IndexError), back off and
    retry. A failed pull never reaches the snapshot layer, so reruns resume."""
    global _last_request_at
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        wait = _POLITE_DELAY_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        try:
            df = fetch(start, end)
        except (IndexError, ValueError, KeyError) as exc:
            df = None
            print(f"    blocked/error ({type(exc).__name__}), attempt {attempt}")
        if df is not None and not df.empty:
            return df
        if attempt < _MAX_ATTEMPTS:
            time.sleep(_BACKOFF_S * attempt)
    raise RuntimeError(f"fetch failed after {_MAX_ATTEMPTS} attempts: {start}..{end}")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and collapse multi-team players to one row per mlbID.

    Traded players can appear once per team in a range pull; summing the
    counting stats and keeping identity fields from the highest-PA stint is
    the honest collapse for rate-stat modeling.
    """
    df = df.copy()
    df["mlbID"] = pd.to_numeric(df["mlbID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["mlbID"])
    for col in COUNTING_COLS + ["Age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = (
        df.sort_values("PA", ascending=False)
        .groupby("mlbID", as_index=False)
        .agg(
            {**{c: "sum" for c in COUNTING_COLS}, **{c: "first" for c in IDENTITY_COLS}}
        )
    )
    return df


def fetch_range(start: str, end: str) -> pd.DataFrame:
    """Batting totals for all MLB players between start and end (inclusive)."""
    name = f"batting_{start}_{end}"
    return snapshots.get_or_fetch(
        name, lambda: _clean(_polite_get(batting_stats_range, start, end))
    )


def fetch_season(season: int) -> pd.DataFrame:
    start, end = SEASON_DATES[season]
    return fetch_range(start, end)


# ---------------------------------------------------------------- pitching

PITCHING_COUNTING_COLS = [
    "G",
    "GS",
    "W",
    "L",
    "SV",
    "outs",
    "H",
    "R",
    "ER",
    "BB",
    "SO",
    "HR",
    "HBP",
    "AB",
    "2B",
    "3B",
    "IBB",
    "GDP",
    "SF",
    "SB",
    "CS",
    "BF",
]


def ip_to_outs(ip: pd.Series) -> pd.Series:
    """Baseball-notation IP (4.2 = four and two-thirds) -> outs.

    Summing raw IP floats corrupts fractions (0.2 + 0.2 = 0.4, not 1.1), so
    conversion must happen before any aggregation.
    """
    ip = pd.to_numeric(ip, errors="coerce").fillna(0)
    whole = ip.astype(int)
    return whole * 3 + ((ip - whole) * 10).round().astype(int)


def _clean_pitching(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mlbID"] = pd.to_numeric(df["mlbID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["mlbID"])
    df["outs"] = ip_to_outs(df["IP"])
    for col in PITCHING_COUNTING_COLS + ["Age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = (
        df.sort_values("BF", ascending=False)
        .groupby("mlbID", as_index=False)
        .agg(
            {
                **{c: "sum" for c in PITCHING_COUNTING_COLS},
                **{c: "first" for c in IDENTITY_COLS},
            }
        )
    )
    df["ERA"] = 27 * df["ER"] / df["outs"].clip(lower=1)
    return df


def fetch_pitching_range(start: str, end: str) -> pd.DataFrame:
    """Pitching totals for all MLB pitchers between start and end (inclusive)."""
    name = f"pitching_{start}_{end}"
    return snapshots.get_or_fetch(
        name, lambda: _clean_pitching(_polite_get(pitching_stats_range, start, end))
    )


def fetch_pitching_season(season: int) -> pd.DataFrame:
    start, end = SEASON_DATES[season]
    return fetch_pitching_range(start, end)
