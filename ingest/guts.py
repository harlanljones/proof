"""Season run-environment constants from the FanGraphs "guts" page.

wOBA is only defined relative to a season's run environment, so recombining
projected components requires that season's linear weights; FIP needs the
season cFIP constant so league FIP equals league ERA. The table is one tiny
public page, fetched once and pinned like everything else. If the fetch
fails (markup drift), we fall back to hardcoded recent values and say so —
the year-to-year variation is immaterial next to model error, and every
system in a comparison shares the same constants.

Snapshot history: `woba_guts` (v1, wOBA weights only) is preserved untouched
per the write-once rule; `fg_guts_v2` adds cFIP for the pitcher module.
"""

from __future__ import annotations

import time

import pandas as pd

from ingest import snapshots

GUTS_URL = "https://www.fangraphs.com/guts.aspx?type=cn"
KEEP_COLS = [
    "Season",
    "wOBAScale",
    "wBB",
    "wHBP",
    "w1B",
    "w2B",
    "w3B",
    "wHR",
    "wOBA",
    "cFIP",
]

# Fallback: 2024 season values, used only if the live fetch fails.
_FALLBACK = {
    "Season": 2024,
    "wOBAScale": 1.0196,
    "wBB": 0.689,
    "wHBP": 0.720,
    "w1B": 0.879,
    "w2B": 1.241,
    "w3B": 1.569,
    "wHR": 2.064,
    "wOBA": 0.310,
    "cFIP": 3.161,
    "source": "hardcoded-2024",
}


def _scrape_guts() -> pd.DataFrame:
    tables = pd.read_html(GUTS_URL)
    df = max(tables, key=len)  # the guts table is the large one
    df = df.rename(columns=lambda c: str(c).strip())
    df = df[KEEP_COLS].rename(columns={"wOBA": "lg_wOBA"})
    df = df[pd.to_numeric(df["Season"], errors="coerce").notna()]
    df = df.astype(float)
    df["Season"] = df["Season"].astype(int)
    df["source"] = "fangraphs-guts"
    time.sleep(5)  # polite: one request, but still
    return df.reset_index(drop=True)


def season_weights(season: int) -> dict:
    """Run-environment constants for a season (wOBA weights + cFIP).

    Falls back to hardcoded 2024 values with a source flag."""
    df = snapshots.get_or_fetch("fg_guts_v2", _scrape_guts)
    row = df[df["Season"] == season]
    if row.empty:
        return {**_FALLBACK, "Season": season}
    return row.iloc[0].to_dict()
