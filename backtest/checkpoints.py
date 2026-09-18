"""Pre-registered walk-forward evaluation grid.

Checkpoints are the 1st of every month, May through September, fixed before
any model tuning — evaluating at many ad-hoc dates and reporting the best
would be checkpoint shopping.
"""

from __future__ import annotations

import pandas as pd

from ingest.baseball_ref import SEASON_DATES

CHECKPOINT_MONTHS = [5, 6, 7, 8, 9]


def grid(season: int) -> list[dict]:
    start, end = SEASON_DATES[season]
    out = []
    for month in CHECKPOINT_MONTHS:
        cp = pd.Timestamp(season, month, 1)
        if cp <= pd.Timestamp(start) or cp > pd.Timestamp(end):
            continue
        out.append(
            {
                "season": season,
                "checkpoint": cp.date().isoformat(),
                "ytd_start": start,
                "ytd_end": (cp - pd.Timedelta(days=1)).date().isoformat(),
                "ros_start": cp.date().isoformat(),
                "ros_end": end,
            }
        )
    return out
