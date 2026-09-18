"""Validate Savant-reconstructed pitching lines against pinned BR snapshots.

    uv run python -m ingest.validate_statcast

The pitcher pipeline consumes the two sources interchangeably; that is only
honest if they agree up to known definitional quirks (BR BF can differ by a
batter on interference/OBP quirks; outs reconstruction misses inning-ending
caught-stealings). This prints agreement stats on every window where both
sources exist and exits nonzero if agreement degrades.
"""

from __future__ import annotations

import sys

import pandas as pd

from backtest.checkpoints import grid
from ingest import snapshots
from ingest.baseball_ref import SEASON_DATES
from ingest.statcast import pitching_lines

CHECK_COLS = ["BF", "SO", "BB", "IBB", "HBP", "HR", "outs", "G", "GS"]
TOL_CORR = 0.995
TOL_MED_ABS_RATE = 0.01  # per-BF


def compare(season: int, start: str, end: str) -> pd.DataFrame:
    br = snapshots.load(f"pitching_{start}_{end}")
    sc = pitching_lines(season, start, end)
    m = br.merge(sc, on="mlbID", suffixes=("_br", "_sc"), how="inner")
    rows = []
    for col in CHECK_COLS:
        a, b = m[f"{col}_br"], m[f"{col}_sc"]
        corr = a.corr(b) if len(m) > 2 else float("nan")
        med_rate = ((a - b).abs() / m["BF_br"].clip(lower=1)).median()
        rows.append(
            {
                "col": col,
                "corr": round(float(corr), 5),
                "med_abs_rate": round(float(med_rate), 5),
                "max_abs": int((a - b).abs().max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    windows = [(2023, *SEASON_DATES[2023])]
    windows += [(2023, cp["ytd_start"], cp["ytd_end"]) for cp in grid(2023)[:2]]
    bad = False
    for season, start, end in windows:
        if not snapshots.exists(f"pitching_{start}_{end}"):
            continue
        rep = compare(season, start, end)
        ev = rep[rep["col"].isin(["BF", "SO", "BB", "IBB", "HBP", "HR"])]
        ok = bool((ev["corr"] >= TOL_CORR).all())
        print(f"{start}..{end}  {'OK ' if ok else 'FAIL'}")
        print(rep.to_string(index=False))
        bad |= not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
