"""Pre-pull every snapshot the backtests need, politely.

    uv run python -m ingest.prefetch          # everything missing
    uv run python -m ingest.prefetch --pitchers-only

Baseball-Reference 429s aggressive clients (we earned one during development;
the write-once snapshot policy meant zero data corruption and an exact
resume point). This runner spaces requests far apart and gives up gracefully
so it can be re-run until complete.
"""

from __future__ import annotations

import argparse
import sys
import time

from backtest.checkpoints import grid
from ingest import snapshots
from ingest.baseball_ref import (
    SEASON_DATES,
    fetch_pitching_range,
    fetch_range,
)

BACKTEST_SEASONS = [2023, 2024, 2025]
PRIOR_SEASONS = [2020, 2021, 2022]
SPACING_S = 30.0


def _needed_ranges() -> list[tuple[str, str]]:
    ranges = set()
    for s in BACKTEST_SEASONS + PRIOR_SEASONS:
        ranges.add(SEASON_DATES[s])
    for s in BACKTEST_SEASONS:
        for cp in grid(s):
            ranges.add((cp["ytd_start"], cp["ytd_end"]))
            ranges.add((cp["ros_start"], cp["ros_end"]))
    return sorted(ranges)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitchers-only", action="store_true")
    args = ap.parse_args()

    kinds = [("pitching", fetch_pitching_range)]
    if not args.pitchers_only:
        kinds.insert(0, ("batting", fetch_range))

    missing = 0
    for kind, fetch in kinds:
        for start, end in _needed_ranges():
            name = f"{kind}_{start}_{end}"
            if snapshots.exists(name):
                continue
            print(f"pulling {name} ...", flush=True)
            try:
                fetch(start, end)
            except RuntimeError as exc:
                print(
                    f"FAILED persistently ({exc}); stopping — rerun later.", flush=True
                )
                sys.exit(1)
            missing += 1
            time.sleep(SPACING_S)
    print(f"done. {missing} snapshots pulled.")


if __name__ == "__main__":
    main()
