"""Write-once parquet snapshot store.

Backtests are only reproducible if the underlying data never changes under
your feet: Baseball-Reference and FanGraphs both retroactively restate
stats, and projection leaderboards are overwritten daily. So every external
pull is written once, dated, and never re-fetched. If you need fresher data,
you use a new snapshot name — you never mutate history.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SNAPSHOT_ROOT = Path(__file__).resolve().parent.parent / "data" / "snapshots"
REGISTRY = SNAPSHOT_ROOT / "registry.json"


def _registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {}


def _register(name: str, path: Path, rows: int) -> None:
    reg = _registry()
    reg[name] = {
        "file": path.name,
        "rows": rows,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    REGISTRY.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")


def exists(name: str) -> bool:
    return (SNAPSHOT_ROOT / f"{name}.parquet").exists()


def load(name: str) -> pd.DataFrame:
    path = SNAPSHOT_ROOT / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"snapshot '{name}' not found at {path}; run the ingest step that creates it"
        )
    return pd.read_parquet(path)


def get_or_fetch(name: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return the snapshot if it exists, otherwise fetch once and pin it.

    Write-once is deliberate: silently re-pulling restated data is how a
    backtest stops being reproducible.
    """
    path = SNAPSHOT_ROOT / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    if df.empty:
        raise ValueError(
            f"fetch for snapshot '{name}' returned 0 rows; refusing to pin"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    _register(name, path, len(df))
    return df


def list_snapshots() -> pd.DataFrame:
    reg = _registry()
    if not reg:
        return pd.DataFrame(columns=["name", "file", "rows", "created_utc"])
    return pd.DataFrame(reg).T.rename_axis("name").reset_index()
