"""Player identity (name, birthdate) from the MLB Stats API.

Baseball-Reference owns the stats tables but Stats API (statsapi.mlb.com) is
a separate host with a bulk player endpoint — one call per season returns
the full pool with birthdates. Used to attach Name/Age to Savant
reconstructions, which carry only mlbIDs.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from ingest import snapshots

PLAYERS_URL = "https://statsapi.mlb.com/api/v1/sports/1/players"


def fetch_identity(season: int) -> pd.DataFrame:
    name = f"identity_{season}"

    def _fetch() -> pd.DataFrame:
        r = requests.get(
            PLAYERS_URL,
            params={"season": season, "hydrate": "person"},
            timeout=60,
        )
        r.raise_for_status()
        rows = [
            {
                "mlbID": p["id"],
                "Name": p.get("fullName"),
                "birthDate": p.get("birthDate"),
            }
            for p in r.json()["people"]
        ]
        time.sleep(2)
        return pd.DataFrame(rows)

    return snapshots.get_or_fetch(name, _fetch)


def identity_for_seasons(seasons: list[int]) -> pd.DataFrame:
    """Union of season pools; one row per mlbID with birthDate."""
    frames = [fetch_identity(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("mlbID", keep="last")
    df["birthDate"] = pd.to_datetime(df["birthDate"])
    return df


def with_identity(
    lines: pd.DataFrame, season: int, pool_seasons: list[int]
) -> pd.DataFrame:
    """Attach Name and BR-convention Age (age on June 30 of `season`)."""
    ids = identity_for_seasons(pool_seasons)
    out = lines.merge(ids, on="mlbID", how="left")
    june30 = pd.Timestamp(season, 6, 30)
    age = (june30 - out["birthDate"]).dt.days // 365
    out["Age"] = age.fillna(28).astype(int)  # unknown DOB: league-ish default
    out["Name"] = out["Name"].fillna("mlbID " + out["mlbID"].astype(str))
    return out.drop(columns=["birthDate"])
