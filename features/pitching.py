"""Pitcher component space: per-batter-faced rates + FIP/ERA construction.

Per the spec, FIP components are projected separately and translated to ERA.
Correction carried from the audit (item 8/9): with BR-only data there is no
fly-ball denominator, so HR is modeled per BF, making this component-FIP,
not xFIP. ERA = FIP + a heavily-shrunk personal ERA-FIP gap.

Components (all per BF):
  * SO      — strikeouts; stabilizes fastest (~70 BF)
  * uBBHBP  — unintentional walks + hit batters (~170 BF)
  * HR      — home runs allowed; very slow (~800 BF)
  * bf_per_ip — batters faced per inning (sequencing/efficiency tendency)
"""

from __future__ import annotations

import pandas as pd

PITCH_EVENTS = ["SO", "uBBHBP", "HR"]
FIP_WEIGHT = {"SO": -2.0, "uBBHBP": 3.0, "HR": 13.0}


def add_pitch_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["uBBHBP"] = out["BB"] - out["IBB"] + out["HBP"]
    bf = out["BF"].clip(lower=1)
    for ev in PITCH_EVENTS:
        out[f"r_{ev}"] = out[ev] / bf
    out["r_bfip"] = 3 * out["BF"] / out["outs"].clip(lower=1)
    out["role"] = [
        "SP" if g > 0 and gs / g >= 0.5 else "RP" for gs, g in zip(out["GS"], out["G"])
    ]
    return out


def fip_from_rates(df: pd.DataFrame, cfip: float, suffix: str = "") -> pd.Series:
    """FIP from per-BF event rates: (13HR + 3uBBHBP - 2SO)/BF x BF/IP + cFIP.

    `suffix` selects the rate columns: "" for observed, "_prior", "_ros"."""
    per_bf = sum(df[f"r_{ev}{suffix}"] * w for ev, w in FIP_WEIGHT.items())
    return per_bf * df[f"r_bfip{suffix}"] + cfip


def league_pitching_rates(df: pd.DataFrame) -> dict:
    """BF-weighted league rates overall and by role (SP/RP run environments differ)."""
    out = {}
    for scope, sub in [("all", df)] + list(df.groupby("role")):
        bf = sub["BF"].sum()
        for ev in PITCH_EVENTS:
            out[f"{scope}_{ev}"] = sub[ev].sum() / bf
        out[f"{scope}_bfip"] = 3 * sub["BF"].sum() / sub["outs"].sum()
    return out
