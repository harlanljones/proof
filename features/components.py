"""Turn raw counting stats into the event-rate space the model lives in.

The build spec asks to project wOBA components (BB%, K%, ISO, BABIP)
separately and recombine. One correction made here: those four rates do not
uniquely determine wOBA — ISO is one equation over {2B, 3B, HR}, so two
players with identical BB%/K%/ISO/BABIP can have different wOBAs. The
minimal sufficient set is the per-PA event distribution:

    {uBB, HBP, SO, 1B, 2B, 3B, HR, other-out}

which is exactly what wOBA linear weights consume, and still stabilizes at
different rates per event — the same idea, done so recombination is exact.
BB%, K%, ISO, BABIP are reported as diagnostics.
"""

from __future__ import annotations

import pandas as pd

EVENTS = ["uBB", "HBP", "SO", "1B", "2B", "3B", "HR"]
WEIGHT_FOR = {
    "uBB": "wBB",
    "HBP": "wHBP",
    "1B": "w1B",
    "2B": "w2B",
    "3B": "w3B",
    "HR": "wHR",
}  # SO and outs: weight 0


def add_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add event counts/rates per PA, the wOBA denominator rate, and diagnostics."""
    out = df.copy()
    out["uBB"] = out["BB"] - out["IBB"]
    out["1B"] = out["H"] - out["2B"] - out["3B"] - out["HR"]
    pa = out["PA"].clip(lower=1)
    for ev in EVENTS:
        out[f"r_{ev}"] = out[ev] / pa
    # wOBA denominator per PA: (AB + BB - IBB + HBP + SF) / PA = 1 - (IBB + SH)/PA
    out["r_denom"] = 1.0 - (out["IBB"] + out["SH"]) / pa
    out["r_IBBSH"] = (out["IBB"] + out["SH"]) / pa

    # Diagnostics (the spec's named components)
    ab = out["AB"].clip(lower=1)
    out["BB%"] = out["BB"] / pa
    out["K%"] = out["SO"] / pa
    out["ISO"] = (out["1B"] * 0 + out["2B"] + 2 * out["3B"] + 3 * out["HR"]) / ab
    bip = (out["AB"] - out["SO"] - out["HR"] + out["SF"]).clip(lower=1)
    out["BABIP"] = (out["H"] - out["HR"]) / bip
    return out


def woba_from_rates(
    df: pd.DataFrame, weights: dict, prefix: str = "r_", denom_col: str = "r_denom"
) -> pd.Series:
    """wOBA from per-PA event rates and season weights.

    wOBA = Σ w_i·rate_i / denom_rate, where denom_rate ≈ 1 - (IBB+SH)/PA.
    """
    numer = sum(
        df[f"{prefix}{ev}"] * weights[WEIGHT_FOR[ev]]
        for ev in EVENTS
        if ev in WEIGHT_FOR
    )
    return numer / df[denom_col].clip(lower=0.9)


def add_woba(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = df.copy()
    out["wOBA"] = woba_from_rates(out, weights)
    return out


def league_rates(df: pd.DataFrame) -> dict:
    """PA-weighted league event rates for a season slice (the regression target)."""
    pa = df["PA"].sum()
    rates = {ev: df[ev].sum() / pa for ev in EVENTS}
    rates["IBBSH"] = (df["IBB"] + df["SH"]).sum() / pa
    return rates
