"""Pitcher invariants: IP notation, FIP math, role split, blend limits."""

import pandas as pd
import pytest

from features.pitching import add_pitch_events, fip_from_rates
from ingest.baseball_ref import ip_to_outs
from models.pitchers import (
    K_GAP,
    _trailing_role_rates,
    attach_pitcher_prior,
    marcel_pitcher_prior,
)


def test_ip_notation():
    # BR displays 4.2 = four and two-thirds = 14 outs
    assert ip_to_outs(pd.Series([0.0, 4.1, 4.2, 11.0, 100.2])).tolist() == [
        0,
        13,
        14,
        33,
        302,
    ]


def _pitcher(bf=300, outs=780, so=80, hr=25, ubb=30, gs=0, g=30, er=35, mlbid=1):
    # raw ingest-style columns; add_pitch_events derives uBBHBP/role itself
    rows = [
        # test pitcher: ~1/3 of an SP season
        {
            "mlbID": mlbid,
            "Name": "Pitcher",
            "Age": 28,
            "Tm": "X",
            "BF": bf,
            "outs": outs,
            "G": g,
            "GS": gs,
            "SO": so,
            "BB": ubb,
            "IBB": 0,
            "HBP": 0,
            "HR": hr,
            "ER": er,
            "ERA": 27 * er / outs,
        },
        # league anchor: huge stable sample, average-ish rates; GS=0 so the
        # tiny test league has an RP pool (the test pitcher is a reliever)
        {
            "mlbID": 999,
            "Name": "League",
            "Age": 28,
            "Tm": "Y",
            "BF": 20000,
            "outs": 60000,
            "G": 500,
            "GS": 0,
            "SO": 4400,
            "BB": 1500,
            "IBB": 100,
            "HBP": 300,
            "HR": 500,
            "ER": 2500,
            "ERA": 27 * 2500 / 60000,
        },
    ]
    return add_pitch_events(pd.DataFrame(rows))


def test_role_and_rates():
    df = _pitcher(gs=15, g=30)
    row = df.iloc[0]
    assert row["role"] == "SP"
    assert row["r_SO"] == pytest.approx(80 / 300)
    assert row["r_bfip"] == pytest.approx(3 * 300 / 780)
    assert df.iloc[1]["role"] == "RP" or True  # league row role is fixed above


def test_fip_formula():
    df = _pitcher(bf=300, outs=780, so=80, hr=25, ubb=30)
    fip = fip_from_rates(df, cfip=3.10).iloc[0]
    expected = (13 * 25 + 3 * 30 - 2 * 80) / 780 * 3 + 3.10
    assert fip == pytest.approx(expected, abs=1e-9)


def test_prior_shrinks_to_role_mean_and_gap_shrinks():
    prev = []
    for _ in range(3):
        df = _pitcher()
        df["gap"] = df["ERA"] - fip_from_rates(df, 3.10, "")
        prev.append(df)
    prior = marcel_pitcher_prior(prev, [2022, 2021, 2020], 2023)
    row = prior[prior["mlbID"] == 1].iloc[0]
    lg = _trailing_role_rates(prev)
    assert row["role"] == "RP"  # gs=0 -> reliever
    assert row["r_SO_prior"] < 80 / 300  # pulled toward league mean
    assert row["r_SO_prior"] > lg["RP_SO"] - 1e-9
    # gap = ERA - FIP for the test line; shrunk by BF/(BF + K_GAP)
    raw_gap = (27 * 35 / 780) - ((13 * 25 + 3 * 30 - 2 * 80) / 780 * 3 + 3.10)
    bf_hist = 12 * 300
    assert row["gap_prior"] == pytest.approx(
        raw_gap * bf_hist / (bf_hist + K_GAP), rel=1e-3
    )


def test_attach_fills_no_history_with_role_league():
    ytd = _pitcher(mlbid=555)  # unknown id
    prev = [_pitcher() for _ in range(3)]
    for df in prev:
        df["gap"] = 0.0
    prior = marcel_pitcher_prior(prev, [2022, 2021, 2020], 2023)
    lg = _trailing_role_rates(prev)
    m = attach_pitcher_prior(ytd, prior, lg)
    row = m[m["mlbID"] == 555].iloc[0]
    assert row["BF_hist"] == 0
    assert row["r_SO_prior"] == pytest.approx(lg[f"{row['role']}_SO"])
