"""Component math: wOBA reconstruction must match the textbook formula."""

import pandas as pd
import pytest

from features.components import add_events, woba_from_rates

W = {"wBB": 0.69, "wHBP": 0.72, "w1B": 0.88, "w2B": 1.24, "w3B": 1.57, "wHR": 2.06}


@pytest.fixture
def player():
    return pd.DataFrame(
        [
            {
                "PA": 96,
                "AB": 84,
                "H": 25,
                "2B": 4,
                "3B": 1,
                "HR": 5,
                "BB": 10,
                "IBB": 0,
                "SO": 20,
                "HBP": 1,
                "SH": 0,
                "SF": 1,
            }
        ]
    )


def test_event_rates(player):
    df = add_events(player)
    row = df.iloc[0]
    assert row["uBB"] == 10
    assert row["1B"] == 15  # H - 2B - 3B - HR
    assert row["r_HR"] == pytest.approx(5 / 96)
    assert row["r_denom"] == pytest.approx(1.0)  # no IBB/SH


def test_woba_matches_formula(player):
    df = add_events(player)
    got = woba_from_rates(df, W).iloc[0]
    # textbook: (wBB*uBB + wHBP*HBP + w1B*1B + w2B*2B + w3B*3B + wHR*HR)
    #           / (AB + BB - IBB + HBP + SF)
    numer = 0.69 * 10 + 0.72 * 1 + 0.88 * 15 + 1.24 * 4 + 1.57 * 1 + 2.06 * 5
    denom = 84 + 10 - 0 + 1 + 1
    assert got == pytest.approx(numer / denom, abs=1e-9)


def test_iso_babip(player):
    df = add_events(player)
    row = df.iloc[0]
    assert row["ISO"] == pytest.approx((4 + 2 * 1 + 3 * 5) / 84)
    assert row["BABIP"] == pytest.approx((25 - 5) / (84 - 20 - 5 + 1))
