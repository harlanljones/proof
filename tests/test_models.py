"""Model invariants: regression direction, blend limits, no-history players."""

import pandas as pd
import pytest

from features.components import EVENTS, add_events
from models.blend import K_SHRINK, _blend_rates, attach_prior
from models.priors import REGRESSION_PA, marcel_prior, trailing_league_rates


def _season(pa, hr, so, age=27, mlbid=1, name="Test"):
    # The "League" row has huge, fixed PA so the league mean is stable and
    # the test player's contribution to it is negligible.
    rows = [
        {
            "mlbID": mlbid,
            "Name": name,
            "Age": age,
            "Tm": "X",
            "PA": pa,
            "AB": pa - 30,
            "R": 10,
            "H": 40,
            "2B": 5,
            "3B": 1,
            "HR": hr,
            "RBI": 20,
            "BB": 25,
            "IBB": 0,
            "SO": so,
            "HBP": 5,
            "SH": 0,
            "SF": 0,
            "GDP": 2,
            "SB": 1,
            "CS": 0,
        },
        {
            "mlbID": 999,
            "Name": "League",
            "Age": 28,
            "Tm": "Y",
            "PA": 10000,
            "AB": 8900,
            "R": 1300,
            "H": 2500,
            "2B": 450,
            "3B": 40,
            "HR": 300,
            "RBI": 1300,
            "BB": 800,
            "IBB": 20,
            "SO": 2200,
            "HBP": 110,
            "SH": 30,
            "SF": 60,
            "GDP": 200,
            "SB": 100,
            "CS": 30,
        },
    ]
    return add_events(pd.DataFrame(rows))


W = {"wBB": 0.69, "wHBP": 0.72, "w1B": 0.88, "w2B": 1.24, "w3B": 1.57, "wHR": 2.06}


def test_prior_regresses_toward_league():
    # age=29 -> aging factor 1.0, so this tests regression alone
    prev = [
        _season(700, hr=30, so=100, age=29),
        _season(350, hr=30, so=100, age=29),
        _season(700, hr=30, so=100, age=29),
    ]
    prev[2] = prev[2].iloc[1:]  # player absent 3 seasons ago; league row stays
    prior = marcel_prior(prev, [2022, 2021, 2020], 2023, W)
    row = prior[prior["mlbID"] == 1].iloc[0]
    lg = trailing_league_rates(prev)
    observed = (5 * 30 + 4 * 30) / (5 * 700 + 4 * 350)  # 5/4/3-weighted HR rate
    assert lg["HR"] < row["r_HR_prior"] < observed  # pulled toward, not past, league


def test_prior_regresses_small_samples_harder():
    prior = marcel_prior(
        [_season(600, hr=30, so=60, age=29)] * 3, [2022, 2021, 2020], 2023, W
    )
    full = prior[prior["mlbID"] == 1].iloc[0]["r_HR_prior"]
    small = _season(60, hr=3, so=6, age=29)  # same HR rate, tiny PA
    prior_s = marcel_prior([small] * 3, [2022, 2021, 2020], 2023, W)
    shrunk = prior_s[prior_s["mlbID"] == 1].iloc[0]["r_HR_prior"]
    assert shrunk < full  # same rate, less PA -> more shrinkage toward league


def test_prior_ages_down_veterans():
    prev = [_season(700, hr=30, so=100, age=35)] * 3
    prior = marcel_prior(prev, [2022, 2021, 2020], 2023, W)
    row = prior[prior["mlbID"] == 1].iloc[0]
    assert row["Age"] == 36


def test_blend_limits():
    ytd = _season(0, hr=0, so=0).iloc[[0]]  # 0 PA YTD
    ytd["PA"] = 0
    prev = [_season(700, hr=30, so=100)] * 3
    prior = marcel_prior(prev, [2022, 2021, 2020], 2023, W)
    lg = trailing_league_rates(prev)
    m = attach_prior(ytd, prior, lg)
    blended = _blend_rates(m)
    for ev in EVENTS:
        assert blended[f"r_{ev}_ros"].iloc[0] == pytest.approx(
            m[f"r_{ev}_prior"].iloc[0]
        )  # no data -> prior, exactly


def test_rookie_gets_league_prior():
    ytd = _season(100, hr=5, so=10, mlbid=555)  # unknown mlbID
    prev = [_season(700, hr=30, so=100)] * 3
    prior = marcel_prior(prev, [2022, 2021, 2020], 2023, W)
    lg = trailing_league_rates(prev)
    m = attach_prior(ytd, prior, lg)
    assert m["PA_hist"].iloc[0] == 0
    for ev in EVENTS:
        assert m[f"r_{ev}_prior"].iloc[0] == pytest.approx(lg[ev])


def test_blend_weight_formula():
    ytd = _season(240, hr=4, so=20).iloc[[0]]
    prev = [_season(700, hr=30, so=100)] * 3
    prior = marcel_prior(prev, [2022, 2021, 2020], 2023, W)
    m = attach_prior(ytd, prior, trailing_league_rates(prev))
    blended = _blend_rates(m)
    k = K_SHRINK["SO"]
    expected = (240 * m["r_SO"].iloc[0] + k * m["r_SO_prior"].iloc[0]) / (240 + k)
    assert blended["r_SO_ros"].iloc[0] == pytest.approx(expected)
    assert REGRESSION_PA == 1200  # Marcel hitter constant
