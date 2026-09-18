"""The PROOF engine: per-component empirical-Bayes shrinkage of YTD toward prior.

Each event rate is blended as

    ros_rate = (PA_ytd * ytd_rate + k_event * prior_rate) / (PA_ytd + k_event)

where k_event is a Carleton-style stabilization point (the PA at which the
observed rate is ~half signal). Different k per event is the whole point:
K% stabilizes in ~60 PA, singles take ~290, doubles/triples effectively
never within one season.

Honest caveats, stated here so the write-up can quote them:
  * These k's were derived against league-mean priors. A player-specific
    prior has lower variance, so the optimal k is somewhat larger — v1
    leans slightly toward the data. The backtest measures the cost.
  * 2B/3B exact values are immaterial: any k far above one season of PA
    means "shrink to prior," which is what both do.
  * Players with no prior history get the trailing league mean as their
    prior (designed behavior — rookies are league-average until seen).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.components import EVENTS, WEIGHT_FOR

K_SHRINK = {
    "SO": 60,
    "uBB": 120,
    "HR": 170,
    "HBP": 240,
    "1B": 290,
    "2B": 1450,
    "3B": 1450,
}
K_IBBSH = 1000  # tiny rates, tiny wOBA impact; shrink hard

# Irreducible within-season talent/role/injury variation: even with infinite
# prior information, a player's rest-of-season true wOBA is not fixed.
# Method-of-moments estimate on 2023-2024 training seasons only
# (residual^2 minus modeled variance, PA-weighted), at k_mult=4.0.
# Without this term the tuned model's intervals are overconfident.
SIGMA_DRIFT2 = 0.000237  # ~0.015 wOBA of "players change" sd


def attach_prior(
    ytd: pd.DataFrame, prior: pd.DataFrame, league_prior: dict
) -> pd.DataFrame:
    """Left-join the prior onto YTD stats; missing history -> league mean."""
    m = ytd.merge(prior, on="mlbID", how="left", suffixes=("", "_p"))
    m["PA_hist"] = m["PA_hist"].fillna(0)
    m["Name"] = m["Name"].fillna(m["Name_p"])
    m["Age"] = m["Age"].fillna(m["Age_p"])
    for ev in EVENTS:
        m[f"r_{ev}_prior"] = m[f"r_{ev}_prior"].fillna(league_prior[ev])
    m["r_IBBSH_prior"] = m["r_IBBSH_prior"].fillna(league_prior["IBBSH"])
    return m.drop(columns=[c for c in m.columns if c.endswith("_p")])


def _blend_rates(
    m: pd.DataFrame,
    prior_suffix: str = "_prior",
    league: dict | None = None,
    k_mult: float = 1.0,
) -> pd.DataFrame:
    out = m[["mlbID"]].copy()
    for ev in EVENTS:
        k = K_SHRINK[ev] * k_mult
        pcol = f"r_{ev}{prior_suffix}"
        prate = league[ev] if league is not None else m[pcol]
        out[f"r_{ev}_ros"] = (m["PA"] * m[f"r_{ev}"] + k * prate) / (m["PA"] + k)
    pibb = league["IBBSH"] if league is not None else m[f"r_IBBSH{prior_suffix}"]
    k_ibbsh = K_IBBSH * k_mult
    out["r_IBBSH_ros"] = (m["PA"] * m["r_IBBSH"] + k_ibbsh * pibb) / (m["PA"] + k_ibbsh)
    return out


def _woba(df: pd.DataFrame, weights: dict, suffix: str) -> pd.Series:
    numer = sum(
        df[f"r_{ev}{suffix}"] * weights[WEIGHT_FOR[ev]]
        for ev in EVENTS
        if ev in WEIGHT_FOR
    )
    denom = (1 - df[f"r_IBBSH{suffix}"]).clip(lower=0.9)
    return numer / denom


def _predictive_sd(
    m: pd.DataFrame,
    blended: pd.DataFrame,
    weights: dict,
    pa_ros: pd.Series,
    k_mult: float = 1.0,
) -> pd.Series:
    """Normal-approx predictive sd for ROS wOBA.

    var = per-PA multinomial variance of the wOBA numerator
          x (talent uncertainty 1/(PA_ytd + k_eff) + sampling 1/PA_ros).

    `pa_ros` is the *realized* ROS playing time in the backtest; a live
    system substitutes projected PA. The mean projection never sees it.
    """
    p = {ev: blended[f"r_{ev}_ros"] for ev in EVENTS}
    denom = (1 - blended["r_IBBSH_ros"]).clip(lower=0.9)
    c = {ev: weights[WEIGHT_FOR[ev]] / denom for ev in WEIGHT_FOR}  # SO/outs: 0
    mean = sum(p[ev] * c[ev] for ev in WEIGHT_FOR)
    var_pa = sum(p[ev] * (c[ev] - mean) ** 2 for ev in WEIGHT_FOR)
    p_out = (1 - sum(p.values())).clip(lower=0)
    var_pa = var_pa + p_out * mean**2  # outs contribute (0 - mean)^2

    k_eff = sum(p[ev] * K_SHRINK[ev] for ev in EVENTS) * k_mult
    var = var_pa * (1.0 / (m["PA"] + k_eff) + 1.0 / pa_ros.clip(lower=1)) + SIGMA_DRIFT2
    return np.sqrt(var)


def project_all(
    m: pd.DataFrame,
    weights: dict,
    league_prior: dict,
    pa_ros: pd.Series,
    k_mult: float = 1.0,
) -> pd.DataFrame:
    """All four systems on one merged frame; returns predictions + sd."""
    out = m[["mlbID", "Name", "Age", "PA"]].rename(columns={"PA": "PA_ytd"})

    out["pred_woba_naive"] = _woba(m, weights, "")
    out["pred_woba_prior"] = _woba(m, weights, "_prior")

    blended = _blend_rates(m, k_mult=k_mult)
    out["pred_woba_proof"] = _woba(blended, weights, "_ros")

    blended_lg = _blend_rates(m, league=league_prior, k_mult=k_mult)
    out["pred_woba_eb_league"] = _woba(blended_lg, weights, "_ros")

    out["sd_proof"] = _predictive_sd(m, blended, weights, pa_ros, k_mult)
    return out
