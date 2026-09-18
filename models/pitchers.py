"""Pitcher prior (Marcel-style) and the EB blend for rest-of-season ERA.

Mirrors the hitter architecture with three pitcher-specific choices:

  * The regression target is the pitcher's *role* league mean (SP vs RP run
    environments differ meaningfully). Role comes from prior-season GS/G;
    no-history arms use their current-season role.
  * The ERA-FIP gap is a separate, heavily-shrunk component: real for a
    minority of pitchers, but mostly defense/park/sequencing noise.
  * Aging scales K by f and BB/HR by 1/f (same Marcel constants as hitters;
    crude — empirical aging curves are phase 2).

Regression constant: 600 BF of role-league mean (Marcel-analogous; pitcher
prior regression has no canonical constant the way 1200 PA does for
hitters — a sensitivity note for the write-up).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.pitching import FIP_WEIGHT, PITCH_EVENTS, fip_from_rates
from models.priors import SEASON_WEIGHTS, aging_factor

REGRESSION_BF = 600

# Per-component shrinkage (BF of prior added). Carleton-style stabilization
# points, same provenance caveat as the hitter constants.
K_SHRINK_P = {"SO": 70, "uBBHBP": 170, "HR": 800}
K_BFIP = 1500
K_GAP = 1200  # ERA-FIP gap is mostly noise: shrink it hard

# Extra predictive variance beyond the component model (role change, injury,
# arsenal drift). Method-of-moments on 2023-2024 training residuals only, at
# the tuned k_mult=2.0: residual^2 minus modeled variance, BF-weighted.
# ~0.23 FIP of "pitchers change" sd — over 15x the hitter drift term in
# scale-adjusted terms, which is the finding, not a nuisance.
SIGMA_P_EXTRA2 = 0.0517


def _weighted(prev: list[pd.DataFrame], seasons: list[int]) -> pd.DataFrame:
    frames = []
    for w, season, df in zip(SEASON_WEIGHTS, seasons, prev):
        keep = ["mlbID", "Name", "Age", "BF", "G", "GS", "outs", "role", "gap"]
        f = df[keep + PITCH_EVENTS].copy()
        # Savant-sourced frames carry no earned runs; gap is already zeroed.
        f["ER"] = df["ER"] if "ER" in df.columns else 0
        f["w"] = w
        f["seen_season"] = season
        frames.append(f)
    long = pd.concat(frames, ignore_index=True).sort_values(
        "seen_season", ascending=False
    )
    ident = long.groupby("mlbID").first()[["Name", "Age", "seen_season"]]

    out = ident.copy()
    out["BF_hist"] = (long["BF"] * long["w"]).groupby(long["mlbID"]).sum()
    for col in PITCH_EVENTS + ["ER", "outs", "GS", "G"]:
        out[f"wcount_{col}"] = (long[col] * long["w"]).groupby(long["mlbID"]).sum()
    out["wgap"] = (long["gap"] * long["BF"] * long["w"]).groupby(long["mlbID"]).sum()
    return out.reset_index()


def _trailing_role_rates(prev: list[pd.DataFrame]) -> dict:
    """5/4/3-weighted trailing league rates by role — the regression target."""
    out = {}
    for role in ["SP", "RP"]:
        num = dict.fromkeys(PITCH_EVENTS, 0.0)
        den = 0.0
        bfip_num = 0.0
        outs_sum = 0.0
        for w, df in zip(SEASON_WEIGHTS, prev):
            sub = df[df["role"] == role]
            den += w * sub["BF"].sum()
            outs_sum += w * sub["outs"].sum()
            bfip_num += w * 3 * sub["BF"].sum()
            for ev in PITCH_EVENTS:
                num[ev] += w * sub[ev].sum()
        for ev in PITCH_EVENTS:
            out[f"{role}_{ev}"] = num[ev] / den
        out[f"{role}_bfip"] = bfip_num / outs_sum
    return out


def marcel_pitcher_prior(
    prev: list[pd.DataFrame], seasons: list[int], target_season: int
) -> pd.DataFrame:
    """Prior for `target_season` from 3 prior seasons (most-recent-first).

    Frames must carry event counts, role, and a `gap` column (ERA - FIP,
    computed with that season's own cFIP) — see backtest/walkforward_pitchers.
    """
    ws = _weighted(prev, seasons)
    role_rates = _trailing_role_rates(prev)

    # Role from weighted games-started share.
    gs_share = ws["wcount_GS"] / ws["wcount_G"].clip(lower=1)
    ws["role"] = np.where((ws["wcount_G"] > 0) & (gs_share >= 0.5), "SP", "RP")

    for ev in PITCH_EVENTS:
        target = ws["role"].map(lambda r, ev=ev: role_rates[f"{r}_{ev}"])
        ws[f"r_{ev}_prior"] = (ws[f"wcount_{ev}"] + REGRESSION_BF * target) / (
            ws["BF_hist"] + REGRESSION_BF
        )

    obs_bfip = 3 * ws["BF_hist"] / ws["wcount_outs"].clip(lower=1)
    role_bfip = ws["role"].map(lambda r: role_rates[f"{r}_bfip"])
    ws["r_bfip_prior"] = (ws["BF_hist"] * obs_bfip + K_BFIP * role_bfip) / (
        ws["BF_hist"] + K_BFIP
    )

    # ERA-FIP gap: weighted history, shrunk hard toward 0 (mostly noise).
    ws["gap_prior"] = ws["wgap"] / (ws["BF_hist"] + K_GAP)

    ws["Age"] = ws["Age"] + (target_season - ws["seen_season"])
    f = aging_factor(ws["Age"])
    ws["r_SO_prior"] = ws["r_SO_prior"] * f
    ws["r_uBBHBP_prior"] = ws["r_uBBHBP_prior"] / f
    ws["r_HR_prior"] = ws["r_HR_prior"] / f

    keep = ["mlbID", "Name", "Age", "role", "BF_hist", "r_bfip_prior", "gap_prior"] + [
        f"r_{ev}_prior" for ev in PITCH_EVENTS
    ]
    return ws[keep].reset_index(drop=True)


def attach_pitcher_prior(
    ytd: pd.DataFrame, prior: pd.DataFrame, league_by_role: dict
) -> pd.DataFrame:
    """Left-join prior onto YTD; no-history arms get their YTD role's league mean."""
    m = ytd.merge(prior, on="mlbID", how="left", suffixes=("", "_p"))
    m["BF_hist"] = m["BF_hist"].fillna(0)
    m["Name"] = m["Name"].fillna(m["Name_p"])
    m["Age"] = m["Age"].fillna(m["Age_p"])
    m["role"] = m["role"].fillna(m["role_p"])
    for ev in PITCH_EVENTS + ["bfip"]:
        m[f"r_{ev}_prior"] = m[f"r_{ev}_prior"].fillna(
            m["role"].map(lambda r, ev=ev: league_by_role[f"{r}_{ev}"])
        )
    m["gap_prior"] = m["gap_prior"].fillna(0.0)
    return m.drop(columns=[c for c in m.columns if c.endswith("_p")])


def _blend(m: pd.DataFrame, k_mult: float, league_by_role: dict | None = None):
    out = m[["mlbID"]].copy()
    for ev in PITCH_EVENTS:
        k = K_SHRINK_P[ev] * k_mult
        prate = (
            m["role"].map(lambda r, ev=ev: league_by_role[f"{r}_{ev}"])
            if league_by_role is not None
            else m[f"r_{ev}_prior"]
        )
        out[f"r_{ev}_ros"] = (m["BF"] * m[f"r_{ev}"] + k * prate) / (m["BF"] + k)
    pbfip = (
        m["role"].map(lambda r: league_by_role[f"{r}_bfip"])
        if league_by_role is not None
        else m["r_bfip_prior"]
    )
    out["r_bfip_ros"] = (m["BF"] * m["r_bfip"] + K_BFIP * pbfip) / (m["BF"] + K_BFIP)
    return out


def _fip_sd(m, blended, bf_ros, k_mult):
    """Per-BF multinomial variance of the FIP numerator, scaled by BF/IP."""
    p = {ev: blended[f"r_{ev}_ros"] for ev in PITCH_EVENTS}
    c = {ev: FIP_WEIGHT[ev] * blended["r_bfip_ros"] for ev in PITCH_EVENTS}
    mean = sum(p[ev] * c[ev] for ev in PITCH_EVENTS)
    var_bf = sum(p[ev] * (c[ev] - mean) ** 2 for ev in PITCH_EVENTS)
    p_other = (1 - sum(p.values())).clip(lower=0)
    var_bf = var_bf + p_other * mean**2
    k_eff = sum(p[ev] * K_SHRINK_P[ev] for ev in PITCH_EVENTS) * k_mult
    var = (
        var_bf * (1.0 / (m["BF"] + k_eff) + 1.0 / bf_ros.clip(lower=1)) + SIGMA_P_EXTRA2
    )
    return np.sqrt(var)


def project_pitchers(
    m: pd.DataFrame,
    cfip: float,
    league_by_role: dict,
    bf_ros: pd.Series,
    k_mult: float = 1.0,
) -> pd.DataFrame:
    """ROS FIP and ERA projections for all four systems.

    ERA = FIP + heavily-shrunk personal ERA-FIP gap. Callers without earned
    runs (Savant path) pass ERA=NaN / gap_ytd=0 and simply ignore era cols.
    """
    out = m[["mlbID", "Name", "Age", "role", "BF"]].rename(columns={"BF": "BF_ytd"})

    blended = _blend(m, k_mult)
    blended_lg = _blend(m, k_mult, league_by_role=league_by_role)
    gap_ros = (m["BF"] * m["gap_ytd"] + K_GAP * m["gap_prior"]) / (m["BF"] + K_GAP)

    out["pred_fip_naive"] = fip_from_rates(m, cfip, "")
    out["pred_fip_prior"] = fip_from_rates(m, cfip, "_prior")
    out["pred_fip_proof"] = fip_from_rates(blended, cfip, "_ros")
    out["pred_fip_eb_league"] = fip_from_rates(blended_lg, cfip, "_ros")

    out["pred_era_naive"] = m["ERA"]
    out["pred_era_prior"] = out["pred_fip_prior"] + m["gap_prior"]
    out["pred_era_proof"] = out["pred_fip_proof"] + gap_ros
    out["pred_era_eb_league"] = out["pred_fip_eb_league"]

    out["sd_proof"] = _fip_sd(m, blended, bf_ros, k_mult)
    return out
