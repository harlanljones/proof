"""PROOF dashboard — `make dashboard` (streamlit run app/dashboard.py).

Sortable rest-of-season leaderboards, 80% intervals, and risers/fallers vs.
preseason for hitters (wOBA) and pitchers (FIP/ERA). Deliberately no WAR:
the audit scoped it out — real WAR needs playing-time, defense, and
replacement-level models this system doesn't build.
"""

from __future__ import annotations

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from app.projections import build_all

st.set_page_config(page_title="PROOF — rest-of-season projections", layout="wide")


@st.cache_data(show_spinner="Pulling today's data and projecting (first run ~1 min)...")
def load(as_of_iso: str):
    h, p = build_all(dt.date.fromisoformat(as_of_iso))
    return h, p


as_of = dt.date.today() - dt.timedelta(days=1)
hitters, pitchers = load(as_of.isoformat())

st.title("PROOF: Probabilistic Rest-Of-season Objective Forecasts")
st.caption(
    f"Projections through {as_of:%B %d, %Y}. Empirical-Bayes blend of a "
    "2023–25 Marcel prior with season-to-date performance, per-component "
    "shrinkage. Backtested 2023–25: beats naive extrapolation ~29% (hitters) "
    "and ~19% (pitchers) on weighted RMSE; 80% intervals cover ~80%. "
    "Methodology and honest failure modes: writeup/ in the repo."
)

min_pa = st.sidebar.slider("Min YTD PA (hitters)", 10, 400, 100, step=10)
min_bf = st.sidebar.slider("Min YTD BF (pitchers)", 10, 700, 150, step=10)
search = st.sidebar.text_input("Player search").strip().lower()

tab_h, tab_p, tab_about = st.tabs(["Hitters", "Pitchers", "About"])


def _filter(df: pd.DataFrame, playing_col: str, minimum: int) -> pd.DataFrame:
    out = df[df[playing_col] >= minimum]
    if search:
        out = out[out["Name"].str.lower().str.contains(search)]
    return out


def _risers_chart(df: pd.DataFrame, value_col: str, title: str) -> alt.Chart:
    top = pd.concat([df.nlargest(8, "delta"), df.nsmallest(8, "delta")])
    top = top.assign(direction=["riser"] * 8 + ["faller"] * 8)
    return (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("delta:Q", title="ROS projection − preseason prior"),
            y=alt.Y("Name:N", sort="-x"),
            color=alt.Color(
                "direction:N",
                legend=None,
                scale=alt.Scale(range=["#c0392b", "#1e8449"]),
            ),
            tooltip=["Name", alt.Tooltip("delta:Q", format=".4f")],
        )
        .properties(title=title, height=380)
    )


with tab_h:
    h = _filter(hitters, "PA_ytd", min_pa)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Rest-of-season wOBA")
        st.dataframe(
            h[
                [
                    "Name",
                    "Tm",
                    "Age",
                    "PA_ytd",
                    "pred_woba_prior",
                    "pred_woba_proof",
                    "lo80",
                    "hi80",
                    "delta",
                    "BB%",
                    "K%",
                    "ISO",
                    "BABIP",
                ]
            ]
            .rename(
                columns={
                    "PA_ytd": "PA",
                    "pred_woba_prior": "Preseason",
                    "pred_woba_proof": "ROS wOBA",
                    "lo80": "lo 80%",
                    "hi80": "hi 80%",
                    "delta": "Δ vs preseason",
                }
            )
            .style.format(
                {
                    "Preseason": "{:.3f}",
                    "ROS wOBA": "{:.3f}",
                    "lo 80%": "{:.3f}",
                    "hi 80%": "{:.3f}",
                    "Δ vs preseason": "{:+.3f}",
                    "BB%": "{:.1%}",
                    "K%": "{:.1%}",
                    "ISO": "{:.3f}",
                    "BABIP": "{:.3f}",
                }
            ),
            width="stretch",
            height=560,
            hide_index=True,
        )
    with right:
        st.altair_chart(
            _risers_chart(h, "delta", "Risers & fallers vs preseason"), width="stretch"
        )
        st.caption(
            "Δ = current true-talent estimate minus the March projection. "
            "Rookies and no-history players carry a league-average prior."
        )

with tab_p:
    p = _filter(pitchers, "BF_ytd", min_bf)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Rest-of-season FIP / ERA")
        st.dataframe(
            p[
                [
                    "Name",
                    "Tm",
                    "Age",
                    "role",
                    "IP",
                    "FIP_ytd",
                    "ERA_ytd",
                    "pred_fip_prior",
                    "pred_fip_proof",
                    "lo80",
                    "hi80",
                    "era_ros",
                    "delta",
                ]
            ]
            .rename(
                columns={
                    "role": "Role",
                    "pred_fip_prior": "Preseason FIP",
                    "pred_fip_proof": "ROS FIP",
                    "lo80": "lo 80%",
                    "hi80": "hi 80%",
                    "era_ros": "ROS ERA",
                    "delta": "Δ vs preseason",
                }
            )
            .style.format(
                {
                    "IP": "{:.1f}",
                    "FIP_ytd": "{:.2f}",
                    "ERA_ytd": "{:.2f}",
                    "Preseason FIP": "{:.2f}",
                    "ROS FIP": "{:.2f}",
                    "lo 80%": "{:.2f}",
                    "hi 80%": "{:.2f}",
                    "ROS ERA": "{:.2f}",
                    "Δ vs preseason": "{:+.2f}",
                }
            ),
            width="stretch",
            height=560,
            hide_index=True,
        )
    with right:
        st.altair_chart(
            _risers_chart(p, "delta", "Risers & fallers vs preseason"), width="stretch"
        )
        st.caption(
            "ROS ERA = projected FIP + heavily-shrunk personal ERA–FIP gap, "
            "on the current observed run-environment scale."
        )

with tab_about:
    st.markdown(
        """
**What this is.** A public, reproducible mini-projection system
(Marcel-style priors + per-component empirical-Bayes updating), built as a
research portfolio piece. Everything regenerates with `make backtest` and
`make dashboard` from pinned data snapshots.

**What the numbers mean.** "ROS wOBA/FIP" is the projected rate for the
remainder of the season; the 80% interval is the model's predictive
interval assuming the player's current playing-time rate continues. In the
2023–25 walk-forward backtest, 80% intervals covered 80.6% (hitters) /
80.5% (pitcher FIP).

**Known blind spots (by design admission).** No-history players project as
league average; injuries, trades, and role changes are invisible; the model
slightly over-projects players who stay on the field while declining
(+0.004–0.007 wOBA bias, documented in writeup/spec-audit.md).

**Not included, on purpose:** WAR (needs defense, baserunning, and
replacement-level modeling — scoped out in the spec audit, item 11).
        """
    )
