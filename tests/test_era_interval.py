"""Tests for the ERA predictive interval (defense/sequencing drift term).

ERA outcomes carry variance beyond FIP (defense, park, sequencing, unearned
runs). The ERA interval must therefore be wider than the FIP interval by an
additive variance term estimated on 2023-24 training residuals only.
"""

import numpy as np
import pandas as pd

from features.pitching import PITCH_EVENTS
from models.pitchers import SIGMA_ERA_EXTRA2, project_pitchers


def _pitcher_frame():
    row = {
        "mlbID": "p1",
        "Name": "Test Arm",
        "Age": 27,
        "role": "SP",
        "BF": 400,
        "outs": 360,
        "gap_ytd": 0.1,
        "ERA": 3.9,
        "BF_hist": 500.0,
        "gap_prior": 0.05,
        "r_bfip_prior": 4.3,
        "r_bfip": 4.3,
    }
    for ev in PITCH_EVENTS:
        row[f"r_{ev}"] = {"SO": 0.22, "uBBHBP": 0.08, "HR": 0.03}[ev]
        row[f"r_{ev}_prior"] = row[f"r_{ev}"]
    return pd.DataFrame([row])


def test_project_pitchers_emits_era_sd_wider_than_fip_sd():
    m = _pitcher_frame()
    league = {f"SP_{ev}": m[f"r_{ev}_prior"][0] for ev in PITCH_EVENTS}
    league["SP_bfip"] = 4.3
    out = project_pitchers(m, cfip=1.0, league_by_role=league, bf_ros=m["BF"])
    assert "sd_proof_era" in out.columns
    # The ERA interval adds irreducible defense/sequencing variance, so it
    # must strictly exceed the FIP interval.
    assert (out["sd_proof_era"] > out["sd_proof"]).all()
    # Exact additive-variance relationship.
    expected = np.sqrt(out["sd_proof"] ** 2 + SIGMA_ERA_EXTRA2)
    assert np.allclose(out["sd_proof_era"], expected)
