"""Statcast reconstruction invariants: outs attribution and starter detection."""

import pandas as pd

from ingest.statcast import _attach_outs


def _pa(game, ab, inning, half, outs_up, event, pitcher):
    return {
        "game_pk": game,
        "at_bat_number": ab,
        "inning": inning,
        "inning_topbot": half,
        "outs_when_up": outs_up,
        "events": event,
        "pitcher": pitcher,
    }


def test_outs_delta_complete_game():
    # Starter A: innings 1-2 complete; reliever B finishes the 3rd after A
    # leaves mid-inning with 1 out. Home team wins 9-0 after 8.5 -> the
    # final half-inning (Bot 3) never happens... use 3-inning toy game:
    rows = [
        _pa(1, 1, 1, "Top", 0, "strikeout", 100),
        _pa(1, 2, 1, "Top", 1, "single", 100),
        _pa(1, 3, 1, "Top", 1, "grounded_into_double_play", 100),  # 3 outs
        _pa(1, 4, 1, "Bot", 0, "field_out", 200),
        _pa(1, 5, 1, "Bot", 1, "field_out", 200),
        _pa(1, 6, 1, "Bot", 2, "strikeout", 200),
        _pa(1, 7, 2, "Top", 0, "field_out", 100),
        _pa(1, 8, 2, "Top", 1, "field_out", 100),
        _pa(1, 9, 2, "Top", 2, "strikeout", 100),
        # Bot 2: A pulled with 1 out; B records the last 2.
        _pa(1, 10, 2, "Bot", 0, "field_out", 200),
        _pa(1, 11, 2, "Bot", 1, "field_out", 200),
        _pa(1, 12, 2, "Bot", 2, "strikeout", 200),
        _pa(1, 13, 3, "Top", 0, "field_out", 100),
        _pa(1, 14, 3, "Top", 1, "field_out", 100),
        _pa(1, 15, 3, "Top", 2, "field_out", 100),
    ]
    pa = pd.DataFrame(rows)
    pa["outs_rec"] = _attach_outs(pa)
    by_pitcher = pa.groupby("pitcher")["outs_rec"].sum()
    assert by_pitcher[100] == 9  # Top 1-3, complete
    assert by_pitcher[200] == 6  # Bot 1-2 only; Bot 3 never happens
    assert pa["outs_rec"].sum() == 15  # five completed half-innings


def test_outs_delta_walkoff_shortened_final_half():
    rows = [
        _pa(1, 1, 1, "Top", 0, "strikeout", 100),
        _pa(1, 2, 1, "Top", 1, "strikeout", 100),
        _pa(1, 3, 1, "Top", 2, "strikeout", 100),
        # Bot 1 ends on a walk-off HR with 1 out: pitcher records just 1.
        _pa(1, 4, 1, "Bot", 0, "field_out", 100),
        _pa(1, 5, 1, "Bot", 1, "home_run", 100),
    ]
    pa = pd.DataFrame(rows)
    pa["outs_rec"] = _attach_outs(pa)
    assert pa["outs_rec"].sum() == 4  # 3 (top) + 1 (walkoff bottom)
    assert pa.iloc[3]["outs_rec"] == 1
    assert pa.iloc[4]["outs_rec"] == 0


def test_inning_ending_double_play():
    rows = [
        _pa(1, 1, 1, "Top", 0, "single", 100),
        _pa(1, 2, 1, "Top", 0, "grounded_into_double_play", 100),  # 2 outs
        _pa(1, 3, 1, "Top", 2, "field_out", 100),
    ]
    pa = pd.DataFrame(rows)
    pa["outs_rec"] = _attach_outs(pa)
    assert pa["outs_rec"].tolist() == [0, 2, 1]
