"""Tests that the gates in build.py catch a mislabeled player id.

Run: python test_build.py  (two small real requests: one StatsAPI /people, one Savant search).
The first test replays the earlier version's error: id 461325 labelled as Hisashi Iwakuma is
Tyler Clippard, and the build must reject Savant's rows for it.
"""
import pandas as pd

import build as b


def person(pid: int) -> dict:
    return b.api("/people", personIds=str(pid))["people"][0]


def test_wrong_id_is_rejected_by_name() -> None:
    iwakuma = person(547874)
    rows = b.fetch_savant("pitcher", 461325, 2015)  # what the earlier version stored as Iwakuma 2015
    assert len(rows) > 0
    errs = b.check_chunk(rows, "pitcher", 461325, 2015, b.savant_names(iwakuma))
    assert any("does not match StatsAPI's name" in e for e in errs), errs
    # The same rows pass against the real owner of the id, so the failure is the name, not the data.
    assert b.check_chunk(rows, "pitcher", 461325, 2015, b.savant_names(person(461325))) == []


def test_rows_of_another_player_are_rejected_by_id() -> None:
    df = pd.DataFrame({"game_year": ["2016"] * 2, "game_type": ["R"] * 2, "pitcher": ["547874", "461325"],
                       "player_name": ["Iwakuma, Hisashi"] * 2, "game_pk": ["1", "1"],
                       "at_bat_number": ["1", "2"], "pitch_number": ["1", "1"], "description": ["ball"] * 2,
                       "launch_speed": [""] * 2, "hyper_speed": [""] * 2})
    errs = b.check_chunk(df, "pitcher", 547874, 2016, {b.norm("Iwakuma, Hisashi")})
    assert any("pitcher ids" in e for e in errs), errs


def test_other_season_and_game_type_are_rejected() -> None:
    df = pd.DataFrame({"game_year": ["2025"], "game_type": ["S"], "batter": ["660271"],
                       "player_name": ["Ohtani, Shohei"], "game_pk": ["1"], "at_bat_number": ["1"],
                       "pitch_number": ["1"], "description": ["ball"], "launch_speed": [""], "hyper_speed": [""]})
    errs = b.check_chunk(df, "batter", 660271, 2026, {b.norm("Ohtani, Shohei")})
    assert any("game_year" in e for e in errs) and any("game_type" in e for e in errs), errs


def test_previous_ids_must_be_kept_or_corrected() -> None:
    old = {461325, 680686, 660271}
    good = {547874, 673513, 660271}
    assert b.old_ids_errors(old, good) == []
    assert b.old_ids_errors(old, good - {547874})  # corrected id missing
    assert b.old_ids_errors(old, good | {461325})  # wrong id still built
    assert b.old_ids_errors(old | {111}, good)  # an earlier player silently dropped


def test_pitch_count() -> None:
    df = pd.DataFrame({"description": ["ball", "automatic_ball", "automatic_ball", "foul"]})
    assert b.pitched(df) == 2  # automatic balls are Savant rows, not pitches
    assert b.count_error(2, 2) is None
    assert "not loaded" in b.count_error(1, 2)  # a game Savant has not loaded yet
    assert b.count_error(3, 2)


def _hyper(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["game_year", "description", "launch_speed", "hyper_speed"])


def test_hyper_speed_rule() -> None:
    ok = _hyper([["2020", "hit_into_play", "63.6", "88"], ["2020", "hit_into_play", "99.3", "99.3"],
                 ["2020", "ball", "", ""], ["2026", "foul_bunt", "", "88.0"],
                 ["2016", "called_strike", "", "102.0"], ["2016", "swinging_strike", "", "95.1"]])
    assert b.hyper_speed_error(ok) is None
    for rows in ([["2020", "hit_into_play", "63.6", "63.6"]], [["2020", "ball", "", "90"]],
                 [["2020", "hit_into_play", "99.3", ""]], [["2018", "called_strike", "", "102.0"]],
                 [["2016", "foul", "", "102.0"]], [["2016", "called_strike", "", "102.0"]] * 3):
        assert b.hyper_speed_error(_hyper(rows)), rows


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} tests passed")
