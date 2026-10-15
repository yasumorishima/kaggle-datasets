# %% [markdown]
# # Confirming the Last-Outing Signal on Unseen Seasons (MLB pitchers, 2021-2023)
#
# The eighth pre-registered pitcher study ([FREEZE_pitch_after.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_after.md), frozen `5bbe6ed`) found on MLB 2024-2026 that the last outing before an arm placement is slower than the pitcher's own recent normal (P1: AUC 0.606, above every rotation of its floor). This notebook tests that once on MLB 2021-2023, seasons where no outing-day velocity had met a placement.
#
# It runs the frozen notebook (checked by md5) with replacements that change the seasons and remove Triple-A, and reads the three outing columns (`inning_topbot`, `bat_score`, `post_bat_score`) from a dataset built for this test, joined by pitch key; the dataset's 23 other columns must equal the pinned 2021-2023 subset's, key by key. **Before any 2021-2023 row meets a label, the same code, going through the same join, is run on the frozen 2024-2026 files and must reproduce the frozen P1 exactly.** The league constants of the score (the velocity scale and the median pitch count, measured on MLB 2025) are taken from that regression run. Everything was fixed before any 2021-2023 outing-day feature was compared with a placement: [FREEZE_pitch_after_confirm.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_after_confirm.md).

# %%
import gc
import hashlib

import numpy as np
import pandas as pd
import requests

AFTER_SHA = "5bbe6ed4df1249c8673600367d9c9941a86531a6"
AFTER_URL = f"https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/{AFTER_SHA}/pitcher-after-il/pitcher_after_il.py"
AFTER = requests.get(AFTER_URL, timeout=60).text
assert hashlib.md5(AFTER.encode()).hexdigest() == "a96dcbef03d67e4cfc9404279ac2e566", "not the frozen last-outing notebook"

# yasunorim/mlb-statcast-2021-2023-pitch-subset (GHA 38483678623; pinned by FREEZE_pitch_ssw_confirm.md)
CONFIRM_MD5 = {2021: "51c22a049453bb1ba5102d66e728a304", 2022: "ac2f365b4d00f0ab139b0a844615a3fc", 2023: "4bba35933470f5a8d6549059a486d59a"}
# yasunorim/mlb-statcast-2021-2023-outing-columns (pitcher-confirm-outing-2021-2023/build.py)
OUTING_MD5 = {2021: "4865ef17b5d991047404ca06ae5589a9", 2022: "7a2f8eeb6336cc826369c4346d30a256", 2023: "0ee0149b93c8cacca91e075fd3465202"}
# The frozen run (GHA 38657482176, after_verdict.csv and run.txt)
FROZEN = {"rows": (61966, 1180, 53384, 3553), "b0": 0.5644912856858865, "auc": 0.6059334572407536,
          "lo_bonf": 0.5628716317318954, "pos_auc": 0.6065369360478149, "positives": 314,
          "rows_p1": 30492, "floor_mean": "0.5334", "floor_975": "0.5587"}
KEY = ["game_pk", "at_bat_number", "pitch_number"]
OUT_COLS = ["inning_topbot", "bat_score", "post_bat_score"]

# %% [markdown]
# ## The replacements

# %%
AAA_SCHEDULE = '''def aaa_season_dates(years):
    """First and last date of completed Triple-A regular-season games from the StatsAPI schedule (the frozen code took
    them from Triple-A Statcast pitches, which are not loaded here)."""
    first, last = {}, {}
    for y in years:
        for k in range(4):
            try:
                r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={"sportId": 11, "season": y, "gameType": "R"},
                                 headers={"User-Agent": "kaggle-notebook-research/1.0"}, timeout=60)
                r.raise_for_status()
                js = r.json()
                break
            except requests.RequestException:
                time.sleep(5 * (k + 1))
        else:
            raise RuntimeError(f"StatsAPI schedule {y} failed")
        ds = [d["date"] for d in js.get("dates", []) for g in d["games"]
              if g["status"]["abstractGameState"] == "Final" and g["status"]["detailedState"] in ("Final", "Completed Early", "Game Over")]
        if ds:
            first[y], last[y] = pd.Timestamp(min(ds)), pd.Timestamp(max(ds))
    return (pd.Series(first, dtype="datetime64[ns]").rename_axis("season"),
            pd.Series(last, dtype="datetime64[ns]").rename_axis("season"))


_aaa_first, _aaa_last = aaa_season_dates([ROW_SEASONS[0] - 1] + ROW_SEASONS)
print("Triple-A regular season (StatsAPI schedule):", {y: (f"{_aaa_first[y]:%Y-%m-%d}", f"{_aaa_last[y]:%Y-%m-%d}") for y in _aaa_last.index})
season_last = {"mlb": mlb.groupby("season").game_date.max(), "aaa": _aaa_last}
season_first = {"mlb": mlb.groupby("season").game_date.min(), "aaa": _aaa_first}
'''

LOAD_JOINED = '''
def _read_one(pattern, y, cols, md5s):
    p = glob.glob(os.path.join(INPUT_ROOT, "**", pattern.format(y=y)), recursive=True)
    assert len(p) == 1, (pattern.format(y=y), p)
    assert FAKE or hashlib.md5(open(p[0], "rb").read()).hexdigest() == md5s.get(y), ("not the pinned file", p[0])
    return pd.read_parquet(p[0], columns=cols)


def join_outing(left, right):
    """The three outing columns joined to the pitch rows by key: one-to-one, every key on both sides."""
    m = left.merge(right, on=KEY, how="outer", validate="one_to_one", indicator=True)
    assert m._merge.eq("both").all(), ("keys differ", m._merge.value_counts().to_dict())
    return m.drop(columns="_merge")


def load_mlb_joined():
    """MLB pitches for the frozen notebook (XCOLS + XSCORE), built by the join in both modes. Regression: the frozen
    2024-2026 files split into the pitch columns and the outing columns and joined back (must equal the file).
    Confirmation: the pinned subset's pitch columns joined to the outing dataset's three columns, after checking that
    the outing dataset's 23 shared columns equal the subset's on every key. Both modes split the columns the same way:
    the pitch side is XCOLS without the three outing columns, the outing side is the key and the three outing columns."""
    XPITCH = [c for c in XCOLS if c not in OUT_COLS]
    assert sorted(XPITCH + OUT_COLS) == sorted(XCOLS + XSCORE) and set(KEY) <= set(XPITCH), (XPITCH, XCOLS, XSCORE)
    frames = []
    for y in ROW_SEASONS:
        if MODE == "regression":
            whole = _read_one("statcast_{y}.parquet", y, XCOLS + XSCORE, MLB_MD5)
            d = join_outing(whole[XPITCH], whole[KEY + OUT_COLS])
            a_ = whole.sort_values(KEY).reset_index(drop=True)
            b_ = d[XCOLS + XSCORE].sort_values(KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(a_, b_, check_dtype=True)
        else:
            sub_cols = list(pd.read_parquet(glob.glob(os.path.join(INPUT_ROOT, "**", f"statcast_{y}.parquet"), recursive=True)[0]).columns) \\
                if FAKE else SUBSET_COLUMNS
            sub = _read_one("statcast_{y}.parquet", y, sub_cols, MLB_MD5)
            out = _read_one("outing_{y}.parquet", y, sub_cols + OUT_COLS, OUTING_MD5)
            a_ = sub.sort_values(KEY).reset_index(drop=True)
            b_ = out[sub_cols].sort_values(KEY).reset_index(drop=True)
            assert len(a_) == len(b_) and a_[KEY].equals(b_[KEY]), f"{y}: the outing dataset's keys differ from the subset's"
            for c in sub_cols:
                assert a_[c].equals(b_[c]), f"{y}: column {c} differs between the outing dataset and the subset"
            print(f"MLB {y}: outing dataset equals the pinned subset on {len(a_):,} keys and {len(sub_cols)} columns")
            assert set(XPITCH) <= set(sub_cols), f"{y}: pitch columns missing from the subset: {set(XPITCH) - set(sub_cols)}"
            d = join_outing(sub[XPITCH], out[KEY + OUT_COLS])
            # the builder's gates, again in the notebook: Top/Bot only, one half per at-bat, scores never fall within a half
            assert d.inning_topbot.isin(["Top", "Bot"]).all(), f"{y}: a half that is not Top or Bot"
            assert d.groupby(["game_pk", "at_bat_number"]).inning_topbot.nunique().max() == 1, f"{y}: an at-bat in both halves"
            assert d.bat_score.notna().all() and not (d.post_bat_score < d.bat_score).any(), f"{y}: post_bat_score below bat_score"
            _st = d.sort_values(KEY)
            _fall = (_st.groupby(["game_pk", "inning", "inning_topbot"]).bat_score.diff() < 0).any()
            assert not _fall, f"{y}: bat_score falls within a half"
        d["season"] = y
        frames.append(d[XCOLS + XSCORE + ["season"]])
    out_ = pd.concat(frames, ignore_index=True)
    out_["game_date"] = pd.to_datetime(out_["game_date"])
    out_ = out_[out_.game_type == "R"]
    return out_.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)


'''

SUBSET_COLUMNS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number", "pitch_type", "release_speed",
                  "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z", "arm_angle", "pfx_x", "pfx_z",
                  "zone", "description", "events", "type", "launch_speed", "woba_denom", "inning", "game_type", "spin_axis"]


def between(s, a, b):
    i = s.index(a)
    return s[i:s.index(b, i)]


def base_replacements(mode):
    """Applied by the frozen notebook to the frozen pitcher notebook (HEAD + DEFS) before it runs it."""
    rep = []
    if mode == "confirm":
        rep += [
            ("seasons", "ROW_SEASONS, TRAIN, TEST = [2024, 2025, 2026], [2024, 2025], 2026",
             "ROW_SEASONS, TRAIN, TEST = [2021, 2022, 2023], [2021, 2022], 2023"),
            ("MLB md5", "MLB_MD5 = {2024:", f"MLB_MD5 = {CONFIRM_MD5!r}\n_UNUSED_MD5 = {{2024:"),
            ("league season of the frozen anomaly scales", 'rows_m = build_rows(mlb, "mlb", 2025, "aaa")',
             'rows_m = build_rows(mlb, "mlb", 2022, "aaa")'),
        ]
    rep += [
        ("no Triple-A pitches", 'aaa = load("aaa_pitching_{y}.parquet", [2023] + ROW_SEASONS, AAA_MD5)\n',
         "aaa = mlb.iloc[:0].copy()  # MLB only: no Triple-A pitch data (none exists for 2021)\n"),
        ("fill table MLB only", 'for name, d in (("MLB", mlb), ("Triple-A", aaa)):\n    fill =', 'for name, d in (("MLB", mlb),):\n    fill ='),
        ("Triple-A season dates", None, AAA_SCHEDULE),
        ("history years", "HIST_YEARS = list(range(2022, TEST + 1))", "HIST_YEARS = list(range(ROW_SEASONS[0] - 2, TEST + 1))"),
        ("game-log years", "    for y in [2023] + ROW_SEASONS:\n", "    for y in [ROW_SEASONS[0] - 1] + ROW_SEASONS:\n"),
        ("empty rehab table", "no_rehab_tx = ~pd.Series([k in rehab_seasons for k in rem.index], index=rem.index)",
         "no_rehab_tx = ~pd.Series([k in rehab_seasons for k in rem.index], index=rem.index, dtype=bool)"),
        ("fill gates MLB only", 'for name, d in (("mlb", mlb), ("aaa", aaa)):', 'for name, d in (("mlb", mlb),):'),
        ("no Triple-A rows", 'rows_a = build_rows(aaa, "aaa", 2023, "mlb")\n', ""),
        ("fake labels MLB only", "    for rows_x in (rows_m, rows_a):\n", "    for rows_x in (rows_m,):\n"),
        ("no Triple-A labels", 'rows_a = attach_labels(rows_a, "mlb")\n', ""),
        ("placement counts MLB only", 'for name, rows_x in (("MLB", rows_m), ("Triple-A", rows_a)):',
         'for name, rows_x in (("MLB", rows_m),):'),
        ("placement count text", 'print(f"{name}: placements of these pitchers in 2024-2026 {len(lab_y):,};',
         'print(f"{name}: placements of these pitchers in {ROW_SEASONS[0]}-{TEST} {len(lab_y):,};'),
    ]
    return rep


_RB, _CB = base_replacements("regression"), base_replacements("confirm")
assert _CB[3:] == _RB and [n for n, *_ in _CB[:3]] == ["seasons", "MLB md5", "league season of the frozen anomaly scales"],     "undeclared confirm-only replacement"


def patch_base(src, mode):
    for name, old, new in base_replacements(mode):
        if old is None:  # the Triple-A season-date block
            old = between(src, 'season_last = {"mlb":', "if not SMOKE:\n    for lv in season_last:")
        assert src.count(old) == 1, f"base replacement '{name}' matches {src.count(old)} times"
        src = src.replace(old, new)
    assert "rows_a" not in src and "aaa_pitching" not in src, "Triple-A left in the base"
    return src


EXEC_LINE = 'exec(compile(HEAD + "\\n\\n" + DEFS, "frozen_pitcher_up_to_tests", "exec"))\n'
EQ_GATE = between(AFTER, "if not FAKE:\n    _got = (len(rows_m)", "# %% [markdown]\n# ## The outing of the row's own day")
EQ_NEW = '''_tr = (rows_m.season.isin(TRAIN) & rows_m.use14).to_numpy()
_te = ((rows_m.season == TEST) & rows_m.use14).to_numpy()
_got = (len(rows_m), rows_m.pitcher.nunique(), int(rows_m.use14.sum()), int(rows_m.il14[rows_m.use14].sum()))
_b0 = auc(rows_m.il14[_te].to_numpy(), predict(fit(CTX, rows_m[_tr], rows_m.il14[_tr].to_numpy()), CTX, rows_m[_te]))
print(f"{MODE}: rows and labels {_got}; AUC(B0) in {TEST} {_b0:.16f}")
if MODE == "regression" and not FAKE:
    assert _got == FROZEN["rows"] and abs(_b0 - FROZEN["b0"]) < 1e-12, "the MLB-only code does not reproduce the frozen rows or rival"
assert sorted(rows_m.season.unique()) == ROW_SEASONS, "row seasons"


'''
LG_OLD = '    lg = one[one.season.eq(LEAGUE[level])].groupby("o_start").o_pitches.median()\n'
LG_NEW = '''    lg = one[one.season.eq(LEAGUE[level])].groupby("o_start").o_pitches.median()
    if FIXED.get("lg") is not None:  # the confirmation uses the regression run's MLB 2025 values
        _mid = one[one.season.eq(ROW_SEASONS[1])].groupby("o_start").o_pitches.median()
        print(f"{level}: league median pitch count by start flag, used (MLB 2025): {FIXED['lg'].to_dict()}; reported, "
              f"season {ROW_SEASONS[1]}: {_mid.to_dict()}")
        lg = FIXED["lg"].copy()
    assert set(lg.index) == {0.0, 1.0} and np.isfinite(lg.to_numpy()).all(), ("league median pitch count missing", lg.to_dict())
    SEEN["lg"] = lg.copy()
'''
LSC_OLD = '    assert all(np.isfinite(v) and v > 0 for v in lsc.values()), "a league-typical scale is missing"\n'
LSC_NEW = '''    if FIXED.get("lsc") is not None:
        print(f"{level}: used league scale (MLB 2025): {FIXED['lsc']}")
        lsc = dict(FIXED["lsc"])
    assert all(np.isfinite(v) and v > 0 for v in lsc.values()), "a league-typical scale is missing"
    SEEN["lsc"] = dict(lsc)
'''
TAIL = between(AFTER, "N_ROT = 1000\n", "def rotation_test(")


def make(mode):
    assert mode in ("regression", "confirm")
    rep = [
        ("base replacements", EXEC_LINE,
         'exec(compile(patch_base(HEAD + "\\n\\n" + DEFS, MODE), "frozen_pitcher_up_to_tests", "exec"))\n'),
        ("equality gates", EQ_GATE, EQ_NEW),
        ("league constants: median pitch count", LG_OLD, LG_NEW),
        ("league constants: velocity scale", LSC_OLD, LSC_NEW),
        ("MLB pitches by the join, no Triple-A context",
         between(AFTER, 'ctx = {"mlb": load_ctx(', "\n\n\ndef outing_items"),
         LOAD_JOINED + 'ctx = {"mlb": load_mlb_joined()}'),
        ("no Triple-A outings", 'rows_a = attach_after(rows_a, "aaa")\n', ""),
        ("no Triple-A unseen days", 'unseen_appearances(rows_a, "aaa", "Triple-A")\n', ""),
        ("no Triple-A target", 'rows_a["y_last"] = last_outing_target(rows_a, "Triple-A")\n', ""),
        ("target gates MLB only", 'for _nm, _r in (("MLB", rows_m), ("Triple-A", rows_a)):', 'for _nm, _r in (("MLB", rows_m),):'),
    ]
    src = AFTER
    for name, old, new in rep:
        assert src.count(old) == 1, f"replacement '{name}' matches {src.count(old)} times"
        src = src.replace(old, new)
    cut = src.index("# %% [markdown]\n# ## P1: does the last outing before an arm placement look slow")
    src = src[:cut] + "\n" + TAIL
    assert "rows_a" not in src and "aaa_pitching" not in src, "Triple-A left in the notebook"
    print(f"{mode}: {len(rep)} replacements in the notebook, {len(base_replacements(mode))} in the base")
    return src


# %% [markdown]
# ## The P1 statistic (the frozen floor, the frozen bootstrap draws)

# %%


GAP_CLASSES = ["<=7", "8-15", "16-30", "31-60", ">60", "none-early", "none-late"]


def position_frame(rows, season_end):
    """Label-free position of every MLB row: fifth of the pitcher-season by row order, days to his next MLB row, and
    whether a last row of the season is within the last 14 days of the regular season (season_end: last MLB date)."""
    r = rows.sort_values(["pitcher", "season", "day"], kind="stable")
    g = r.groupby(["pitcher", "season"])
    q = g.cumcount() / g.day.transform("size")
    gap = (g.day.shift(-1) - r.day).dt.days
    end = pd.to_datetime(r.season.map(season_end))
    assert end.notna().all(), "a season without its last MLB date"
    late = (end - r.day).dt.days <= 13
    cls = np.select([gap <= 7, gap <= 15, gap <= 30, gap <= 60, gap.notna()], [0, 1, 2, 3, 4], np.where(late, 6, 5))
    fifth = np.minimum((q * 5).astype(int), 4).to_numpy()
    out = pd.DataFrame({"q": q.to_numpy(), "gap": gap.to_numpy(), "late": late.to_numpy(), "fifth": fifth, "cls": cls,
                        "key": fifth * 7 + cls}, index=r.index)
    return out.loc[rows.index]


def merge_bins(key, y):
    """PLAN v5: while a bin with k positives has fewer than 4k rows, the smallest such bin joins the next shorter gap
    class of its fifth; a <=7 bin joins the <=7 bin of the previous fifth (the next for the first fifth)."""
    key = np.asarray(key).copy()
    while True:
        u = np.unique(key)
        if len(u) == 1:
            return key
        bad = [k for k in u if y[key == k].sum() > 0 and (key == k).sum() < 4 * y[key == k].sum()]
        if not bad:
            return key
        k = min(bad)
        f, c = divmod(int(k), 7)
        t = f * 7 + (5 if c == 6 else c - 1) if c > 0 else ((f - 1) * 7 if f > 0 else 7)
        while t not in u:  # follow the same order through classes that hold no rows
            f2, c2 = divmod(int(t), 7)
            t = f2 * 7 + (5 if c2 == 6 else c2 - 1) if c2 > 0 else ((f2 - 1) * 7 if f2 > 0 else (f2 + 1) * 7)
            assert 0 <= t < 35, "bin merge ran off the grid"
        key[key == k] = t


def _check_position_rules():
    """Label-free unit gate on the bin definition and the merge order (PLAN v5)."""
    d0 = pd.Timestamp("2021-04-01")
    gaps = [7, 8, 15, 16, 30, 31, 60, 61]  # one pitcher: rows at these gaps, then a last row
    days = [d0]
    for g_ in gaps:
        days.append(days[-1] + pd.Timedelta(days=g_))
    r = pd.DataFrame({"pitcher": 1, "season": 2021, "day": days})
    r2 = pd.DataFrame({"pitcher": 2, "season": 2021, "day": [d0, pd.Timestamp("2021-12-18")]})  # last row 13 days before the end
    r3 = pd.DataFrame({"pitcher": 3, "season": 2021, "day": [d0, pd.Timestamp("2021-12-17")]})  # last row 14 days before
    rr = pd.concat([r, r2, r3], ignore_index=True)
    pf = position_frame(rr, pd.Series({2021: pd.Timestamp("2021-12-31")}))
    assert pf.cls.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 5, 4, 6, 4, 5], ("gap classes", pf.cls.tolist())
    assert pf.fifth.tolist()[:9] == [0, 0, 1, 1, 2, 2, 3, 3, 4], ("fifths", pf.fifth.tolist())
    k = np.array([4 * 7 + 6] * 2 + [4 * 7 + 5] * 20 + [7] * 3 + [0] * 30)
    yk = np.array([1, 1] + [0] * 20 + [1, 0, 0] + [0] * 30, dtype=bool)
    mk = merge_bins(k, yk)
    assert (mk[:2] == 4 * 7 + 5).all() and (mk[22:25] == 0).all(), ("merge order", np.unique(mk))


_check_position_rules()


def bin_draws(y, key, auc_fn, s, seeds):
    """Exact stratified label permutation: in each merged bin, as many rows as real positives, uniformly without
    replacement from all of the bin's rows (real positives included)."""
    return np.array([auc_fn(yy, s) for yy in _draw_masks(y, key, seeds)])


def p1_stat(ns, score):
    rows_m, auc, rotated_group, floor_structure = ns["rows_m"], ns["auc"], ns["rotated_group"], ns["floor_structure"]
    np, N_BOOT = ns["np"], ns["N_BOOT"]
    no_hook = rows_m.early_hook.eq(0.0).to_numpy()
    POS = position_frame(rows_m, ns["season_last"]["mlb"])
    sel = no_hook & rows_m.y_last.notna().to_numpy() & np.isfinite(rows_m[score]).to_numpy()
    df = rows_m[sel].reset_index(drop=True)
    pos_df = POS[sel].reset_index(drop=True)
    y, s = df.y_last.to_numpy() == 1.0, df[score].to_numpy()
    key = merge_bins(pos_df.key.to_numpy(), y)
    for k in np.unique(key):
        n_k, y_k = int((key == k).sum()), int(y[key == k].sum())
        assert y_k == 0 or n_k >= 4 * y_k or len(np.unique(key)) == 1, "a merged bin breaks the 4x rule"
    fl_bin = bin_draws(y, key, auc, s, range(7000, 8000))
    gix = list(df.groupby(["pitcher", "season"]).indices.values())
    a = auc(y, s)
    fl = np.array([auc(y, rotated_group(s, gix, sd)) for sd in range(7000, 8000)])
    p = (1 + int((fl >= a).sum())) / 1001
    floor_structure(df, gix, 7000)
    for i in range(3):
        ri = rotated_group(np.arange(len(df), dtype=float), gix, 7000 + i).astype(int)
        assert fl[i] == auc(y, s[ri]), "the floor is not the whole-group rotation"
    idx_by = list(df.groupby("pitcher").indices.values())
    rng = np.random.default_rng(2)
    v = []
    for _ in range(N_BOOT):  # the frozen boot() draws (seed 2), kept as an array
        idx = np.concatenate([idx_by[i] for i in rng.integers(0, len(idx_by), len(idx_by))])
        v.append(auc(y[idx], s[idx]))
    v = np.array(v)
    v = v[np.isfinite(v)]
    p_bin = (1 + int((fl_bin >= a).sum())) / 1001
    # reported: within-bin stratified Mann-Whitney (sum over bins of the positives' within-bin rank sums), same draws
    wr = pd.Series(s).groupby(key).rank().to_numpy()  # within-bin ranks (do not depend on the labels)
    t_real = wr[y].sum()
    t_fl = np.array([wr[_d].sum() for _d in _draw_masks(y, key, range(7000, 7200))])
    tab = pd.DataFrame({"bin": key, "y": y}).groupby("bin").agg(rows=("y", "size"), positives=("y", "sum"))
    return {"auc": a, "floor": fl, "p": p, "lo95": np.percentile(v, 2.5), "lo_bonf": np.percentile(v, ns["LO"]),
            "floor_bin": fl_bin, "p_bin": p_bin, "p_strat": (1 + int((t_fl >= t_real).sum())) / 201, "bins": tab,
            "pos": int(y.sum()), "rows": len(df), "pitchers": df.pitcher.nunique(), "df": df, "key": key,
            "index": rows_m.index[sel]}


def _draw_masks(y, key, seeds):
    groups = [(np.flatnonzero(key == k), int(y[key == k].sum())) for k in np.unique(key)]
    for sd in seeds:
        rng = np.random.default_rng(sd)
        yy = np.zeros(len(y), dtype=bool)
        for ix, k in groups:
            if k:
                yy[rng.choice(ix, k, replace=False)] = True
        assert all(yy[ix].sum() == k for ix, k in groups), "a draw does not hold every bin's positive count"
        yield yy


# %% [markdown]
# ## 1. Regression gate: the frozen 2024-2026 run, reproduced through the join

# %%
FIXED, SEEN = {}, {}
ns = {"__name__": "after_regression", "MODE": "regression", "FIXED": FIXED, "SEEN": SEEN, "patch_base": patch_base,
      "FROZEN": FROZEN, "KEY": KEY, "OUT_COLS": OUT_COLS, "SUBSET_COLUMNS": SUBSET_COLUMNS, "OUTING_MD5": OUTING_MD5}
exec(compile(make("regression"), "after_regression", "exec"), ns)
R1 = p1_stat(ns, "A_rank")
R1P = p1_stat(ns, "A_rank_pos")
print(f"regression P1: AUC {R1['auc']:.16f} (frozen {FROZEN['auc']}), Bonferroni lower {R1['lo_bonf']:.16f}, floor mean "
      f"{R1['floor'].mean():.4f}, 97.5th {np.percentile(R1['floor'], 97.5):.4f}, p {R1['p']:.4f}, positives {R1['pos']}, rows "
      f"{R1['rows']}; position-adjusted {R1P['auc']:.16f}")
_bd = R1["df"]
_bt = ns["boot"](_bd, _bd.y_last.to_numpy() == 1.0, {"A": _bd.A_rank.to_numpy()}, [])
assert _bt.loc["A", "lo95"] == R1["lo95"] and _bt.loc["A", "lo_bonf"] == R1["lo_bonf"], "the bootstrap is not the frozen one"
print(f"reported (labels read before): the frozen P1 under the confirmation's position-matched floor: A_rank floor mean "
      f"{R1['floor_bin'].mean():.4f}, 97.5th {np.percentile(R1['floor_bin'], 97.5):.4f}, p {R1['p_bin']:.4f}; A_rank_pos p "
      f"{R1P['p_bin']:.4f}; stratified Mann-Whitney p {R1['p_strat']:.4f} / {R1P['p_strat']:.4f}")
print(R1["bins"].T.to_string())
print(f"regression: the 95% and Bonferroni bounds equal the frozen boot() ({R1['lo95']:.16f}, {R1['lo_bonf']:.16f})")
LEAGUE_2025 = {"lsc": dict(SEEN["lsc"]), "lg": SEEN["lg"].copy()}
print("league constants from MLB 2025 (passed to the confirmation):", LEAGUE_2025["lsc"], LEAGUE_2025["lg"].to_dict())
if not ns["FAKE"]:
    assert abs(R1["auc"] - FROZEN["auc"]) < 1e-12 and abs(R1P["auc"] - FROZEN["pos_auc"]) < 1e-12, "P1 not reproduced"
    assert abs(R1["lo_bonf"] - FROZEN["lo_bonf"]) < 1e-12, "the bootstrap is not the frozen one"
    assert (R1["pos"], R1["rows"]) == (FROZEN["positives"], FROZEN["rows_p1"]), "P1 rows not reproduced"
    assert f"{R1['floor'].mean():.4f}" == FROZEN["floor_mean"] and f"{np.percentile(R1['floor'], 97.5):.4f}" == FROZEN["floor_975"], \
        "the floor is not reproduced"
    print("regression gate passed")
FAKE_RUN = ns["FAKE"]
del ns, R1, R1P
gc.collect()

# %% [markdown]
# ## 2. Confirmation: MLB 2021-2023

# %%
FIXED.update(LEAGUE_2025)
SEEN.clear()
ns = {"__name__": "after_confirm", "MODE": "confirm", "FIXED": FIXED, "SEEN": SEEN, "patch_base": patch_base,
      "FROZEN": FROZEN, "KEY": KEY, "OUT_COLS": OUT_COLS, "SUBSET_COLUMNS": SUBSET_COLUMNS, "OUTING_MD5": OUTING_MD5}
exec(compile(make("confirm"), "after_confirm", "exec"), ns)
assert ns["ROW_SEASONS"] == [2021, 2022, 2023] and ns["TEST"] == 2023, "the confirmation is not on 2021-2023"
assert SEEN["lsc"] == LEAGUE_2025["lsc"] and SEEN["lg"].equals(LEAGUE_2025["lg"]), "the confirmation did not use the 2025 constants"
C2 = p1_stat(ns, "A_rank")
C2P = p1_stat(ns, "A_rank_pos")
def decide(r):
    return bool(r["p"] <= 0.025 and r["p_bin"] <= 0.025)


def verdict_of(r_main, r_pos):
    return "SUPPORTED" if decide(r_main) and decide(r_pos) else "NOT SUPPORTED"


assert decide({"p": 0.001, "p_bin": 0.001}) and decide({"p": 0.025, "p_bin": 0.025}), "the rule does not pass at its bound"
assert not decide({"p": 0.001, "p_bin": 0.026}) and not decide({"p": 0.026, "p_bin": 0.001}), "the rule passes with one floor beaten"
ok_main, ok_pos = decide(C2), decide(C2P)
print("C2 rows and positives per merged position bin (key = fifth * 7 + gap class; classes " + ", ".join(GAP_CLASSES) + "):")
print(C2["bins"].T.to_string())
for nm, r_, ok in (("C2 MLB 2021-2023, last outing before an arm placement, no early hook [A_rank]", C2, ok_main),
                   ("C2 position-adjusted [A_rank_pos]", C2P, ok_pos)):
    print(f"{nm}: AUC {r_['auc']:.4f}; 95% lower {r_['lo95']:.4f} (Bonferroni 0.625% {r_['lo_bonf']:.4f}); whole-group "
          f"floor mean {r_['floor'].mean():.4f}, 97.5th {np.percentile(r_['floor'], 97.5):.4f}, max {r_['floor'].max():.4f}, "
          f"p {r_['p']:.4f}; position-matched floor mean {r_['floor_bin'].mean():.4f}, 97.5th "
          f"{np.percentile(r_['floor_bin'], 97.5):.4f}, p {r_['p_bin']:.4f} (reported: stratified Mann-Whitney p "
          f"{r_['p_strat']:.4f}); positives {r_['pos']:,}, rows {r_['rows']:,}, pitchers {r_['pitchers']:,} -> "
          f"{'passes' if ok else 'does not pass'}")
verdict = verdict_of(C2, C2P)
print(f"{'FAKE LABELS (code test, not a result) ' if FAKE_RUN else ''}C2: {verdict} (both floors, both lines)")
if FAKE_RUN:  # a pure position score (the merged bin) must sit exactly on the position-matched floor
    _rm = ns["rows_m"]
    _rm["_kscore"] = np.nan
    _rm.loc[C2["index"], "_kscore"] = C2["key"].astype(float)
    _k = p1_stat(ns, "_kscore")
    assert np.array_equal(_k["key"], C2["key"]) and _k["p_bin"] == 1.0, ("the position-matched floor does not hold position", _k["p_bin"])
    print(f"position-only score: AUC {_k['auc']:.4f}, position-matched p {_k['p_bin']:.4f} (must be 1), rotation p {_k['p']:.4f}")
    _rm.drop(columns="_kscore", inplace=True)
if FAKE_RUN:  # the verdict must be reachable by a planted signal, and only when both lines carry it
    _rm = ns["rows_m"]
    for _sc in ("A_rank", "A_rank_pos"):
        _rm[_sc + "_up"] = np.where(_rm.y_last.eq(1.0), _rm[_sc] + 0.3, _rm[_sc])
        _rm[_sc + "_down"] = np.where(_rm.y_last.eq(1.0), _rm[_sc] - 0.3, _rm[_sc])
    _up, _upp = p1_stat(ns, "A_rank_up"), p1_stat(ns, "A_rank_pos_up")
    _dn, _dnp = p1_stat(ns, "A_rank_down"), p1_stat(ns, "A_rank_pos_down")
    print(f"planted +0.3: AUC {_up['auc']:.4f} / {_upp['auc']:.4f}, p {_up['p']:.4f} / {_upp['p']:.4f}, lower {_up['lo95']:.4f} / "
          f"{_upp['lo95']:.4f}, position-matched p {_up['p_bin']:.4f} / {_upp['p_bin']:.4f}; planted -0.3: AUC {_dn['auc']:.4f} / {_dnp['auc']:.4f}")
    assert verdict_of(_up, _upp) == "SUPPORTED", "the verdict cannot be reached by a planted signal"
    assert verdict_of(_up, _dnp) == "NOT SUPPORTED" and verdict_of(_dn, _upp) == "NOT SUPPORTED", \
        "the verdict passes with one line against it"
    assert verdict_of(_dn, _dnp) == "NOT SUPPORTED", "the verdict passes a planted reverse signal"
    _rm.drop(columns=[c for c in _rm.columns if c.endswith(("_up", "_down"))], inplace=True)

# %% [markdown]
# ## Reported, not tested

# %%
rows_c = ns["rows_m"]
auc_c = ns["auc"]
D = C2["df"]
pos, neg = D[D.y_last.eq(1.0)], D[D.y_last.eq(0.0)]
for nm, m in (("effective the next day", pos.days_to_il.eq(1)), ("effective 2 to 14 days later", pos.days_to_il.ge(2))):
    sub = pd.concat([pos[m], neg])
    print(f"reported: {nm}: positives {int(m.sum())}, AUC {auc_c(sub.y_last.to_numpy() == 1.0, sub.A_rank.to_numpy()):.4f}")
for nm, m in [(f"season {y}", D.season.eq(y)) for y in (2021, 2022, 2023)] + [("starters", D.role.eq(1.0)), ("relievers", D.role.eq(0.0))]:
    sub = D[m]
    print(f"reported: {nm}: positives {int(sub.y_last.eq(1.0).sum())}, AUC {auc_c(sub.y_last.to_numpy() == 1.0, sub.A_rank.to_numpy()):.4f}")
_all = rows_c[rows_c.y_last.notna() & np.isfinite(rows_c.A_rank)]
print(f"reported: all outings (manipulation check): positives {int(_all.y_last.eq(1.0).sum())}, AUC "
      f"{auc_c(_all.y_last.to_numpy() == 1.0, _all.A_rank.to_numpy()):.4f}")
print(f"reported: dev (mph) in the C2 rows: positives mean {pos.dev.mean():+.3f}, others {neg.dev.mean():+.3f}; early hooks "
      f"among positives / negatives {np.nanmean(rows_c.early_hook[rows_c.y_last.eq(1.0)]):.4f} / "
      f"{np.nanmean(rows_c.early_hook[rows_c.y_last.eq(0.0)]):.4f}")

summary = pd.DataFrame([{"test": "C2 MLB 2021-2023 last outing before an arm placement, no early hook, daily rank of A",
                         "value": C2["auc"], "lo95": C2["lo95"], "p_rotation": C2["p"], "position_adjusted": C2P["auc"],
                         "position_adjusted_lo95": C2P["lo95"], "position_adjusted_p": C2P["p"], "p_position_matched": C2["p_bin"],
                         "position_adjusted_p_position_matched": C2P["p_bin"], "positives": C2["pos"],
                         "verdict": verdict}])
if FAKE_RUN:
    print("FAKE LABELS: code test only, no verdict written")
    print(summary.to_string(index=False))
else:
    summary.to_csv("after_confirm_verdict.csv", index=False)
    print(summary.to_string(index=False))
