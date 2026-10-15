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


GAP_CLASSES = ["8-15", "16-30", "31-60", ">60", "none"]


def pool_frame(rows, season_first, season_last):
    """Label-free position of every MLB row (PLAN v7.1): days to his next MLB row (over all of his MLB rows), the gap
    class (-1 = within 7 days, not in the pool; 0-4 = GAP_CLASSES), the calendar fifth of the regular season and the
    frozen role (the pre-day start share)."""
    r = rows.sort_values(["pitcher", "season", "day"], kind="stable")
    gap = (r.groupby(["pitcher", "season"]).day.shift(-1) - r.day).dt.days
    first = pd.to_datetime(r.season.map(season_first))
    last = pd.to_datetime(r.season.map(season_last))
    assert first.notna().all() and last.notna().all(), "a season without its first or last MLB date"
    frac = (r.day - first).dt.days / ((last - first).dt.days + 1)
    assert ((frac >= 0) & (frac < 1)).all(), "a row outside its regular season"
    fifth = np.minimum((frac * 5).astype(int), 4)
    cls = np.select([gap <= 7, gap <= 15, gap <= 30, gap <= 60, gap.notna()], [-1, 0, 1, 2, 3], 4)
    assert r.role.isin([0.0, 1.0]).all(), "a row without a role"
    out = pd.DataFrame({"gap": gap.to_numpy(), "frac": frac.to_numpy(), "fifth": fifth.to_numpy(), "cls": cls,
                        "role": r.role.to_numpy()}, index=r.index)
    return out.loc[rows.index]


def pool_base(rows_m, P):
    """The pool's row filter: the positives' filter (use14 = ok14 and no other-level outing within 14 days), no early
    hook, both scores finite, next MLB row 8 or more days later or none."""
    return (rows_m.use14 & rows_m.early_hook.eq(0.0) & np.isfinite(rows_m.A_rank) & np.isfinite(rows_m.A_rank_pos)
            & P.cls.ge(0)).to_numpy()


def placed_within(rows_m, il_by, which):
    """True where any placement of the kind (which: "arm" or "notext") is effective 0-14 days after the row."""
    out = np.zeros(len(rows_m), bool)
    days = rows_m.day.to_numpy()
    for b, ix in rows_m.groupby("pitcher").indices.items():
        eff, arm, txt = il_by.get(b, (np.array([], "datetime64[ns]"), np.array([], bool), np.array([], bool)))
        m = arm if which == "arm" else ~np.asarray(txt, bool)
        e = np.sort(np.asarray(eff)[np.asarray(m, bool)])
        if len(e) == 0:
            continue
        d = days[ix]
        lo = np.searchsorted(e, d, side="left")
        hi = np.searchsorted(e, d + np.timedelta64(14, "D"), side="right")
        out[ix] = hi > lo
    return out


def pool_labels(rows_m, P, il_by):
    """PLAN v7.1 item 1: 1 = a frozen P1 positive in the pool; 0 = every other pool row (non-arm placements, no
    placement, demotions); NaN = outside the pool, or a row with an arm placement 0-14 days later that is not a
    positive, or a row before a placement without injury text within 14 days."""
    base = pool_base(rows_m, P)
    y = np.where(base, 0.0, np.nan)
    y[(rows_m.il14 & (rows_m.next_arm | ~rows_m.next_text)).to_numpy()] = np.nan
    y[placed_within(rows_m, il_by, "arm") | placed_within(rows_m, il_by, "notext")] = np.nan
    pos = rows_m.y_last.eq(1.0).to_numpy()
    y[pos & base] = 1.0
    hooked = pos & rows_m.early_hook.ne(0.0).to_numpy()
    near = pos & ~hooked & ~base & P.cls.eq(-1).to_numpy()
    return y, int(near.sum())


def violates_4x(n, k):
    return k > 0 and n < 4 * k


def violates_both(n, k):  # PLAN v7.2, the arm vs non-arm secondary: a stratum with arm rows needs a non-arm row
    return k > 0 and n - k < 1


def merge_strata(role, cls, fifth, y, rule=violates_4x):
    """PLAN v7.1 item 2, counts only: while a stratum with k positives has fewer than 4k rows (positives included), the
    smallest violating (role, class, first fifth) merges with the adjacent fifth group of its role and class (the next
    one; the previous one if it is the last); a class that is one group joins the next shorter class of its role that
    holds rows (the next longer if none is shorter; 8-15 joins 16-30); never across roles; a role with one stratum stops."""
    role, cls, fifth = np.asarray(role, float), np.asarray(cls, int), np.asarray(fifth, int)
    y = np.asarray(y, bool)
    gid = (role * 100 + cls * 10 + fifth).astype(int)
    g_cls = {g: int(c) for g, c in zip(gid, cls)}
    g_f0 = {g: int(f) for g, f in zip(gid, fifth)}
    g_role = {g: float(r) for g, r in zip(gid, role)}
    while True:
        u = np.unique(gid)
        n = {g: int((gid == g).sum()) for g in u}
        k = {g: int(y[gid == g].sum()) for g in u}
        single = {r for r in set(g_role[g] for g in u) if sum(g_role[g] == r for g in u) == 1}
        bad = [g for g in u if rule(n[g], k[g]) and g_role[g] not in single]
        if not bad:
            return gid
        g = min(bad, key=lambda x: (g_role[x], g_cls[x], g_f0[x]))
        same = sorted([h for h in u if g_role[h] == g_role[g] and g_cls[h] == g_cls[g]], key=lambda x: g_f0[x])
        if len(same) > 1:
            i = same.index(g)
            t = same[i + 1] if i + 1 < len(same) else same[i - 1]
            keep = t if g_f0[t] < g_f0[g] else g
            gone = g if keep == t else t
            gid[gid == gone] = keep
            continue
        have = sorted({g_cls[h] for h in u if g_role[h] == g_role[g] and h != g})
        c = g_cls[g]
        shorter = [x for x in have if x < c]
        tc = max(shorter) if shorter else min(x for x in have if x > c)
        tg = [h for h in u if g_role[h] == g_role[g] and g_cls[h] == tc]
        keep = min(tg, key=lambda x: g_f0[x])
        for h in tg + [g]:
            gid[gid == h] = keep
        g_f0[keep] = min(g_f0[h] for h in tg + [g])


def _check_pool_rules():
    """Label-free unit gates on the gap classes, the calendar fifths and the merge order (PLAN v7.1)."""
    d0 = pd.Timestamp("2021-04-01")
    gaps = [7, 8, 15, 16, 30, 31, 60, 61]
    days = [d0]
    for g_ in gaps:
        days.append(days[-1] + pd.Timedelta(days=g_))
    rr = pd.DataFrame({"pitcher": 1, "season": 2021, "day": days, "role": 0.0})
    pf = pool_frame(rr, pd.Series({2021: d0}), pd.Series({2021: pd.Timestamp("2021-12-31")}))
    assert pf.cls.tolist() == [-1, 0, 0, 1, 1, 2, 2, 3, 4], ("gap classes", pf.cls.tolist())
    # 275 days: day 0 -> fifth 0, day 55 -> fifth 1 (55 / 275 = 0.2), day 274 -> fifth 4
    rf = pd.DataFrame({"pitcher": 2, "season": 2021, "day": [d0, d0 + pd.Timedelta(days=54), d0 + pd.Timedelta(days=55),
                                                              d0 + pd.Timedelta(days=274)], "role": 1.0})
    pf2 = pool_frame(rf, pd.Series({2021: d0}), pd.Series({2021: d0 + pd.Timedelta(days=274)}))
    assert pf2.fifth.tolist() == [0, 0, 1, 4], ("calendar fifths", pf2.fifth.tolist())
    # merge: role 0, class 0, fifths 0 and 1 (2 positives in 3 rows -> violates) merge first across fifths
    role = np.array([0.0] * 3 + [0.0] * 10 + [1.0] * 2 + [1.0] * 20)
    cls = np.array([0] * 3 + [0] * 10 + [4] * 2 + [3] * 20)
    fif = np.array([0] * 3 + [1] * 10 + [2] * 2 + [2] * 20)
    yy = np.array([1, 1, 0] + [0] * 10 + [1, 1] + [0] * 20, dtype=bool)
    m = merge_strata(role, cls, fif, yy)
    assert len(np.unique(m[:13])) == 1 and m[0] == 0, ("merge across fifths", np.unique(m[:13]))
    assert len(np.unique(m[13:])) == 1 and m[13] != m[0], ("class merge within the role only", np.unique(m))
    # the last fifth of a chain merges with the previous one
    m3 = merge_strata(np.zeros(12), np.zeros(12, int), np.array([3] * 10 + [4] * 2), np.array([0] * 10 + [1, 1], bool))
    assert len(np.unique(m3)) == 1 and m3[0] == 3, ("last fifth", np.unique(m3))
    # two violators: the smallest (role, class, fifth) first; fifths 0 and 2 violate, 0 takes 1, then 2 takes 3
    m4 = merge_strata(np.zeros(24), np.zeros(24, int), np.array([0] * 2 + [1] * 10 + [2] * 2 + [3] * 10),
                      np.array([1, 1] + [0] * 10 + [1, 1] + [0] * 10, bool))
    assert m4[0] == m4[2] == 0 and m4[12] == m4[14] == 2 and m4[0] != m4[12], ("two violators", np.unique(m4))
    # the secondary rule merges only until a stratum with arm rows has a non-arm row
    m5 = merge_strata(np.zeros(4), np.zeros(4, int), np.array([0, 1, 2, 2]), np.array([1, 0, 1, 0], bool), violates_both)
    assert m5.tolist() == [0, 0, 2, 2], ("secondary rule", m5.tolist())
    # a role with a single violating stratum is left as it is
    m2 = merge_strata(np.zeros(3), np.zeros(3, int), np.zeros(3, int), np.array([1, 1, 0], bool))
    assert len(np.unique(m2)) == 1, "a single-stratum role must stop"


_check_pool_rules()


def _draw_masks(y, key, seeds):
    groups = [(np.flatnonzero(key == k), int(y[key == k].sum())) for k in np.unique(key)]
    for sd in seeds:
        rng = np.random.default_rng(sd)
        yy = np.zeros(len(y), dtype=bool)
        for ix, k in groups:
            if k:
                yy[rng.choice(ix, k, replace=False)] = True
        assert all(yy[ix].sum() == k for ix, k in groups), "a draw does not hold every stratum's positive count"
        yield yy


class Strat:
    """Stratified AUC = sum over strata of k_s * AUC_s / K, with within-stratum average ranks computed once (they do not
    depend on the labels)."""

    def __init__(self, key, s):
        self.code = pd.factorize(key)[0]
        self.n = np.bincount(self.code)
        self.wr = pd.Series(s).groupby(self.code).rank().to_numpy()

    def auc(self, y):
        k = np.bincount(self.code, weights=y.astype(float), minlength=len(self.n))
        sr = np.bincount(self.code, weights=self.wr * y, minlength=len(self.n))
        m = (k > 0) & (self.n - k > 0)
        self.dropped = int(((k > 0) & ~m).sum())
        if not m.any():  # no stratum holds both classes: not computable (the rule then fails, see pool_test)
            return np.nan
        return float(((sr[m] - k[m] * (k[m] + 1) / 2) / (self.n[m] - k[m])).sum() / k[m].sum())


def pool_test(ns, score, y=None, n_draws=1000, seed0=7000, two_sided=False, rule=None):
    """PLAN v7.1 items 2-3 on the pool: merged strata (printed table), stratified AUC, exact stratified permutation."""
    rows_m = ns["rows_m"]
    P = pool_frame(rows_m, ns["season_first"]["mlb"], ns["season_last"]["mlb"])
    near = None
    if y is None:
        y, near = pool_labels(rows_m, P, ns["il_by"])
    sel = np.isfinite(y) & np.isfinite(rows_m[score]).to_numpy()
    yy = y[sel] == 1.0
    Ps = P[sel]
    rule = rule or violates_4x
    key = merge_strata(Ps.role.to_numpy(), Ps.cls.to_numpy(), Ps.fifth.to_numpy(), yy, rule)
    tab = pd.DataFrame({"stratum": key, "y": yy}).groupby("stratum").agg(rows=("y", "size"), positives=("y", "sum"))
    single = set(Ps.role[pd.Series(key, index=Ps.index).groupby(Ps.role).transform("nunique").eq(1)].unique())
    for k_, row in tab.iterrows():
        assert not rule(row.rows, row.positives) or (k_ // 100) in single, "a stratum breaks the merge rule"
    s = rows_m[score].to_numpy()[sel]
    st = Strat(key, s)
    a = st.auc(yy)
    fl = np.array([st.auc(m) for m in _draw_masks(yy, key, range(seed0, seed0 + n_draws))])
    if not np.isfinite(a):  # fail closed: a NaN statistic must not compare as beating every draw
        p = np.nan
    elif two_sided:
        c = fl.mean()
        p = (1 + int((np.abs(fl - c) >= abs(a - c)).sum())) / (n_draws + 1)
    else:
        p = (1 + int((fl >= a).sum())) / (n_draws + 1)
    return {"auc": a, "floor": fl, "p": p, "table": tab, "pos": int(yy.sum()), "rows": int(sel.sum()), "key": key,
            "dropped": st.dropped,
            "index": rows_m.index[sel], "near_left_out": near, "y": yy}


def p1_stat(ns, score):
    """The frozen statistic and the frozen whole-group rotation floor on the frozen C2 rows, the frozen bootstrap draws."""
    rows_m, auc, rotated_group, floor_structure = ns["rows_m"], ns["auc"], ns["rotated_group"], ns["floor_structure"]
    np, N_BOOT = ns["np"], ns["N_BOOT"]
    no_hook = rows_m.early_hook.eq(0.0).to_numpy()
    sel = no_hook & rows_m.y_last.notna().to_numpy() & np.isfinite(rows_m[score]).to_numpy()
    df = rows_m[sel].reset_index(drop=True)
    y, s = df.y_last.to_numpy() == 1.0, df[score].to_numpy()
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
    return {"auc": a, "floor": fl, "p": p, "lo95": np.percentile(v, 2.5), "lo_bonf": np.percentile(v, ns["LO"]),
            "pos": int(y.sum()), "rows": len(df), "pitchers": df.pitcher.nunique(), "df": df}


def covid_flag(rows_m, il_lab):
    """True where the next placement within 14 days has COVID in its description (none in fake mode)."""
    if "description" not in il_lab.columns:
        return np.zeros(len(rows_m), bool)
    cov = il_lab[il_lab.description.str.contains("covid", case=False, na=False)]
    keys = set(zip(cov.pitcher, pd.to_datetime(cov.effective)))
    pl = rows_m.day + pd.to_timedelta(rows_m.days_to_il.where(np.isfinite(rows_m.days_to_il), 0), unit="D")
    return (rows_m.il14 & pd.Series([(b, d) in keys for b, d in zip(rows_m.pitcher, pl)], index=rows_m.index)).to_numpy()


def arm_vs_nonarm_labels(rows_m, P, il_lab, il_by):
    """PLAN v7.1 item 7: within the pool filter, 1 = a frozen P1 positive, 0 = the last row 1-14 days before a non-arm
    placement with injury text, COVID-related placements left out; NaN otherwise."""
    base = pool_base(rows_m, P)
    cov = covid_flag(rows_m, il_lab)
    r = rows_m[rows_m.il14 & ~rows_m.next_arm & rows_m.next_text].copy()
    r["pl"] = r.day + pd.to_timedelta(r.days_to_il, unit="D")
    last = r.sort_values("day").groupby(["pitcher", "pl"]).tail(1)
    last = last[last.days_to_il.ge(1)].index  # as the frozen arm rule: the last row, then 1-14 days
    y = np.full(len(rows_m), np.nan)
    ilast = rows_m.index.get_indexer(last)
    y[ilast] = 0.0
    y[~base | cov | placed_within(rows_m, il_by, "arm") | placed_within(rows_m, il_by, "notext")] = np.nan
    y[rows_m.y_last.eq(1.0).to_numpy() & base] = 1.0
    return y, int((cov & rows_m.next_arm.eq(False).to_numpy()).sum())


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
R1Q, R1QP = pool_test(ns, "A_rank"), pool_test(ns, "A_rank_pos")
print(f"reported (labels read before): the frozen P1 under the pool floor (PLAN v7.1): stratified AUC {R1Q['auc']:.4f} / "
      f"{R1QP['auc']:.4f}, floor mean {R1Q['floor'].mean():.4f} / {R1QP['floor'].mean():.4f}, 97.5th "
      f"{np.percentile(R1Q['floor'], 97.5):.4f} / {np.percentile(R1QP['floor'], 97.5):.4f}, p {R1Q['p']:.4f} / {R1QP['p']:.4f}; "
      f"positives {R1Q['pos']} in {R1Q['rows']} pool rows (positives within 7 days of the next row, left out: {R1Q['near_left_out']})")
print(R1Q["table"].T.to_string())
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
del ns, R1, R1P, R1Q, R1QP
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
Q2, Q2P = pool_test(ns, "A_rank"), pool_test(ns, "A_rank_pos")


def decide(r_rot, r_pool):
    return bool(r_rot["p"] <= 0.025 and r_pool["p"] <= 0.025)


def verdict_of(main, pos):
    return "SUPPORTED" if decide(*main) and decide(*pos) else "NOT SUPPORTED"


assert decide({"p": 0.001}, {"p": 0.001}) and decide({"p": 0.025}, {"p": 0.025}), "the rule does not pass at its bound"
assert not decide({"p": 0.001}, {"p": 0.026}) and not decide({"p": 0.026}, {"p": 0.001}), "the rule passes with one floor beaten"
assert not decide({"p": 0.001}, {"p": np.nan}) and not decide({"p": np.nan}, {"p": 0.001}), "the rule passes a NaN p"
ok_main, ok_pos = decide(C2, Q2), decide(C2P, Q2P)
print("C2 pool rows and positives per merged stratum (stratum = role * 100 + gap class * 10 + first calendar fifth; "
      "classes " + ", ".join(GAP_CLASSES) + f"); positives within 7 days of the next row, left out: {Q2['near_left_out']}")
print(Q2["table"].T.to_string())
for nm, r_, q_, ok in (("C2 MLB 2021-2023, last outing before an arm placement, no early hook [A_rank]", C2, Q2, ok_main),
                       ("C2 position-adjusted [A_rank_pos]", C2P, Q2P, ok_pos)):
    print(f"{nm}: AUC {r_['auc']:.4f}; 95% lower {r_['lo95']:.4f} (Bonferroni 0.625% {r_['lo_bonf']:.4f}); whole-group "
          f"floor mean {r_['floor'].mean():.4f}, 97.5th {np.percentile(r_['floor'], 97.5):.4f}, max {r_['floor'].max():.4f}, "
          f"p {r_['p']:.4f}; pool: stratified AUC {q_['auc']:.4f}, floor mean {q_['floor'].mean():.4f}, 97.5th "
          f"{np.percentile(q_['floor'], 97.5):.4f}, p {q_['p']:.4f} (positives {q_['pos']:,} in {q_['rows']:,} pool rows); "
          f"positives {r_['pos']:,}, rows {r_['rows']:,}, pitchers {r_['pitchers']:,} -> {'passes' if ok else 'does not pass'}")
verdict = verdict_of((C2, Q2), (C2P, Q2P))
print(f"{'FAKE LABELS (code test, not a result) ' if FAKE_RUN else ''}C2: {verdict} (both floors, both lines)")
if FAKE_RUN:  # fake placements are random days the pitcher keeps pitching after, so few reach the pool: the gates below
    # use 30 random pool rows as the pool's positives (the frozen fake labels stay on the rotation side)
    _Pf = pool_frame(ns["rows_m"], ns["season_first"]["mlb"], ns["season_last"]["mlb"])
    _yf, _ = pool_labels(ns["rows_m"], _Pf, ns["il_by"])
    _yf[_yf == 1.0] = 0.0
    _cand = np.flatnonzero(_yf == 0.0)
    _yf[np.random.default_rng(5).choice(_cand, min(30, len(_cand) // 5), replace=False)] = 1.0
    print(f"fake pool positives for the gates: {int((_yf == 1.0).sum())} of {len(_cand)} pool rows")
    QF = pool_test(ns, "A_rank", y=_yf)
    _rm = ns["rows_m"]
    _rm["_kscore"] = np.nan
    _rm.loc[QF["index"], "_kscore"] = QF["key"].astype(float)
    _k = pool_test(ns, "_kscore", y=_yf)
    assert np.array_equal(_k["key"], QF["key"]) and _k["p"] == 1.0, ("the pool floor does not hold position", _k["p"])
    _ks = QF["key"].astype(float)
    _pa = {round(ns["auc"](m_, _ks), 12) for m_ in _draw_masks(QF["y"], QF["key"], range(7000, 7100))}
    assert _pa == {round(ns["auc"](QF["y"], _ks), 12)}, ("a draw moves positives across strata", len(_pa))
    print(f"position-only score: stratified AUC {_k['auc']:.4f}, pool p {_k['p']:.4f} (must be 1); its plain AUC is the "
          f"same in 100 draws")
    _rm.drop(columns="_kscore", inplace=True)
    _yy, _kk = QF["y"], QF["key"]
    _sv = ns["rows_m"].loc[QF["index"], "A_rank"].to_numpy()
    _direct = sum(int(_yy[_kk == k_].sum()) * ns["auc"](_yy[_kk == k_], _sv[_kk == k_])
                  for k_ in np.unique(_kk) if _yy[_kk == k_].sum() > 0) / _yy.sum()
    assert abs(QF["auc"] - _direct) < 1e-12, ("the stratified AUC differs from the per-stratum AUCs", QF["auc"], _direct)
    print(f"stratified AUC equals the weighted per-stratum AUCs ({_direct:.12f})")
if FAKE_RUN:  # the verdict must be reachable by a planted signal, and only when both lines carry it
    _rm = ns["rows_m"]
    _mark = _rm.y_last.eq(1.0).to_numpy() | (_yf == 1.0)
    for _sc in ("A_rank", "A_rank_pos"):
        _rm[_sc + "_up"] = np.where(_mark, _rm[_sc] + 0.3, _rm[_sc])
        _rm[_sc + "_down"] = np.where(_mark, _rm[_sc] - 0.3, _rm[_sc])
    _up, _upp = p1_stat(ns, "A_rank_up"), p1_stat(ns, "A_rank_pos_up")
    _dn, _dnp = p1_stat(ns, "A_rank_down"), p1_stat(ns, "A_rank_pos_down")
    _qu, _qup = pool_test(ns, "A_rank_up", y=_yf), pool_test(ns, "A_rank_pos_up", y=_yf)
    _qd, _qdp = pool_test(ns, "A_rank_down", y=_yf), pool_test(ns, "A_rank_pos_down", y=_yf)
    print(f"planted +0.3: AUC {_up['auc']:.4f} / {_upp['auc']:.4f}, p {_up['p']:.4f} / {_upp['p']:.4f}, pool AUC {_qu['auc']:.4f} / "
          f"{_qup['auc']:.4f}, pool p {_qu['p']:.4f} / {_qup['p']:.4f}; planted -0.3: AUC {_dn['auc']:.4f} / {_dnp['auc']:.4f}, "
          f"pool AUC {_qd['auc']:.4f} / {_qdp['auc']:.4f}")
    assert verdict_of((_up, _qu), (_upp, _qup)) == "SUPPORTED", "the verdict cannot be reached by a planted signal"
    assert verdict_of((_up, _qu), (_dnp, _qdp)) == "NOT SUPPORTED" and verdict_of((_dn, _qd), (_upp, _qup)) == "NOT SUPPORTED", \
        "the verdict passes with one line against it"
    assert verdict_of((_dn, _qd), (_dnp, _qdp)) == "NOT SUPPORTED", "the verdict passes a planted reverse signal"
    _rm.drop(columns=[c for c in _rm.columns if c.endswith(("_up", "_down"))], inplace=True)
_P = pool_frame(ns["rows_m"], ns["season_first"]["mlb"], ns["season_last"]["mlb"])
_ya, _ncov = arm_vs_nonarm_labels(ns["rows_m"], _P, ns["il_lab"], ns["il_by"])
S2, S2P = pool_test(ns, "A_rank", y=_ya, rule=violates_both), pool_test(ns, "A_rank_pos", y=_ya, rule=violates_both)
print(f"reported (secondary, arm vs non-arm last outings, COVID-related rows left out: {_ncov}): stratified AUC {S2['auc']:.4f} / "
      f"{S2P['auc']:.4f}, floor mean {S2['floor'].mean():.4f} / {S2P['floor'].mean():.4f}, p {S2['p']:.4f} / {S2P['p']:.4f}; "
      f"arm {S2['pos']}, non-arm {S2['rows'] - S2['pos']} -> "
      f"{'arm-specific' if S2['p'] <= 0.025 and S2P['p'] <= 0.025 else 'not shown to be arm-specific'}")
if FAKE_RUN:  # the secondary's path on computable labels: the 30 fake pool positives against all pool 0 rows, its own rule
    _ys = np.where(_yf == 0.0, 0.0, np.nan)
    _ys[_yf == 1.0] = 1.0
    _sf = pool_test(ns, "A_rank", y=_ys, rule=violates_both)
    assert np.isfinite(_sf["auc"]) and np.isfinite(_sf["p"]) and _sf["dropped"] == 0, ("the secondary path", _sf["auc"], _sf["dropped"])
    print(f"secondary path on fake labels: stratified AUC {_sf['auc']:.4f}, p {_sf['p']:.4f}, strata {len(np.unique(_sf['key']))}")
    _nan = pool_test(ns, "A_rank", y=np.where(np.isfinite(_ys), 1.0, np.nan), rule=violates_both)
    assert np.isnan(_nan["auc"]) and np.isnan(_nan["p"]), "a statistic with no computable stratum must give p NaN"

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
                         "position_adjusted_lo95": C2P["lo95"], "position_adjusted_p": C2P["p"],
                         "pool_stratified_auc": Q2["auc"], "p_pool": Q2["p"], "position_adjusted_pool_stratified_auc": Q2P["auc"],
                         "position_adjusted_p_pool": Q2P["p"], "positives": C2["pos"], "pool_positives": Q2["pos"],
                         "secondary_arm_vs_nonarm_auc": S2["auc"], "secondary_p": S2["p"], "secondary_p_pos": S2P["p"],
                         "verdict": verdict}])
if FAKE_RUN:
    print("FAKE LABELS: code test only, no verdict written")
    print(summary.to_string(index=False))
else:
    summary.to_csv("after_confirm_verdict.csv", index=False)
    print(summary.to_string(index=False))
