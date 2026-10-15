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
          "rows_p1": 30492, "floor_mean": "0.5334", "floor_975": "0.5587", "game_log_out": 1}
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


def p1_stat(ns, score):
    rows_m, auc, rotated_group, floor_structure = ns["rows_m"], ns["auc"], ns["rotated_group"], ns["floor_structure"]
    np, N_BOOT = ns["np"], ns["N_BOOT"]
    no_hook = rows_m.early_hook.eq(0.0).to_numpy()
    df = rows_m[no_hook & rows_m.y_last.notna() & np.isfinite(rows_m[score])].reset_index(drop=True)
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
ok_main = bool(C2["lo95"] > 0.5 and C2["p"] <= 0.025)
ok_pos = bool(C2P["lo95"] > 0.5 and C2P["p"] <= 0.025)
for nm, r_, ok in (("C2 MLB 2021-2023, last outing before an arm placement, no early hook [A_rank]", C2, ok_main),
                   ("C2 position-adjusted [A_rank_pos]", C2P, ok_pos)):
    print(f"{nm}: AUC {r_['auc']:.4f}; 95% lower {r_['lo95']:.4f} (Bonferroni 0.625% {r_['lo_bonf']:.4f}); whole-group "
          f"floor mean {r_['floor'].mean():.4f}, 97.5th {np.percentile(r_['floor'], 97.5):.4f}, max {r_['floor'].max():.4f}, "
          f"p {r_['p']:.4f}; positives {r_['pos']:,}, rows {r_['rows']:,}, pitchers {r_['pitchers']:,} -> "
          f"{'passes' if ok else 'does not pass'}")
verdict = "SUPPORTED" if ok_main and ok_pos else "NOT SUPPORTED"
print(f"C2: {verdict} (both lines must pass)")

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
                         "position_adjusted_lo95": C2P["lo95"], "position_adjusted_p": C2P["p"], "positives": C2["pos"],
                         "verdict": verdict}])
if FAKE_RUN:
    print("FAKE LABELS: code test only, no verdict written")
    print(summary.to_string(index=False))
else:
    summary.to_csv("after_confirm_verdict.csv", index=False)
    print(summary.to_string(index=False))
